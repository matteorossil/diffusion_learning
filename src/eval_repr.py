# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------

import argparse
import json
import os
import numpy as np
from pathlib import Path
import omegaconf
import time

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torchvision import models
import torchvision

from torchvision import transforms as pth_transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from utils import fix_random_seeds, init_distributed_mode, get_rank, accuracy, get_train_test_datasets,\
                     MetricLogger, SmoothedValue, get_world_size, is_main_process

from eps_models.init_light import MyResNet as Init3
from eps_models.init_vqvae import VQVAE

def get_args_parser():
    parser = argparse.ArgumentParser('Linear probing for image classification', add_help=False)
    parser.add_argument('--batch_size', default=512, type=int, help='total batch size')
    parser.add_argument('--epochs', default=100, type=int)

    # Model parameters
    parser.add_argument('--resume', default='', help='resume from checkpoint')

    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=0.0005, metavar='LR', help='learning rate (absolute lr)')

    # Dataset parameters
    parser.add_argument('--num_labels', default=1000, type=int, help='number of classes')
    parser.add_argument('--output_dir', default='./output_dir', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--data_path', default='', type=str)
    parser.add_argument('--split', default=False, action='store_true', help='whether to manually split dataset into train-val')
    parser.add_argument('--subsample', default=False, action='store_true', help='whether to subsample the data')

    # training parameters
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument("--save_prefix", default="", type=str, help="""prefix for saving checkpoint and log files""")
    parser.add_argument("--frac_retained", default=1.0, type=float, choices=[0.010147, 0.02, 0.03, 0.05, 0.1, 1.0], help="""Fraction of train data retained for linear probing""")

    return parser


def main(args):
    init_distributed_mode(args)
    fix_random_seeds(args.random_seed)
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    resolved_args = omegaconf.OmegaConf.to_container(args, resolve=True, throw_on_missing=True)
    print("{}".format(resolved_args).replace(', ', ',\n'))

    device = torch.device(args.device)
    cudnn.benchmark = True

    # ============ building network ... ============
    
    # Load weights to evaluate
    if args.arch == 'init_light':
        model = Init3()
        model.cuda()
        model.eval()
        embed_dim = 384
        ckpt_initc_path = f'{args.ckpt_path}/ckpt_initc_{args.ckpt_step}.pt'
        state_dict = torch.load(ckpt_initc_path, map_location="cpu")
        msg = model.load_state_dict(state_dict, strict=False)
        model = model.encoder # Remove decoder
        print('Pretrained weights found at {} and loaded with msg: {}'.format(ckpt_initc_path, msg))
    elif args.arch == 'vqvae':
        assert args.dataset == "cifar10"
        model = VQVAE(input_channels=3, hidden_channels=16, embedding_dim=384, num_embeddings=64, commitment_cost=0.25)
        model.cuda()
        model.eval()
        embed_dim = 384
        ckpt_initc_path = f'{args.ckpt_path}/ckpt_initc_{args.ckpt_step}.pt'
        state_dict = torch.load(ckpt_initc_path, map_location="cpu")
        msg = model.load_state_dict(state_dict, strict=False)
        model = model.encoder # Remove decoder
        print('Pretrained weights found at {} and loaded with msg: {}'.format(ckpt_initc_path, msg))

    elif args.arch == 'vqvae_unloaded':
        assert args.dataset == "cifar10"
        model = VQVAE(input_channels=3, hidden_channels=16, embedding_dim=384, num_embeddings=64, commitment_cost=0.25)
        model.cuda()
        model.eval()
        embed_dim = 384
        model = model.encoder # Remove decoder
        print('No weights loaded!')


    elif args.arch == 'init_light_unloaded':
        model = Init3()
        model.cuda()
        model.eval()
        embed_dim = 384
        model = model.encoder # Remove decoder
        print('No weights loaded!')
    elif args.arch == 'resnet_50':
        resnet50 = models.resnet50(pretrained=True)
        model = nn.Sequential(*list(resnet50.children())[:-2], nn.Flatten(1))
        model.cuda()
        model.eval()
        embed_dim = 100352
        print('Pretrained resnet50 weights loaded')
    else:
        # Dummy model
        model = nn.Sequential(nn.AdaptiveAvgPool2d((16, 8)), nn.Flatten(1))
        model.cuda()
        model.eval()
        embed_dim = 384

    linear_classifier = LinearClassifier(embed_dim, num_labels=args.num_labels)
    linear_classifier = linear_classifier.cuda()
    linear_classifier = nn.parallel.DistributedDataParallel(linear_classifier, device_ids=[args.gpu])

    # ============ preparing data ... ============
    # validation transforms
    val_transform = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # training transforms
    train_transform = pth_transforms.Compose([
        pth_transforms.RandomResizedCrop(224),
        pth_transforms.RandomHorizontalFlip(),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    if args.dataset == "cifar10":
        dataset_val = torchvision.datasets.CIFAR10(root=args.data_path, train=False,
                                        download=True, transform=train_transform)

        evens = list(range(0, len(dataset_val), 2))
        odds = list(range(1, len(dataset_val), 2))
        train_dataset = torch.utils.data.Subset(dataset_val, evens)
        val_dataset = torch.utils.data.Subset(dataset_val, odds)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                    shuffle=True, num_workers=2)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                                    shuffle=False, num_workers=args.num_workers)
    elif args.subclass_sampling:
        train_dataset, val_dataset = get_train_test_datasets(args, train_transform, val_transform)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True, seed=args.random_seed)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)
    else:
        train_path = os.path.join(args.data_path, 'train')
        val_path = os.path.join(args.data_path, 'val')
        val_dataset = ImageFolder(val_path, transform=val_transform)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        train_dataset = ImageFolder(train_path, transform=train_transform)

        # few-shot finetuning
        if args.frac_retained < 1.0:
            print('Fraction of train data retained:', args.frac_retained)
            num_train = len(train_dataset)
            indices = list(range(num_train))
            np.random.seed(0)
            np.random.shuffle(indices)
            train_idx = indices[:int(args.frac_retained * num_train)]
            # train_dataset = torch.utils.data.Subset(train_dataset, train_idx)
            train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)
        else:
            print('Using all of train data')
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True, seed=args.random_seed)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=True, sampler=train_sampler)    
    

    print(f"Data loaded with {len(train_dataset)} train and {len(val_dataset)} val imgs.")
    print(f"{len(train_loader)} train and {len(val_loader)} val iterations per epoch.")

    if args.eval:
        state_dict = torch.load(args.lin_ckpt_path, map_location="cpu")
        msg = linear_classifier.load_state_dict(state_dict['state_dict'], strict=False)
        print('Linear classifier weights found at {} and loaded with msg: {}'.format(args.lin_ckpt_path, msg))
        test_stats = validate_network(val_loader, model, linear_classifier)
        print(f"Accuracy of the network on the {len(val_dataset)} test images: {test_stats['acc1']:.1f}%")
        return


    # set optimizer
    optimizer = torch.optim.SGD(
        linear_classifier.parameters(),
        args.lr * (args.batch_size * get_world_size()) / 256., # linear scaling rule
        momentum=0.9,
        weight_decay=0, # we do not apply weight decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

    start_time = time.time()
    best_acc = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        # train_loader.sampler.set_epoch(epoch)

        train_stats = train(model, linear_classifier, optimizer, train_loader, epoch)
        scheduler.step()

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            test_stats = validate_network(val_loader, model, linear_classifier)
            print(f"Accuracy at epoch {epoch} of the network on the {len(val_dataset)} test images: {test_stats['acc1']:.1f}%")
            best_acc = max(best_acc, test_stats["acc1"])
            print(f'Max accuracy so far: {best_acc:.2f}%')
            log_stats = {**{k: v for k, v in log_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()}}
        if is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": linear_classifier.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_acc": best_acc,
            }
            torch.save(save_dict, os.path.join(args.output_dir, "checkpoint.pth"))
            if epoch % args.save_freq == 0 or epoch == args.epochs - 1:
                torch.save(save_dict, os.path.join(args.output_dir, "checkpoint_{}.pth".format(epoch)))
            print("Elapsed time: ", (time.time() - start_time)/60, " minutes")
    print("Training of the supervised linear classifier on frozen features completed.\n"
                "Top-1 test accuracy: {acc:.1f}".format(acc=best_acc))


def train(model, linear_classifier, optimizer, loader, epoch):
    linear_classifier.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    for (inp, target) in metric_logger.log_every(loader, 20, header):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        with torch.no_grad():
            output = model(inp)
        output = linear_classifier(output)

        # compute cross entropy loss
        loss = nn.CrossEntropyLoss()(output, target)

        # compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # step
        optimizer.step()

        # log 
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate_network(val_loader, model, linear_classifier):
    linear_classifier.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'
    for inp, target in metric_logger.log_every(val_loader, 20, header):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        with torch.no_grad():
            inp = model(inp)
        
        output = linear_classifier(inp)
        # print("output")
        # print(torch.argmax(output, dim=1)[:10])
        # print("target")
        # print(target[:10])
        loss = nn.CrossEntropyLoss()(output, target)

        if linear_classifier.module.num_labels >= 5:
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
        else:
            acc1, = accuracy(output, target, topk=(1,))

        batch_size = inp.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        if linear_classifier.module.num_labels >= 5:
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    if linear_classifier.module.num_labels >= 5:
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    else:
        print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, losses=metric_logger.loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}    


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)