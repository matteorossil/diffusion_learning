# Modules
from eps_models.unet_conditioned import UNet as Denoiser, UNet2 as Denoiser2 #
from eps_models.init_reps import UNet as Init
from eps_models.init_conv import ConvNetConditioner as Init2
from eps_models.init_light import MyResNet as Init3
from eps_models.init_vqvae import VQVAE
from diffusion.ddpm_conditioned import DenoiseDiffusion, DenoiseDiffusion2 #
from utils import init_distributed_mode, fix_random_seeds, get_proc_mem, get_GPU_mem

# Torch
import torch
from torch import nn
import torchvision
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
import torch.nn.functional as F

# Numpy
import numpy as np
#from numpy import savetxt

# Other
import os
from typing import List
from pathlib import Path
import datetime
import wandb
import matplotlib.pyplot as plt
import argparse
import pickle
import omegaconf
import time
import logging

# DDP
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import destroy_process_group

def get_exp_path(path=''):
    exp_path = os.path.join(path, datetime.datetime.now().strftime("%m%d%Y_%H%M%S"))
    Path(exp_path).mkdir(parents=True, exist_ok=True)
    return exp_path

class Trainer():
    """
    ## Configurations
    """
    def __init__(self, argv):
        # Number of channels in the image. 3 for RGB.
        self.image_channels: int = 3
        # Image size
        self.image_size: int = 224
        # Number of channels in the initial feature map
        self.n_channels: int = 32
        # The list of channel numbers at each resolution.
        # The number of channels is `channel_multipliers[i] * n_channels`
        self.channel_multipliers: List[int] = [1, 2, 2, 3]
        # The list of booleans that indicate whether to use attention at each resolution
        self.is_attention: List[int] = [False, False, False, False]
        # Number of time steps $T$
        self.n_steps: int = 1_000
        # noise scheduler Beta_0
        self.beta_0 = 1e-6 # 0.000001
        # noise scheduler Beta_T
        self.beta_T = 1e-2 # 0.01
        # Batch size
        self.batch_size: int = argv.batch_size
        # L2 loss
        self.alpha = argv.l2_loss
        # Threshold Regularizer
        self.threshold = argv.threshold
        # Learning rate D
        self.learning_rate: float = argv.d_lr
        # Learning rate G
        self.learning_rate_init: float = argv.g_lr
        # Weight decay rate
        self.weight_decay_rate: float = 1e-3
        # ema decay
        self.betas = (0.9, 0.999)
        # Number of training epochs
        self.epochs: int = argv.epochs
        # Number of samples (evaluation)
        self.n_samples: int = argv.sample_size
        # Use wandb
        self.wandb: bool = argv.wandb
        # load from a checkpoint
        self.ckpt_step: int = argv.ckpt_step
        # paths 
        self.store_checkpoints: str = argv.output_dir
        self.dataset_t: str = argv.dataset_t
        self.dataset_v: str = argv.dataset_v
        self.ckpt_denoiser: str = f'{argv.ckpt_path}/ckpt_denoiser_{self.ckpt_step}.pt'
        self.ckpt_initc: str = f'{argv.ckpt_path}/ckpt_initc_{self.ckpt_step}.pt'
        self.ckpt_metrics_: str = f'{argv.ckpt_path}/metrics_step{self.ckpt_step}.pt'

        # multiplier for virtual dataset
        self.multiplier = argv.multiplier
        # dataloader workers
        self.num_workers = argv.num_workers
        # how often to sample
        self.sampling_interval = argv.sampling_interval
        # how often to log
        self.log_interval = argv.log_interval
        # how often to log
        self.save_interval = argv.save_interval
        # random seed for evaluation
        self.seed = argv.random_seed
        # whether to sample or not
        self.sample = argv.sample
        # import metrics from chpt
        self.ckpt_metrics = argv.ckpt_metrics
        # perform crops on eval
        self.crop_eval = argv.crop_eval
        # training step start
        self.step = self.ckpt_step
        # path
        self.exp_path = get_exp_path(path=self.store_checkpoints)
        # Use embedding
        self.use_emb = argv.use_emb
        # conditioner type
        self.cond_type = argv.cond_type
        # dataset
        self.dataset = argv.dataset
        # data path
        self.data_path = argv.data_path
        # Sample only
        self.sample_only = argv.sample_only

    def init_train(self, rank: int, world_size: int):
        # gpu id
        self.gpu_id = rank
        # world_size
        self.world_size = world_size

        if self.use_emb:
            self.denoiser = Denoiser2(
                image_channels= self.image_channels,
                n_channels=self.n_channels,
                ch_mults=self.channel_multipliers,
                is_attn=self.is_attention
            ).to(self.gpu_id)
        else:
            self.denoiser = Denoiser(
                image_channels= self.image_channels*2,
                n_channels=self.n_channels,
                ch_mults=self.channel_multipliers,
                is_attn=self.is_attention
            ).to(self.gpu_id)
        
        if self.cond_type == "unet":
            self.initc = Init(3, 4, (1, 2, 3, 1, 1, 1), (False, False, False, False, False, False), 1).to(self.gpu_id)
        elif self.cond_type == "light":
            self.initc = Init3().to(self.gpu_id)
        elif self.cond_type == "conv":
            self.initc = Init2().to(self.gpu_id)
        elif self.cond_type == "vqvae":
            assert self.dataset == "cifar10"
            self.initc = VQVAE(input_channels=3, hidden_channels=16, embedding_dim=384, num_embeddings=64, commitment_cost=0.25).to(self.gpu_id)

        self.denoiser = DDP(self.denoiser, device_ids=[self.gpu_id])
        self.initc = DDP(self.initc, device_ids=[self.gpu_id])

        # load checkpoints
        if self.ckpt_step != 0:
            checkpoint_d = torch.load(self.ckpt_denoiser)
            self.denoiser.module.load_state_dict(checkpoint_d)
            checkpoint_i = torch.load(self.ckpt_initc)
            self.initc.module.load_state_dict(checkpoint_i)

        # Create DDPM class
        if self.use_emb:
            self.diffusion = DenoiseDiffusion2(
                eps_model=self.denoiser,
                conditioner=self.initc,
                n_steps=self.n_steps,
                device=self.gpu_id,
                beta_0=self.beta_0,
                beta_T=self.beta_T
            )
        else:
            self.diffusion = DenoiseDiffusion(
                eps_model=self.denoiser,
                conditioner=self.initc,
                n_steps=self.n_steps,
                device=self.gpu_id,
                beta_0=self.beta_0,
                beta_T=self.beta_T
            )

        if self.dataset == "saycam":
            # simple augmentation
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=3), 
                transforms.RandomHorizontalFlip(), 
                transforms.ToTensor(), 
                #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            
            self.transform_val = transforms.Compose([
                transforms.ToTensor(), 
                #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            
            dataset_train = ImageFolder(self.dataset_t, transform=transform)
            # print(dataset_train)

            sampler = DistributedSampler(dataset_train, shuffle=True, seed=self.seed)
            self.dataloader_train = DataLoader(dataset=dataset_train,
                                                batch_size=self.batch_size // self.world_size, 
                                                num_workers=self.num_workers, #os.cpu_count() // 2,
                                                drop_last=True, 
                                                pin_memory=False,
                                                sampler=sampler)
        elif self.dataset == "cifar10":
            self.transform_val = transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            
            self.inv_transform = transforms.Compose(
                [transforms.Normalize((-1, -1, -1), (2, 2, 2))])

            dataset_train = torchvision.datasets.CIFAR10(root=self.data_path, train=True,
                                        download=True, transform=transform)
            sampler = DistributedSampler(dataset_train, shuffle=True, seed=self.seed)
            self.dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=self.batch_size // self.world_size, 
                                                    pin_memory=False, num_workers=self.num_workers, sampler=sampler)


        # Num params of models
        params_denoiser = list(self.denoiser.parameters())
        params_init = list(self.initc.parameters())
        num_params_denoiser = sum(p.numel() for p in params_denoiser if p.requires_grad)
        self.num_params_init = sum(p.numel() for p in params_init if p.requires_grad)
        print("Number of parameters conditioner : ", self.num_params_init)
        print("Number of parameters Denoiser : ", num_params_denoiser)

        # Create optimizers
        self.optimizer = torch.optim.AdamW(self.denoiser.parameters(), lr=self.learning_rate, weight_decay= self.weight_decay_rate, betas=self.betas)
        self.optimizer2 = torch.optim.AdamW(self.initc.parameters(), lr=self.learning_rate_init, weight_decay= self.weight_decay_rate, betas=self.betas)

    def sample_(self, name):
        
        if self.dataset == "saycam":
            dataset_val = ImageFolder(self.dataset_v, transform=self.transform_val)
            dataloader = DataLoader(dataset=dataset_val, batch_size=self.n_samples, num_workers=0, drop_last=False, shuffle=True, pin_memory=False)
        elif self.dataset == "cifar10":
            dataset_val = torchvision.datasets.CIFAR10(root=self.data_path, train=False,
                                        download=False, transform=self.transform_val)
            dataloader = torch.utils.data.DataLoader(dataset_val, batch_size=self.n_samples,
                                                    shuffle=True, num_workers=0)
        with torch.no_grad():

            img = next(iter(dataloader))
            
            img = img[0].to(self.gpu_id)
            #print(img.shape)

            # compute initial conditioning
            if self.cond_type == "vqvae": # Returns an extra loss
                if self.use_emb:
                    _, init, _ = self.diffusion.conditioner(img)
                else:
                    init, _, _ = self.diffusion.conditioner(img)
            else:
                if self.use_emb:
                    _, init = self.diffusion.conditioner(img) # init is the embedding
                else:
                    init, _ = self.diffusion.conditioner(img)

            # Sample X from Gaussian Noise
            X = torch.randn([self.n_samples, self.image_channels, img.shape[2], img.shape[3]], device=self.gpu_id)

            # Remove noise for $T$ steps
            for t_ in range(self.n_steps):
                    
                # e.g. t_ from 999 to 0 for 1_000 time steps
                t = self.n_steps - t_ - 1

                # create a t for every sample in batch
                t_vec = X.new_full((self.n_samples,), t, dtype=torch.long)

                # take one denoising step
                X = self.diffusion.p_sample(X, init, t_vec)
            
            if self.dataset == "cifar10":
                img = self.inv_transform(img)
                X = self.inv_transform(X)

            #save initial image
            save_image(img, os.path.join(self.exp_path, f'img_step{name}.png'))
            # save initial conditioning
            if not self.use_emb:
                save_image(init, os.path.join(self.exp_path, f'img_init_step{name}.png'))
            # save sampled residual
            save_image(X, os.path.join(self.exp_path, f'img_sampled_step{name}.png'))


    def train(self):
        """
        ### Train
        """

        # Iterate through the dataset
        for _, img in enumerate(self.dataloader_train):

            # Increment global step
            self.step += 1

            # Move data to device
            img = img[0].to(self.gpu_id)

            # save image
            #save_image(img, os.path.join(self.exp_path, f'img_train_step{self.step}.png'))

            # get initial prediction and loss
            if self.cond_type == "vqvae": # Returns an extra loss
                if self.use_emb:
                    init, emb, vqloss = self.diffusion.conditioner(img)
                    denoiser_loss = self.diffusion.loss(img, emb)
                    regression_loss = (F.mse_loss(img, init) + vqloss) * 0.1
                else:
                    init, emb, vqloss = self.diffusion.conditioner(img)
                    denoiser_loss = self.diffusion.loss(img, init)
                    regression_loss = (F.mse_loss(img, init) + vqloss) * 0.1
            else:
                if self.use_emb:
                    init, emb = self.diffusion.conditioner(img)
                    denoiser_loss = self.diffusion.loss(img, emb)
                    regression_loss = F.mse_loss(img, init) * 0.1
                else:    
                    init, _ = self.diffusion.conditioner(img)
                    denoiser_loss = self.diffusion.loss(img, init)
                    regression_loss = F.mse_loss(img, init) * 0.1

            #init = torch.cat((init, torch.zeros(img.shape[0], 2, img.shape[2], img.shape[3], device=self.gpu_id)), dim=1)
            #save_image(init, os.path.join(self.exp_path, f'init_step{self.step}.png'))
            # Make the gradients zero
            self.optimizer.zero_grad()
            self.optimizer2.zero_grad()

            # final loss
            loss = denoiser_loss + regression_loss

            if self.gpu_id == 0 and self.step % self.log_interval == 0:
                logging.info("elapsed: {}, step: {}, mem: {:.03f}GB, GPUmem: {:.03f}GB, loss: {:.04f}, D_loss: {:.4f}, G_loss: {:.4f}".format(
                        str(datetime.timedelta(seconds=(time.time() - self.start_time)))[:-3],
                        self.step, get_proc_mem(), get_GPU_mem(), loss.item(), denoiser_loss.item(), regression_loss.item()))

            # Compute gradients
            loss.backward()

            #print("############ GRAD OUTPUT ############")
            #print("Grad bias denoiser:", self.denoiser.module.final.bias.grad)
            #print("Grad bias init:", self.initc.module.final.bias.grad)

            # clip gradients
            nn.utils.clip_grad_norm_(self.denoiser.parameters(), 0.01)
            nn.utils.clip_grad_norm_(self.initc.parameters(), 0.01)

            # Take an optimization step
            self.optimizer.step()
            self.optimizer2.step()

            # Track the loss with WANDB
            if self.wandb and self.gpu_id == 0:
                wandb.log({'loss': loss.item()}, step=self.step)
                wandb.log({'Denoiser Loss': denoiser_loss.item()}, step=self.step)
                wandb.log({'Regression Loss': regression_loss.item()}, step=self.step)

            if (self.sample) and (self.gpu_id == 0) and self.step % self.sampling_interval == 0:
                self.sample_(self.step)
            
            if self.step % self.save_interval == 0:
                torch.save(self.denoiser.module.state_dict(), os.path.join(self.exp_path, f'ckpt_denoiser_{self.step}.pt'))
                torch.save(self.initc.module.state_dict(), os.path.join(self.exp_path, f'ckpt_initc_{self.step}.pt'))


    def run(self):

        if self.sample_only:
            for epoch in range(self.epochs):
                print("Sampling step : "+str(epoch))
                self.sample_(epoch)
            
            return


        # sample at step 0
        if (self.sample) and (self.gpu_id == 0):
            self.sample_(0)

        self.start_time = time.time()

        for epoch in range(self.epochs):
            self.dataloader_train.sampler.set_epoch(epoch)
            # train
            self.train()

def main(args):
    # print("{}".format(args).replace(', ', ',\n'))
    init_distributed_mode(args)
    fix_random_seeds(args.random_seed)
    resolved_args = omegaconf.OmegaConf.to_container(args, resolve=True, throw_on_missing=True)
    print("{}".format(resolved_args).replace(', ', ',\n'))

    trainer = Trainer(args)
    trainer.init_train(args.rank, args.world_size) # initialize trainer class

    #### Track Hyperparameters with WANDB####
    if trainer.wandb and args.rank == 0:
        
        wandb.init(
            project="diff_rep_learning",
            name=args.name,
            args=resolved_args
        )

    trainer.run() # perform training
    destroy_process_group()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--sample_size', type=int, default=32)
    parser.add_argument('--d_lr', type=float, default=1e-4)
    parser.add_argument('--g_lr', type=float, default=1e-4)
    parser.add_argument('--threshold', type=float, default=0.02)
    parser.add_argument('--l2_loss', type=float, default=0.)
    parser.add_argument('--dataset_t', type=str, default="gopro")
    parser.add_argument('--dataset_v', type=str, default="gopro_128")
    parser.add_argument('--ckpt_step', type=int, default=0)
    parser.add_argument('--ckpt_path', type=str, default="")
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--multiplier', type=int, default=1)
    parser.add_argument('--sampling_interval', type=int, default=10_000)
    parser.add_argument('--random_seed', type=int, default=7)
    parser.add_argument('--name', type=str, default="conditioned")
    parser.add_argument('--wandb', action="store_true")
    parser.add_argument('--sample', action="store_true")
    parser.add_argument('--ckpt_metrics', action="store_true")
    parser.add_argument('--crop_eval', action="store_true")
    argv = parser.parse_args()

    print('port:', argv.port, type(argv.port))
    print('batch_size:', argv.batch_size, type(argv.batch_size))
    print('sample_size:', argv.sample_size, type(argv.sample_size))
    print('d_lr:', argv.d_lr, type(argv.d_lr))
    print('g_lr:', argv.g_lr, type(argv.g_lr))
    print('threshold:', argv.threshold, type(argv.threshold))
    print('l2_loss:', argv.l2_loss, type(argv.l2_loss))
    print('dataset_t:', argv.dataset_t, type(argv.dataset_t))
    print('dataset_v:', argv.dataset_v, type(argv.dataset_v))
    print('ckpt_step:', argv.ckpt_step, type(argv.ckpt_step))
    print('ckpt_path:', argv.ckpt_path, type(argv.ckpt_path))
    print('num_workers:', argv.num_workers, type(argv.num_workers))
    print('multiplier:', argv.multiplier, type(argv.multiplier))
    print('sampling_interval:', argv.sampling_interval, type(argv.sampling_interval))
    print('random_seed:', argv.random_seed, type(argv.random_seed))
    print('name:', argv.name, type(argv.name))
    print('wandb:', argv.wandb, type(argv.wandb))
    print('hpc:', argv.hpc, type(argv.hpc))
    print('sample:', argv.sample, type(argv.sample))
    print('ckpt_metrics:', argv.ckpt_metrics, type(argv.ckpt_metrics))
    print('crop_eval:', argv.crop_eval, type(argv.crop_eval))

    world_size = torch.cuda.device_count() # how many GPUs available in the machine
    mp.spawn(main, args=(argv), nprocs=world_size)