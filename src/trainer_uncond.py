# Modules
from diffusion.ddpm_unconditioned import DenoiseDiffusion
from eps_models.unet_unconditioned import UNet as Denoiser
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
        self.image_size: int = 32
        # Number of channels in the initial feature map
        self.n_channels: int = 32
        # The list of channel numbers at each resolution.
        # The number of channels is `channel_multipliers[i] * n_channels`
        self.channel_multipliers: List[int] = [1, 2, 3, 4]
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
        # Initial learning rate
        self.learning_rate: float = argv.lr
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
        # training step start
        self.step = self.ckpt_step
        # path
        self.exp_path = get_exp_path(path=self.store_checkpoints)
        # dataset
        self.dataset = argv.dataset
        # Sample only
        self.sample_only = argv.sample_only


    def init_train(self, rank: int, world_size: int):
        # gpu id
        self.gpu_id = rank
        # world_size
        self.world_size = world_size

        self.denoiser = Denoiser(
            image_channels= self.image_channels,
            n_channels=self.n_channels,
            ch_mults=self.channel_multipliers,
            is_attn=self.is_attention
        ).to(self.gpu_id)

        self.denoiser = DDP(self.denoiser, device_ids=[self.gpu_id])

        # load checkpoints
        if self.ckpt_step != 0:
            checkpoint_d = torch.load(self.ckpt_denoiser)
            self.denoiser.module.load_state_dict(checkpoint_d)

        # Create DDPM class
        self.diffusion = DenoiseDiffusion(
                eps_model=self.denoiser,
                n_steps=self.n_steps,
                device=self.gpu_id,
                beta_0=self.beta_0,
                beta_T=self.beta_T
        )

        # Num params of models
        params_denoiser = list(self.denoiser.parameters())
        num_params_denoiser = sum(p.numel() for p in params_denoiser if p.requires_grad)
        print(num_params_denoiser)

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

            dataset_train = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
            sampler = DistributedSampler(dataset_train, shuffle=True, seed=self.seed)
            self.dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=self.batch_size // self.world_size, 
                                                    pin_memory=False, num_workers=self.num_workers, sampler=sampler)

        # Create optimizers
        self.optimizer = torch.optim.AdamW(self.denoiser.parameters(), lr=self.learning_rate, weight_decay= self.weight_decay_rate, betas=self.betas)

    def sample_(self, name):
        
        if self.dataset == "saycam":
            dataset_val = ImageFolder(self.dataset_v, transform=self.transform_val)
            dataloader = DataLoader(dataset=dataset_val, batch_size=self.n_samples, num_workers=0, drop_last=False, shuffle=True, pin_memory=False)
        elif self.dataset == "cifar10":
            dataset_val = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=self.transform_val)
            dataloader = torch.utils.data.DataLoader(dataset_val, batch_size=self.n_samples,
                                                    shuffle=True, num_workers=0)

        with torch.no_grad():

            img = next(iter(dataloader))
            img = img[0].to(self.gpu_id)

            # Sample X from Gaussian Noise
            X = torch.randn([self.n_samples, self.image_channels, img.shape[2], img.shape[3]], device=self.gpu_id)

            # Remove noise for $T$ steps
            for t_ in range(self.n_steps):
                    
                # e.g. t_ from 999 to 0 for 1_000 time steps
                t = self.n_steps - t_ - 1

                # create a t for every sample in batch
                t_vec = X.new_full((self.n_samples,), t, dtype=torch.long)

                # take one denoising step
                X = self.diffusion.p_sample(X, t_vec)
            
            if self.dataset == "cifar10":
                img = self.inv_transform(img)
                X = self.inv_transform(X)
            #save initial image
            save_image(img, os.path.join(self.exp_path, f'img_step{name}.png'))

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

            # Make the gradients zero
            self.optimizer.zero_grad()

            # final loss
            loss = self.diffusion.loss(img)

            if self.gpu_id == 0 and self.step % self.log_interval == 0:
                logging.info("elapsed: {}, step: {}, mem: {:.03f}GB, GPUmem: {:.03f}GB, loss: {:.04f}".format(
                        str(datetime.timedelta(seconds=(time.time() - self.start_time)))[:-3],
                        self.step, get_proc_mem(), get_GPU_mem(), loss.item()))

            # Compute gradients
            loss.backward()

            # clip gradients
            nn.utils.clip_grad_norm_(self.denoiser.parameters(), 0.01)

            # Take an optimization step
            self.optimizer.step()

            # Track the loss with WANDB
            if self.wandb and self.gpu_id == 0:
                wandb.log({'loss': loss}, step=self.step)

            if (self.sample) and (self.gpu_id == 0) and self.step % self.sampling_interval == 0:
                self.sample_(self.step)
            
            if self.step % self.save_interval == 0:
                torch.save(self.denoiser.module.state_dict(), os.path.join(self.exp_path, f'ckpt_denoiser_{self.step}.pt'))


    def run(self):
        # sample at step 0
        if self.sample_only:
            for epoch in range(self.epochs):
                 self.sample_(epoch)
            return

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
            project="unconditional_diffusion",
            name=args.name,
            config=
            {
            "GPUs": args.world_size,
            "GPU Type": torch.cuda.get_device_name(args.rank),
            "Denoiser params": trainer.num_params_denoiser,
            "Denoiser LR": trainer.learning_rate,
            "Batch size": trainer.batch_size,
            "Sample size": trainer.n_samples,
            "L2 Loss": trainer.alpha > 0,
            "L2 param": trainer.alpha,
            "Regularizer": trainer.threshold <= 0.5,
            "Regularizer Threshold": trainer.threshold,
            "Dataset_t": trainer.dataset_t,
            "Dataset_v": trainer.dataset_v,
            "Path": trainer.exp_path,
            "Port": argv.port,
            "Ckpt step": trainer.ckpt_step,
            "Ckpt path": argv.ckpt_path,
            "Workers": trainer.num_workers,
            "Dataset multiplier": trainer.multiplier,
            "Sampling interval": trainer.sampling_interval,
            "Random seed eval": trainer.seed,
            "Sampling": trainer.sample,
            }
        )

    trainer.run() # perform training
    destroy_process_group()