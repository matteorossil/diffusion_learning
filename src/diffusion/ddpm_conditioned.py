from typing import Tuple, Optional

import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn
from copy import deepcopy
from torchvision.utils import save_image
import os

from eps_models.unet_conditioned import UNet, UNet2

from utils import gather # Used for Image Data
# from utils import gather2d as gather # Used for Gaussian 2D Data

import torchvision.transforms as T

class BaseDenoiseDiffusion:
    def __init__(self, n_steps: int, device: torch.device, beta_0: float, beta_T: float):
        """
        * `eps_model` is $\textcolor{lightgreen}{\epsilon_\theta}(x_t, t)$ model
        * `n_steps` is $t$
        * `device` is the device to place constants on
        """
        super().__init__()

        # device id
        self.device = device

        # Create linearly increasing variance schedule
        self.beta = torch.linspace(beta_0, beta_T, n_steps).to(device)

        # $\alpha_t = 1 - \beta_t$
        self.alpha = 1. - self.beta
        # $\bar\alpha_t = \prod_{s=1}^t \alpha_s$
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        # $T$
        self.n_steps = n_steps
        # $\sigma^2 = \beta$
        self.sigma2 = self.beta
        self.has_copy = False

        self.R = []
        self.G = []
        self.B = []

        self.R_std = []
        self.G_std = []
        self.B_std = []

        self.R_min = []
        self.R_max = []

        self.G_min = []
        self.G_max = []

        self.B_min = []
        self.B_max = []

    def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        #### Get q(x_t|x_0) distribution
        """
        # compute mean
        mean = gather(self.alpha_bar, t) ** 0.5 * x0
        #save_image(x0, os.path.join(self.path, f'x0_{self.t_step}_{t.item()}.png'))
        #save_image(mean, os.path.join(self.path, f'mean_{self.t_step}_{t.item()}.png'))

        # compute variance
        var = 1 - gather(self.alpha_bar, t)
        return mean, var

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor]):
        """
        #### Sample from $q(x_t|x_0)$
        """
        # get q(x_t|x_0)
        mean, var = self.q_xt_x0(x0, t)

        # Sample from q(x_t|x_0)
        return mean + (var ** 0.5) * eps

    def loss(self, img: torch.Tensor, cond: torch.Tensor, noise: Optional[torch.Tensor] = None):
        raise NotImplementedError
    
    def p_sample(self, xt: torch.Tensor, cond: torch.Tensor, t: torch.Tensor):
        raise NotImplementedError


class DenoiseDiffusion(BaseDenoiseDiffusion):
    """
    This diffusion model takes input the concatenated image and decoded embedding from the conditioner
    
    init, emb = self.diffusion.conditioner(img)
    img_c = torch.cat((img, init), dim=1) # init is conditioning, which is same size as image
    eps_theta = self.eps_model(img_c, t)
    return F.mse_loss(noise, eps_theta)

    """
    def __init__(self, eps_model: nn.Module, conditioner: nn.Module, n_steps: int, device: torch.device, beta_0: float, beta_T: float):
        super().__init__(n_steps, device, beta_0, beta_T)

        assert type(eps_model.module).__name__ == "UNet"
        # denoiser model
        self.eps_model = eps_model

        # initial conditioner
        self.conditioner = conditioner

    def loss(self, img: torch.Tensor, cond: torch.Tensor, noise: Optional[torch.Tensor] = None):
        """
        #### Simplified Loss

        $$L_{\text{simple}}(\theta) = \mathbb{E}_{t,x_0, \epsilon} \Bigg[ \bigg\Vert
        \epsilon - \textcolor{lightgreen}{\epsilon_\theta}(\sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon, t)
        \bigg\Vert^2 \Bigg]$$
        """
        # Get batch size
        batch_size = img.shape[0]

        # Get random $t$ for each sample in the batch
        t = torch.randint(0, self.n_steps, (batch_size,), device=img.device, dtype=torch.long)
        #print("sampled time:", t.item())

        # generate noise if None
        if noise is None:
            noise = torch.randn_like(img)

        xt = self.q_sample(img, t, eps=noise)
        #save_image(xt, os.path.join(self.path, f'xt_{self.t_step}_{t.item()}.png'))
        #save_image(noise, os.path.join(self.path, f'noise_{self.t_step}_{t.item()}.png'))

        # concatenate channel wise for conditioning
        xt_ = torch.cat((xt, cond), dim=1) # or xt_ = torch.cat((xt, init), dim=1), different conditioning

        # predict noise
        eps_theta = self.eps_model(xt_, t)
        #save_image(eps_theta, os.path.join(self.path, f'predicted_noise_{self.t_step}_{t.item()}.png'))

        return F.mse_loss(noise, eps_theta)
    
    def p_sample(self, xt: torch.Tensor, cond: torch.Tensor, t: torch.Tensor):
        """
        #### Sample from p_theta(x_t-1|x_t)
        """
        # $\textcolor{lightgreen}{\epsilon_\theta}(x_t, t)$
        xt_y = torch.cat((xt, cond), dim=1)
        
        eps_theta = self.eps_model(xt_y, t)
        
        alpha_bar = gather(self.alpha_bar, t)
        # $\alpha_t$
        alpha = gather(self.alpha, t)
        # $\frac{\beta}{\sqrt{1-\bar\alpha_t}}$
        eps_coef = (1 - alpha) / (1 - alpha_bar) ** .5
        # $$\frac{1}{\sqrt{\alpha_t}} \Big(x_t -
        #      \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\textcolor{lightgreen}{\epsilon_\theta}(x_t, t) \Big)$$
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        # $\sigma^2$
        var = gather(self.sigma2, t)

        # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
        eps = torch.randn(xt.shape, device=xt.device)
        # Sample
        # import pdb; pdb.set_trace()

        return mean + (var ** .5) * eps

    def save_model_copy(self):
        with torch.no_grad():
            self.eps_model_copy = deepcopy(self.eps_model)
        self.has_copy = True

    def generate_auxilary_data(self, x0: torch.Tensor):
        with torch.no_grad():
            batch_size, image_size = x0.shape[0], x0.shape[1:]
            self.aux_t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long)
            self.aux_data = torch.randn((batch_size, *image_size), dtype=x0.dtype, device=x0.device) # TODO: Other methods of generating auxilary data
            self.aux_noise = self.eps_model_copy(self.aux_data, self.aux_t)

    def distillation_loss(self):
        eps_theta = self.eps_model(self.aux_data, self.aux_t)
        return F.l1_loss(self.aux_noise, eps_theta)


class DenoiseDiffusion2(BaseDenoiseDiffusion):
    """
    This diffusion model takes input the image and decoded embedding from the conditioner
    
    init, emb = self.diffusion.conditioner(img)
    eps_theta = self.eps_model(xt_y, t, emb) # emb is conditioning, which is the representation from conditioner
    return F.mse_loss(noise, eps_theta)

    """
    def __init__(self, eps_model: nn.Module, conditioner: nn.Module, n_steps: int, device: torch.device, beta_0: float, beta_T: float):
        super().__init__(n_steps, device, beta_0, beta_T)
        
        
        # assert type(eps_model.module).__name__ == "UNet2"

        # denoiser model
        self.eps_model = eps_model

        # initial conditioner
        self.conditioner = conditioner

    def loss(self, img: torch.Tensor, cond: torch.Tensor, noise: Optional[torch.Tensor] = None):
        """
        #### Simplified Loss

        $$L_{\text{simple}}(\theta) = \mathbb{E}_{t,x_0, \epsilon} \Bigg[ \bigg\Vert
        \epsilon - \textcolor{lightgreen}{\epsilon_\theta}(\sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon, t)
        \bigg\Vert^2 \Bigg]$$
        """
        # Get batch size
        batch_size = img.shape[0]

        # Get random $t$ for each sample in the batch
        t = torch.randint(0, self.n_steps, (batch_size,), device=img.device, dtype=torch.long)
        #print("sampled time:", t.item())

        # generate noise if None
        if noise is None:
            noise = torch.randn_like(img)

        xt = self.q_sample(img, t, eps=noise)
        #save_image(xt, os.path.join(self.path, f'xt_{self.t_step}_{t.item()}.png'))
        #save_image(noise, os.path.join(self.path, f'noise_{self.t_step}_{t.item()}.png'))

        # predict noise
        eps_theta = self.eps_model(xt, t, cond)
        #save_image(eps_theta, os.path.join(self.path, f'predicted_noise_{self.t_step}_{t.item()}.png'))
        return F.mse_loss(noise, eps_theta)
    
    def p_sample(self, xt: torch.Tensor, cond: torch.Tensor, t: torch.Tensor):
        """
        #### Sample from p_theta(x_t-1|x_t)
        """
        # $\textcolor{lightgreen}{\epsilon_\theta}(x_t, t)$

        eps_theta = self.eps_model(xt, t, cond)
        
        alpha_bar = gather(self.alpha_bar, t)
        # $\alpha_t$
        alpha = gather(self.alpha, t)
        # $\frac{\beta}{\sqrt{1-\bar\alpha_t}}$
        eps_coef = (1 - alpha) / (1 - alpha_bar) ** .5
        # $$\frac{1}{\sqrt{\alpha_t}} \Big(x_t -
        #      \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\textcolor{lightgreen}{\epsilon_\theta}(x_t, t) \Big)$$
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        # $\sigma^2$
        var = gather(self.sigma2, t)

        # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
        eps = torch.randn(xt.shape, device=xt.device)
        # Sample
        # import pdb; pdb.set_trace()

        return mean + (var ** .5) * eps

def show_image(tensor):
    transform = T.ToPILImage()
    img = transform(tensor.squeeze())
    img.show()