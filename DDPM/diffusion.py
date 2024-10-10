import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import wandb

def extract(a, t, x_shape):
    """
    Extracts the tensor at the given time step.
    Args:
        a: A tensor contains the values of all time steps.
        t: The time step to extract.
        x_shape: The reference shape.
    Returns:
        The extracted tensor.
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def cosine_schedule(timesteps, s=0.008):
    """
    Defines the cosine schedule for the diffusion process
    Args:
        timesteps: The number of timesteps.
        s: The strength of the schedule.
    Returns:
        The computed alpha.
    """
    steps = timesteps + 1
    x = torch.linspace(0, steps, steps)
    alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    return torch.clip(alphas, 0.001, 1)


# normalization functions
def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


# DDPM implementation
class Diffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        channels=3,
        timesteps=1000,
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.model = model
        self.num_timesteps = int(timesteps)

        """
        Initializes the diffusion process.
            1. Setup the schedule for the diffusion process.
            2. Define the coefficients for the diffusion process.
        Args:
            model: The model to use for the diffusion process.
            image_size: The size of the images.
            channels: The number of channels in the images.
            timesteps: The number of timesteps for the diffusion process.
        """
        self.scheduler = cosine_schedule(self.num_timesteps).to(next(model.parameters()).device)
        self.scheduler_hat = torch.cumprod(self.scheduler, dim=0).to(next(model.parameters()).device)
        self.scheduler_hat_minusone = torch.cat([torch.tensor([1.0], device=self.scheduler.device), self.scheduler_hat[:-1]])
        # ###########################################################

    def noise_like(self, shape, device):
        """
        Generates noise with the same shape as the input.
        Args:
            shape: The shape of the noise.
            device: The device on which to create the noise.
        Returns:
            The generated noise.
        """
        noise = lambda: torch.randn(shape, device=device)
        return noise()

    # backward diffusion
    @torch.no_grad()
    def p_sample(self, x, t, t_index):
        """
        Samples from the reverse diffusion process at time t_index.
        Args:
            x: The initial image.
            t: a tensor of the time index to sample at.
            t_index: a scalar of the index of the time step.
        Returns:
            The sampled image.
        """
        alpha_t = extract(self.scheduler, t, x.shape).to(x.device)
        alpha_hat_t = extract(self.scheduler_hat, t, x.shape).to(x.device)
        alpha_hat_tminus = extract(self.scheduler_hat_minusone, t, x.shape).to(x.device)

        pred_noise = self.model(x, t)
       
        sqrt_alpha_hat = torch.sqrt(alpha_hat_t)
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - alpha_hat_t)
        x0_est = (x - sqrt_one_minus_alpha_hat * pred_noise) / sqrt_alpha_hat
        x0_est = torch.clamp(x0_est, -1.0, 1.0)

        sqrt_alpha = torch.sqrt(alpha_t)
        one_minus_alpha_hat_prev = 1 - alpha_hat_tminus
        one_minus_alpha_hat = 1 - alpha_hat_t

        sqrt_alpha_hat_prev = torch.sqrt(alpha_hat_tminus)
        one_minus_alpha = 1 - alpha_t
        
        mu_t = ((sqrt_alpha * one_minus_alpha_hat_prev / one_minus_alpha_hat) * x + 
                (sqrt_alpha_hat_prev * one_minus_alpha / one_minus_alpha_hat) * x0_est)
                
        if t_index == 0:
            return mu_t
        else:
            post_var = (one_minus_alpha_hat_prev / one_minus_alpha_hat) * one_minus_alpha
            z = self.noise_like(x.shape, x.device)
            return mu_t + torch.sqrt(post_var) * z

        # ####################################################

    @torch.no_grad()
    def p_sample_loop(self, img):
        """
        Samples from the noise distribution at each time step.
        Args:
            img: The initial image that randomly sampled from the noise distribution.
        Returns:
            The sampled image.
        """
        b = img.shape[0]
        for t_index in torch.arange(self.num_timesteps-1, -1, -1, device=img.device):
            t = torch.full((b,), t_index, device=img.device, dtype=torch.long)
            img = self.p_sample(img, t, t_index)
        img = torch.clamp(img, -1.0, 1.0)
        return unnormalize_to_zero_to_one(img)
        # ####################################################

    @torch.no_grad()
    def sample(self, batch_size):
        """
        Samples from the noise distribution at each time step.
        Args:
            batch_size: The number of images to sample.
        Returns:
            The sampled images.
        """
        self.model.eval()
        img_shape = (batch_size, self.channels, self.image_size, self.image_size)
        img = self.noise_like(img_shape, device=next(self.model.parameters()).device)
        return self.p_sample_loop(img)

    # forward diffusion
    def q_sample(self, x_0, t, noise):
        """
        Samples from the noise distribution at time t. Simply apply alpha interpolation between x_0 and noise.
        Args:
            x_0: The initial image.
            t: The time index to sample at.
            noise: The noise tensor to sample from.
        Returns:
            The sampled image.
        """
        alpha_hat_t = extract(self.scheduler_hat, t, x_0.shape).to(x_0.device)
        x_t = torch.sqrt(alpha_hat_t) * x_0 + torch.sqrt(1 - alpha_hat_t) * noise
        return x_t

    def p_losses(self, x_0, t, noise):
        """
        Computes the loss for the forward diffusion.
        Args:
            x_0: The initial image.
            t: The time index to compute the loss at.
            noise: The noise tensor to use.
        Returns:
            The computed loss.
        """
        x_t = self.q_sample(x_0, t, noise)
        pred_noise = self.model(x_t, t)
        loss = F.l1_loss(pred_noise, noise)
        return loss
        # ####################################################

    def forward(self, x_0, noise):
        """
        Computes the loss for the forward diffusion.
        Args:
            x_0: The initial image.
            noise: The noise tensor to use.
        Returns:
            The computed loss.
        """
        b, c, h, w, device, img_size, = *x_0.shape, x_0.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x_0, t, noise)
