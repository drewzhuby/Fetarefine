"""Ablation Study 1: DiffUNet Conditional Denoising Diffusion Segmentation.

Adapted from the official BTCV branch to FeTA 7-foreground structure.
Uses 1000 linear diffusion timesteps during training, and fast 10-step DDIM (eta=0) sampling during inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .diffunet_vendor.unet.basic_unet import BasicUNetEncoder
from .diffunet_vendor.unet.basic_unet_denose import BasicUNetDe
from .diffunet_vendor.guided_diffusion.gaussian_diffusion import (
    LossType,
    ModelMeanType,
    ModelVarType,
    get_named_beta_schedule,
)
from .diffunet_vendor.guided_diffusion.resample import UniformSampler
from .diffunet_vendor.guided_diffusion.respace import SpacedDiffusion, space_timesteps


class DiffUNetSegmentation(nn.Module):
    """DiffUNet conditional diffusion segmentation model."""

    def __init__(
        self,
        features=(64, 64, 128, 256, 512, 64),
        ddim_steps: int = 10,
        timesteps: int = 1000,
    ):
        super().__init__()
        # Image condition encoder: 1-channel MRI -> feature embeddings
        self.embed_model = BasicUNetEncoder(3, 1, 2, features)
        # Denoiser U-Net: inputs 7-channel noisy state + 1-channel image + time embedding -> 7-channel prediction
        self.model = BasicUNetDe(
            3,
            8,
            7,
            features,
            act=("LeakyReLU", {"negative_slope": 0.1, "inplace": False}),
        )

        betas = get_named_beta_schedule("linear", timesteps)
        kwargs = dict(
            betas=betas,
            model_mean_type=ModelMeanType.START_X,
            model_var_type=ModelVarType.FIXED_LARGE,
            loss_type=LossType.MSE,
        )
        self.diffusion = SpacedDiffusion(use_timesteps=space_timesteps(timesteps, [timesteps]), **kwargs)
        self.sample_diffusion = SpacedDiffusion(use_timesteps=space_timesteps(timesteps, [ddim_steps]), **kwargs)
        self.sampler = UniformSampler(timesteps)

    def forward(self, image: torch.Tensor, target_onehot7: torch.Tensor) -> torch.Tensor:
        """Training forward pass: samples t, adds noise to target, predicts denoised state."""
        state = target_onehot7.float() * 2.0 - 1.0  # Scale to [-1, +1]
        t, _ = self.sampler.sample(image.shape[0], image.device)
        noise = torch.randn_like(state)
        noisy_state = self.diffusion.q_sample(state, t, noise=noise)

        embeddings = self.embed_model(image)
        pred_xstart = self.model(noisy_state, t=t, image=image, embeddings=embeddings)
        return pred_xstart

    @torch.inference_mode()
    def sample(self, image: torch.Tensor, seed: int = 20260918) -> torch.Tensor:
        """Inference forward pass using 10-step DDIM deterministic trajectory."""
        torch.manual_seed(seed)
        shape = (image.shape[0], 7, *image.shape[2:])
        noise = torch.randn(shape, device=image.device)
        embeddings = self.embed_model(image)

        def model_fn(x, t):
            return self.model(x, t=t, image=image, embeddings=embeddings)

        sample = self.sample_diffusion.ddim_sample_loop(
            model_fn,
            shape=shape,
            noise=noise,
            clip_denoised=True,
            eta=0.0,
            device=image.device,
        )
        return sample["model_output"]
