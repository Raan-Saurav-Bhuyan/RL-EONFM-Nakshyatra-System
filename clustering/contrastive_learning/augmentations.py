import torch
import random

class Augmentation:
    """
    Applies stochastic augmentations (noise and masking) to tabular OPM data,
    as described in the SSL framework.
    """
    def __init__(self, noise_std=0.05, mask_ratio=0.1, device='cpu'):
        self.noise_std = noise_std
        self.mask_ratio = mask_ratio
        self.device = device
        self.transforms = [self._corrupt, self._mask]

    def _corrupt(self, x):
        """Feature Corruption (Noise Injection): Adds Gaussian noise."""
        noise = torch.randn_like(x) * self.noise_std
        return x + noise

    def _mask(self, x):
        """Feature Masking: Randomly zeros out a fraction of features."""
        mask = torch.bernoulli(torch.full_like(x, 1 - self.mask_ratio))
        return x * mask

    def __call__(self, x):
        """
        Generates two distinct augmented views of the input tensor by randomly
        sampling from the available transformations.
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32).to(self.device)

        t1 = random.choice(self.transforms)
        t2 = random.choice(self.transforms)

        return t1(x), t2(x)
