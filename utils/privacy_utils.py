import torch
import numpy as np

def add_laplace_noise(tensor, noise_scale=1.0):
    noise = torch.from_numpy(np.random.laplace(0, noise_scale, tensor.size())).float()
    noisy_tensor = tensor + noise
    return noisy_tensor

def add_laplace_noise_device(tensor, noise_scale=1.0):
    noise = torch.from_numpy(np.random.laplace(0, noise_scale, tensor.size())).float().to(tensor.device)
    noisy_tensor = tensor + noise
    return noisy_tensor