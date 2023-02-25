from einops import rearrange
import numpy as np
import requests
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import typing as t
import torch



def download_images(n: int, res: int) -> t.List[np.ndarray]:    
    url = f"https://picsum.photos/{res}/{res}"
    imgs = []
    for _ in range(n):
        response = requests.get(url)
        img = np.array(Image.open(BytesIO(response.content)))
        imgs.append(img)
    return np.array(imgs)


def normalize_image(image: np.ndarray) -> np.ndarray:
    assert image.dtype == np.uint8, "Image must be uint8"
    x = 2. * (image / 255) - 1 
    return x


def denormalize_image(x: np.ndarray) -> np.ndarray:
    assert x.dtype == np.float64, "Image must be float64"
    image = (np.clip(x, -1, 1) + 1) / 2 * 255
    return image.astype(np.uint8)


def X2imgs(X: torch.Tensor) -> np.ndarray:
    X = torch.clamp((X + 1.0) / 2.0, min=0.0, max=1.0)
    X = 255. * rearrange(X.cpu().numpy(), 'b c h w -> b h w c')
    imgs = X.astype(np.uint8)
    return imgs
    

def imgs2X(imgs: np.ndarray, device: torch.device) -> torch.Tensor:
    X = rearrange(imgs, 'b h w c -> b c h w') 
    X = 2. * (X/ 255.) - 1.
    X = torch.from_numpy(X).to(device)
    return X.float()



class ForwardDiffusionProcessor:

    def __init__(self,  n_total_timesteps: int = 1000, beta_start: float = 0.00085, beta_end: float = 0.012):
        self.n_total_timesteps = n_total_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self._precompute()

    def _precompute(self):
        self.total_timesteps = np.arange(self.n_total_timesteps) + 1
        self._beta = np.linspace(self.beta_start ** 0.5, self.beta_end ** 0.5, self.n_total_timesteps) ** 2
        self._alpha = 1 - self._beta
        self._alpha_bar = np.cumprod(self._alpha)
    
    def get_timesteps(self, n_denoising_timesteps: int) -> np.ndarray: 
        d_steps = self.n_total_timesteps // n_denoising_timesteps
        timesteps = np.arange(1, self.n_total_timesteps, d_steps)
        return timesteps

    def get_beta(self, step: int) -> float:
        return self._beta[step-1]

    def get_alpha(self, step: int) -> float:
        return self._alpha[step-1]

    def get_alpha_bar(self, step: int) -> float:
        return self._alpha_bar[step-1]

    def single_forward_diffusion_step(self, x_prev: np.ndarray, timestep: int) -> t.Tuple[np.ndarray, np.ndarray]:
        alpha = self.get_alpha(timestep)
        epsilon = np.random.normal(size=x_prev.shape)
        x = np.sqrt(alpha) * x_prev + np.sqrt(1 - alpha) * epsilon
        return x, epsilon

    def defuse_sample(self, x0: np.ndarray, timestep: int) -> np.ndarray:
        ahlpha_bar = self.get_alpha_bar(timestep)
        epsilon = np.random.normal(size=x0.shape)
        x = np.sqrt(ahlpha_bar) * x0 + np.sqrt(1 - ahlpha_bar) * epsilon
        return x

    def plot_schedule(self):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5, 8), sharex=True)
        ax1.plot(self.total_timesteps, self._beta, color="tab:blue")
        ax1.set_title("$\\beta$ schedule")
        
        ax2.plot(self.total_timesteps, self._alpha, color="tab:red")
        ax2.set_title("$\\alpha$ schedule")

        ax3.plot(self.total_timesteps, self._alpha_bar, color="tab:green")
        ax3.set_title("$\\bar{\\alpha}$ schedule")
        
        return fig