import numpy as np
import torch
import torch.nn.functional as F
import config

COARSE_FACTOR = config.COARSE_FACTOR
NOISE_STD     = config.NOISE_STD      # in normalised units, training inputs only
# Mirror symmetries of the NS equations, for channels [u, v, p, ω] indexed [x, y].
_FLIP_SIGNS_Y = np.array([ 1.,-1.,1.,-1.], dtype=np.float32)  # y -> -y (axis=2)
_FLIP_SIGNS_X = np.array([-1.,1.,1.,-1.], dtype=np.float32)  # x -> -x (axis=1)


def mirror(fine: np.ndarray, axis: str) -> np.ndarray:
    C = fine.shape[0]
    if axis == "y":
        fine, signs = fine[:, :, ::-1], _FLIP_SIGNS_Y[:C]
    elif axis == "x":
        fine, signs = fine[:, ::-1, :], _FLIP_SIGNS_X[:C]
    else:
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")
    return (fine * signs.reshape(C, 1, 1)).astype(np.float32)

def random_flip(fine: np.ndarray, p: float = 0.5) -> np.ndarray:
    """With probability p, mirror a physical-unit field in x or y (equally likely)."""
    if np.random.rand() > p: return fine
    return mirror(fine, "y" if np.random.rand() < 0.5 else "x")

def random_shift(fine: np.ndarray) -> np.ndarray:
    """Random periodic translation (periodic domains only)."""
    sx, sy = np.random.randint(0, fine.shape[1]), np.random.randint(0, fine.shape[2])
    return np.roll(fine, (sx, sy), axis=(1, 2))

def block_average(fine: torch.Tensor, factor: int = COARSE_FACTOR) -> torch.Tensor:
    """(B, C, H, W) -> (B, C, H/factor, W/factor) by averaging factor×factor blocks."""
    H, W = fine.shape[-2:]
    assert H % factor == 0 and W % factor == 0, (f"fine spatial dims ({H},{W}) must be divisible by {factor}")
    return F.avg_pool2d(fine, kernel_size=factor)

def upsample(coarse: torch.Tensor, size: int, mode: str = "bilinear", periodic: bool = False) -> torch.Tensor:
    # align_corners=False puts each coarse value at the centre of the block it
    # averages, so the upsampled field lines up with the fine grid. For a periodic
    # domain, wrap two coarse cells around the edges first and crop afterwards.
    if not periodic:
        return F.interpolate(coarse, size=(size, size), mode=mode, align_corners=False)
    pad = 2
    factor = size // coarse.shape[-1]
    up = F.interpolate(F.pad(coarse, (pad, pad, pad, pad), mode="circular"),
                       scale_factor=factor, mode=mode, align_corners=False)
    c = pad * factor
    return up[..., c:c + size, c:c + size]

def make_coarse_field(fine: torch.Tensor, factor: int = COARSE_FACTOR, noise_std: float = 0.0,
                      mode: str = "bilinear", periodic: bool = False) -> torch.Tensor:
    """Model input: block-average by `factor`, then upsample back. Accepts (C,H,W) or (B,C,H,W)."""
    squeeze = fine.dim() == 3
    x = fine[None] if squeeze else fine
    out = upsample(block_average(x, factor), x.shape[-1], mode, periodic)
    if noise_std > 0.0:
        out = out + noise_std * torch.randn_like(out)
    return out[0] if squeeze else out
