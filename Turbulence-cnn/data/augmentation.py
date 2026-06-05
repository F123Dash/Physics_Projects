import numpy as np
from scipy.ndimage import zoom

COARSE_SIZE = 16
NOISE_STD   = 0.05
_FLIP_SIGNS_Y = np.array([ 1.,-1.,1.,-1.], dtype=np.float32)  # y-flip (axis=2)
_FLIP_SIGNS_X = np.array([-1.,1.,1.,-1.], dtype=np.float32)  # x-flip (axis=1)


def random_flip(fine: np.ndarray, p: float = 0.5) -> np.ndarray:
    if np.random.rand() > p: return fine
    fine = fine.copy()
    C = fine.shape[0]
    if np.random.rand() < 0.5:
        fine = fine[:, :, ::-1].copy()
        signs = _FLIP_SIGNS_Y[:C].reshape(C, 1, 1)
    else:
        fine = fine[:, ::-1, :].copy()
        signs = _FLIP_SIGNS_X[:C].reshape(C, 1, 1)
    fine = fine * signs
    return fine.astype(np.float32)

def make_coarse_field(fine:np.ndarray,coarse_size:int = COARSE_SIZE,noise_std:float = NOISE_STD,) -> np.ndarray:
    C, H, W = fine.shape
    assert H % coarse_size == 0 and W % coarse_size == 0, (f"fine spatial dims ({H},{W}) must be divisible by coarse_size={coarse_size}")
    factor_h = H // coarse_size
    factor_w = W // coarse_size
    coarse_small = fine.reshape(C, coarse_size, factor_h, coarse_size, factor_w).mean(axis=(2, 4))
    coarse_up = np.stack([zoom(coarse_small[c],(factor_h, factor_w),order=1) for c in range(C)],axis=0,)
    if noise_std > 0.0:coarse_up = coarse_up + noise_std * np.random.randn(*coarse_up.shape)
    return coarse_up.astype(np.float32)