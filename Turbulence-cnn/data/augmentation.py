import numpy as np
from scipy.ndimage import zoom

# Defaults used by the dataset module (kept here for convenience)
COARSE_SIZE = 16
NOISE_STD = 0.05


def random_flip(fine: np.ndarray, p: float = 0.5) -> np.ndarray:
    if np.random.rand() > p:
        return fine
    fine = fine.copy()
    fine = fine[:, :, ::-1].copy()
    fine[0] = -fine[0]
    return fine


def make_coarse_field(fine: np.ndarray, coarse_size: int = COARSE_SIZE, noise_std: float = NOISE_STD) -> np.ndarray:
    C, H, W = fine.shape
    factor = H // coarse_size

    coarse_small = fine.reshape(
        C,
        coarse_size, factor,
        coarse_size, factor
    ).mean(axis=(2, 4))

    zoom_factor = H / coarse_size
    coarse_up = np.stack([
        zoom(coarse_small[c], zoom_factor, order=1)
        for c in range(C)
    ], axis=0)

    noise = np.random.randn(*coarse_up.shape).astype(np.float32)
    coarse_noisy = coarse_up + noise_std * noise

    return coarse_noisy.astype(np.float32)
