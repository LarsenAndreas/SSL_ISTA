import numpy as np


def getDSMatrix(res_init: int, ds_factor: int = 2) -> np.ndarray:
    """Generates a downsampling matrix equivalent to a convolution with an averaging kernel. The downsampling matrix is designed to be multiplied on a flattened image.

    Only works for square images!

    Args:
        `res_init` (int): Number of pixels in one dimension.
        `ds_factor` (int, optional): Downsampling factor. Defaults to 2.

    Returns:
        `D` (np.ndarray): Downsampling matrix. Shape: [res_init²/ds_factor, res_init²]
    """
    w, ds = res_init, ds_factor
    p = w**2
    scale = 1 / ds**2
    w_ds = w // ds

    if p % ds != 0:
        raise Exception(f"{ds=} must be a factor of {p=} | {ds/p=}")

    D = np.zeros(shape=(p // ds**2, p))
    for i, row in enumerate(D):
        start = i * ds
        n = (i // w_ds) * w * (ds - 1)
        for j in range(ds):
            shift = w * j
            row[start + shift + n : start + ds + shift + n] = scale
    return D
