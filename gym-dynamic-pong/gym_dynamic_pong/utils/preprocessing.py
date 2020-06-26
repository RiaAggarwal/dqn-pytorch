import numpy as np


def bool_array_to_rgb(array: np.ndarray) -> np.ndarray:
    """
    Converts a boolean numpy array to an RGB tensor with equal values in each of the colors.

    :param array: 2d boolean numpy array
    :return: HxWx3 tensor
    """
    assert array.dtype == np.bool, "array must be boolean"
    array = array.astype(np.uint8) * (2 ** 8 - 1)
    empty = np.zeros_like(array)
    return np.stack([array, empty, empty], axis=2)


# noinspection PyArgumentList
def rgb_array_to_binary(array: np.ndarray) -> np.ndarray:
    """
    Set non-zero to 255 and set the values of all color layers if one is set.

    :param array: numpy array
    :return: boolean numpy array
    """
    assert array.ndim == 3, "array must have 3 dimensions"
    assert array.shape[2] == 3, "The third dimension must be of size 3"

    array[array != 0] = 2**8 - 1
    array = array[:, :, 0] | array[:, :, 1] | array[:, :, 2]
    return array.astype(np.uint8)
