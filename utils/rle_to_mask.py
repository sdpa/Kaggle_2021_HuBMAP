import numpy as np


def rle_to_mask(encoding, size):
    # encoding is a string. convert it to int.
    mask = np.zeros(size[0] * size[1], dtype=np.uint8)
    encoding = np.fromstring(encoding, dtype=int, sep=' ')
    starts = np.array(encoding)[::2]
    lengths = np.array(encoding)[1::2]
    for start, length in zip(starts, lengths):
        mask[start:start + length] = 1
    return mask.reshape(size, order='F')