import numpy as np

def rle_encode_less_memory(img):
    pixels = img.T.flatten()
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle_to_mask(_encoding, size):
    # encoding is a string. convert it to int.
    mask = np.zeros(size[0] * size[1], dtype=np.uint8)
    encoding_np = np.fromstring(_encoding, dtype=int, sep=' ')
    starts = np.array(encoding_np)[::2]
    lengths = np.array(encoding_np)[1::2]
    for start, length in zip(starts, lengths):
        mask[start:start + length] = 1
    return mask.reshape(size, order='F')

def global_shift_mask(maskpred1, y_shift, x_shift):
    """
    applies a global shift to a mask by
    padding one side and cropping from the other
    """
    if y_shift < 0 and x_shift >=0:
        maskpred2 = np.pad(maskpred1,
                           [(0,abs(y_shift)), (abs(x_shift), 0)],
                           mode='constant', constant_values=0)
        maskpred3 = maskpred2[abs(y_shift):, :maskpred1.shape[1]]
    elif y_shift >=0 and x_shift <0:
        maskpred2 = np.pad(maskpred1,
                           [(abs(y_shift),0), (0, abs(x_shift))],
                           mode='constant', constant_values=0)
        maskpred3 = maskpred2[:maskpred1.shape[0], abs(x_shift):]
    elif y_shift >=0 and x_shift >=0:
        maskpred2 = np.pad(maskpred1,
                           [(abs(y_shift),0), (abs(x_shift), 0)],
                           mode='constant', constant_values=0)
        maskpred3 = maskpred2[:maskpred1.shape[0], :maskpred1.shape[1]]
    elif y_shift < 0 and x_shift < 0:
        maskpred2 = np.pad(maskpred1,
                           [(0, abs(y_shift)), (0, abs(x_shift))],
                           mode='constant', constant_values=0)
        maskpred3 = maskpred2[abs(y_shift):, abs(x_shift):]
    return maskpred3