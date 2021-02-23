import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import errno


def dice(im1, im2, empty_score=1.0):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        Both are empty (sum eq to zero) = empty_score

    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return (2. * intersection.sum() / im_sum)

def plot_img_and_mask(img, mask, orig_mask, output, name, single):
    classes = mask.shape[2] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 3)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i+1].set_title(f'Output mask (class {i+1})')
            ax[i+1].imshow(mask[:, :, i])
    else:
        mask = Image.fromarray(np.array(mask))
        mask = mask.resize(orig_mask.size)
        dice_s = dice(np.array(orig_mask), np.array(mask))
        ax[1].set_title(f'Original mask')
        ax[1].imshow(orig_mask, cmap='Greys_r')
        ax[2].set_title(f'Output mask')
        ax[2].imshow(mask)
        ax[3].set_title(f'Output (Light) and Original (Dark), Superimposed')
        ax[3].imshow(img)
        ax[3].imshow(mask, 'Reds', interpolation='none', alpha=0.5)
        ax[3].imshow(orig_mask, 'Blues', interpolation='none', alpha=0.5)
    if single:
        fig.suptitle(name[0] + ', Dice: ' + str(round(dice_s, 3)), fontsize=20)
        fig.set_size_inches(20, 6)
        plt.xticks([]), plt.yticks([])
        plt.savefig(name[0])
        plt.show()
    else:
        filename = output + name
        if not os.path.exists(os.path.dirname(filename)):
            try:
                os.makedirs(os.path.dirname(filename))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        fig.suptitle(name + ', Dice: ' + str(round(dice_s, 3)), fontsize=20)
        fig.set_size_inches(20, 6)
        plt.xticks([]), plt.yticks([])
        plt.savefig(output + name)
        plt.close()
