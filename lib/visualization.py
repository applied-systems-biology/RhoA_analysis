from __future__ import division

import numpy as np
from skimage.segmentation import find_boundaries


def draw_outlines(mask, img, color=(255, 255, 255), borders=True, channel=0):
    """
    Draw outlines of the input mask over the input image

    Parameters
    ----------
    mask : ndarray of shape MxN
        Input binary mask
    img : ndarray of shape MxN or MxNx3
        Input image to be used as the background
    color : tuple of shape 3, optional
        Color of the outline in RGB mode.
        Defauls is (255,255,255), which corresponds to white
    borders : bool, optional
        Whether to compute the borders of the mask. If False, the whole mask is overlaid.
        Default is True
    channel : int (0,1,or 2)
        RGB channel to which the background is assigned.
        Default is 0, which corresponds to the red channel

    Return
    ------
    im : ndarray of shape MxNx3, type uint8
        Output image with overlaid mask outlines
    """
    if borders:
        borders = find_boundaries(mask.astype(np.uint8))
    else:
        borders = mask

    ind = np.where(borders > 0)

    img = img*255./img.max()

    if len(img.shape) == 3 and img.shape[-1] == 3:
        im = img.copy()
    else:
        im = np.zeros([img.shape[0], img.shape[1], 3])
        if channel >= 0:
            im[:, :, channel] = img
        else:
            im[:, :, 0] = im[:, :, 1] = im[:, :, 2] = img

    im[ind] = color
    return im.astype(np.uint8)


def merge_RGB(red=None, green=None, blue=None):

    imRGB = None
    for img in [red, green, blue]:
        if img is not None:
            imRGB = np.zeros(img.shape + (3,))

    if imRGB is not None:
        if red is not None:
            imRGB[:, :, 0] = red
        if green is not None:
            imRGB[:, :, 1] = green
        if blue is not None:
            imRGB[:, :, 2] = blue

    return imRGB







