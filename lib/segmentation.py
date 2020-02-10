# -*- coding: utf-8 -*-
"""
Auxiliary functions for image segmentation

:Author:
  `Anna Medyukhina`_
  email: anna.medyukhina@leibniz-hki.de or anna.medyukhina@gmail.com	

:Organization:
  Applied Systems Biology Group, Leibniz Institute for Natural Product Research and Infection Biology - Hans Knöll Institute (HKI)

Copyright (c) 2014-2018, 
Leibniz Institute for Natural Product Research and Infection Biology – 
Hans Knöll Institute (HKI)

Licence: BSD-3-Clause, see ./LICENSE or 
https://opensource.org/licenses/BSD-3-Clause for full details

Requirements
------------
* `Python 2.7.3  <http://www.python.org>`_

"""
from __future__ import division

import numpy as np
from scipy import ndimage

from skimage import morphology
from skimage.measure import label
from skimage.filters import threshold_otsu


def normalize(img, per_low=0.0, per_high=100.0, val_low=0.0, val_high=1.0):
    """
    This function normalizes the input image to given maximum and minimum value, 
    excludes given percent of extreme values

    Parameters
    ----------
    img : ndarray
        Input image to normalize
    per_low : float
        Percentile for the minimum intensity. 
        All intensities lower than this percentile are set to this percentile value 
    per_high : float
        Percentile for the maximum intensity. 
        All intensities higher than this percentile are set to this percentile value 
    val_low : float
        New minimum value of the array
    val_high : float
        New maximum value of the array

    Returns
    -------
    normalize : ndarray
        Output normalized image of the same shape as 'img'
    """

    low = np.percentile(img, per_low)
    high = np.percentile(img, per_high)*1.
    img[np.where(img < low)] = low
    img[np.where(img > high)] = high
    img = img - low
    img = img/img.max()
    img = img*(val_high - val_low)
    img = img + val_low
    return img


def preprocess(img, norm=False, median=None, gauss=None, **kwargs):
    """
    Smoothing and denoising of the input image

    Parameters
    ----------
    img : ndarray
        Input image
    norm : Boolean
        Whether to normalize the image
        Default is True
    median : int
        Radius for the median filter. None means no median filter
        Default is None
    gauss : scalar or sequence of scalars
        Standard deviation for Gaussian kernel. The standard
        deviations of the Gaussian filter are given for each axis as a
        sequence, or as a single number, in which case it is equal for
        all axes. None means no Gaussian filer.
        Default is None

    kwargs : keyword arguments for the normalization function

    Returns
    -------
    img : ndarray
    """
    if norm:
        img = normalize(img, **kwargs)

    if median is not None:
        img = ndimage.median_filter(img, median)

    if gauss is not None:
        img = ndimage.gaussian_filter(img, gauss)

    return img


def segment(img, thr=None, relative_thr=False, preproc=False, morph=False, fill_holes=False, invert=False, **kwargs):
    """
    This function segments the input image with given settings of pre- and post-processing

    Parameters
    ----------
    img : ndarray
        Input image
    thr : float, optional
        Threshold value for segmentation. If None, automatic Otsu thresholding is applied
        Default: None
    relative_thr : bool, optional
        If True, the threshold is computed relative to the maximum image intensity, 
        i.e. 'thr' is multiplied by the maximum image intensity
        Default: False
    preproc : bool, optional
        If True, preprocessing with median and/or Gaussian filter(s) is done
        Default: False
    morph : bool, optional
        If True, postprocessing with morphological opening and closing is done
        Default: False
    fill_holes : bool, optional
        If True, binary hole filling is done in the segmented image
        Default: False
    invert : bool, optional
        If True, the output binary mask is inverted
        Default: False   
    kwargs : key, value pairings
        Keyword arguments are passed through the preprocess function

    Returns
    -------
    segment : ndarray
        Returns segmented image of the same shape as 'img'
    """

    if np.max(img) > 0:
        if preproc:
            img = preprocess(img, **kwargs)

        if thr is None:
            thr = threshold_otsu(img)

        else:
            if relative_thr:
                thr = thr*np.max(img)

        mask = (img > thr)*1.

        if morph:
            mask = morphology.opening(mask)
            mask = morphology.closing(mask)

        if fill_holes:
            mask = ndimage.binary_fill_holes(mask)

    else:
        mask = img

    mask = (mask > 0)*1.

    if invert:
        mask = 1 - mask
    return mask


def remove_large_objects(mask, max_size, labeled=False, connectivity=1):

    if labeled is False:
        labels = label(mask > 0, connectivity=connectivity)

    else:
        labels = mask

    llist = np.unique(labels)
    size = ndimage.sum(labels > 0, labels, llist)

    bv = llist[np.where(size > max_size)]
    ix = np.in1d(labels.ravel(), bv).reshape(labels.shape)
    mask[ix] = 0

    return mask


def filter_by_size(labels, labeled=False, minsize=0, maxsize=None):

    if labeled is False:
        labels = label(labels)

    llist = np.unique(labels)
    llist = llist[llist > 0]
    if len(llist) > 0:
        size = ndimage.sum(labels > 0, labels, llist)

        if maxsize is None:
            maxsize = np.max(size)

        bv = llist[np.where((size < minsize) | (size > maxsize))]
        ix = np.in1d(labels.ravel(), bv).reshape(labels.shape)
        labels[ix] = 0

    return labels








