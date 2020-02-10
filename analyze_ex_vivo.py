# -*- coding: utf-8 -*-
from __future__ import division
import os
import sys
import re
import mkl

mkl.set_num_threads(1)

import numpy as np
from skimage import io
import pandas as pd
import warnings
from scipy import ndimage
from scipy.ndimage.morphology import binary_fill_holes
from skimage import morphology
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops

from helper_lib import filelib
from helper_lib import parallel

import lib.segmentation as sgm
from lib.visualization import draw_outlines
from lib import plotting


########################################


def manders(x, y):
    return np.sum(x * y) / np.sqrt(np.sum(x ** 2) * np.sum(y ** 2))


def find_thickness(mask, percentile=0.5):
    """
    Finds a thickness value, for which a given percentage of the objects in the image are as thick or thinner.

    Parameters
    ----------
    mask : numpy.ndarray
        Input binary mask
    percentile : float
        percentage of objects that should be as thick or thinner than the returned thickness value

    Returns
    -------
    int : object thickness in pixels
    """
    mask = mask > 0
    m = []
    m.append(mask.sum())
    d = []
    d.append(0)

    for i in range(1, 10):
        o = morphology.opening(mask, morphology.disk(i))
        m.append(o.sum())
        d.append(2 * i)

    if m[0] > 0:
        m = np.array(m) * 1. / m[0]
        d = np.array(d)

        m = m <= percentile
        davg = d[m.argmax()]
    else:
        davg = 0

    return davg


def analyze_parallel(debug=False, **kwargs):
    files = filelib.list_subfolders(kwargs['inputfolder'])

    if debug:
        kwargs['item'] = files[0]
        analyze(**kwargs)
    else:
        kwargs['items'] = files
        parallel.run_parallel(process=analyze, **kwargs)
        filelib.combine_statistics(kwargs.get('outputfolder') + 'image_statistics/')
        filelib.combine_statistics(kwargs.get('outputfolder') + 'roi_statistics/')


def analyze(item, inputfolder, outputfolder, channels, colors, thresholds=None, bg_thr=None, cell_radius=None,
            bg_estimator=np.median, bg_channels=3, bg_sigma=1, min_diam_canaliculi=1., max_diam_canaliculi=20,
            min_height_canaliculi=2, **kwargs):
    filelib.make_folders([outputfolder + 'image_statistics/', outputfolder + 'roi_statistics/'])
    img = io.imread(inputfolder + item)  # [20:30]

    metadata = pd.read_csv(inputfolder + item[:-4] + '.txt', sep='\t', index_col=0, header=-1).transpose()
    xpix = float(metadata.voxel_size_xy)
    zpix = float(metadata.voxel_size_z)
    bg_sigma = bg_sigma / np.array([zpix, xpix, xpix])

    colors = np.array(colors)
    ind = [np.where(colors == 0)[0][0], np.where(colors == 1)[0][0]]
    if 2 in colors:
        ind.append(np.where(colors == 2)[0][0])

    # preprocess
    img = img.astype(np.float)

    for i in range(img.shape[-1]):
        img[:, :, :, i] = sgm.preprocess(img[:, :, :, i], median=3)

    # RGB image for outlines
    im = np.zeros_like(img)
    for i, c in enumerate(colors):
        im[:, :, :, c] = sgm.normalize(img[:, :, :, i]) * 255

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(len(im)):
            filelib.imsave(outputfolder + 'RGB/' + item[:-4] + '_%02d.tif' % i, im[i].astype(np.uint8))

    # segment background
    bg = np.ones_like(img[:, :, :, 0])
    for i in ind[:bg_channels]:
        gray = sgm.preprocess(img[:, :, :, i], gauss=bg_sigma)
        if bg_thr is None:
            thr = threshold_otsu(gray[np.where(gray < threshold_otsu(gray))])
        else:
            thr = bg_thr
        mask = gray < thr
        bg = bg * mask

    bg = sgm.segment(bg, thr=0, morph=True) * 255

    # remove small areas from the background and foreground
    if cell_radius is not None:
        minarea = np.pi * (cell_radius / xpix) ** 2
        for i in range(len(bg)):
            bg[i] = sgm.filter_by_size(bg[i], minsize=minarea) == 0
        minarea = 2 * np.pi * (cell_radius / xpix) ** 2
        for i in range(len(bg)):
            bg[i] = sgm.filter_by_size(bg[i], minsize=minarea) == 0
        bg = bg * 255

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(len(im)):
            filelib.imsave(outputfolder + 'background/' + item[:-4] + '_%02d.tif' % i, draw_outlines(bg[i], im[i]))

    # compute foreground volume
    stat = pd.DataFrame()
    stat['Image_name'] = [item]
    stat['Group'] = item.split('/')[0]
    stat['Foreground volume, $\mu$m$^3$'] = np.sum((bg == 0) * 1.) * xpix ** 2 * zpix
    stat['Foreground volume fraction'] = np.sum((bg == 0) * 1.) * 1. / (bg.shape[0] * bg.shape[1] * bg.shape[2])

    # segment positive areas in different channels
    masks = []
    for i in ind:
        if thresholds is not None:
            avg = bg_estimator(img[:, :, :, i][np.where(bg == 0)])
            if avg > 0:
                mask = img[:, :, :, i] > avg * thresholds[i]
            else:
                mask = np.zeros_like(img[:, :, :, i])
        else:
            thr = threshold_otsu(img[:, :, :, i][np.where(img[:, :, :, i] > threshold_otsu(img[:, :, :, i]))])
            mask = img[:, :, :, i] > thr

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for j in range(len(im)):
                filelib.imsave(outputfolder + 'outlines_' + channels[i] + '/' + item[:-4] + '_%02d.tif' % j,
                               draw_outlines(mask[j], img[j, :, :, i], channel=colors[i]))
        masks.append(mask)
        stat[channels[i] + ' volume, $\mu$m$^3$'] = np.sum((mask > 0) * 1.) * xpix ** 2 * zpix
        if np.sum((bg == 0) * 1.) > 0:
            stat[channels[i] + ' volume normalized'] = np.sum((mask > 0) * 1.) / np.sum((bg == 0) * 1.)
            stat['Mean intensity of ' + channels[i] + ' in the foreground'] = \
                np.mean(img[:, :, :, i][np.where(bg == 0)])
            stat['Integrated intensity of ' + channels[i] + ' in the foreground'] = \
                np.sum(img[:, :, :, i][np.where(bg == 0)])
        if mask.max() > 0:
            stat['Mean intensity of ' + channels[i] + '-positive area'] = np.mean(img[:, :, :, i][np.where(mask > 0)])
            stat['Integrated intensity of ' + channels[i]
                 + '-positive area'] = np.sum(img[:, :, :, i][np.where(mask > 0)])

    # remove nuclei positive regions from MRP2 and F-Actin channels
    if masks[2].max() > 0:
        masks[0][np.where(masks[2] > 0)] = 0
        masks[1][np.where(masks[2] > 0)] = 0

    # compute mask overlap and save
    positive = np.where((masks[0] > 0) | (masks[1] > 0), 1, 0)
    overlap = np.where((masks[0] > 0) & (masks[1] > 0), 255, 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(len(im)):
            filelib.imsave(outputfolder + 'outlines_overlap/' + item[:-4] + '_%02d.tif' % i,
                           draw_outlines(overlap[i], im[i]))
    stat['Overlap volume, $\mu$m$^3$'] = np.sum((overlap > 0) * 1.) * xpix ** 2 * zpix
    if np.sum((bg == 0) * 1.) > 0:
        stat['Overlap volume normalized over foreground'] = np.sum((overlap > 0) * 1.) / np.sum((bg == 0) * 1.)
    if np.sum(positive) > 0:
        stat['Overlap volume normalized over positive area'] = np.sum((overlap > 0) * 1.) / np.sum(positive)

    if overlap.max() > 0:
        for i in ind:
            stat['Mean intensity of ' + channels[i] +
                 ' in the overlap area'] = np.mean(img[:, :, :, i][np.where(overlap > 0)])
            stat['Integrated intensity of ' + channels[i] +
                 ' in the overlap area'] = np.sum(img[:, :, :, i][np.where(overlap > 0)])

    # compute correlations
    if np.sum((bg == 0) * 1.) > 0:
        stat['Pearson correlation in foreground'] = np.corrcoef(img[:, :, :, ind[0]][np.where(bg == 0)],
                                                                img[:, :, :, ind[1]][np.where(bg == 0)])[0, 1]

    stat['Manders overlap'] = manders(np.where(masks[0] > 0, img[:, :, :, ind[0]], 0),
                                      np.where(masks[1] > 0, img[:, :, :, ind[1]], 0))

    # analyze canaliculi

    canaliculi_stat = pd.DataFrame()
    canaliculi = label(masks[1])
    canaliculi_filled = np.zeros_like(masks[1]).astype(np.uint32)  # channel 1 corresponds to MRP2
    for i in range(len(masks[1])):
        regions = regionprops(canaliculi[i])
        for reg in regions:
            minr, minc, maxr, maxc = reg.bbox
            canaliculi_filled[i][minr:maxr, minc:maxc][np.where(reg.convex_image > 0)] = reg.label
            canaliculi_stat = pd.concat([canaliculi_stat, pd.DataFrame({'Label': [reg.label],
                                                                        'Z': [i],
                                                                        'Dimensions XY, $\mu m$':
                                                                            [np.sqrt(reg.convex_area
                                                                                     / np.pi)*2*xpix],
                                                                        'Area': [reg.area],
                                                                        'Convex area': reg.convex_area})])

    if np.sum(np.where(canaliculi_filled > 0, 1, 0)) > 0:
        stat['MRP2 positive volume fraction'] = np.sum(np.where(masks[1] > 0, 1, 0)) \
                                                / np.sum(np.where(canaliculi_filled > 0, 1, 0))
    if len(canaliculi_stat) > 0:
        spotstat = canaliculi_stat.groupby('Label').mean()
        spotstat['Height, $\mu m$'] = (canaliculi_stat.groupby('Label').max()['Z']
                                       - canaliculi_stat.groupby('Label').min()['Z'] + 1) * zpix
        spotstat['Solidity'] = canaliculi_stat.groupby('Label').sum()['Area'] / \
                               canaliculi_stat.groupby('Label').sum()['Convex area']
        spotstat['Convex volume, $\mu m^3$'] = canaliculi_stat.groupby('Label').sum()['Convex area'] * xpix**2 * zpix
        spotstat['Volume, $\mu m^3$'] = canaliculi_stat.groupby('Label').sum()['Area'] * xpix**2 * zpix

        spotstat = spotstat.drop(columns=['Area', 'Convex area', 'Z'])
        llist = np.array(spotstat.index)
        spotstat['F-Actin positive volume fraction'] = ndimage.sum(masks[0] > 0, canaliculi_filled,
                                                                   llist) * xpix**2 * zpix \
                                                       / spotstat['Convex volume, $\mu m^3$']
        spotstat['Overlap volume fraction'] = ndimage.sum(overlap > 0, canaliculi_filled, llist) * xpix**2 * zpix / \
                                              spotstat['Convex volume, $\mu m^3$']
        spotstat = spotstat[(spotstat['Dimensions XY, $\mu m$'] > min_diam_canaliculi) &
                            (spotstat['Dimensions XY, $\mu m$'] < max_diam_canaliculi) &
                            (spotstat['Height, $\mu m$'] >= min_height_canaliculi) &
                            (spotstat['F-Actin positive volume fraction'] > 0)]

        llist = np.array(spotstat.index)

        ix = np.in1d(canaliculi_filled.ravel(), llist).reshape(canaliculi_filled.shape)
        canaliculi_filled = np.zeros_like(canaliculi_filled)
        canaliculi_filled[ix] = 255
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(len(im)):
                filelib.imsave(outputfolder + 'outlines_canaliculi/' + item[:-4] + '_%02d.tif' % i,
                               draw_outlines(canaliculi_filled[i], img[i, :, :, ind[1]], channel=colors[ind[1]]))

        spotstat['Image_name'] = item
        spotstat['Group'] = item.split('/')[0]

        filelib.make_folders([outputfolder + 'roi_statistics/' + os.path.dirname(item)])
        spotstat.to_csv(outputfolder + 'roi_statistics/' + item[:-4] + '.csv', sep='\t')

    # analyze F-Actin small particles
    minsize = 4. / 3 * np.pi * 2.5 ** 3
    for i in range(len(canaliculi_filled)):
        canaliculi_filled[i] = morphology.dilation(canaliculi_filled[i], morphology.disk(int(round(0.5 / xpix))))
    masks[0][np.where(canaliculi_filled > 0)] = 0  # remove MRP2 positive regions from the F-Actin channel
    f_actin_large = morphology.remove_small_objects(masks[0] > 0, min_size=minsize / (xpix ** 2 * zpix))
    f_actin_small = np.where((masks[0] > 0) & (f_actin_large == 0), 1, 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(len(im)):
            filelib.imsave(outputfolder + 'outlines_F-Actin_large_fibers/' + item[:-4] + '_%02d.tif' % i,
                           draw_outlines(f_actin_large[i] * 255, img[i, :, :, ind[0]], channel=colors[ind[0]]))
            filelib.imsave(outputfolder + 'outlines_F-Actin_small_particles/' + item[:-4] + '_%02d.tif' % i,
                           draw_outlines(f_actin_small[i] * 255, img[i, :, :, ind[0]], channel=colors[ind[0]]))

    if np.sum((masks[0] > 0) * 1.):
        stat['F-Actin small particles volume fraction'] = np.sum((f_actin_small > 0) * 1.) / np.sum((masks[0] > 0) * 1.)

    # compute the thickness of F-Actin fibers
    # for percentile in [0.5, 0.8, 0.9, 0.95, 0.99]:
    for percentile in [0.5]:
        thickness = []
        for i in range(len(masks[0])):
            thickness.append(find_thickness(masks[0][i], percentile) * xpix)

        stat['F-Actin fiber thickness' + str(percentile) + ' median, $\mu m$'] = np.median(thickness)
        stat['F-Actin fiber thickness' + str(percentile) + ' mean, $\mu m$'] = np.mean(thickness)
    filelib.make_folders([outputfolder + 'image_statistics/' + os.path.dirname(item)])
    stat.to_csv(outputfolder + 'image_statistics/' + item[:-4] + '.csv', sep='\t')


############################################

p = re.compile('\d*\.*\d+')
args = sys.argv[1:]
if len(args) > 0:
    path = args[0]
    dataset = p.findall(path)[-1]

    if dataset == '1':
        kwargs = {'max_threads': 30,
                  'channels': ['MRP2', 'DAPI', 'F-Actin'],
                  'colors': [1, 2, 0],
                  'thresholds': [4.5, 2, 3.5],
                  'bg_sigma': 1.5, 'cell_radius': 6, 'bg_estimator': np.mean, 'permax': 100,
                  'min_diam_canaliculi': 1.2,
                  'max_diam_canaliculi': 40,
                  'min_height_canaliculi': 1,
                  'debug': False}

    elif dataset == '2':
        kwargs = {'max_threads': 30,
                  'channels': ['F-Actin', 'DAPI', 'MRP2'],
                  'colors': [0, 2, 1],
                  'thresholds': [3.5, 2, 4.5],
                  'bg_sigma': 1.5, 'cell_radius': 6, 'bg_estimator': np.mean, 'permax': 100,
                  'min_diam_canaliculi': 1.2,
                  'max_diam_canaliculi': 40,
                  'min_height_canaliculi': 1,
                  'debug': False}

    else:
        kwargs = None

    if kwargs is not None:
        analyze_parallel(inputfolder=path + 'input/', outputfolder=path + 'output/', **kwargs)

        plotting.plot_boxplots(path + 'output/image_statistics.csv', path + 'output/plots/', height=4,
                               aspect=0.8, kind='box', notch=False, rotation=0)

        plotting.plot_swarmplots(path + 'output/roi_statistics.csv', path + 'output/plots/swarmplots/')













