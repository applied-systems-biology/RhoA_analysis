# -*- coding: utf-8 -*-
from __future__ import division
import os

import mkl
mkl.set_num_threads(1)
import numpy as np
from skimage import io
import pandas as pd
import warnings
from skimage.filters import threshold_otsu

from helper_lib import filelib
from helper_lib import parallel

import lib.segmentation as sgm
from lib.visualization import draw_outlines
from lib import plotting

########################################


def manders(x,y):
    return np.sum(x*y)/np.sqrt(np.sum(x**2)*np.sum(y**2))


def analyze_parallel(debug=False, **kwargs):

    files = filelib.list_subfolders(kwargs['inputfolder'])

    if debug:
        kwargs['item'] = files[0]
        analyze(**kwargs)
    else:
        kwargs['items'] = files
        parallel.run_parallel(process=analyze, **kwargs)
        filelib.combine_statistics(kwargs.get('outputfolder') + 'image_statistics/')


def analyze(item, inputfolder, outputfolder, channels, colors, thresholds=None, bg_thr=None, cell_radius=None,
            bg_estimator=np.median, bg_channels=3, bg_sigma=1, **kwargs):
    filelib.make_folders([outputfolder + 'image_statistics/'])
    img_series = io.imread(inputfolder + item)#[0:3]

    metadata = pd.read_csv(inputfolder + item[:-4] + '.txt', sep='\t', index_col=0, header=-1).transpose()

    xpix = float(metadata.voxel_size_xy)
    bg_sigma = bg_sigma / xpix

    colors = np.array(colors)
    ind = [np.where(colors == 0)[0][0], np.where(colors == 1)[0][0]]
    if 2 in colors:
        ind.append(np.where(colors == 2)[0][0])

    stat = pd.DataFrame()
    for image_index in range(img_series.shape[0]):
        img = img_series[image_index]
        if img.max() > 0:

            # preprocess
            img = img.astype(np.float)

            for i in range(img.shape[-1]):
                img[:, :, i] = sgm.preprocess(img[:, :, i], median=3)

            # RGB image for outlines
            im = np.zeros([img.shape[0], img.shape[1], 3])

            for i, c in enumerate(colors):
                if c >= 0:
                    im[:, :, c] = sgm.normalize(img[:, :, i]) * 255

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                filelib.imsave(outputfolder + 'RGB/' + item[:-4] + '_%02d.tif' % image_index, im.astype(np.uint8))

            # segment background
            bg = np.ones_like(img[:, :, 0])
            for i in ind[:bg_channels]:
                gray = sgm.preprocess(img[:, :, i], gauss=bg_sigma)
                if bg_thr is None:
                    thr = threshold_otsu(gray[np.where(gray < threshold_otsu(gray))])
                else:
                    thr = bg_thr
                mask = gray < thr
                bg = bg * mask

            bg = sgm.segment(bg, thr=0, morph=True) * 255

            # # remove small areas from the background and foreground
            if cell_radius is not None:
                minarea = np.pi * (cell_radius / xpix) ** 2
                bg = sgm.filter_by_size(bg, minsize=minarea) == 0
                minarea = 2 * np.pi * (cell_radius / xpix) ** 2
                bg = sgm.filter_by_size(bg, minsize=minarea) == 0
                bg = bg * 255

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                filelib.imsave(outputfolder + 'background/' + item[:-4] + '_%02d.tif' % image_index, draw_outlines(bg, im))

            # compute foreground area
            curstat = pd.DataFrame()
            curstat['Image_name'] = [item[:-4] + '_%02d.tif' % image_index]
            curstat['Foreground area, $\mu$m$^2$'] = np.sum((bg == 0) * 1.) * xpix ** 2
            curstat['Foreground area fraction'] = np.sum((bg == 0) * 1.) * 1. / (bg.shape[0] * bg.shape[1])

            # segment positive areas in different channels
            masks = []
            for i in ind:
                if thresholds is not None:
                    avg = bg_estimator(img[:, :, i][np.where(bg == 0)])
                    if avg > 0:
                        mask = img[:, :, i] > avg * thresholds[i]
                    else:
                        mask = np.zeros_like(img[:, :, i])
                else:
                    thr = threshold_otsu(img[:, :, i][np.where(img[:, :, i] > threshold_otsu(img[:, :, i]))])
                    mask = img[:, :, i] > thr

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    filelib.imsave(outputfolder + 'outlines_' + channels[i] + '/' + item[:-4] + '_%02d.tif' % image_index,
                                       draw_outlines(mask, img[:, :, i], channel=colors[i]))
                masks.append(mask)
                curstat[channels[i] + ' area, $\mu$m$^2$'] = np.sum((mask > 0)*1.)*xpix**2
                if np.sum((bg == 0)*1.) > 0:
                    curstat[channels[i] + ' area normalized'] = np.sum((mask > 0)*1.) / np.sum((bg == 0)*1.)
                    curstat['Mean intensity of ' + channels[i] + ' in the foreground'] = np.mean(img[:, :, i][np.where(bg == 0)])

                    curstat['Integrated intensity of ' + channels[i]
                         + ' in the foreground'] = np.sum(img[:, :, i][np.where(bg == 0)])
                if mask.max() > 0:
                    curstat['Mean intensity of ' + channels[i] + '-positive area'] = \
                        np.mean(img[:, :, i][np.where(mask > 0)])
                    curstat['Integrated intensity of ' + channels[i]
                            + '-positive area'] = np.sum(img[:, :, i][np.where(mask > 0)])

            # compute mask overlap and save
            positive = np.where((masks[0] > 0) | (masks[1] > 0), 1, 0)
            overlap = np.where((masks[0] > 0) & (masks[1] > 0), 255, 0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                filelib.imsave(outputfolder + 'outlines_overlap/' + item[:-4] + '_%02d.tif' % image_index,
                                   draw_outlines(overlap, im))
            curstat['Overlap area, $\mu$m$^2$'] = np.sum((overlap > 0) * 1.) * xpix ** 2
            if np.sum((bg == 0)*1.) > 0:
                curstat['Overlap area normalized over foreground'] = np.sum((overlap > 0) * 1.) / np.sum((bg == 0) * 1.)
            if np.sum(positive) > 0:
                curstat['Overlap are normalized over positive area'] = np.sum((overlap > 0) * 1.) / np.sum(positive)

            if overlap.max() > 0:
                for i in ind:
                    curstat['Mean intensity of ' + channels[i] +
                            ' in the overlap area'] = np.mean(img[:, :, i][np.where(overlap > 0)])
                    curstat['Integrated intensity of ' + channels[i] +
                            ' in the overlap area'] = np.sum(img[:, :, i][np.where(overlap > 0)])

            # compute correlations

            if np.sum((bg == 0)*1.) > 0:
                curstat['Pearson correlation in foreground'] = np.corrcoef(img[:, :, ind[0]][np.where(bg == 0)],
                                                                           img[:, :, ind[1]][np.where(bg == 0)])[0, 1]

            if masks[0].max() > 0 and masks[1].max() > 0:
                curstat['Manders overlap'] = manders(np.where(masks[0] > 0, img[:, :, ind[0]], 0),
                                                     np.where(masks[1] > 0, img[:, :, ind[1]], 0))
            stat = pd.concat([stat, curstat], ignore_index=True, sort=False)

    stat['Group'] = item[:-4]
    filelib.make_folders([outputfolder + 'image_statistics/' + os.path.dirname(item)])
    stat.to_csv(outputfolder + 'image_statistics/' + item[:-4] + '.csv', sep='\t')


############################################

path = '../Data/in_vivo/'
kwargs = {'max_threads': 15,
          'channels': ['F-Actin', 'DPPIV', 'DAPI', 'MRP2'],
          'colors': [0, -1, 2, 1],
          'thresholds': [2, 2, 2, 2],
          'bg_sigma': 1.5, 'cell_radius': 6,
          'bg_estimator': np.mean,
          'bg_thr': None,
          'permax': 100,
          'debug': False}


analyze_parallel(inputfolder=path+'input/', outputfolder=path+'output/', **kwargs)

plotting.plot_boxplots(path+'output/image_statistics.csv', path + 'output/plots/', height=4,
                       aspect=3, kind='box', notch=False, rotation=90)










