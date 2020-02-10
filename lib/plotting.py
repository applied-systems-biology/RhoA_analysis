from __future__ import division

import os
import numpy as np
import pandas as pd
from scipy import stats

import seaborn as sns
import pylab as plt

from helper_lib import filelib


def distributions(inputfile, outputfolder, condition='Group', exclude=None, sort=False, sort_key=None,
                  bins=10, normed=False, maxper=100, separate_plots=False, **kwargs):
    filelib.make_folders([outputfolder])

    extension = '.' + kwargs.pop('extension', 'png')
    if exclude is None:
        exclude = [condition]

    stat = pd.read_csv(inputfile, sep='\t')

    if sort:
        if sort_key is None:
            sort_key = [condition]
        stat = stat.sort(sort_key).reset_index(drop=True)

    ncols = kwargs.get('ncols', 7)

    for value in stat.columns:
        if value not in exclude:
            r = (stat[value].min(), np.nanpercentile(stat[value], maxper))
            if separate_plots:
                for c in stat[condition].unique():
                    curstat = stat[stat[condition] == c]
                    plt.hist(curstat[value], bins=bins, density=normed, range=r)
                    plt.xlabel(value + ' (' + c + ')')
                    name = value.replace(' ', '_').replace('$', '').replace('/', '').replace('\\', '').replace(',', '') \
                           + '_' + c
                    plt.savefig(outputfolder + name + extension)
                    plt.close()

            else:
                g = sns.FacetGrid(stat, col=condition, col_wrap=ncols)
                g = g.map(plt.hist, value, bins=bins, density=normed, range=r)
                g.set_titles(col_template="{col_name}", fontsize=18)

                name = value.replace(' ', '_').replace('$', '').replace('/', '').replace('\\', '').replace(',', '')
                g.fig.savefig(outputfolder + name + extension)
                plt.close()


##################################################################################


def plot_stat(x, y, stat, outputname, statmethod=stats.ranksums, bonf=1, **kwargs):

    filelib.make_folders([os.path.dirname(outputname)])

    # sns.set(font='arial', style="ticks")
    margins = kwargs.pop('margins', None)

    data = stat.reset_index(drop=True)
    data[x] = data[x].astype(np.str)

    perhigh = []
    for cx in data[x].unique():
        curdata = data[data[x] == cx]
        if len(curdata) > 0:
            perhigh.append(min(np.median(curdata[y]) + 2*(np.percentile(curdata[y],75) - np.median(curdata[y])), np.max(curdata[y])))

        ind = data[data[x] == cx].index
        n = len(curdata)
        data[x].iloc[ind] = np.array(data[x].iloc[ind]) + '\nn = ' + str(n)

    pl = sns.catplot(data=data, x=x, y=y, **kwargs)

    f = open(outputname + '.csv', 'w')

    maxlim = np.max(perhigh)
    step = 0.1*maxlim
    nstep = 2

    for cx1 in data[x].unique():
        f.write('\t' + str(cx1.split('\n')[0]).replace(' ','_'))
    f.write('\n')

    for i in range(len(data[x].unique())):
        cx1 = data[x].unique()[i]
        f.write(str(cx1.split('\n')[0]).replace(' ','_'))
        for j in range(len(data[x].unique())):
            cx2 = data[x].unique()[j]

            d1 = data[data[x] == cx1][y]
            d2 = data[data[x] == cx2][y]
            if statmethod == stats.ttest_ind:
                z, p = statmethod(d1, d2)

            else:

                try:
                    z,p = statmethod(d1, d2)
                except:
                    p = 1

            p = p*bonf

            f.write('\t' + str(p))

            if j > i:
                tx = 'n.s.'
                if p <= 0.05:
                    tx = '*'
                if p <= 0.01:
                    tx = '**'
                if p <= 0.001:
                    tx = '***'

                X = [i, j]
                Y = [maxlim + nstep*step]*2
                nstep +=1

                plt.plot(X, Y, 'k')
                plt.text(np.mean(X), Y[0], tx, verticalalignment='bottom', horizontalalignment='center', fontsize=9)

        f.write('\n')

    f.close()
    if margins is not None:
        plt.subplots_adjust(**margins)
    pl.savefig(outputname)
    plt.close()


def plot_boxplots(statname, outputfolder, **kwargs):

    # sns.set(style="ticks")
    filelib.make_folders([outputfolder])
    margins = kwargs.pop('margins', None)

    exclude = ['Image_name', 'Group']

    stat = pd.read_csv(statname, sep='\t')

    stat = stat.sort_values(['Group']).reset_index(drop=True)
    rotation = kwargs.pop('rotation', 0)

    columns = []
    for c in stat.columns:
        if c not in exclude:
            columns.append(c)

    for c in columns:
        substat = stat[stat[c] > -10000]

        pl = sns.catplot(x='Group', y=c, data=substat, **kwargs)
        pl.set_xticklabels(rotation=rotation)
        if margins is not None:
            plt.subplots_adjust(**margins)
        pl.savefig(outputfolder + c.replace(' ','_') + '.png')
        plt.close()

        # plot_stat(x='Group', y=c, stat=substat, outputname=outputfolder+c.replace(' ', '_')+'_stat.png',
        #           statmethod=stats.ranksums, bonf=1, **kwargs)


def plot_swarmplots(statname, outputfolder, **kwargs):

    # sns.set(style="ticks")
    filelib.make_folders([outputfolder])

    exclude = ['Image_name', 'Group']

    stat = pd.read_csv(statname, sep='\t')

    stat = stat.sort_values(['Group']).reset_index(drop=True)

    columns = []
    for c in stat.columns:
        if c not in exclude:
            columns.append(c)

    for c in columns:
        substat = stat[stat[c] > -10000]

        sns.swarmplot(x='Group', y=c, data=substat, **kwargs)
        margins = {'left': 0.1, 'right': 0.95, 'top': 0.9, 'bottom': 0.2}
        plt.subplots_adjust(**margins)
        plt.savefig(outputfolder + c.replace(' ', '_') + '.png')
        plt.close()





















