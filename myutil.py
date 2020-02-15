from __future__ import print_function
from builtins import input
from builtins import range

#import pyfftw   # See https://github.com/pyFFTW/pyFFTW/issues/40
import numpy as np
import functools
import operator
import matplotlib.pyplot as mplot
mplot.rcParams["axes.grid"] = False
import math
import pprint
import os
import shutil
import time

import numpy as np
from scipy.linalg import toeplitz
from sporco.dictlrn import cbpdndl
from sporco.admm import cbpdn
from sporco import util
from sporco import plot
from sporco import cnvrep
import sporco.linalg as sl
import sporco.metric as sm
from sporco.admm import ccmod

from skimage.measure import compare_ssim
from skimage.measure import compare_psnr
plot.config_notebook_plotting()

def l2norm(A):
    l2norm = np.sum( abs(A)*abs(A) )
    return l2norm

def l0norm(A, threshold):
    return np.where(abs(A) < threshold, 0, 1).sum()

def strict_l0norm(A):
    return np.where(A == 0, 0, 1).sum()

def smoothedl0norm(A, sigma):
    N = functools.reduce(operator.mul, A.shape)
    # exp = np.sum( np.exp(-(A*A)/(2*sigma*sigma)) )
    # print(exp)
    # l0_norm = N - exp
    EPS = 0.0000001
    A_ = A.flatten()
    l0_norm = 0
    for a in A_:
        if a > EPS:
            l0_norm += 1
    return l0_norm

def getimages():
    exim = util.ExampleImages(scaled=True, zoom=0.5, gray=True)
    S1 = exim.image('barbara.png', idxexp=np.s_[10:522, 100:612])
    S2 = exim.image('kodim23.png', idxexp=np.s_[:, 60:572])
    S3 = exim.image('monarch.png', idxexp=np.s_[:, 160:672])
    S4 = exim.image('sail.png', idxexp=np.s_[:, 210:722])
    S5 = exim.image('tulips.png', idxexp=np.s_[:, 30:542])
    return np.dstack((S1, S2, S3, S4, S5))

def getdict(dsz):
    if type(dsz[0]) is int:
        return np.random.standard_normal(dsz)
    else:
        num_filt = 0
        tmp = [0 for i in range(len(dsz[0]))]
        for i in dsz:
            i = list(i)
            num_filt = num_filt + i[len(i)-1]
            if i[0] > tmp[0]:
                tmp = i
        tmp[len(tmp)-1] = num_filt
        return cnvrep.bcrop(np.random.standard_normal(tuple(tmp)),dsz)
    
def saveimg(img, filename, title=None):
    fig = plot.figure(figsize=(7, 7))
    plot.imview(img, fig=fig)
    fig.savefig(filename)
    plot.close()
    mplot.close()

# imgs.shape == (R, C, imgR, imgC) or (C, imgR, imgC)
def saveimg2D(imgs, filename, titles=None):
    if imgs.ndim == 3:
        imgs = np.array([imgs])
    if titles is not None and titles.ndim == 3:
        titles = np.array([titles])
    R = imgs.shape[0]
    C = imgs.shape[1]
    fig = plot.figure(figsize=(7*C, 7*R))
    for r in range(R):
        for c in range(C):
            ax = fig.add_subplot(R, C, r*C + c + 1)
            s = None
            if titles is not None:
                s = titles[r][c]
            plot.imview(imgs[r][c], title=s, fig=fig, ax=ax)
    plot.savefig(filename)
    plot.close()
    mplot.close()

# be careful of non-robust implementation
def format_sig(signal):
    return np.transpose(signal, (3, 0, 1, 2, 4)).squeeze()

def saveXimg(cri, Xr, filename):
    # print(Xr.shape)
    X = np.sum(abs(Xr), axis=cri.axisM).squeeze()
    fig = plot.figure(figsize=(7, 7))
    plot.imview(X, cmap=plot.cm.Blues, fig=fig)
    fig.savefig(filename)
    plot.close()
    mplot.close()

def saveXhist(Xr, filename):
    Xr_ = abs(Xr.flatten())
    fig = plot.figure(figsize=(7*10, 7))
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(Xr_, bins=500, density=True)
    fig.savefig(filename)
    plot.close()
    mplot.close()

def save_result(D0, D, X, S, S_reconstructed, filename):
    titles = [[], []]
    r1 = []
    for k in range(S.shape[-1]):
        r1.append(S.T[k].T)
        titles[0].append('')
    r1.append(util.tiledict(D0))
    titles[0].append('')
    r2 = []
    for k in range(S.shape[-1]):
        r2.append(S_reconstructed.T[k].T)
        psnr = sm.psnr(S.T[k].T, S_reconstructed.T[k].T)
        ssim = compare_ssim(S.T[k].T, S_reconstructed.T[k].T)
        l0 = strict_l0norm(np.rollaxis(X, 2)[k])
        titles[1].append("PSNR: %.3fdb\nSSIM: %.4f\nl0norm: %d" % (psnr, ssim, l0))
    r2.append(util.tiledict(D))
    titles[1].append('')
    saveimg2D(np.array([r1, r2]), filename, np.array(titles))

def compressedXk(Xrk, size_rate):
    Xrk = Xrk.copy()
    X_flat = np.ravel(Xrk)
    n = math.ceil(X_flat.size*(1 - size_rate))
    print(str(X_flat.size) + " -> " + str(X_flat.size - n))
    for i in np.argsort(abs(X_flat))[0:n]:
        X_flat[i] = 0
    return Xrk

def to_inative(X, sigma):
    return np.where(X < sigma, 0, X)

# a specific axis to 1-length
# copied
def compress_axis(A, axis, i):
    idx = [slice(None)]*A.ndim
    idx[axis] = slice(i, i + 1)
    return A[tuple(idx)]

def compress_axis_op(A, axis, i):
    idx = [slice(None)]*A.ndim
    idx[axis] = slice(i, i + 1)
    return tuple(idx)

def reconstruct(cri, Dr, Xr):
    Xf = sl.rfftn(Xr, s=cri.Nv, axes=cri.axisN)
    Df = sl.rfftn(Dr, s=cri.Nv, axes=cri.axisN)
    return sl.irfftn(sl.inner(Df, Xf, axis=cri.axisM), s=cri.Nv, axes=cri.axisN)

def save_reconstructed(cri, Dr, Xr, Sr, filename, Sr_add=None):
    Sr_ = reconstruct(cri, Dr, Xr)
    if Sr_add is None:
        Sr_add = np.zeros_like(Sr)
    img = np.stack((format_sig(Sr + Sr_add), format_sig(Sr_ + Sr_add)), axis=1)
    saveimg2D(img, filename)

def compressedX(cri, Xr, Sr, size_rate):
    Xr_cmp = Xr.copy()
    for k in range(cri.K):
        s = compress_axis_op(Xr_cmp, cri.axisK, k)
        Xr_cmp[s] = compressedXk(Xr_cmp[s], (Sr.size / Xr.size)*size_rate)
    return Xr_cmp

def calcXr(cri, Dr, Sr, lmbda=5e-2):
    opt = cbpdn.ConvBPDN.Options({'Verbose': True, 'MaxMainIter': 200,
                                  'RelStopTol': 5e-3, 'AuxVarObj': False})
    b = cbpdn.ConvBPDN(Dr.squeeze(), Sr.squeeze(), lmbda, opt, dimK=cri.dimK, dimN=cri.dimN)
    Xr = b.solve()
    return Xr

def evaluate_result(cri, Dr0, Dr, Sr, Sr_add=None, lmbda=5e-2, title='result.png'):
    Xr_ = calcXr(cri, Dr, Sr, lmbda)
    print("strict l0 norm", strict_l0norm(Xr_))
    print("l2norm: ", l2norm(Xr_))
    for k in range(cri.K):
        print("image %d: strict l0 norm %f" % (k, strict_l0norm(compress_axis(Xr_, cri.axisK, k))))
    if Sr_add is None:
        Sr_add = np.zeros_like(Sr)
    save_result(Dr0.squeeze(), Dr.squeeze(), Xr_.squeeze(), (Sr + Sr_add).squeeze(), (reconstruct(cri, Dr, Xr_) + Sr_add).squeeze(), title)

def l2norm_minimize(cri, Dr, Sr):
    Df = sl.rfftn(Dr, s=cri.Nv, axes=cri.axisN) # implicitly zero-padding
    Sf = sl.rfftn(Sr, s=cri.Nv, axes=cri.axisN) # implicitly zero-padding
    Xf = np.conj(Df) / sl.inner(Df, np.conj(Df), axis=cri.axisM) * Sf
    Xr = sl.irfftn(Xf, s=cri.Nv, axes=cri.axisN)

    return Xr

def derivD_spdomain(cri, Xr, Sr, Df, Xf, dict_Nv):
    B = sl.irfftn(sl.inner(Df, Xf, axis=cri.axisM), s=cri.Nv, axes=cri.axisN) - Sr
    B = B[np.newaxis, np.newaxis,]
    Xshifted = np.ones(dict_Nv + Xr.shape) * Xr
    
    N1 = 0
    N2 = 1
    I = 2
    J = 3

    print("start shifting")
    for n1 in range(dict_Nv[0]):
        for n2 in range(dict_Nv[1]):
            Xshifted[n1][n2] = np.roll(Xshifted[n1][n2], (n1, n2), axis=(I, J))
            # print("shifted ", (n1, n2))
    ret = np.sum(np.conj(B) * Xshifted, axis=(I, J, 2 + cri.axisK), keepdims=True)
    print(ret.shape)
    ret = ret[:, :, 0, 0]
    print(ret.shape)
    return ret

def goldenRatioSearch(function, rng, cnt):
    # 黄金探索法によるステップ幅の最適化
    gamma = (-1+np.sqrt(5))/2
    a = rng[0]
    b = rng[1]
    p = b-gamma*(b-a)
    q = a+gamma*(b-a)
    Fp = function(p)
    Fq = function(q)
    width = 1e8
    for i in range(cnt):
        if Fp <= Fq:
            b = q
            q = p
            Fq = Fp
            p = b-gamma*(b-a)
            Fp = function(p)
        else:
            a = p
            p = q
            Fp = Fq
            q = a+gamma*(b-a)
            Fq = function(q)
            width = abs(b-a)/2
    alpha = (a+b)/2
    return alpha

# 下に凸
def ternary_search(f, rng, cnt):
    left = rng[0]
    right = rng[1]
    for i in range(cnt):
        if f((left * 2 + right) / 3) > f((left + right * 2) / 3):
            left = (left * 2 + right) / 3
        else:
            right = (left + right * 2) / 3
    return (left + right) / 2

def min_max(x, axis=None):
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    return (x - min) / (max - min)

def zscore(x, axis = None):
    xmean = x.mean(axis=axis, keepdims=True)
    xstd  = np.std(x, axis=axis, keepdims=True)
    zscore = (x-xmean)/xstd
    return zscore

def normalize(v, axis=-1, order=2):
    l2 = np.linalg.norm(v, ord=order, axis=axis, keepdims=True)
    l2[l2==0] = 1
    return v/l2
