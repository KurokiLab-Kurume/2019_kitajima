import numpy as np
import sporco.linalg as sl
from sporco import util
from sporco import cnvrep
import myutil 

def l2norm_minimize(cri, Dr, Sr):

    Df = sl.rfftn(Dr, s=cri.Nv, axes=cri.axisN) # implicitly zero-padding
    Sf = sl.rfftn(Sr, s=cri.Nv, axes=cri.axisN) # implicitly zero-padding
    Xf = np.conj(Df) / sl.inner(Df, np.conj(Df), axis=cri.axisM) * Sf

    return sl.irfftn(Xf, s=cri.Nv, axes=cri.axisN)

def convBPDN(
    cri, Dr0, Sr,
    final_sigma,
    maxitr,
    non_nega ,
    param_mu = 1,
    debug_dir = None
):
    Dr = Dr0.copy()
    Sr = Sr.copy()

    # 係数をl2ノルム最小解で初期化
    Xr = l2norm_minimize(cri, Dr, Sr)

    # 2次元離散フーリエ変換
    Df = sl.rfftn(Dr, s=cri.Nv, axes=cri.axisN)
    Sf = sl.rfftn(Sr, s=cri.Nv, axes=cri.axisN)
    Xf = sl.rfftn(Xr, s=cri.Nv, axes=cri.axisN)
    alpha = 1e0

    # sigma set
    first_sigma = Xr.max()*4
    # σを更新する定数c(c<1)の決定
    c = (final_sigma / first_sigma) ** (1/(maxitr - 1))
    print("c = %.8f" % c)
    sigma_list = []
    sigma_list.append(first_sigma)
    for i in range(maxitr - 1):
        sigma_list.append(sigma_list[i]*c)
    
    updcnt = 0
    for sigma in sigma_list:
        print("sigma = %.8f" % sigma)
        
        # 係数の勾配降下
        delta = Xr * np.exp(-(Xr*Xr) / (2*sigma*sigma))
        Xr = Xr - param_mu*delta
        Xf = sl.rfftn(Xr, cri.Nv, cri.axisN)
        print("l0norm = %i" % np.where(abs(Xr.transpose(3,4,2,0,1).squeeze()[0]) < final_sigma, 0, 1).sum(), end=" ")

        # 係数の射影
        Df = sl.rfftn(Dr, s=cri.Nv, axes=cri.axisN)
        b = sl.inner(Df, Xf, axis=cri.axisM) - Sf
        c = sl.inner(Df, np.conj(Df), axis=cri.axisM)
        Xf = Xf - np.conj(Df) / c * b
        Xr = sl.irfftn(Xf, s=cri.Nv, axes=cri.axisN)
        # Xr = np.where(Xr < 0, 0, Xr) # 非負制約
        print("l0norm = %i" % np.where(abs(Xr.transpose(3,4,2,0,1).squeeze()[0]) < final_sigma, 0, 1).sum(), end=" ")
       
        # Xr = np.where(Xr < 1e-7, 0, Xr)
        updcnt += 1
    
    return Xr

def convDictLearn(
    cri, Dr0, dsz, Sr,
    final_sigma,
    maxitr,
    non_nega,
    param_mu = 1,
    debug_dir = None
):
    Dr = Dr0.copy()
    Sr = Sr.copy()

    # 係数をl2ノルム最小解で初期化
    Xr = l2norm_minimize(cri, Dr, Sr)

    # 2次元離散フーリエ変換
    Df = sl.rfftn(Dr, s=cri.Nv, axes=cri.axisN)
    Sf = sl.rfftn(Sr, s=cri.Nv, axes=cri.axisN)
    Xf = sl.rfftn(Xr, s=cri.Nv, axes=cri.axisN)
    alpha = 1e0

    # sigma set
    first_sigma = Xr.max()*4
    # σを更新する定数c(c<1)の決定
    c = (final_sigma / first_sigma) ** (1/(maxitr - 1))
    print("c = %.8f" % c)
    sigma_list = []
    sigma_list.append(first_sigma)
    for i in range(maxitr - 1):
        sigma_list.append(sigma_list[i]*c)
    
    # 辞書のクロップする領域を添え字で指定
    crop_op = []
    for l in Dr.shape:
        crop_op.append(slice(0, l))
    crop_op = tuple(crop_op)
    
    # 射影関数のインスタンス化
    Pcn = cnvrep.getPcn(dsz, cri.Nv, cri.dimN, cri.dimCd, zm=False)

    updcnt = 0
    for sigma in sigma_list:
        print("sigma = %.8f" % sigma, end=" ")
        
        # 係数の勾配降下
        delta = Xr * np.exp(-(Xr*Xr) / (2*sigma*sigma))
        Xr = Xr - param_mu*delta
        Xf = sl.rfftn(Xr, cri.Nv, cri.axisN)
        print("l0norm = %i" % np.where(abs(Xr.transpose(3,4,2,0,1).squeeze()[0]) < final_sigma, 0, 1).sum(), end=" ")

        # 辞書の勾配降下
        B = sl.inner(Xf, Df, axis=cri.axisM) - Sf
        derivDf = sl.inner(np.conj(Xf), B, axis=cri.axisK)
        def func(alpha):
            Df_ = Df - alpha * derivDf
            Dr_ = sl.irfftn(Df_, s=cri.Nv, axes=cri.axisN)[crop_op]
            Df_ = sl.rfftn(Dr_, s=cri.Nv, axes=cri.axisN)
            Sf_ = sl.inner(Df_, Xf, axis=cri.axisM)
            return myutil.l2norm(Sr - sl.irfftn(Sf_, s=cri.Nv, axes=cri.axisN))
        
        error_list = np.array([func(alpha / 2), func(alpha), func(alpha * 2)])
        choice = error_list.argmin()
        alpha *= [0.5, 1, 2][choice]
        print("alpha = %.8f" % alpha, end=" ")
        print("error = %5.5f" % error_list[choice], end=" ")
        Df = Df - alpha * derivDf

        # 辞書の射影
        Dr = Pcn(sl.irfftn(Df, s=cri.Nv, axes=cri.axisN))[crop_op] # 正規化とゼロパディングを同時に行う
        # print(myutil.l2norm(Dr.T[0]))


        # 係数の射影
        Df = sl.rfftn(Dr, s=cri.Nv, axes=cri.axisN)
        b = sl.inner(Df, Xf, axis=cri.axisM) - Sf
        c = sl.inner(Df, np.conj(Df), axis=cri.axisM)
        Xf = Xf - np.conj(Df) / c * b
        Xr = sl.irfftn(Xf, s=cri.Nv, axes=cri.axisN)

        if(non_nega): Xr = np.where(Xr < 0, 0, Xr) # 非負制約

        # Xr = np.where(Xr < 1e-6, 0, Xr)
        print("l0norm_projected = %i" % np.where(abs(Xr.transpose(3,4,2,0,1).squeeze()[0]) < final_sigma, 0, 1).sum())
        updcnt += 1
    
    return Dr, Xr