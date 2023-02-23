from typing import Any, List, Tuple, Union

import numpy as np
from numpy.typing import NDArray


def log(x: NDArray[np.double], eps: float = 1.0e-8) -> NDArray[np.double]:
    return np.log(np.maximum(x * 255.0, eps))


def var_log(mu, var, eps: float = 1.0e-8):
    """2nd-order delta method"""
    mu2 = mu * mu
    return var / (mu2 + eps) + (var * var) / (2.0 * mu2 * mu2 + eps)


def lin_log(x, threshold=20):
    """
    linear mapping + logarithmic mapping.
    :param x: float or ndarray
        the input linear value in range 0-255 TODO assumes 8 bit
    :param threshold: float threshold 0-255
        the threshold for transition from linear to log mapping
    Returns: the log value
    """
    # converting x into np.float64.
    if x.dtype is not np.float64:  # note float64 to get rounding to work
        x = x.astype(np.float64)

    x = np.maximum(x * 255.0, 1.0e-8)
    f = (1. / threshold) * np.log(threshold)
    y = np.where(x <= threshold, x * f, np.log(x))

    rounding = 1e8
    y = np.round(y * rounding) / rounding

    return y.astype(x.dtype)


def var_lin_log(mu, var, threshold=20):
    if mu.dtype is not np.float64:
        mu = mu.astype(np.float64)

    if var.dtype is not np.float64:
        var = var.astype(np.float64)

    mu = np.maximum(mu * 255.0, 1.0e-8)
    var = var * 255.0 * 255.0
    f = (1. / threshold) * np.log(threshold)
    ret = np.where(mu <= threshold, f * f * var, var_log(mu, var))

    rounding = 1e8
    ret = np.round(ret * rounding) / rounding

    ret = ret.astype(mu.dtype)
    return ret


def wgt(x: NDArray[np.double], b: Union[float, NDArray[np.double]]) -> NDArray[np.double]:
    """Epanechnikov kernel"""
    # assert x.ndim == 2
    # assert isinstance(b, float) or x.shape[1] == len(b)

    eps = 1.0e-8
    t = x / b
    w = np.maximum(0.75 * (1.0 - t * t), eps).prod(axis=1)
    return w / w.sum()


def lstsq(X: NDArray[np.double], Y: NDArray[np.double], W: NDArray[np.double]) -> NDArray[np.double]:
    """
    Solve weighted least squares following the original WLR paper
    """
    XtW = X.T @ W
    A = XtW @ X
    b = XtW @ Y
    return np.linalg.pinv(A, rcond=1.0e-8) @ b


def residue(X_diff: NDArray[np.double], Y: NDArray[np.double], alph: float, beta: NDArray[np.double],
            bb: Union[float, NDArray[np.double]]) -> float:
    ww = wgt(X_diff, bb)
    diff = Y - alph - X_diff @ beta
    return (ww * diff * diff).sum()


def solve_wlr_simple(x: NDArray[np.double],
                     Y: NDArray[np.double],
                     i_ctr: int,
                     bb=None,
                     x_var=None,
                     Y_var=None,
                     spp=None):
    """
    X: [n, D]
    Y: [n, 1]
    """
    n = x.shape[0]
    bb = 1.0
    ones = np.ones((n, 1), dtype=x.dtype)
    x_diff = x - x[i_ctr:i_ctr + 1, :]

    X = np.concatenate([ones, x_diff], axis=1)
    ww = wgt(x_diff, bb)
    ans = lstsq(X, Y, np.diag(ww))
    alph = ans[0]
    beta = ans[1:]
    resd = residue(x_diff, Y, alph, beta, bb)
    return alph, beta, resd, None, bb


def solve_wlr_tsvd(x: NDArray[np.double], Y: NDArray[np.double], i_ctr: int, bb=None, x_var=None, Y_var=None, spp=None):
    n, D = x.shape
    C = 2.0
    assert x_var is not None, '"tsvd" WLR solver requires feature variances!'

    x_mean = np.mean(x, axis=0, keepdims=True)
    _, S, Vh = np.linalg.svd(x - x_mean)

    # PCA
    # NOTE:
    # When we define "E" as in the original WLR paper, we obtained extremely
    # blurry results, so we divide it with the window size.
    E = np.sqrt(x_var) / n
    thres = np.linalg.norm(E, 2) * C
    k = np.sum(S > thres)
    Vp = Vh.T[:, :k]
    x_diff = x - x[i_ctr:i_ctr + 1, :]
    z_diff = x_diff @ Vp

    # WLR to determine b_j
    ones = np.ones((n, 1), dtype=z_diff.dtype)
    if bb is None:
        Z = np.concatenate([ones, z_diff, z_diff * z_diff], axis=1)
        ww = wgt(z_diff, 1.0)
        ans = lstsq(Z, Y, np.diag(ww))
        gamma = ans[1 + k:]
        bb = 1.0 / (np.sqrt(np.abs(2.0 * gamma)) + 1.0e-8)

    # WLR
    Z = np.concatenate([ones, z_diff], axis=1)
    ww = wgt(z_diff, bb)
    ans = lstsq(Z, Y, np.diag(ww))
    alph = ans[0]
    beta = ans[1:]
    resd = residue(z_diff, Y, alph, beta, bb)
    return alph, beta, resd, Vp, bb


def solve_wlr_varopt(x: NDArray[np.double],
                     Y: NDArray[np.double],
                     i_ctr: int,
                     bb=None,
                     x_var=None,
                     Y_var=None,
                     spp=None):
    n, D = x.shape
    C = 2.0
    assert x_var is not None, '"varopt" WLR solver requires feature variances!'
    assert Y_var is not None, '"varopt" WLR solver requires brightness variances!'
    assert spp > 0, '"varopt" WLR solver requires samples per pixel!'

    x_mean = np.mean(x, axis=0, keepdims=True)
    _, S, Vh = np.linalg.svd(x - x_mean)

    # PCA
    # NOTE:
    # When we define "E" as in the original WLR paper, we obtained extremely
    # blurry results, so we divide it with the window size.
    E = np.sqrt(x_var) / n
    thres = np.linalg.norm(E, 2) * C
    k = np.sum(S > thres)
    Vp = Vh.T[:, :k]
    x_diff = x - x[i_ctr:i_ctr + 1, :]
    z_diff = x_diff @ Vp

    # WLR to determine b_j
    ones = np.ones((n, 1), dtype=z_diff.dtype)
    if bb is None:
        Z = np.concatenate([ones, z_diff, z_diff * z_diff], axis=1)
        ww = wgt(z_diff, 1.0)
        ans = lstsq(Z, Y, np.diag(ww))
        gamma = ans[1 + k:]
        bb = 1.0 / (np.sqrt(np.abs(2.0 * gamma)) + 1.0e-8)

    # Optimize h
    h_list = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    var_list = []
    bias_list = []
    for h in h_list:
        Z = np.concatenate([ones, z_diff], axis=1)
        ww = wgt(z_diff, bb * h)
        W = np.diag(ww)
        A = np.linalg.pinv(Z.T @ W @ Z, rcond=1.0e-8) @ (Z.T @ W)
        ls = A[0, :]
        var = np.sum(ls * ls * Y_var)
        bias = ls @ Y - Y[i_ctr]
        var_list.append(var)
        bias_list.append(bias)

    var_list = np.array(var_list)
    bias_list = np.array(bias_list)

    h2_list = h_list * h_list
    h2_mean = np.mean(h2_list)
    bias_mean = np.mean(bias_list)
    lmb1 = np.sum((bias_list - bias_mean) * (h2_list - h2_mean)) \
        / (np.sum((h2_list - h2_mean)**2) + 1.0e-8)
    # lmb0 = bias_mean - lmb1 * h2_mean

    hk_inv_list = 1.0 / (h_list**k)
    hk_inv_mean = np.mean(hk_inv_list)
    var_mean = np.mean(var_list)
    kap1 = spp * np.sum((var_list - var_mean) * (hk_inv_list - hk_inv_mean)) \
        / (np.sum((hk_inv_list - hk_inv_mean)**2) + 1.0e-8)
    # kap0 = var_mean - kap0 * hk_inv_mean

    h_opt = (max(0.0, k * kap1) / (4.0 * lmb1 * lmb1 * spp + 1.0e-8))**(1.0 / (k + 4))
    h_opt = np.clip(h_opt, h_list[0], h_list[-1])

    # WLR
    Z = np.concatenate([ones, z_diff], axis=1)
    ww = wgt(z_diff, bb * h_opt)
    ans = lstsq(Z, Y, np.diag(ww))
    alph = ans[0]
    beta = ans[1:]
    resd = residue(z_diff, Y, alph, beta, bb * h_opt)
    return alph, beta, resd, Vp, bb * h_opt


def wlr_denoise(gray_imgs: NDArray[np.double],
                gbuf_imgs: NDArray[np.double],
                var_gray_imgs: NDArray[np.double],
                var_gbuf_imgs: NDArray[np.double],
                pixel: Tuple[int, int],
                ksize: int = 5,
                spp: int = -1,
                method: str = 'varopt'):

    N, H, W, D = gbuf_imgs.shape
    px, py = pixel
    kh = (ksize - 1) // 2
    kl = max(0, px - kh)
    kr = min(px + kh, W - 1) + 1
    kb = max(0, py - kh)
    kt = min(py + kh, H - 1) + 1

    cx = px - kl
    cy = py - kb
    i_ctr = cy * (kr - kl) + cx

    gray_samples = gray_imgs[:, kb:kt, kl:kr]
    gbuf_samples = gbuf_imgs[:, kb:kt, kl:kr, :]
    var_gray_samples = var_gray_imgs[:, kb:kt, kl:kr]
    var_gbuf_samples = var_gbuf_imgs[:, kb:kt, kl:kr, :]

    gbuf_min = np.amin(gbuf_samples, axis=(1, 2), keepdims=True).reshape((N, -1, D))
    gbuf_max = np.amax(gbuf_samples, axis=(1, 2), keepdims=True).reshape((N, -1, D))

    gray_samples = gray_samples.reshape((N, -1))
    var_gray_samples = var_gray_samples.reshape((N, -1))
    gbuf_samples = gbuf_samples.reshape((N, -1, D))
    var_gbuf_samples = var_gbuf_samples.reshape((N, -1, D))

    gbuf_samples = (gbuf_samples - gbuf_min) / (gbuf_max - gbuf_min + 1.0e-8)
    var_gbuf_samples = var_gbuf_samples / ((gbuf_max - gbuf_min)**2 + 1.0e-8)

    if method == 'simple':
        wlr_method = solve_wlr_simple
    elif method == 'tsvd':
        wlr_method = solve_wlr_tsvd
    elif method == 'varopt':
        wlr_method = solve_wlr_varopt
    else:
        raise RuntimeError('Unknown WLR type: ' + method)

    alph_values = []
    for gbuf, gray, var_gbuf, var_gray in zip(gbuf_samples, gray_samples, var_gbuf_samples, var_gray_samples):
        wlr_res = wlr_method(gbuf, gray, i_ctr, x_var=var_gbuf, Y_var=var_gray, spp=spp)
        alph_values.append(wlr_res[0])

    alph_values = np.array(alph_values, dtype=gray_imgs.dtype)
    return alph_values


def esim_simple(gray_values: NDArray[np.double],
                times: NDArray[np.double],
                pixel: Tuple[float, float],
                threshold: float = 0.2) -> List[Any]:
    px, py = pixel
    log_gray_values = lin_log(gray_values)
    prev_log_gray = log_gray_values[0]
    event_list = []
    for t, log_gray in zip(times, log_gray_values):
        current = log_gray - prev_log_gray
        if abs(current) > threshold:
            prev_log_gray = log_gray
            event_list.append((px, py, t, np.sign(current)))

    return event_list


def esim_ours(log_gray_imgs: NDArray[np.double],
              gbuf_imgs: NDArray[np.double],
              var_log_gray_imgs: NDArray[np.double],
              var_gbuf_imgs: NDArray[np.double],
              times: NDArray[np.double],
              pixel: Tuple[int, int],
              ksize: int = 5,
              threshold: float = 0.2,
              spp: int = -1,
              method: str = 'varopt') -> Tuple[List[Any], int]:

    N, H, W, D = gbuf_imgs.shape
    px, py = pixel
    kh = (ksize - 1) // 2
    kl = max(0, px - kh)
    kr = min(px + kh, W - 1) + 1
    kb = max(0, py - kh)
    kt = min(py + kh, H - 1) + 1

    cx = px - kl
    cy = py - kb
    i_ctr = cy * (kr - kl) + cx

    log_gray_samples = log_gray_imgs[:, kb:kt, kl:kr]
    var_log_gray_samples = var_log_gray_imgs[:, kb:kt, kl:kr]
    gbuf_samples = gbuf_imgs[:, kb:kt, kl:kr, :]
    var_gbuf_samples = var_gbuf_imgs[:, kb:kt, kl:kr, :]

    gbuf_min = np.amin(gbuf_samples, axis=(1, 2)).reshape((N, -1, D))
    gbuf_max = np.amax(gbuf_samples, axis=(1, 2)).reshape((N, -1, D))

    log_gray_samples = log_gray_samples.reshape((N, -1))
    var_log_gray_samples = var_log_gray_samples.reshape((N, -1))
    gbuf_samples = gbuf_samples.reshape((N, -1, D))
    gbuf_samples = (gbuf_samples - gbuf_min) / (gbuf_max - gbuf_min + 1.0e-8)
    gbuf_centers = gbuf_samples[:, i_ctr:i_ctr + 1, :]

    var_gbuf_samples = var_gbuf_samples.reshape((N, -1, D))
    var_gbuf_samples = var_gbuf_samples / ((gbuf_max - gbuf_min)**2 + 1.0e-8)

    if method == 'simple':
        wlr_method = solve_wlr_simple
    elif method == 'tsvd':
        wlr_method = solve_wlr_tsvd
    elif method == 'varopt':
        wlr_method = solve_wlr_varopt
    else:
        raise RuntimeError('Unknown WLR type: ' + method)

    n_wlr = 0
    thres2 = threshold**2
    prev_log_alph, prev_beta, prev_res, prev_Vp, prev_bb =\
         wlr_method(gbuf_samples[0], log_gray_samples[0], i_ctr, x_var=var_gbuf_samples[0],
                    Y_var=var_log_gray_samples[0], spp=spp)

    prev_i = 0
    event_list = []
    for i, t in enumerate(times):
        gbuf_diff_i = gbuf_samples[prev_i] - gbuf_centers[i]
        if prev_Vp is not None:
            gbuf_diff_i = gbuf_diff_i @ prev_Vp
        res = residue(gbuf_diff_i, log_gray_samples[prev_i], prev_log_alph, prev_beta, prev_bb)

        if abs(res - prev_res) > thres2:
            n_wlr += 1
            log_alph, beta, res, Vp, bb = wlr_method(gbuf_samples[i],
                                                     log_gray_samples[i],
                                                     i_ctr,
                                                     x_var=var_gbuf_samples[i],
                                                     Y_var=var_log_gray_samples[i],
                                                     spp=spp)

            current = log_alph - prev_log_alph
            if abs(current) > threshold:
                prev_log_alph = log_alph
                prev_beta = beta
                prev_res = res
                prev_Vp = Vp
                prev_bb = bb
                prev_i = i
                event_list.append((px, py, t, np.sign(current)))

    return event_list, n_wlr
