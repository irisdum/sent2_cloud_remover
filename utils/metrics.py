# File where the metrics to compare the simulated image quality are implemented

import numpy as np
from skimage.measure import compare_ssim as ssim


def calculate_psnr(img1, img2, max_value=255):
    """"Calculating peak signal-to-noise ratio (PSNR) between two images."""
    mse = np.mean((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse)))


def psnr_by_band(im1, im2, max_value):
    """Compute the psnr on each band of the two images"""
    assert im1.shape == im2.shape, "PSNR input image does not have the same shape {} {}".format(im1.shape, im2.shape)
    assert len(im1.shape) == 3, "wrong dimension input should be (n*n*nchannel) not {}".format(im1.shape)
    nb = im1.shape[2]
    l = [0] * nb
    for b in range(nb):  # go over all the bands
        l[b] = calculate_psnr(im1[:, :, b], im2[:, :, b], max_value)
    return l, np.mean(l)


def batch_psnr(batch1, batch2, max_value=1):
    """Compute the psnr value on a batch of images"""
    assert batch1.shape == batch2.shape, "Batch 1 {} does not have the same dim as batch 2 {}".format(batch1.shape,
                                                                                                      batch2.shape)
    assert len(batch1.shape) == 4, "Wrong batch dim should be (num_batch,n,n,nchannel)"
    psnr_batch = []
    for i in range(batch1.shape[0]):  # go over all the images in the batch
        _, psnr = psnr_by_band(batch1[i, :, :, :], batch2[i, :, :, :], max_value)
        psnr_batch += [psnr]

    return psnr_batch, np.mean(psnr_batch)


def ssim_batch(batch1, batch2):
    assert batch1.shape == batch2.shape, "Batch 1 {} does not have the same dim as batch 2 {}".format(batch1.shape,
                                                                                                      batch2.shape)
    assert len(batch1.shape) == 4, "Wrong batch dim should be (num_batch,n,n,nchannel)"
    lssim_batch = []
    for i in range(batch1.shape[0]):  # go over all the images in the batch
        lssim_batch += [ssim(batch1[i, :, :, :], batch2[i, :, :, :], multichannel=True)]
    return lssim_batch, np.mean(lssim_batch)


def compute_metric(gt, gen_img):
    _, psnr = batch_psnr(gt, gen_img)
    _, ssim = ssim_batch(gt, gen_img)
    l_name=["psnr", "ssim"]
    l_val=[psnr, ssim]
    return l_name,l_val
