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


def batch_sam(batch1, batch2):
    assert batch1.shape == batch2.shape, "Batch 1 {} does not have the same dim as batch 2 {}".format(batch1.shape,
                                                                                                      batch2.shape)
    assert len(batch1.shape) == 4, "Wrong batch dim should be (num_batch,n,n,nchannel)"
    sam_batch = []
    for i in range(batch1.shape[0]):  # go over all the images in the batch
        _, sam_val = sam(batch1[i, :, :, :], batch2[i, :, :, :])
        sam_batch += [sam_val]
    return sam_batch, np.mean(sam_batch)


def ssim_batch(batch1, batch2):
    assert batch1.shape == batch2.shape, "Batch 1 {} does not have the same dim as batch 2 {}".format(batch1.shape,
                                                                                                      batch2.shape)
    assert len(batch1.shape) == 4, "Wrong batch dim should be (num_batch,n,n,nchannel)"
    lssim_batch = []
    for i in range(batch1.shape[0]):  # go over all the images in the batch
        lssim_batch += [ssim(batch1[i, :, :, :], batch2[i, :, :, :], multichannel=True)]
    return lssim_batch, np.mean(lssim_batch)


def sam(im1, im2):
    """Spectral Angle Mapper"""
    assert im1.shape == im2.shape, "SAM input image does not have the same shape {} {}".format(im1.shape, im2.shape)
    assert len(im1.shape) == 3, "wrong dimension input should be (n*n*nchannel) not {}".format(im1.shape)
    alpha_array = np.zeros((im1.shape[0], im2.shape[0]))
    for i in range(im1.shape[0]):
        for j in range(im1.shape[1]):
            pixel1 = im1[i, j, :]
            pixel2 = im2[i, j, :]
            alpha_array[i, j] = pixels_sam(pixel1, pixel2)
    return alpha_array, np.mean(alpha_array)


def pixels_sam(pixel1, pixel2):
    """pixel 1 and pixel 2 are two array"""
    assert pixel1.shape == pixel2.shape, "The two pixels does not have the same dimension pix 1 : {} pix 2 {}".format(
        pixel1.shape, pixel2.shape)
    assert len(pixel2.shape) == 1, "Wrong dimension input should be a vector not {}".format(pixel2.shape)
    alpha = np.arccos(np.dot(pixel1, pixel2) / (np.linalg.norm(pixel1) * np.linalg.norm(pixel2)))
    return alpha


def compute_metric(gt, gen_img,compute_sam=False):
    _, psnr = batch_psnr(gt, gen_img)
    _, ssim = ssim_batch(gt, gen_img)
    if compute_sam:
        _,bsam=batch_sam(gt,gen_img)
    _, bsam = 2,1
    l_name = ["psnr", "ssim", "sam"]
    l_val = [psnr, ssim, bsam]
    return l_name, l_val
