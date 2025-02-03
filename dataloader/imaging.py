import os
import numpy as np
import pywt.data
from scipy import signal
from pyts.image import GramianAngularField, MarkovTransitionField, RecurrencePlot
from tqdm import tqdm

import torch

"""
https://github.com/PyWavelets/pywt
https://pyts.readthedocs.io/en/stable/modules/image.html
https://github.com/johannfaouzi/pyts
"""


def gray(x):
    num = x.shape[0]
    images = []
    length = int(np.sqrt(x.shape[1]))
    width = int(x.shape[1] / length)
    for i in range(num):
        x_imaging = np.zeros((width, length))
        for j in range(width):
            x_imaging[j, :] = x[i][j*length: (j+1)*length]
        images.append(x_imaging)
    return np.array(images)


def stft(x, fs=1.0, window='hann', nperseg=60, noverlap=50):
    """
    fs: sampling frequency of the time series,
    nperseg: length of each segment,
    noverlap: number of points of overlap between segments. If none then noverlap = nperseg / 2
    """
    num = x.shape[0]
    images = []
    for i in range(num):
        f, t, nd = signal.stft(x[i], fs, window, nperseg, noverlap)
        images.append(abs(nd))
    return np.array(images)


def cwt(x, sampling_rate=1024, totalscal=32, wavename='cmor'):  # cgau8
    num = x.shape[0]
    images = []
    for i in range(num):
        # Central frequency
        fc = pywt.central_frequency(wavename)
        # Calculate the wavelet scale for the corresponding frequency
        cparam = 2 * fc * totalscal
        scales = cparam / np.arange(totalscal, 0, -1)
        [cwtmatr, frequencies] = pywt.cwt(x[i].numpy(), scales, wavename, 1.0 / sampling_rate)
        images.append(abs(cwtmatr))
    return np.array(images)


# def cwt(x, totalscal=32):
#     num = x.shape[0]
#     images = []
#     for i in range(num):
#         cwtmatr = signal.cwt(x[i].numpy(), signal.morlet2, np.arange(1, totalscal+1))
#         images.append(abs(cwtmatr))
#     return np.array(images)


def rp(x):
    imaging_rp = RecurrencePlot(dimension=3, time_delay=2)
    return imaging_rp.transform(x)


def gasf(x):
    imaging_gasf = GramianAngularField(method='summation')
    return imaging_gasf.transform(x)


def gadf(x):
    imaging_gadf = GramianAngularField(method='difference')
    return imaging_gadf.transform(x)


def mtf(x):
    imaging_mtf = MarkovTransitionField()
    return imaging_mtf.transform(x)


def patch_average(images, size=100):
    num = images.shape[0]
    batch = int(images.shape[1] / size)
    patch_images = []
    for i in range(num):
        image = images[i]
        patch = []
        for p in range(size):
            for q in range(size):
                patch.append(np.mean(image[p * batch:(p + 1) * batch, q * batch:(q + 1) * batch]))
        # reshape
        patch_image = np.array(patch).reshape(size, size)
        patch_images.append(patch_image)
    return np.array(patch_images)


def patch_average_optimized(images, size=100):
    # Calculate the size of the patches
    batch = images.shape[1] // size

    # Reshape the images to a new shape that splits the image into patch-sized sub-arrays
    reshaped = images.reshape(images.shape[0], size, batch, size, batch)

    # Calculate the mean across the patch dimensions (2 and 4) and keep the first and third dimensions
    patch_means = reshaped.mean(axis=(2, 4))

    return patch_means


def imaging(inputs, imaging_fn, patch_size=0):
    batch_size = inputs.shape[0]
    images = []
    for i in tqdm(range(batch_size)):
        x = inputs[i]
        x_imaging = imaging_fn(x)
        if patch_size:
            x_imaging = patch_average(x_imaging, size=100)
        images.append(x_imaging)
    return np.array(images)


def pre_imaging(dataset, imaging_fn, data_folder, sub_file, file_index, patch_size=0):
    inputs = dataset['samples']
    labels = dataset['labels']
    num = inputs.shape[0]
    os.makedirs(os.path.join(data_folder, sub_file, str(file_index), str(0)), exist_ok=True)
    os.makedirs(os.path.join(data_folder, sub_file, str(file_index), str(1)), exist_ok=True)
    os.makedirs(os.path.join(data_folder, sub_file, str(file_index), str(2)), exist_ok=True)
    for i in tqdm(range(num)):
        x = inputs[i]
        x_imaging = imaging_fn(x)
        if patch_size:
            x_imaging = patch_average_optimized(x_imaging, size=patch_size)
        x_imaging = torch.from_numpy(x_imaging).half()
        torch.save(x_imaging, os.path.join(data_folder, sub_file, str(file_index), str(labels[i].item()), f"images_{i}.pt"))

