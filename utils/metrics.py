"""
Name:        "metrics.py"
Author:      Marco Olivieri
Date:        23 November 2021
Description: Complex metrics

Note: this module comes from the KHCNN code by M. Olivieri
(Olivieri et al., Sensors 2021, doi:10.3390/s21237834).
Dataset-level helpers (`nmse_dataset_complex`, `ncc_dataset_complex`) added
for the test and Grad-CAM scripts.
"""

import numpy as np


def nmse(x_hat, x):
    """
    Compute NMSE in complex number of a single image
    :param x_hat: prediction
    :param x: groundtruth
    :return: nmse dB
    """
    x = x.flatten()
    x_hat = x_hat.flatten()
    err = x_hat - x
    nmse = np.sum(err * np.conj(err)) / np.sum(x * np.conj(x))
    nmse = nmse.real  # imaginary part is zero
    return 10 * np.log10(nmse)


def ncc(x_hat, x):
    """
    Compute NCC in complex number of a single image
    :param x_hat: prediction
    :param x: groundtruth
    :return: ncc
    """
    x_hat = x_hat.flatten()
    x = x.flatten()

    num = abs(np.sum(x_hat * np.conj(x)))
    den = np.linalg.norm(x_hat) * np.linalg.norm(x)

    return num/den


def nmse_dataset_complex(out, gt):
    """
    Compute NMSE for every sample of a dataset
    :param out: iterable of predictions
    :param gt: iterable of groundtruths
    :return: (mean NMSE in dB computed on the linear values, per-sample NMSE in dB)
    """
    nmse_db = np.array([nmse(out[i], gt[i]) for i in range(len(out))])
    nmse_mean_db = 10 * np.log10(np.mean(10 ** (nmse_db / 10)))
    return nmse_mean_db, nmse_db


def ncc_dataset_complex(out, gt):
    """
    Compute NCC for every sample of a dataset
    :param out: iterable of predictions
    :param gt: iterable of groundtruths
    :return: (mean NCC, per-sample NCC)
    """
    ncc_values = np.array([ncc(out[i], gt[i]) for i in range(len(out))])
    return np.mean(ncc_values), ncc_values
