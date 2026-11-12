"""
Plotting helpers for KHCNN test scripts.

Based on the KHCNN code by M. Olivieri
(Olivieri et al., Sensors 2021, doi:10.3390/s21237834).
"""

import os
import sys

this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(this_dir, "../"))  # to import other module

import matplotlib.pyplot as plt
import numpy as np


def plot_history(history):
    """Plot the validation loss curves stored in the training history dict."""
    plt.figure()
    plt.plot(history["val_real_p_h_hat_loss"], label="p_h_hat_val_real")
    plt.plot(history["val_imag_p_h_hat_loss"], label="p_h_hat_val_imag")
    plt.plot(history["val_real_v_s_hat_loss"], label="v_s_hat_val_real")
    plt.plot(history["val_imag_v_s_hat_loss"], label="v_s_hat_val_imag")
    plt.legend()
    plt.title("Losses")
    plt.xlabel("epochs")
    plt.ylabel("loss [log-scale]")
    plt.yscale('log')
    plt.grid()
    plt.show()


def plot_real_imag(pred, gt, quantity, index, size):
    """
    Plot the real and imag part of prediction and ground truth
    :param pred: prediction dataset
    :param gt: groundtruth dataset
    :param quantity: string specifing physics
    :param index: idx of dataset to plot
    :param size: size of the pressure image
    :return:
    """
    plt.figure()
    plt.subplot(221)
    plt.imshow(np.real(pred[index]).reshape(size[0], size[1]), extent=(0, 63, 15, 0)), plt.colorbar()
    plt.title("real_" + quantity + "_hat")
    plt.subplot(222)
    plt.imshow(np.real(gt[index]).reshape(size[0], size[1]), extent=(0, 63, 15, 0)), plt.colorbar()
    plt.title("real_" + quantity + "_gt")
    plt.subplot(223)
    plt.imshow(np.imag(pred[index]).reshape(size[0], size[1]), extent=(0, 63, 15, 0)), plt.colorbar()
    plt.title("imag_" + quantity + "_hat")
    plt.subplot(224)
    plt.imshow(np.imag(gt[index]).reshape(size[0], size[1]), extent=(0, 63, 15, 0)), plt.colorbar()
    plt.title("imag_" + quantity + "_gt")
    plt.show()


def plot_abs_phase(pred, gt, quantity, index, size):
    """
    Plot the abs and phase part of prediction and ground truth
    :param pred: prediction dataset
    :param gt: groundtruth dataset
    :param quantity: string specifing physics
    :param index: idx of dataset to plot
    :param size: size of the pressure image
    :return:
    """
    plt.figure()
    plt.subplot(221)
    plt.imshow(np.abs(pred[index]).reshape(size[0], size[1]), extent=(0, 63, 15, 0)), plt.colorbar()
    plt.title("abs_" + quantity + "_hat")
    plt.subplot(222)
    plt.imshow(np.abs(gt[index]).reshape(size[0], size[1]), extent=(0, 63, 15, 0)), plt.colorbar()
    plt.title("abs_" + quantity + "_gt")
    plt.subplot(223)
    plt.imshow(np.angle(pred[index]).reshape(size[0], size[1]), extent=(0, 63, 15, 0)), plt.colorbar()
    plt.title("phase_" + quantity + "_hat")
    plt.subplot(224)
    plt.imshow(np.angle(gt[index]).reshape(size[0], size[1]), extent=(0, 63, 15, 0)), plt.colorbar()
    plt.title("phase_" + quantity + "_gt")
    plt.show()


def print_information(i, mode_test=[], freq_test=[], num_plate_test=[], boundary_test=[], plate_dimensions_test=[],
                      snr_test=[]):
    """
    Print and return all the information relate to a specified test case. If pass "" -> not print that information
    :param i: index to print information
    :param all the test set infotmation
    :return: information
    """
    information = ""
    if len(boundary_test) != 0:
        information += f"bound:\t{boundary_test[i]}\n"
    if len(freq_test) != 0:
        information += f"freq:\t{np.round(freq_test[i], 2)} Hz\n"
    if len(mode_test) != 0:
        information += f"mode:\t{mode_test[i]}\n"
    if len(num_plate_test) != 0:
        information += f"plate:\t{num_plate_test[i]}\n"
    if len(plate_dimensions_test) != 0:
        information += f"dim:\t{plate_dimensions_test[i]} m\n"
    if len(snr_test) != 0:
        information += f"snr:\t{snr_test[i]} dB\n"

    information += f"index:\t{i}\n"

    print(information)
    return information


def plot_NMSE_complex(nmse_mean_db, nmse_db, quantity):
    """Plot the sorted per-sample NMSE values together with their mean."""
    fig_nmse = plt.figure()
    print(f"Mean NMSE = {nmse_mean_db} dB")
    plt.plot(np.sort(nmse_db), label="nmse")
    plt.plot(np.ones(nmse_db.shape) * nmse_mean_db, label="mean(nmse)")
    plt.title("NMSE({})".format(quantity))
    plt.legend()
    plt.grid()
    plt.show()

    return fig_nmse

def plot_NCC_complex(ncc_mean, ncc, quantity):
    """Plot the sorted per-sample NCC values together with their mean."""
    fig_ncc = plt.figure()
    print(f"Mean NCC = {ncc_mean}")
    plt.plot(np.sort(ncc), label="ncc")
    plt.plot(np.ones(ncc.shape) * ncc_mean, label="mean(ncc)")
    plt.title("NCC({})".format(quantity))
    plt.legend()
    plt.grid()
    plt.show()

    return fig_ncc


def plot_metrics_for_bands(metric, freqs, band_limits, quantity):
    """
    Plot the metrics as a function of frequency bands
    :param metric: per-sample metric values
    :param freqs: per-sample frequencies
    :param band_limits: frequency band edges [Hz]
    :param quantity: "NMSE" or "NCC"
    :return:
    """
    metrics_band = {}

    for b_id in range(len(band_limits) - 1):
        index = np.where((freqs > band_limits[b_id]) & (freqs < band_limits[b_id+1]))
        metrics_band[b_id] = metric[index]

    means_band = {}
    mean = []
    std = []

    for b_id in range(len(metrics_band)):
        if quantity == "NCC":
            mean_value = np.mean(metrics_band[b_id])
            means_band[b_id] = mean_value
            mean.append(mean_value)
            std.append(np.std(metrics_band[b_id]))
        elif quantity == "NMSE":
            linear_nmse = 10 ** (metrics_band[b_id] / 10)
            mean_value = 10 * np.log10(np.mean(linear_nmse))
            means_band[b_id] = mean_value
            mean.append(mean_value)
            std.append(np.std(metrics_band[b_id]))

    x = np.arange(len(mean))
    xticks = []
    for b_id in range(len(band_limits) - 1):
        xticks.append(f"[{band_limits[b_id]}, {band_limits[b_id+1]}]")
    fig = plt.figure()
    plt.errorbar(x, mean, std, fmt='o-', markersize=8, capsize=10)
    plt.title(f"mean & std ({quantity}) as function of freq bands")
    plt.xticks(x, xticks)
    plt.xlabel("Freq bands [Hz]")
    if quantity == "NCC":
        plt.ylabel("[-]")
    elif quantity == "NMSE":
        plt.ylabel("[dB]")
    plt.grid()
    plt.show()
