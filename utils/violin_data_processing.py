"""
Data processing helpers for the violin top-plate dataset.

Note: this module comes from the KHCNN code by M. Olivieri
(Olivieri et al., Sensors 2021, doi:10.3390/s21237834).
"""

import numpy as np


# normalized complex_data wrt to max(abs)
def normalise_wrt_abs_velocity_violin(complex_velocity):
    """
    Normalisation considering NaN value in complex velocity
    :param complex_velocity:
    :return:
    """
    V = complex_velocity.copy()

    for i in range(V.shape[0]):
        v = V[i].flatten()
        arg = np.argwhere(~np.isnan(v))
        v[arg] = v[arg] / np.max(abs(v[arg]))
        V[i] = v.reshape(V[i].shape)

    V_norm = V.reshape(complex_velocity.shape)

    return V_norm


def _nan_mask_for(reference, target_shape):
    """Boolean NaN mask of `reference`, reshaped to `target_shape`."""
    mask = np.isnan(np.asarray(reference))
    if mask.shape != target_shape:
        if mask.size == int(np.prod(target_shape)):
            mask = mask.reshape(target_shape)
        else:
            mask = np.broadcast_to(mask, target_shape)
    return mask


def apply_nan_mask(v_s_hat, V_test, type="complex"):
    """
    Apply mask to the velocity prediction based on the velocity ground truth shapes
    :param v_s_hat: velocity prediciton from network
    :param V_test: velocity ground truth
    :return: prediction with NaN mask
    """
    v_hat_mask = v_s_hat.copy()
    mask = _nan_mask_for(V_test, v_hat_mask.shape)

    if type == "complex":
        v_hat_mask[mask] = np.nan + 1j*np.nan
    else:
        v_hat_mask[mask] = np.nan
    return v_hat_mask


def apply_zero_mask(v_s_hat, V_test):
    """
    Apply mask to the velocity prediction based on the velocity ground truth shapes
    :param v_s_hat: velocity prediciton from network
    :param V_test: velocity ground truth
    :return: prediction with zero mask
    """
    v_hat_mask = v_s_hat.copy()
    mask = _nan_mask_for(V_test, v_hat_mask.shape)
    v_hat_mask[mask] = 0 + 1j*0
    return v_hat_mask


def get_binary_mask(vel_test):
    """
    Compute binary mask from the NaN values contained in velocities ground truth (where NaN put zeros)
    :param vel_test: velocity ground truth containing NaN values
    :return: binary masks of violins
    """

    bm = np.ones(shape=vel_test.shape)
    bm[_nan_mask_for(vel_test, bm.shape)] = 0

    return bm


def get_internal_points(data):
    """
    Filter data only for the internal points of the shapes
    :param data: data with NaN values in outer points
    :return: data_internal_points list with data values only internal
    """

    data_internal_points = []
    for img in data:
        i = img.flatten()
        outer_points = np.where(np.isnan(i))
        filter_i = np.delete(i, outer_points)
        data_internal_points.append(filter_i)

    return data_internal_points

def split_data_from_indexes(indexes, acpr_test, acpr_downsampling_test, vel_test, bm_test, mode_test, freq_test, num_violin_test, boundary_test, \
h_points_test, h_points_downsampling_test, s_points_test, plate_dimensions_test):
    start=indexes[0]
    end = indexes[-1]
    acpr = acpr_test[start:end]
    acpr_downsampling = acpr_downsampling_test[start:end]
    vel = vel_test[start:end]
    bm = bm_test[start:end]
    mode = mode_test[start:end]
    freq = freq_test[start:end]
    num_violin = num_violin_test[start:end]
    boundary = boundary_test[start:end]
    h_points = h_points_test[start:end]
    h_points_downsampling = h_points_downsampling_test[start:end]
    s_points = s_points_test[start:end]
    plate_dimensions = plate_dimensions_test[start:end]

    return acpr, acpr_downsampling, vel, bm, mode, freq, num_violin, boundary, h_points, h_points_downsampling, s_points, plate_dimensions


def get_data_from_network_ds(dict_ds):
    """
    Get all data from the dictionary dataset used for the networks
    :param dict_ds: dictionary dataset
    :return: all the data
    """

    acpr = dict_ds["P"]
    acpr_downsampling = dict_ds["P_downsampling"]
    vel = dict_ds["V"]
    bm = dict_ds["bm"]
    mode = dict_ds["mode"]
    freq = dict_ds["freq"]
    num_plate = dict_ds["num_violin"]
    boundary = dict_ds["boundary"]
    h_points = dict_ds["h_points"]
    h_points_downsampling = dict_ds["h_points_downsampling"]
    s_points = dict_ds["s_points"]
    plate_dimensions = np.concatenate((np.expand_dims(dict_ds["Lx"], -1), np.expand_dims(dict_ds["Ly"], -1)), -1)

    return acpr, acpr_downsampling, vel, bm, mode, freq, num_plate, boundary, h_points, h_points_downsampling, s_points, plate_dimensions




