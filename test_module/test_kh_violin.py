"""
Script to test the KHCNN with violin top plates.

Run from the repository root:
    python -m test_module.test_kh_violin

It loads the pretrained violin weights (`kh_violin_model/`) and the test
dataset (`test_set/test_ds_violin_noisy.pickle`), evaluates the network and
plots reconstruction results and NMSE/NCC metrics.

Based on the KHCNN code by M. Olivieri
(Olivieri et al., Sensors 2021, doi:10.3390/s21237834).
"""

import os
import sys

this_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.normpath(os.path.join(this_dir, ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)  # allow running the script directly

from utils import params as PARAMS

os.environ['CUDA_VISIBLE_DEVICES'] = PARAMS.GPUS
os.environ['CUDA_ALLOW_GROWTH'] = 'True'

import pickle

from utils import violin_data_processing as VIOLIN_DATA
from utils import data_processing as DATA

from network_module import khcnn_network
from test_module import plot_results
from tensorflow.keras.models import Model

import numpy as np
from datetime import datetime
from utils import metrics as METRICS

now = datetime.now()
print(now.strftime("%Y-%m-%d %H:%M:%S"))
print(f"TEST: {PARAMS.test_ds_name}")
print(f"64 points: {PARAMS.use_low_P_resolution}")
print(f"WEIGHTS: {PARAMS.test_model_name}")

main_ds_path = os.path.join(repo_root, PARAMS.path_test_set)
test_ds_name = PARAMS.test_ds_name

# load datasets
print("Load datasets...")
if PARAMS.use_addivive_noise:  # load clean or noisy (with additive noise) dataset
    test_ds_name += "_noisy"
print("\ttest set: " + test_ds_name)
with open(os.path.join(main_ds_path, test_ds_name + ".pickle"), "rb") as input_file:
    test_ds_dict = pickle.load(input_file)
if PARAMS.use_addivive_noise:  # retrieve the SNR information used to build the dataset
    snr_db_test = test_ds_dict["snr_db"]
else:
    snr_db_test = []

print("Extract data from datasets...")
acpr_test, acpr_downsampling_test, vel_test, bm_test, mode_test, freq_test, num_violin_test, boundary_test, \
h_points_test, h_points_downsampling_test, s_points_test, plate_dimensions_test = \
    VIOLIN_DATA.get_data_from_network_ds(test_ds_dict)

# get correct binary mask information
bm_test = VIOLIN_DATA.get_binary_mask(vel_test)
# get velocity (outer shapes with zeros -> to compute MSE)
V_test_mask = vel_test.copy()
vel_test = VIOLIN_DATA.apply_zero_mask(vel_test, np.squeeze(vel_test))
s_points_test[np.where(np.isnan(s_points_test))] = 0

# select correct data (hologram pressure at 1024 or 64 points)
p_size = (16, 64)
s_size = (16, 64)
if PARAMS.use_low_P_resolution:
    print("Use downsampled holographic plane")
    acpr_test = acpr_downsampling_test
    h_points_test = h_points_downsampling_test
    p_size = (8, 8)

# data normalisation
if PARAMS.use_normalise_wrt_abs:
    print("Normalise wrt abs value...")
    P_test = DATA.normalise_wrt_abs(acpr_test)
    V_test_mask = VIOLIN_DATA.normalise_wrt_abs_velocity_violin(V_test_mask)
    V_test = DATA.normalise_wrt_abs(vel_test)
else:
    print("Not normalise wrt abs value - original data")
    P_test = acpr_test
    V_test = vel_test

# get real and imags in 2 channels
print("Split real & imag in 2 channels")
P_test_ch = DATA.split_real_imag_channels(P_test)
V_test_ch = DATA.split_real_imag_channels(V_test)
bm_test = np.concatenate((bm_test, bm_test), axis=-1)

# build model
model, _, _ = khcnn_network.build_model_violin(p_size, s_size)

# get weights
print("Load model & weights")
weights_path = os.path.join(repo_root, PARAMS.path_saved_model)
weights_name = PARAMS.test_model_name
model.load_weights(os.path.join(weights_path, weights_name + '.h5'), by_name=True)

print("Get history loss")
history = np.load(os.path.join(weights_path, weights_name + '.npy'), allow_pickle=True)

print("Plot history")
plot_results.plot_history(history)

print("Test phase")
print("\tprediction...")
estimate_real, estimate_imag, v_s_real_hat, v_s_imag_hat = model.predict(
    [P_test_ch, freq_test, h_points_test, s_points_test, bm_test])
p_h_hat = estimate_real + 1j * estimate_imag
v_s_hat = v_s_real_hat + 1j * v_s_imag_hat

v_s_hat_norm = DATA.normalise_wrt_abs(v_s_hat)
v_s_hat_norm_mask = VIOLIN_DATA.apply_nan_mask(v_s_hat_norm, np.squeeze(V_test_mask))

# retrieve internal layers (surface velocity and surface pressure, masked)
v_model = Model(inputs=model.inputs, outputs=model.get_layer("v_s_masked").output)
p_model = Model(inputs=model.inputs, outputs=model.get_layer("p_s_masked").output)

v_s_hat_ch = v_model.predict([P_test_ch, freq_test, h_points_test, s_points_test, bm_test])
v_s_hat_bis = v_s_hat_ch[:, :, :, 0] + 1j * v_s_hat_ch[:, :, :, 1]
p_s_hat_ch = p_model.predict([P_test_ch, freq_test, h_points_test, s_points_test, bm_test])
p_s_hat = p_s_hat_ch[:, :, :, 0] + 1j * p_s_hat_ch[:, :, :, 1]

print("Plot results")  # ================================================================================
i = np.random.randint(len(P_test))
plot_results.print_information(i, mode_test=mode_test, freq_test=freq_test, num_plate_test=num_violin_test,
                               boundary_test=boundary_test, plate_dimensions_test=plate_dimensions_test,
                               snr_test=snr_db_test)
print("\tp_h...")
plot_results.plot_real_imag(p_h_hat, P_test, "p_h", i, p_size)
plot_results.plot_abs_phase(p_h_hat, P_test, "p_h", i, p_size)

print("\tv_s...")
plot_results.plot_real_imag(v_s_hat, V_test, "v_s", i, s_size)
plot_results.plot_abs_phase(v_s_hat_norm_mask, V_test_mask, "v_s", i, s_size)

print("Metrics")  # ================================================================================
print("\tp_h...")
p_nmse_mean_db, p_nmse_db = METRICS.nmse_dataset_complex(out=p_h_hat, gt=P_test)
plot_results.plot_NMSE_complex(nmse_mean_db=p_nmse_mean_db, nmse_db=p_nmse_db, quantity="p_h")
p_ncc_mean, p_ncc = METRICS.ncc_dataset_complex(out=p_h_hat, gt=P_test)
plot_results.plot_NCC_complex(ncc_mean=p_ncc_mean, ncc=p_ncc, quantity="p_h")

print("\tv_s...")
# get only internal points -> reduces images to only the important points
v_s_hat_internal = VIOLIN_DATA.get_internal_points(v_s_hat_norm_mask)
V_test_internal = VIOLIN_DATA.get_internal_points(V_test_mask)

v_nmse_mean_db, v_nmse_db = METRICS.nmse_dataset_complex(out=v_s_hat_internal, gt=V_test_internal)
plot_results.plot_NMSE_complex(nmse_mean_db=v_nmse_mean_db, nmse_db=v_nmse_db, quantity="v_s")
v_ncc_mean, v_ncc = METRICS.ncc_dataset_complex(out=v_s_hat_internal, gt=V_test_internal)
plot_results.plot_NCC_complex(ncc_mean=v_ncc_mean, ncc=v_ncc, quantity="v_s")

v_s_hat_internal_abs = VIOLIN_DATA.get_internal_points(abs(v_s_hat_norm_mask))
V_test_internal_abs = VIOLIN_DATA.get_internal_points(abs(V_test_mask))

# get magnitude metrics
v_nmse_mean_db_abs, v_nmse_db_abs = METRICS.nmse_dataset_complex(out=v_s_hat_internal_abs, gt=V_test_internal_abs)
plot_results.plot_NMSE_complex(nmse_mean_db=v_nmse_mean_db_abs, nmse_db=v_nmse_db_abs, quantity="v_s abs")
v_ncc_mean_abs, v_ncc_abs = METRICS.ncc_dataset_complex(out=v_s_hat_internal_abs, gt=V_test_internal_abs)
plot_results.plot_NCC_complex(ncc_mean=v_ncc_mean_abs, ncc=v_ncc_abs, quantity="v_s")

print("Metrics analysis")
bands = [0, 300, 600, 1000, 1500, 2000]
plot_results.plot_metrics_for_bands(v_nmse_db_abs, freq_test, bands, "NMSE")
plot_results.plot_metrics_for_bands(v_ncc, freq_test, bands, "NCC")

print("Finish")
