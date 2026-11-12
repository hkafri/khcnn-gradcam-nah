"""
Name:        "params.py"
Author:      Marco Olivieri (adapted for the violin test case)
Date:        23 November 2021
Description: Parameters to use

Note: this module comes from the KHCNN code by M. Olivieri
(Olivieri et al., Sensors 2021, doi:10.3390/s21237834), adapted so that all
paths are relative to the repository root.
"""

GPUS = "0"                   # GPU id to use ("" to force CPU)
USE_LOW_P_RESOLUTION = True  # True: hologram pressure with 64 points (8x8), False: 1024 points (16x64)

# Violin test-case settings
test_ds_name = "test_ds_violin"       # test dataset name (without extension / noise suffix)
use_addivive_noise = True             # load the dataset with additive noise ("_noisy" suffix)
use_normalise_wrt_abs = True          # normalise data wrt max(abs) before feeding the network (as in training)
use_low_P_resolution = USE_LOW_P_RESOLUTION

# Paths (relative to the repository root)
path_saved_model = "kh_violin_model"  # folder with the trained weights
test_model_name = "kh_violin_64_noise"
path_test_set = "test_set"            # folder with the test dataset pickle

############################################

c = 343         # speed of sound [m/s]
rho_0 = 1.225   # air mass density [Kg/m^3]
