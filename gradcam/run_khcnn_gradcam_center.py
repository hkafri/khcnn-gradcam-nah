"""
Earlier Grad-CAM variant for the KHCNN violin model (kept for reference).

This older version targets the bottleneck layer (`CENTER_*`) and uses the
mean of the velocity output as score, instead of the paper-aligned loss-based
formulation implemented in `run_khcnn_gradcam.py`.

Run from the repository root:
    python -m gradcam.run_khcnn_gradcam_center
"""

import os
import pickle
import sys

repo_root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)  # allow running the script directly

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("CUDA_ALLOW_GROWTH", "True")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from utils import params as PARAMS
from utils import data_processing as DATA
from utils import violin_data_processing as VIOLIN_DATA
from network_module import khcnn_network


def load_test_sample():
    """Load and preprocess the violin test set."""
    dataset_dir = os.path.join(repo_root, PARAMS.path_test_set)
    dataset_name = PARAMS.test_ds_name + ("_noisy" if PARAMS.use_addivive_noise else "")

    with open(os.path.join(dataset_dir, dataset_name + ".pickle"), "rb") as input_file:
        test_ds_dict = pickle.load(input_file)

    acpr_test, acpr_downsampling_test, vel_test, bm_test, mode_test, freq_test, num_violin_test, boundary_test, \
        h_points_test, h_points_downsampling_test, s_points_test, plate_dimensions_test = \
        VIOLIN_DATA.get_data_from_network_ds(test_ds_dict)

    bm_test = VIOLIN_DATA.get_binary_mask(vel_test)
    vel_test_mask = vel_test.copy()
    vel_test = VIOLIN_DATA.apply_zero_mask(vel_test, np.squeeze(vel_test))
    s_points_test[np.where(np.isnan(s_points_test))] = 0

    p_size = (16, 64)
    s_size = (16, 64)
    if PARAMS.use_low_P_resolution:
        acpr_test = acpr_downsampling_test
        h_points_test = h_points_downsampling_test
        p_size = (8, 8)

    if PARAMS.use_normalise_wrt_abs:
        p_test = DATA.normalise_wrt_abs(acpr_test)
    else:
        p_test = acpr_test

    p_test_ch = DATA.split_real_imag_channels(p_test)
    bm_test = np.concatenate((bm_test, bm_test), axis=-1)

    return p_test_ch, freq_test, h_points_test, s_points_test, bm_test, p_size, s_size


def pick_target_layer(model):
    """Pick the bottleneck (CENTER) layer, falling back to the last convolutions."""
    preferred_names = ["CENTER_complex_64", "CENTER_complex_1024"]
    for name in preferred_names:
        try:
            return model.get_layer(name)
        except Exception:
            continue

    conv_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
    if conv_layers:
        return conv_layers[-2] if len(conv_layers) >= 2 else conv_layers[-1]

    raise RuntimeError("Could not find a convolution layer for Grad-CAM")


def compute_gradcam(model, inputs, target_layer_name, target_output_name):
    """Grad-CAM with the mean of a network output as score."""
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(target_layer_name).output, model.get_layer(target_output_name).output],
    )

    input_tensors = [tf.convert_to_tensor(x, tf.float32) for x in inputs]

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_tensors, training=False)
        target_score = tf.reduce_mean(predictions)

    grads = tape.gradient(target_score, conv_outputs)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.reduce_sum(tf.multiply(weights, conv_outputs[0]), axis=-1)
    cam = tf.nn.relu(cam)
    cam = cam / (tf.reduce_max(cam) + tf.keras.backend.epsilon())
    return cam.numpy()


def main():
    p_test_ch, freq_test, h_points_test, s_points_test, bm_test, p_size, s_size = load_test_sample()

    model, _, _ = khcnn_network.build_model_violin(p_size, s_size)
    weights_path = os.path.join(repo_root, PARAMS.path_saved_model)
    model.load_weights(os.path.join(weights_path, PARAMS.test_model_name + ".h5"), by_name=True)

    sample_idx = 0
    target_layer = pick_target_layer(model)
    target_output_name = "real_v_s"
    cam = compute_gradcam(
        model,
        [
            p_test_ch[sample_idx:sample_idx + 1],
            freq_test[sample_idx:sample_idx + 1],
            h_points_test[sample_idx:sample_idx + 1],
            s_points_test[sample_idx:sample_idx + 1],
            bm_test[sample_idx:sample_idx + 1],
        ],
        target_layer.name,
        target_output_name,
    )

    input_mag = np.linalg.norm(p_test_ch[sample_idx], axis=-1)
    # the CENTER layer may be 1x1 for the 8x8 violin input; keep the map 2-D
    heatmap = np.atleast_2d(cam.squeeze())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    axes[0].imshow(input_mag, cmap="gray", aspect="auto")
    axes[0].set_title("KHCNN input magnitude")
    axes[0].axis("off")

    axes[1].imshow(input_mag, cmap="gray", aspect="auto")
    axes[1].imshow(heatmap, cmap="inferno", alpha=0.6, aspect="auto")
    axes[1].set_title(f"Grad-CAM overlay on {target_layer.name}")
    axes[1].axis("off")

    output_path = os.path.join(repo_root, "results", "khcnn_gradcam_center.jpeg")
    fig.savefig(output_path, dpi=200, bbox_inches="tight", format="jpeg")
    print(output_path)


if __name__ == "__main__":
    main()
