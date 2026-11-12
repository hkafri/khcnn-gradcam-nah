"""
Grad-CAM-inspired interpretation of the KHCNN regression network.

Implementation of the Grad-CAM variant for regression described in:

    H. Kafri, M. Olivieri, F. Antonacci, M. Moradi, A. Sarti, S. Gannot,
    "Grad-CAM-Inspired Interpretation of Nearfield Acoustic Holography using
    Physics-Informed Explainable Neural Network", ICASSP 2023,
    doi:10.1109/ICASSP49357.2023.10097272

Method (Sec. 2.2 and 3.2, Eqs. 3-6 of the paper): the classification score of
Grad-CAM is replaced by the KHCNN training loss

    L = 0.5 * L_Re + 0.5 * L_Im,
    L_{Re/Im} = ||{Re/Im}(V) - {Re/Im}(V_hat)||^2 + ||{Re/Im}(P_H) - {Re/Im}(P_H_hat)||^2

with ground-truth velocity V (FEM simulation) and the KH forward-propagated
pressure P_H_hat.  The weights alpha_k are the global average pooling of
dL/dA^k over the activation maps A^k of the last convolutional layer of the
velocity decoder D2, R = ReLU(sum_k alpha_k A^k) is linearly interpolated to
the input resolution M1 x M2 (8x8 for the violin, 16x64 for rectangular
plates) and scaled to [0, 1] to obtain the heatmap H.

Validation (Sec. 4.3, Eq. 7): the network is re-evaluated on the
heatmap-masked input, V_f_hat = KHCNN(P_H * H), and compared with
V_hat = KHCNN(P_H) through NCC and NMSE (both compare the two network
estimates, not the ground truth).

Options that are NOT specified in the paper are exposed as CLI flags with the
assumed defaults: per-sample gradients (--batch-size 1), min-max scaling
(--scale minmax), bilinear interpolation, gradients of +L (--negate-loss off)
and the noisy violin test set (--dataset noisy; the paper does not mention
noise, the SNR comes from the KHCNN dataset configuration).

Run from the repository root:

    python -m gradcam.run_khcnn_gradcam [--n-samples N] [--sample-idx i j k]
        [--target-layer NAME] [--out-dir DIR] [--cpu] [--batch-size B]
        [--scale {minmax,max}] [--negate-loss] [--dataset {noisy,clean}]
"""

import argparse
import os
import pickle
import sys

repo_root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)  # allow running the script directly

# Frequencies of the three violin samples shown in Fig. 3 of the paper [Hz]
FIG3_FREQUENCIES = (128.7, 247.47, 880.3)


def parse_args():
    parser = argparse.ArgumentParser(description="Grad-CAM for the KHCNN regression network (ICASSP 2023)")
    parser.add_argument("--n-samples", type=int, default=-1,
                        help="number of test samples used for the validation metrics (-1 = whole test set)")
    parser.add_argument("--sample-idx", type=int, nargs="*", default=None,
                        help="indices of the samples shown in the summary figure "
                             "(default: the samples closest to the Fig. 3 frequencies "
                             f"{FIG3_FREQUENCIES} Hz)")
    parser.add_argument("--target-layer", type=str, default=None,
                        help="name of the convolutional layer used for Grad-CAM "
                             "(default: last conv layer of the velocity decoder D2)")
    parser.add_argument("--out-dir", type=str, default=os.path.join(repo_root, "results"),
                        help="output directory for the figure and the metrics file")
    parser.add_argument("--cpu", action="store_true", help="force execution on CPU")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="gradient computation batch size (not specified in the paper; "
                             "mathematically equivalent for any value, default 1 = per sample)")
    parser.add_argument("--scale", choices=["minmax", "max"], default="minmax",
                        help="scaling of the heatmap to [0, 1] (not specified in the paper): "
                             "'minmax' = (R - min) / (max - min), 'max' = R / max")
    parser.add_argument("--negate-loss", action="store_true",
                        help="use gradients of -L instead of L (the paper uses dL/dA; off by default)")
    parser.add_argument("--dataset", choices=["noisy", "clean"], default=None,
                        help="use the noisy or clean violin test set (the paper does not mention "
                             "noise; default: utils/params.py `use_addivive_noise`)")
    return parser.parse_args()


args = parse_args()

from utils import params as PARAMS  # noqa: E402

if args.cpu:
    PARAMS.GPUS = "-1"  # read by the network modules when they set CUDA_VISIBLE_DEVICES
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
if args.dataset is not None:
    PARAMS.use_addivive_noise = args.dataset == "noisy"

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from utils import data_processing as DATA  # noqa: E402
from utils import violin_data_processing as VIOLIN_DATA  # noqa: E402
from network_module import khcnn_network  # noqa: E402


def load_test_set():
    """Load and preprocess the violin test set exactly like the test script."""
    dataset_dir = os.path.join(repo_root, PARAMS.path_test_set)
    dataset_name = PARAMS.test_ds_name + ("_noisy" if PARAMS.use_addivive_noise else "")
    print(f"Dataset: {dataset_name} (additive noise: {PARAMS.use_addivive_noise})")

    with open(os.path.join(dataset_dir, dataset_name + ".pickle"), "rb") as input_file:
        test_ds_dict = pickle.load(input_file)

    if PARAMS.use_addivive_noise and "snr_db" in test_ds_dict:
        snr = np.asarray(test_ds_dict["snr_db"], dtype=float)
        print(f"SNR of the noisy dataset (from the KHCNN dataset config, not specified in the "
              f"ICASSP paper): {snr.min():.1f} .. {snr.max():.1f} dB (mean {snr.mean():.1f} dB)")

    acpr, acpr_down, vel, _, _, freq, _, _, h_points, h_points_down, s_points, _ = \
        VIOLIN_DATA.get_data_from_network_ds(test_ds_dict)

    bm = VIOLIN_DATA.get_binary_mask(vel)
    vel_mask = vel.copy()  # velocity with NaN outside the plate (for plotting)
    vel = VIOLIN_DATA.apply_zero_mask(vel, np.squeeze(vel))
    s_points[np.where(np.isnan(s_points))] = 0

    p_size = (16, 64)
    s_size = (16, 64)
    if PARAMS.use_low_P_resolution:
        acpr = acpr_down
        h_points = h_points_down
        p_size = (8, 8)

    if PARAMS.use_normalise_wrt_abs:
        p = DATA.normalise_wrt_abs(acpr)
        v = DATA.normalise_wrt_abs(vel)
    else:
        p = acpr
        v = vel

    p_ch = DATA.split_real_imag_channels(p)
    bm_ch = np.concatenate((bm, bm), axis=-1)

    return p, p_ch, v, vel_mask, freq, h_points, s_points, bm_ch, p_size, s_size


def find_target_layer(model, name=None):
    """Return the Grad-CAM target layer.

    By default, walk the graph backwards from the 1x1 output convolution of
    the velocity branch (`v_s_output_imgs_*`) to its last convolutional
    activation (the last conv layer of decoder D2 in the paper).
    """
    if name:
        return model.get_layer(name)

    out_conv = next(layer for layer in model.layers if layer.name.startswith("v_s_output_imgs"))
    tensor = out_conv.input
    for _ in range(20):  # bounded backwards traversal
        producer = tensor._keras_history[0]
        if isinstance(producer, tf.keras.layers.Conv2D) and \
                getattr(producer, "activation", None) is tf.keras.activations.relu:
            return producer
        tensor = producer.input
    raise RuntimeError("Could not find the last conv activation of the velocity decoder")


def khcnn_loss(outputs, real_p_gt, imag_p_gt, real_v_gt, imag_v_gt):
    """KHCNN loss L = 0.5 * L_Re + 0.5 * L_Im used as the Grad-CAM score (Eq. 6)."""
    # the KH propagation works in float64, so cast all outputs to a common dtype
    real_p, imag_p, real_v, imag_v = [tf.cast(o, tf.float32) for o in outputs]
    loss_re = tf.reduce_sum(tf.square(real_v_gt - real_v)) + tf.reduce_sum(tf.square(real_p_gt - real_p))
    loss_im = tf.reduce_sum(tf.square(imag_v_gt - imag_v)) + tf.reduce_sum(tf.square(imag_p_gt - imag_p))
    return 0.5 * loss_re + 0.5 * loss_im


def compute_gradcam_heatmaps(model, target_layer, inputs, targets, p_size,
                             batch_size=1, scale="minmax", negate_loss=False):
    """Compute Grad-CAM heatmaps for every sample.

    :param model: KHCNN model
    :param target_layer: convolutional layer whose activations are explained
    :param inputs: list of the 5 network inputs (numpy arrays)
    :param targets: (real_p_gt, imag_p_gt, real_v_gt, imag_v_gt) ground truths
    :param p_size: input (hologram) resolution M1 x M2
    :param batch_size: gradient computation batch size (any value is
        mathematically equivalent since the summed loss keeps per-sample
        gradients independent)
    :param scale: '[0, 1]' scaling: 'minmax' or 'max' (division by the max)
    :param negate_loss: backpropagate -L instead of L
    :return: heatmaps of shape (N, M1, M2) scaled to [0, 1]
    """
    grad_model = tf.keras.Model(inputs=model.inputs,
                                outputs=[target_layer.output] + list(model.outputs))
    n = inputs[0].shape[0]
    heatmaps = []
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        batch_in = [tf.convert_to_tensor(x[start:stop], tf.float32) for x in inputs]
        batch_gt = [tf.convert_to_tensor(t[start:stop], tf.float32) for t in targets]

        with tf.GradientTape() as tape:
            conv_out, *outputs = grad_model(batch_in, training=False)
            loss = khcnn_loss(outputs, *batch_gt)
            if negate_loss:
                loss = -loss

        grads = tape.gradient(loss, conv_out)
        alpha = tf.reduce_mean(grads, axis=(1, 2))                       # alpha_k, per sample
        cam = tf.nn.relu(tf.reduce_sum(alpha[:, None, None, :] * conv_out, axis=-1))
        cam = tf.image.resize(cam[..., None], p_size, method="bilinear")[..., 0]

        cam = cam.numpy()
        cam_max = cam.reshape(cam.shape[0], -1).max(axis=1)[:, None, None]
        if scale == "minmax":
            cam_min = cam.reshape(cam.shape[0], -1).min(axis=1)[:, None, None]
        else:
            cam_min = np.zeros_like(cam_max)
        heatmaps.append((cam - cam_min) / (cam_max - cam_min + 1e-12))
    return np.concatenate(heatmaps, axis=0)


def predict_velocity(model, inputs, batch_size=64):
    """Run the KHCNN and return the complex surface velocity estimate."""
    _, _, real_v, imag_v = model.predict(inputs, batch_size=batch_size, verbose=0)
    return real_v + 1j * imag_v


def validate_heatmaps(model, inputs, heatmaps):
    """Validation of the explanations (Eq. 7 of the paper).

    Feeds the heatmap-masked input P_H * H to the network and compares the
    resulting velocity estimate V_f_hat with the original estimate V_hat
    (both network estimates, not the ground truth) through NCC and NMSE.
    :return: (v_hat, v_f, ncc, nmse_db) with per-sample metric arrays
    """
    v_hat = predict_velocity(model, inputs)

    # the real-valued heatmap multiplies both the real and imaginary channel
    masked_p = inputs[0] * heatmaps[..., None]
    v_f = predict_velocity(model, [masked_p] + inputs[1:])

    n = v_hat.shape[0]
    ncc = np.zeros(n)
    nmse_db = np.zeros(n)
    for i in range(n):
        v = v_hat[i].flatten()
        vf = v_f[i].flatten()
        err = vf - v
        ncc[i] = abs(np.vdot(vf, v)) / (np.linalg.norm(vf) * np.linalg.norm(v) + 1e-12)
        nmse_db[i] = 10 * np.log10(np.real(np.vdot(err, err)) / np.real(np.vdot(v, v)) + 1e-300)
    return v_hat, v_f, ncc, nmse_db


def pick_figure_samples(freq, target_freqs=FIG3_FREQUENCIES):
    """Pick the samples closest to the frequencies shown in Fig. 3 of the paper."""
    freq = np.squeeze(freq)
    return [int(np.argmin(np.abs(freq - f))) for f in target_freqs]


def plot_summary_figure(sample_idx, p, v_hat, v_gt_mask, heatmaps, freq, out_path):
    """Figure in the style of Fig. 3 of the paper.

    Rows: |P_H| (input hologram pressure), |V_hat| (estimated velocity),
    |V| (ground truth velocity), Grad-CAM heatmap H.
    """
    n = len(sample_idx)
    fig, axes = plt.subplots(4, n, figsize=(4 * n, 9), constrained_layout=True)
    axes = np.atleast_2d(axes.reshape(4, n))
    row_titles = [r"$|P_H|$ (input)", r"$|\hat{V}|$ (KHCNN)", r"$|V|$ (ground truth)", r"heatmap $H$"]

    for col, idx in enumerate(sample_idx):
        images = [
            (np.abs(np.squeeze(p[idx])), "inferno"),
            (np.abs(v_hat[idx]), "viridis"),
            (np.abs(np.squeeze(v_gt_mask[idx])), "viridis"),
            (heatmaps[idx], "jet"),
        ]
        for row, (img, cmap) in enumerate(images):
            ax = axes[row, col]
            im = ax.imshow(img, cmap=cmap, aspect="auto")
            fig.colorbar(im, ax=ax, fraction=0.046)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(row_titles[row], fontsize=11)
            if row == 0:
                ax.set_title(f"sample {idx} - {float(np.squeeze(freq[idx])):.1f} Hz", fontsize=11)

    fig.suptitle("Grad-CAM-inspired explanation of KHCNN (violin top plates)", fontsize=13)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading test set...")
    p, p_ch, v, vel_mask, freq, h_points, s_points, bm_ch, p_size, s_size = load_test_set()

    if args.n_samples > 0:
        n = min(args.n_samples, p_ch.shape[0])
        p, p_ch, v, vel_mask = p[:n], p_ch[:n], v[:n], vel_mask[:n]
        freq, h_points, s_points, bm_ch = freq[:n], h_points[:n], s_points[:n], bm_ch[:n]
    print(f"Test samples: {p_ch.shape[0]}")

    print("Building KHCNN and loading weights...")
    model, _, _ = khcnn_network.build_khcnn_model(p_size, s_size)
    weights_file = os.path.join(repo_root, PARAMS.path_saved_model, PARAMS.test_model_name + ".h5")
    model.load_weights(weights_file, by_name=True)

    target_layer = find_target_layer(model, args.target_layer)
    print(f"Grad-CAM target layer: {target_layer.name} {target_layer.output.shape}")

    inputs = [p_ch, freq, h_points, s_points, bm_ch]
    v_sq = np.squeeze(v)  # ground-truth velocity (zeros outside the plate)
    targets = (np.real(np.squeeze(p)), np.imag(np.squeeze(p)), np.real(v_sq), np.imag(v_sq))

    print("Computing Grad-CAM heatmaps...")
    heatmaps = compute_gradcam_heatmaps(model, target_layer, inputs, targets, p_size,
                                        batch_size=args.batch_size, scale=args.scale,
                                        negate_loss=args.negate_loss)

    print("Validating heatmaps (Eq. 7)...")
    v_hat, _, ncc, nmse_db = validate_heatmaps(model, inputs, heatmaps)

    metrics_path = os.path.join(args.out_dir, "gradcam_validation_metrics.csv")
    with open(metrics_path, "w") as f:
        f.write("sample,freq_hz,ncc,nmse_db\n")
        for i in range(len(ncc)):
            f.write(f"{i},{float(np.squeeze(freq[i])):.2f},{ncc[i]:.6f},{nmse_db[i]:.4f}\n")
    print("\nPer-sample validation metrics (first 10):")
    for i in range(min(10, len(ncc))):
        print(f"  sample {i:4d}  f = {float(np.squeeze(freq[i])):8.2f} Hz   "
              f"NCC = {ncc[i]:.4f}   NMSE = {nmse_db[i]:8.3f} dB")
    # the paper does not specify whether the NMSE average is taken over the
    # per-sample dB values or in linear scale, so both are reported
    nmse_mean_db = np.mean(nmse_db)
    nmse_mean_linear_db = 10 * np.log10(np.mean(10 ** (nmse_db / 10)))
    print(f"\nMean over {len(ncc)} samples:")
    print(f"  NCC                      = {np.mean(ncc):.4f}")
    print(f"  NMSE (mean of dB values) = {nmse_mean_db:.3f} dB")
    print(f"  NMSE (linear mean, dB)   = {nmse_mean_linear_db:.3f} dB")
    print("(paper, violin test set: mean NCC = 0.98, mean NMSE = -13.93 dB)")
    print(f"Metrics saved to {metrics_path}")

    sample_idx = args.sample_idx if args.sample_idx else pick_figure_samples(freq)
    fig_path = os.path.join(args.out_dir, "gradcam_violin_summary.png")
    plot_summary_figure(sample_idx, p, v_hat, vel_mask, heatmaps, freq, fig_path)
    print(f"Figure saved to {fig_path}")


if __name__ == "__main__":
    main()
