# evaluation.py
"""
Evaluation Script for 3D CycleGAN Models

Performs various evaluation tasks based on a trained checkpoint:
1.  Visualize sample translations (A->B->A and B->A->B).
2.  Plot training losses and validation metrics from log files.
3.  Generate necessary images (real & fake slices) and calculate FID scores.
4.  Optionally save full 3D synthetic volumes (.npy).
5.  Optionally save full 3D difference volumes (.npy).

Relies on checkpoints, logs, and model definitions from train_cycleGAN_v3.py (or utils.py).
Assumes input data is preprocessed to [0, 1] range initially and handled as [-1, 1] internally.
Uses padding/cropping consistent with training.

---example use---
python evaluation.py ^
    --checkpoint ".\output\t2_flair\checkpoints\t2_flair\checkpoint_t2_flair_epoch003_20250410_174940.pth" ^
    --eval_data_dir ".\data\processed_data\brats128_validation\images" ^
    --output_dir ".\evaluation_run_test_all" ^
    --tasks visualize plot_logs calculate_fid ^
    --vis_samples 2 ^
    --save_synthetic ^
    --save_difference ^
    --save_num_samples 2 ^
    --fid_num_images 10 ^
    --fid_force_regen ^
    --num_workers 2
"""

import os
import torch
import argparse
import numpy as np

from torch.utils.data import DataLoader
import json
import sys
from tqdm import tqdm
import torch.nn.functional as F
import glob
import multiprocessing
import random 

# --- Try importing necessary components from utils ---
try:
    # Assuming utils.py contains these from the refactoring
    from utils import (PairedPreprocessedDataset, Generator3D,
                       load_checkpoint as load_train_checkpoint, # Use the specific loader
                       Logger) # May not be needed directly, but shows dependency
except ImportError as e:
    print(f"Error importing from utils.py: {e}", file=sys.stderr)
    print("Please ensure utils.py (containing models, dataset, etc.) is accessible.", file=sys.stderr)
    sys.exit(1)

# --- Try importing FID and PIL ---
try:
    from pytorch_fid import fid_score
except ImportError:
    # This is handled later if fid task is selected
    fid_score = None

try:
    from PIL import Image
except ImportError:
     # This is handled later if fid task is selected
     Image = None

# ------------------------------
# Model Initialization
# ------------------------------
def init_models(mapping_type, n_resnet, device):
    """Creates and initializes the generator pair."""
    if mapping_type not in ['t2_flair', 't1_contrast']:
        raise ValueError(f"Unknown mapping_type: {mapping_type}")
    in_channels = 1
    out_channels = 1
    g_model_AtoB = Generator3D(in_channels, out_channels, n_resnet=n_resnet).to(device)
    g_model_BtoA = Generator3D(out_channels, in_channels, n_resnet=n_resnet).to(device)
    return g_model_AtoB, g_model_BtoA

# ------------------------------
# Visualization Function
# ------------------------------
def visualize_sample_translations(epoch_label, g_model_AtoB, g_model_BtoA, dataset,
                                  mapping_type, device, n_samples=5, save_dir=None):
    """Visualizes sample translations for both directions."""
    # Ensure matplotlib is imported when function is called
    try:
        import matplotlib.pyplot as plt
        plt.switch_backend('agg') # Use non-interactive backend for saving
        import numpy as np
    except ImportError:
        print("Error: matplotlib not found. Cannot generate plots.", file=sys.stderr)
        return

    print(f"Generating sample translation visualizations for {mapping_type} at {epoch_label}...")
    g_model_AtoB.eval()
    g_model_BtoA.eval()

    old_augment_state = None
    if hasattr(dataset, 'augment'): # Temporarily disable augmentation
        old_augment_state = dataset.augment
        dataset.augment = False

    if len(dataset) == 0:
        print("Warning: Dataset is empty, cannot generate visualization.")
        if old_augment_state is not None: dataset.augment = old_augment_state
        return

    n_samples = min(n_samples, len(dataset))
    if n_samples == 0: return

    indices = np.random.choice(len(dataset), n_samples, replace=False)

    # Define domain names for titles
    domain_map = {'t2_flair': ('T2', 'FLAIR'), 't1_contrast': ('T1', 'T1CE')}
    domain_A, domain_B = domain_map.get(mapping_type, ('A', 'B'))

    # --- AtoB Visualization ---
    fig_AtoB, axs_AtoB = plt.subplots(4, n_samples, figsize=(max(15, 3 * n_samples), 12))
    if n_samples == 1: axs_AtoB = axs_AtoB[:, np.newaxis]

    for i, idx in enumerate(indices):
        sample_data = dataset[idx]
        if sample_data is None or sample_data[0] is None: continue # Skip bad data
        img_A, _, _ = sample_data # Get img_A and ignore img_B, path
        real_A = img_A.unsqueeze(0).to(device) # Add batch dim [1, C, D, H, W]

        with torch.no_grad():
            # Pad real_A, generate fake_B, crop fake_B
            real_A_padded = F.pad(real_A, (0, 0, 0, 0, 0, 1)) # [1, C, D+1, H, W]
            fake_B = g_model_AtoB(real_A_padded)              # [1, C, D+1, H, W]
            fake_B = fake_B[:, :, :-1, :, :]                  # [1, C, D,   H, W]

            fake_B_padded = F.pad(fake_B, (0, 0, 0, 0, 0, 1)) # [1, C, D+1, H, W]
            rec_A = g_model_BtoA(fake_B_padded)              # [1, C, D+1, H, W]
            rec_A = rec_A[:, :, :-1, :, :]                  # [1, C, D,   H, W]

        real_A_np = (real_A.cpu().numpy()[0] + 1.0) / 2.0 # Use real_A directly, Shape [C, D, H, W] -> [0, 1]
        fake_B_np = (fake_B.cpu().numpy()[0] + 1.0) / 2.0 # Shape [C, D, H, W] -> [0, 1]
        rec_A_np  = (rec_A.cpu().numpy()[0] + 1.0) / 2.0 # Shape [C, D, H, W] -> [0, 1]

 
        diff_A = np.abs(real_A_np - rec_A_np)
        d_mid = max(0, real_A_np.shape[1] // 2) # Middle slice index along depth (axis 1)
        if real_A_np.shape[1] == 0: continue

        # Plotting (access channel 0, depth slice d_mid)
        axs_AtoB[0, i].imshow(real_A_np[0, d_mid, :, :], cmap='gray', vmin=0, vmax=1)
        axs_AtoB[0, i].set_title(f"Real {domain_A}")
        axs_AtoB[0, i].axis('off')
        axs_AtoB[1, i].imshow(fake_B_np[0, d_mid, :, :], cmap='gray', vmin=0, vmax=1)
        axs_AtoB[1, i].set_title(f"Generated {domain_B}")
        axs_AtoB[1, i].axis('off')
        axs_AtoB[2, i].imshow(rec_A_np[0, d_mid, :, :], cmap='gray', vmin=0, vmax=1)
        axs_AtoB[2, i].set_title(f"Reconstructed {domain_A}")
        axs_AtoB[2, i].axis('off')
        im = axs_AtoB[3, i].imshow(diff_A[0, d_mid, :, :], cmap='hot', vmin=0, vmax=max(0.1, diff_A.max()))
        axs_AtoB[3, i].set_title("Abs Difference")
        axs_AtoB[3, i].axis('off')

    fig_AtoB.suptitle(f"{mapping_type.upper()}: {domain_A} -> {domain_B} -> {domain_A} ({epoch_label})", fontsize=16)
    try: fig_AtoB.tight_layout(rect=[0, 0.03, 1, 0.95])
    except Exception as e: print(f"Warning: tight_layout failed: {e}", file=sys.stderr)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename_AtoB = os.path.join(save_dir, f'{mapping_type}_AtoB_samples_{epoch_label}.png')
        try: plt.savefig(filename_AtoB); print(f"  Saved AtoB visualization: {filename_AtoB}")
        except Exception as e: print(f"Error saving plot {filename_AtoB}: {e}", file=sys.stderr)
    else: plt.show()
    plt.close(fig_AtoB)

    # --- BtoA Visualization ---
    fig_BtoA, axs_BtoA = plt.subplots(4, n_samples, figsize=(max(15, 3 * n_samples), 12))
    if n_samples == 1: axs_BtoA = axs_BtoA[:, np.newaxis]

    for i, idx in enumerate(indices): # Use same indices for comparison
        sample_data = dataset[idx]
        if sample_data is None or sample_data[1] is None: continue
        _, img_B, _ = sample_data
        real_B = img_B.unsqueeze(0).to(device) # Shape [1, C, D, H, W]

        with torch.no_grad():
             real_B_padded = F.pad(real_B, (0, 0, 0, 0, 0, 1))
             fake_A = g_model_BtoA(real_B_padded)
             fake_A = fake_A[:, :, :-1, :, :]

             fake_A_padded = F.pad(fake_A, (0, 0, 0, 0, 0, 1))
             rec_B = g_model_AtoB(fake_A_padded)
             rec_B = rec_B[:, :, :-1, :, :]

        real_B_np = (real_B.cpu().numpy()[0] + 1.0) / 2.0
        fake_A_np = (fake_A.cpu().numpy()[0] + 1.0) / 2.0
        rec_B_np  = (rec_B.cpu().numpy()[0] + 1.0) / 2.0
        diff_B = np.abs(real_B_np - rec_B_np)

        d_mid = max(0, real_B_np.shape[1] // 2)
        if real_B_np.shape[1] == 0: continue

        axs_BtoA[0, i].imshow(real_B_np[0, d_mid, :, :], cmap='gray', vmin=0, vmax=1)
        axs_BtoA[0, i].set_title(f"Real {domain_B}"); axs_BtoA[0, i].axis('off')
        axs_BtoA[1, i].imshow(fake_A_np[0, d_mid, :, :], cmap='gray', vmin=0, vmax=1)
        axs_BtoA[1, i].set_title(f"Generated {domain_A}"); axs_BtoA[1, i].axis('off')
        axs_BtoA[2, i].imshow(rec_B_np[0, d_mid, :, :], cmap='gray', vmin=0, vmax=1)
        axs_BtoA[2, i].set_title(f"Reconstructed {domain_B}"); axs_BtoA[2, i].axis('off')
        im = axs_BtoA[3, i].imshow(diff_B[0, d_mid, :, :], cmap='hot', vmin=0, vmax=max(0.1, diff_B.max()))
        axs_BtoA[3, i].set_title("Abs Difference"); axs_BtoA[3, i].axis('off')

    fig_BtoA.suptitle(f"{mapping_type.upper()}: {domain_B} -> {domain_A} -> {domain_B} ({epoch_label})", fontsize=16)
    try: fig_BtoA.tight_layout(rect=[0, 0.03, 1, 0.95])
    except Exception as e: print(f"Warning: tight_layout failed: {e}", file=sys.stderr)

    if save_dir:
        filename_BtoA = os.path.join(save_dir, f'{mapping_type}_BtoA_samples_{epoch_label}.png')
        try: plt.savefig(filename_BtoA); print(f"  Saved BtoA visualization: {filename_BtoA}")
        except Exception as e: print(f"Error saving plot {filename_BtoA}: {e}", file=sys.stderr)
    else: plt.show()
    plt.close(fig_BtoA)

    # Restore dataset state
    if old_augment_state is not None:
        dataset.augment = old_augment_state
    # Restore model state
    g_model_AtoB.train()
    g_model_BtoA.train()

# ------------------------------
# Plotting from Logs
# ------------------------------
def load_log_data(log_file):
    """Loads log data from a JSON file."""
    if not log_file or not os.path.exists(log_file):
        print(f"Error: Log file not found at {log_file}", file=sys.stderr)
        return None
    try:
        with open(log_file, 'r') as f: log_data = json.load(f)
        log_data.sort(key=lambda x: (x.get('epoch', 0), x.get('iteration', 0)))
        return log_data
    except Exception as e:
        print(f"Error loading/parsing log file {log_file}: {e}", file=sys.stderr); return None

def plot_losses(log_data, mapping_type, epoch_label, save_dir):
    """Plots training losses from log data."""
    try:
        import matplotlib.pyplot as plt
        plt.switch_backend('agg')
    except ImportError:
        print("Error: matplotlib not found. Cannot plot losses.", file=sys.stderr); return

    print("Plotting training losses...")
    has_iterations = any('iteration' in entry and entry['iteration'] < entry.get('n_steps', float('inf')) for entry in log_data if isinstance(entry, dict))
    epoch_avg_losses = {}; epoch_counts = {}
    loss_keys = ['loss_G_total', 'loss_D_A', 'loss_D_B', 'loss_G_gan', 'loss_G_cycle', 'loss_G_identity', 'loss_G_intensity']

    for entry in log_data:
        epoch = entry.get('epoch'); iteration = entry.get('iteration')
        if epoch is None: continue
        is_iter_log = has_iterations and iteration is not None and 'val_psnr_avg' not in entry

        if epoch not in epoch_avg_losses:
            epoch_avg_losses[epoch] = {key: 0.0 for key in loss_keys}; epoch_counts[epoch] = {key: 0 for key in loss_keys}

        for key in loss_keys:
            if key in entry and entry[key] is not None:
                if is_iter_log:
                    epoch_avg_losses[epoch][key] += entry[key]; epoch_counts[epoch][key] += 1
                elif not has_iterations:
                    epoch_avg_losses[epoch][key] = entry[key]; epoch_counts[epoch][key] = 1

    if has_iterations:
        for epoch in epoch_avg_losses:
            for key in loss_keys:
                if epoch_counts[epoch][key] > 0: epoch_avg_losses[epoch][key] /= epoch_counts[epoch][key]
                else: epoch_avg_losses[epoch][key] = None

    plot_epochs = sorted(epoch_avg_losses.keys())
    losses_to_plot = {}; valid_epochs_for_key = {}
    for key in loss_keys:
        data = [epoch_avg_losses[e].get(key) for e in plot_epochs]
        valid_indices = [i for i, d in enumerate(data) if d is not None]
        if not valid_indices: print(f"  Skipping plot for loss key '{key}' - no data found."); continue
        valid_epochs = [plot_epochs[i] for i in valid_indices]; valid_data = [data[i] for i in valid_indices]
        losses_to_plot[key] = valid_data; valid_epochs_for_key[key] = valid_epochs

    if not losses_to_plot: print("  No valid loss data found to plot."); return

    fig, axs = plt.subplots(2, 2, figsize=(15, 10)); fig.suptitle(f'{mapping_type.upper()} Training Losses ({epoch_label})', fontsize=16)

    def plot_subplot(ax, data_key, title, ylabel='Loss'):
        if data_key in losses_to_plot:
            plot_x_epochs = [e + 1 for e in valid_epochs_for_key[data_key]]
            ax.plot(plot_x_epochs, losses_to_plot[data_key], label=data_key.replace('loss_','').replace('_',' '), marker='.')
            ax.set_title(title); ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel); ax.legend(); ax.grid(True)
            if len(plot_x_epochs) < 15: ax.set_xticks(plot_x_epochs)
            return True
        else: ax.set_title(f'{title} (No Data)'); ax.axis('off'); return False

    plot_subplot(axs[0, 0], 'loss_G_total', 'Generator Loss (Total)')
    d_plotted = False
    if 'loss_D_A' in losses_to_plot: axs[0, 1].plot([e + 1 for e in valid_epochs_for_key['loss_D_A']], losses_to_plot['loss_D_A'], label='D_A Loss', marker='.'); d_plotted=True
    if 'loss_D_B' in losses_to_plot: axs[0, 1].plot([e + 1 for e in valid_epochs_for_key['loss_D_B']], losses_to_plot['loss_D_B'], label='D_B Loss', marker='.'); d_plotted=True
    if d_plotted: axs[0, 1].set_title('Discriminator Losses'); axs[0, 1].set_xlabel('Epoch'); axs[0, 1].set_ylabel('Loss'); axs[0, 1].legend(); axs[0, 1].grid(True);
    else: axs[0, 1].set_title('Discriminator Losses (No Data)'); axs[0, 1].axis('off')
    g_comp_plotted = False
    for key in ['loss_G_gan', 'loss_G_cycle', 'loss_G_identity', 'loss_G_intensity']:
        if key in losses_to_plot: axs[1, 0].plot([e + 1 for e in valid_epochs_for_key[key]], losses_to_plot[key], label=key.replace('loss_G_',''), marker='.'); g_comp_plotted=True
    if g_comp_plotted: axs[1, 0].set_title('Generator Loss Components'); axs[1, 0].set_xlabel('Epoch'); axs[1, 0].set_ylabel('Loss'); axs[1, 0].legend(); axs[1, 0].grid(True)
    else: axs[1, 0].set_title('Generator Loss Components (No Data)'); axs[1, 0].axis('off')

    axs[1, 1].axis('off')
    try: plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    except Exception as e: print(f"Warning: tight_layout failed: {e}", file=sys.stderr)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f'{mapping_type}_training_losses_{epoch_label}.png')
        try: plt.savefig(filename); print(f"  Saved loss plot: {filename}")
        except Exception as e: print(f"Error saving loss plot {filename}: {e}", file=sys.stderr)
    else: plt.show()
    plt.close(fig)

def plot_metrics(log_data, mapping_type, epoch_label, save_dir):
    """Plots validation metrics from log data."""
    try:
        import matplotlib.pyplot as plt
        plt.switch_backend('agg')
    except ImportError:
        print("Error: matplotlib not found. Cannot plot metrics.", file=sys.stderr); return

    print("Plotting validation metrics...")
    val_entries = [entry for entry in log_data if 'val_psnr_avg' in entry and entry['val_psnr_avg'] is not None]
    if not val_entries: print("  No validation metrics found in log file."); return

    val_entries.sort(key=lambda x: x.get('epoch', 0))
    epochs = [entry.get('epoch', 0) for entry in val_entries]
    metric_keys = ['val_psnr_avg', 'val_ssim_avg', 'val_loss_cycle', 'val_loss_id']
    metrics_to_plot = {}; valid_epochs_for_key = {}

    for key in metric_keys:
        data = [entry.get(key) for entry in val_entries]
        valid_indices = [i for i, d in enumerate(data) if d is not None]
        if not valid_indices: print(f"  Skipping plot for metric key '{key}' - no data found."); continue
        valid_epochs = [epochs[i] for i in valid_indices]; valid_data = [data[i] for i in valid_indices]
        metrics_to_plot[key] = valid_data; valid_epochs_for_key[key] = valid_epochs

    if not metrics_to_plot: print("  No valid metric data found to plot."); return

    fig, axs = plt.subplots(2, 2, figsize=(15, 10)); fig.suptitle(f'{mapping_type.upper()} Validation Metrics ({epoch_label})', fontsize=16)

    def plot_metric_subplot(ax, data_key, title, ylabel):
        if data_key in metrics_to_plot:
            plot_x_epochs = [e + 1 for e in valid_epochs_for_key[data_key]]
            ax.plot(plot_x_epochs, metrics_to_plot[data_key], label=data_key.replace('val_','').replace('_avg',' Avg'), marker='.')
            ax.set_title(title); ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel); ax.legend(); ax.grid(True)
            # Set integer ticks if epochs are integers
            if all(isinstance(e, int) for e in plot_x_epochs) and len(plot_x_epochs) > 0:
                 ax.set_xticks(range(min(plot_x_epochs), max(plot_x_epochs) + 1))
        else: ax.set_title(f'{title} (No Data)'); ax.axis('off')

    plot_metric_subplot(axs[0, 0], 'val_psnr_avg', 'Validation PSNR', 'PSNR (dB)')
    plot_metric_subplot(axs[0, 1], 'val_ssim_avg', 'Validation SSIM', 'SSIM')
    plot_metric_subplot(axs[1, 0], 'val_loss_cycle', 'Validation Cycle Loss', 'L1 Loss')
    plot_metric_subplot(axs[1, 1], 'val_loss_id', 'Validation Identity Loss', 'L1 Loss')

    try: plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    except Exception as e: print(f"Warning: tight_layout failed: {e}", file=sys.stderr)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f'{mapping_type}_validation_metrics_{epoch_label}.png')
        try: plt.savefig(filename); print(f"  Saved metrics plot: {filename}")
        except Exception as e: print(f"Error saving metrics plot {filename}: {e}", file=sys.stderr)
    else: plt.show()
    plt.close(fig)

# ----------------------------------------
# Prepare Real Images for FID
# ----------------------------------------
def prepare_real_images_for_fid(dataset, num_images, output_dir_a, output_dir_b, mapping_type):
    """Loads real images from dataset, extracts middle slice, saves as PNG for FID."""
    if Image is None: print("E: Pillow not found.", file=sys.stderr); return False
    dataset_size = len(dataset)
    if dataset_size == 0: print("E: Dataset for FID prep is empty.", file=sys.stderr); return False

    print(f"Preparing real images for FID...")
    os.makedirs(output_dir_a, exist_ok=True); os.makedirs(output_dir_b, exist_ok=True)
    count = 0; processed_indices = set()
    num_images = min(num_images, dataset_size) # Cap at dataset size

    domain_map = {'t2_flair': ('T2', 'FLAIR'), 't1_contrast': ('T1', 'T1CE')}
    domain_A_name, domain_B_name = domain_map.get(mapping_type, ('A', 'B'))

    pbar = tqdm(total=num_images, desc="Prep Real FID Imgs", file=sys.stdout, dynamic_ncols=True)
    while count < num_images and len(processed_indices) < dataset_size:
        idx = random.randint(0, dataset_size - 1) # Use random import
        if idx in processed_indices: continue
        processed_indices.add(idx)

        try:
            sample_data = dataset[idx]
            if sample_data is None or sample_data[0] is None or sample_data[1] is None: continue
            img_A, img_B, file_path = sample_data # These are [-1, 1]
            case_id = os.path.basename(file_path).replace('image_', '').replace('.npy', '')

            # Process/save Domain A (T2 or T1)
            real_A_np = (img_A.cpu().numpy()[0] + 1.0) / 2.0 # [D, H, W], range [0, 1]
            if real_A_np.shape[0] <= 0: continue
            d_mid_A = real_A_np.shape[0] // 2
            slice_A = real_A_np[d_mid_A, :, :]
            slice_A_uint8 = (slice_A * 255).clip(0, 255).astype(np.uint8)
            pil_img_A = Image.fromarray(slice_A_uint8, mode='L')
            save_path_A = os.path.join(output_dir_a, f'real_{domain_A_name}_{case_id}_{idx:05d}.png') # Use index for unique name
            pil_img_A.save(save_path_A)

            # Process/save Domain B (FLAIR or T1CE)
            real_B_np = (img_B.cpu().numpy()[0] + 1.0) / 2.0 # [D, H, W], range [0, 1]
            if real_B_np.shape[0] <= 0: continue
            d_mid_B = real_B_np.shape[0] // 2
            slice_B = real_B_np[d_mid_B, :, :]
            slice_B_uint8 = (slice_B * 255).clip(0, 255).astype(np.uint8)
            pil_img_B = Image.fromarray(slice_B_uint8, mode='L')
            save_path_B = os.path.join(output_dir_b, f'real_{domain_B_name}_{case_id}_{idx:05d}.png') # Use index for unique name
            pil_img_B.save(save_path_B)

            count += 1; pbar.update(1)
        except Exception as e: print(f"E: preparing real slice {idx} for {case_id}: {e}", file=sys.stderr)
    pbar.close()
    print(f"Finished preparing {count} real image pairs for FID.")
    if count < 2: print(f"Warning: Only prepared {count} images. FID requires at least 2.", file=sys.stderr)
    return count >= 2

# ------------------------------
# FID Generation & Calculation
# ------------------------------
def generate_images_for_fid(generator, dataset, num_images, output_dir, device, input_domain='A'):
    """Generates images using the generator and saves middle slice as PNGs for FID calculation."""
    if Image is None: print("E: Pillow not found.", file=sys.stderr); return None
    dataset_size = len(dataset)
    if dataset_size == 0: print("E: Dataset for FID gen is empty.", file=sys.stderr); return None

    print(f"Generating {num_images} images for FID (Input: {input_domain}), saving to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    generator.eval()
    count = 0; processed_indices = set()
    num_images = min(num_images, dataset_size) # Cap at dataset size

    pbar = tqdm(total=num_images, desc=f"Gen FID Imgs (In:{input_domain})", file=sys.stdout, dynamic_ncols=True)
    while count < num_images and len(processed_indices) < dataset_size:
        idx = random.randint(0, dataset_size - 1) # Use random import
        if idx in processed_indices: continue
        processed_indices.add(idx)

        try:
            sample_data = dataset[idx]
            if sample_data is None or sample_data[0] is None or sample_data[1] is None: continue

            img_A, img_B, file_path = sample_data
            case_id = os.path.basename(file_path).replace('image_', '').replace('.npy', '')

            if input_domain == 'A': real_input = img_A.to(device)
            elif input_domain == 'B': real_input = img_B.to(device)
            else: raise ValueError("input_domain must be 'A' or 'B'")

            with torch.no_grad():
                real_input_padded = F.pad(real_input.unsqueeze(0), (0, 0, 0, 0, 0, 1))
                fake_output = generator(real_input_padded)
                fake_output = fake_output[:, :, :-1, :, :] # Crop padding [-1, 1]

            fake_output_np = (fake_output.cpu().numpy()[0] + 1.0) / 2.0 # [C, D, H, W] -> [0, 1]

            if fake_output_np.shape[1] <= 0: continue

            d_mid = fake_output_np.shape[1] // 2
            slice_img = fake_output_np[0, d_mid, :, :]
            slice_img_uint8 = (slice_img * 255).clip(0, 255).astype(np.uint8)

            pil_img = Image.fromarray(slice_img_uint8, mode='L')
            save_path = os.path.join(output_dir, f'generated_{case_id}_{idx:05d}.png') # Use index
            pil_img.save(save_path)
            count += 1; pbar.update(1)
        except Exception as e: print(f"E: saving generated image {idx} for {case_id}: {e}", file=sys.stderr)
    pbar.close()

    generator.train() # Restore train mode
    print(f"Finished generating {count} images.")
    if count < 2: print(f"Warning: Only generated {count} images. FID requires at least 2.", file=sys.stderr); return None
    return output_dir

def calculate_fid_scores(real_images_dir_a, real_images_dir_b, generated_images_dir_a, generated_images_dir_b, epoch_label, mapping_type, save_dir, batch_size=16, device='cpu'):
    """Calculates FID scores between generated and real images."""
    if fid_score is None: print("E: FID calculation skipped, pytorch-fid not installed.", file=sys.stderr); return

    print("Calculating FID scores...")
    fid_results = {}

    def check_fid_dir(name, path):
        if not path or not os.path.isdir(path): print(f"E: {name} directory not found: {path}", file=sys.stderr); return False
        image_files = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if len(image_files) < 2: print(f"E: {name} dir '{path}' needs >= 2 images for FID (found {len(image_files)}).", file=sys.stderr); return False
        return True

    dirs_ok = True
    if not check_fid_dir("Real A", real_images_dir_a): dirs_ok = False
    if not check_fid_dir("Real B", real_images_dir_b): dirs_ok = False
    if not check_fid_dir("Generated A", generated_images_dir_a): dirs_ok = False
    if not check_fid_dir("Generated B", generated_images_dir_b): dirs_ok = False
    if not dirs_ok: print("FID calculation aborted due to missing/invalid image directories.", file=sys.stderr); return

    def compute_fid(name, path1, path2):
        try:
            print(f"  Calculating FID: {name} ({os.path.basename(path1)} vs {os.path.basename(path2)})")
            device_str = str(device)
            fid_value = fid_score.calculate_fid_given_paths([path1, path2], batch_size=batch_size, device=device_str, dims=2048)
            print(f"  FID ({name}): {fid_value:.3f}")
            return fid_value
        except Exception as e: print(f"  E: calculating FID ({name}): {e}", file=sys.stderr); return f'Error: {e}'

    fid_results['Generated_A_vs_Real_A'] = compute_fid("GenA vs RealA", real_images_dir_a, generated_images_dir_a)
    fid_results['Generated_A_vs_Real_B'] = compute_fid("GenA vs RealB", real_images_dir_b, generated_images_dir_a)
    fid_results['Generated_B_vs_Real_B'] = compute_fid("GenB vs RealB", real_images_dir_b, generated_images_dir_b)
    fid_results['Generated_B_vs_Real_A'] = compute_fid("GenB vs RealA", real_images_dir_a, generated_images_dir_b)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f'{mapping_type}_fid_scores_{epoch_label}.json')
        try:
            serializable_results = {k: (str(v) if isinstance(v, Exception) else v) for k, v in fid_results.items()}
            with open(filename, 'w') as f: json.dump(serializable_results, f, indent=4)
            print(f"  Saved FID scores: {filename}")
        except Exception as e: print(f"  E: saving FID scores: {e}", file=sys.stderr)

#-------------------------------
# Save Synthetic Volume
#-------------------------------
def save_synthetic_volume(g_model_AtoB, dataloader, device, output_dir, mapping_type, num_samples=-1):
    """Generates synthetic Domain B volumes and saves as .npy files (range [0, 1])."""
    g_model_AtoB.eval()
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving synthetic volumes to: {output_dir}")
    count = 0
    max_count = num_samples if num_samples >= 0 else float('inf')
    dataset_size = len(dataloader.dataset) if hasattr(dataloader, 'dataset') else 0
    if dataset_size == 0: print("W: Dataset empty, cannot save synthetic volumes.", file=sys.stderr); return
    max_iters = min(int(max_count) if max_count != float('inf') else dataset_size, dataset_size)

    with torch.no_grad():
        # Iterate using index if batch size is 1 and dataset is accessible
        if dataloader.batch_size == 1 and hasattr(dataloader, 'dataset'):
             indices_to_process = list(range(dataset_size))
             random.shuffle(indices_to_process) # Use random import
             pbar = tqdm(indices_to_process, total=max_iters, desc="Saving Synthetic Volumes", file=sys.stdout, dynamic_ncols=True)

             for idx in pbar:
                if count >= max_count: break
                batch = dataloader.dataset[idx]
                if batch is None or batch[0] is None: continue
                real_A, _, file_path = batch

                case_id = os.path.basename(file_path).replace('image_', '').replace('.npy', '')
                current_real_A = real_A.unsqueeze(0).to(device)

                real_A_padded = F.pad(current_real_A, (0, 0, 0, 0, 0, 1))
                fake_B = g_model_AtoB(real_A_padded)
                fake_B = fake_B[:, :, :-1, :, :]
                fake_B_01 = (fake_B + 1.0) / 2.0

                fake_B_np = fake_B_01.squeeze().cpu().numpy()
                if fake_B_np.ndim == 3:
                    fake_B_np_hwd = np.transpose(fake_B_np, (1, 2, 0))
                    output_filename = os.path.join(output_dir, f"synthetic_{mapping_type}_{case_id}.npy")
                    try: np.save(output_filename, fake_B_np_hwd.astype(np.float32)); count += 1
                    except Exception as e: print(f"E: saving synth {case_id}: {e}", file=sys.stderr)
                else: print(f"W: Bad shape {fake_B_np.shape} for {case_id}", file=sys.stderr)
             pbar.close()
        else: # Fallback to iterating dataloader
             print("W: Iterating DataLoader for saving synth, order might not be random if shuffle=False.", file=sys.stderr)
             pbar = tqdm(dataloader, total=min(max_iters, len(dataloader)), desc="Saving Synthetic Volumes", file=sys.stdout, dynamic_ncols=True)
             for i, batch in enumerate(pbar):
                 if count >= max_count: break
                 if batch is None or batch[0] is None: continue
                 real_A, _, file_path_tuple = batch
                 for j in range(real_A.size(0)): # Handle potential batches > 1 in fallback
                      if count >= max_count: break
                      file_path = file_path_tuple[j]
                      case_id = os.path.basename(file_path).replace('image_', '').replace('.npy', '')
                      current_real_A = real_A[j:j+1].to(device)
                      # ... (rest of generation, scaling, saving logic) ...
                      real_A_padded = F.pad(current_real_A, (0, 0, 0, 0, 0, 1))
                      fake_B = g_model_AtoB(real_A_padded)
                      fake_B = fake_B[:, :, :-1, :, :]
                      fake_B_01 = (fake_B + 1.0) / 2.0
                      fake_B_np = fake_B_01.squeeze().cpu().numpy()
                      if fake_B_np.ndim == 3:
                          fake_B_np_hwd = np.transpose(fake_B_np, (1, 2, 0))
                          output_filename = os.path.join(output_dir, f"synthetic_{mapping_type}_{case_id}.npy")
                          try: np.save(output_filename, fake_B_np_hwd.astype(np.float32)); count += 1
                          except Exception as e: print(f"E: saving synth {case_id}: {e}", file=sys.stderr)
                      else: print(f"W: Bad shape {fake_B_np.shape} for {case_id}", file=sys.stderr)
             pbar.close()

    print(f"Finished saving {count} synthetic volumes.")
    g_model_AtoB.train() # Restore train mode

#--------------------------------
# Save Difference Volume
#--------------------------------
def save_difference_volume(g_model_AtoB, dataloader, device, output_dir, mapping_type, num_samples=-1):
    """Generates synthetic B, calculates diff with real B, saves diff volume (range [0, 2])."""
    g_model_AtoB.eval()
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving difference volumes to: {output_dir}")
    count = 0
    max_count = num_samples if num_samples >= 0 else float('inf')
    dataset_size = len(dataloader.dataset) if hasattr(dataloader, 'dataset') else 0
    if dataset_size == 0: print("W: Dataset empty, cannot save difference volumes.", file=sys.stderr); return
    max_iters = min(int(max_count) if max_count != float('inf') else dataset_size, dataset_size)

    with torch.no_grad():
        # Iterate using index if batch size is 1
        if dataloader.batch_size == 1 and hasattr(dataloader, 'dataset'):
            indices_to_process = list(range(dataset_size))
            random.shuffle(indices_to_process) # Use random import
            pbar = tqdm(indices_to_process, total=max_iters, desc="Saving Difference Volumes", file=sys.stdout, dynamic_ncols=True)
            for idx in pbar:
                if count >= max_count: break
                batch = dataloader.dataset[idx]
                if batch is None or batch[0] is None or batch[1] is None: continue
                real_A, real_B, file_path = batch
                case_id = os.path.basename(file_path).replace('image_', '').replace('.npy', '')
                current_real_A = real_A.unsqueeze(0).to(device) # [-1, 1]
                current_real_B = real_B.unsqueeze(0).to(device) # [-1, 1]
                real_A_padded = F.pad(current_real_A, (0, 0, 0, 0, 0, 1))
                fake_B = g_model_AtoB(real_A_padded)
                fake_B = fake_B[:, :, :-1, :, :] # Crop output [-1, 1]
                if fake_B.shape != current_real_B.shape: print(f"W: Shape mismatch {case_id}", file=sys.stderr); continue
                diff_B = torch.abs(fake_B - current_real_B) # Range [0, 2]
                diff_B_np = diff_B.squeeze().cpu().numpy() # Shape (D, H, W)
                if diff_B_np.ndim == 3:
                    diff_B_np_hwd = np.transpose(diff_B_np, (1, 2, 0)) # Shape (H, W, D)
                    output_filename = os.path.join(output_dir, f"difference_{mapping_type}_{case_id}.npy")
                    try: np.save(output_filename, diff_B_np_hwd.astype(np.float32)); count += 1
                    except Exception as e: print(f"E: saving diff {case_id}: {e}", file=sys.stderr)
                else: print(f"W: Bad shape {diff_B_np.shape} for {case_id}", file=sys.stderr)
            pbar.close()
        else: # Fallback
             print("W: Iterating DataLoader for saving diff, order/count might be affected.", file=sys.stderr)
             pbar = tqdm(dataloader, total=min(max_iters, len(dataloader)), desc="Saving Difference Volumes", file=sys.stdout, dynamic_ncols=True)
             for i, batch in enumerate(pbar):
                 if count >= max_count: break
                 if batch is None or batch[0] is None or batch[1] is None: continue
                 real_A, real_B, file_path_tuple = batch
                 for j in range(real_A.size(0)):
                     if count >= max_count: break
                     file_path = file_path_tuple[j]
                     case_id = os.path.basename(file_path).replace('image_', '').replace('.npy', '')
                     current_real_A = real_A[j:j+1].to(device)
                     current_real_B = real_B[j:j+1].to(device)
                     # ... (rest of generation, diff, saving logic) ...
                     real_A_padded = F.pad(current_real_A, (0, 0, 0, 0, 0, 1))
                     fake_B = g_model_AtoB(real_A_padded)
                     fake_B = fake_B[:, :, :-1, :, :]
                     if fake_B.shape != current_real_B.shape: print(f"W: Shape mismatch {case_id}", file=sys.stderr); continue
                     diff_B = torch.abs(fake_B - current_real_B)
                     diff_B_np = diff_B.squeeze().cpu().numpy()
                     if diff_B_np.ndim == 3:
                         diff_B_np_hwd = np.transpose(diff_B_np, (1, 2, 0))
                         output_filename = os.path.join(output_dir, f"difference_{mapping_type}_{case_id}.npy")
                         try: np.save(output_filename, diff_B_np_hwd.astype(np.float32)); count += 1
                         except Exception as e: print(f"E: saving diff {case_id}: {e}", file=sys.stderr)
                     else: print(f"W: Bad shape {diff_B_np.shape} for {case_id}", file=sys.stderr)
             pbar.close()


    print(f"Finished saving {count} difference volumes.")
    g_model_AtoB.train() # Restore train mode


# ------------------------------
# Main Execution Logic
# ------------------------------
def main():
    # --- Device Setup ---
    global device
    print("> Setting up device...")
    if torch.cuda.is_available():
        try:
            device = torch.device("cuda")
            device_name = torch.cuda.get_device_name(device); print(f"  Using device: {device} ({device_name})")
            x = torch.randn(1, device=device); print("  GPU test successful.")
        except Exception as e: print(f"  GPU initialization failed: {e}. Falling back to CPU.", file=sys.stderr); device = torch.device("cpu")
    else: device = torch.device("cpu"); print("  CUDA not available. Using device: cpu")
    print("-------------------------")

    parser = argparse.ArgumentParser(description="Evaluate 3D CycleGAN Model")
    # --- Arguments ---
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the .pth checkpoint file.")

    parser.add_argument("--eval_data_dir", type=str, default="./data/processed_data/brats128_validation/images", help="Path to evaluation data directory ([0,1] normalized .npy files, e.g., validation split).")
    parser.add_argument("--output_dir", type=str, default="./output/evaluation", help="Base directory to save evaluation results.")

    parser.add_argument("--tasks", nargs='+', default=['visualize', 'plot_logs'], choices=['visualize', 'plot_logs', 'calculate_fid'], help="List of standard evaluation tasks to perform.")
    parser.add_argument("--vis_samples", type=int, default=5, help="Number of random samples for visualization task.")
    parser.add_argument("--log_file", type=str, default=None, help="Path to training log JSON file (optional, attempts auto-find for plot_logs).")

    parser.add_argument('--save_synthetic', action='store_true', help='Generate and save full synthetic volumes.')
    parser.add_argument('--save_difference', action='store_true', help='Generate and save full difference volumes.')
    parser.add_argument("--save_num_samples", type=int, default=5, help="Number of validation samples to process for saving volumes (-1 to process all).")
    # --- FID arguments ---
    parser.add_argument("--fid_num_images", type=int, default=50, help="Number of images (slices) to generate/prepare for FID calculation.")
    parser.add_argument("--fid_batch_size", type=int, default=16, help="Batch size for FID calculation.")
    parser.add_argument("--fid_force_regen", action='store_true', help="Force regeneration/preparation of images for FID even if directories exist.")
    # --- Other arguments ---
    parser.add_argument("--num_workers", type=int, default=2, help="Number of DataLoader workers.")


    args = parser.parse_args()

    # --- Load Checkpoint and Extract Info ---
    if not os.path.exists(args.checkpoint): print(f"Error: Checkpoint file not found: {args.checkpoint}", file=sys.stderr); sys.exit(1)
    try:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        start_epoch = checkpoint.get('epoch', -1)
        epoch_label = f"epoch{start_epoch+1:03d}"
        mapping_type = checkpoint.get('mapping_type')
        n_resnet = checkpoint.get('n_resnet')
        if not mapping_type or n_resnet is None: raise KeyError("Checkpoint missing 'mapping_type' or 'n_resnet'.")
        print(f"> Loaded checkpoint: {os.path.basename(args.checkpoint)} (Epoch {start_epoch}, Map: {mapping_type}, ResNet: {n_resnet})")
    except Exception as e: print(f"Error loading checkpoint info: {e}", file=sys.stderr); sys.exit(1)

    # --- Prepare Output Directory ---
    eval_output_dir = os.path.abspath(os.path.expanduser(os.path.join(args.output_dir, mapping_type, epoch_label)))
    os.makedirs(eval_output_dir, exist_ok=True)
    print(f"> Saving evaluation results to: {eval_output_dir}")

    # --- Initialize Models ---
    try:
        g_model_AtoB, g_model_BtoA = init_models(mapping_type, n_resnet, device)
        g_model_AtoB.load_state_dict(checkpoint['g_model_AtoB_state'])
        g_model_BtoA.load_state_dict(checkpoint['g_model_BtoA_state'])
        print("> Generator models initialized and loaded.")
    except Exception as e: print(f"Error initializing/loading models: {e}", file=sys.stderr); sys.exit(1)

    # --- Prepare Dataset ---
    try:
        eval_data_dir_abs = os.path.abspath(os.path.expanduser(args.eval_data_dir))
        if not os.path.isdir(eval_data_dir_abs): raise FileNotFoundError(f"Evaluation data directory not found: {eval_data_dir_abs}")
        eval_dataset = PairedPreprocessedDataset(eval_data_dir_abs, mapping_type, augment=False)
        if len(eval_dataset) == 0: raise ValueError(f"No data found in evaluation directory: {eval_data_dir_abs}")

        def collate_fn_skip_none(batch):
            batch = list(filter(lambda x: x is not None and x[0] is not None and x[1] is not None, batch))
            if not batch: return None
            return torch.utils.data.dataloader.default_collate(batch)

        num_workers = min(args.num_workers, os.cpu_count() // 2 if os.cpu_count() else 1)
        # Use collate_fn in DataLoader
        eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=collate_fn_skip_none)
        print(f"Found {len(eval_dataset)} evaluation samples. Using {num_workers} workers.")
    except Exception as e: print(f"Error creating evaluation dataset/loader: {e}", file=sys.stderr); sys.exit(1)

    # ==========================
    # --- Execute Tasks ---
    # ==========================
    print("\n--- Running Selected Evaluation Tasks ---")
    tasks_to_run = set(args.tasks)

    # --- Task: Visualize Sample Translations ---
    if 'visualize' in tasks_to_run:
        viz_save_dir = os.path.join(eval_output_dir, 'visualizations')
        visualize_sample_translations(epoch_label, g_model_AtoB, g_model_BtoA, eval_dataset, mapping_type, device, n_samples=args.vis_samples, save_dir=viz_save_dir)

    # --- Task: Plot Logs ---
    if 'plot_logs' in tasks_to_run:
        log_file_to_use = args.log_file
        if not log_file_to_use: # Auto-find logic
            try:
                # <<< FIX: Go up 3 levels from checkpoint file path >>>
                checkpoint_dir = os.path.dirname(os.path.abspath(args.checkpoint))
                checkpoints_base_dir = os.path.dirname(checkpoint_dir)
                mapping_type_base_dir = os.path.dirname(checkpoints_base_dir)
                log_dir_guess = os.path.join(mapping_type_base_dir, 'logs')
                if os.path.isdir(log_dir_guess):
                    potential_logs = sorted(glob.glob(os.path.join(log_dir_guess, f'{mapping_type}_log_*.json')))
                    if potential_logs: log_file_to_use = potential_logs[-1]; print(f"  Auto-found log file: {log_file_to_use}")
                    else: print(f"  Could not automatically find log file in {log_dir_guess}")
                else: print(f"  Could not automatically find log directory at {log_dir_guess}")
            except Exception as e: print(f"  Error during log auto-find: {e}", file=sys.stderr)

        if log_file_to_use and os.path.exists(log_file_to_use):
            log_data = load_log_data(log_file_to_use)
            if log_data:
                plot_save_dir = os.path.join(eval_output_dir, 'plots')
                plot_losses(log_data, mapping_type, epoch_label, plot_save_dir)
                plot_metrics(log_data, mapping_type, epoch_label, plot_save_dir)
            else: print("Skipping plot_logs task: Failed to load log data.")
        elif 'plot_logs' in tasks_to_run:
             print(f"Skipping plot_logs task: Log file not specified or found ({args.log_file or 'auto-find failed'}).")

    # --- Task: Calculate FID (with Auto Image Prep) ---
    if 'calculate_fid' in tasks_to_run:
        if fid_score is None or Image is None:
            print("Skipping calculate_fid task: Missing 'pytorch-fid' or 'Pillow' package.", file=sys.stderr)
        else:
            print("\n--- Preparing images for FID ---")
            fid_base_dir = os.path.join(eval_output_dir, 'fid_data')
            # Define paths for automatically generated real and fake images
            fid_gen_a_dir = os.path.join(fid_base_dir, 'generated_A')
            fid_gen_b_dir = os.path.join(fid_base_dir, 'generated_B')
            fid_real_a_dir_auto = os.path.join(fid_base_dir, 'real_A')
            fid_real_b_dir_auto = os.path.join(fid_base_dir, 'real_B')

            # Prepare real images automatically if needed or forced
            real_prep_needed = args.fid_force_regen or not os.path.isdir(fid_real_a_dir_auto) or not os.path.exists(os.path.join(fid_real_a_dir_auto, f'real_{mapping_type.split("_")[0]}_00000.png')) or \
                               not os.path.isdir(fid_real_b_dir_auto) or not os.path.exists(os.path.join(fid_real_b_dir_auto, f'real_{mapping_type.split("_")[1]}_00000.png'))
            real_prep_success = False
            if real_prep_needed:
                real_prep_success = prepare_real_images_for_fid(
                    dataset=eval_dataset,
                    num_images=args.fid_num_images,
                    output_dir_a=fid_real_a_dir_auto,
                    output_dir_b=fid_real_b_dir_auto,
                    mapping_type=mapping_type
                )
            else:
                print("Skipping real image preparation for FID, directories seem populated.")
                real_prep_success = True # Assume existing dirs are okay if not forcing regen

            # Generate fake images if needed or forced
            gen_a_needed = args.fid_force_regen or not os.path.isdir(fid_gen_a_dir) or not os.path.exists(os.path.join(fid_gen_a_dir, f'generated_00000.png'))
            gen_b_needed = args.fid_force_regen or not os.path.isdir(fid_gen_b_dir) or not os.path.exists(os.path.join(fid_gen_b_dir, f'generated_00000.png'))
            gen_a_success = True; gen_b_success = True

            if gen_a_needed:
                print("Generating Domain A images for FID (from B)...")
                gen_a_path = generate_images_for_fid(g_model_BtoA, eval_dataset, args.fid_num_images, fid_gen_a_dir, device, input_domain='B')
                gen_a_success = gen_a_path is not None and len(glob.glob(os.path.join(gen_a_path, '*.png'))) >= 2
            else: print(f"Skipping generation for Domain A, directory exists: {fid_gen_a_dir}")

            if gen_b_needed:
                print("Generating Domain B images for FID (from A)...")
                gen_b_path = generate_images_for_fid(g_model_AtoB, eval_dataset, args.fid_num_images, fid_gen_b_dir, device, input_domain='A')
                gen_b_success = gen_b_path is not None and len(glob.glob(os.path.join(gen_b_path, '*.png'))) >= 2
            else: print(f"Skipping generation for Domain B, directory exists: {fid_gen_b_dir}")

            # Calculate FID if all preparations were successful
            if real_prep_success and gen_a_success and gen_b_success:
                fid_save_dir = os.path.join(eval_output_dir, 'fid_scores')
                calculate_fid_scores(
                    real_images_dir_a=fid_real_a_dir_auto, real_images_dir_b=fid_real_b_dir_auto,
                    generated_images_dir_a=fid_gen_a_dir, generated_images_dir_b=fid_gen_b_dir,
                    epoch_label=epoch_label, mapping_type=mapping_type, save_dir=fid_save_dir,
                    batch_size=args.fid_batch_size, device=device
                )
            else: print("Skipping FID calculation due to errors or insufficient images during preparation/generation.")

    # --- Task: Save Synthetic Volumes ---
    if args.save_synthetic:
        synthetic_output_dir = os.path.join(eval_output_dir, f"synthetic_volumes")
        save_synthetic_volume(g_model_AtoB, eval_dataloader, device, synthetic_output_dir, mapping_type, num_samples=args.save_num_samples)

    # --- Task: Save Difference Volumes ---
    if args.save_difference:
        difference_output_dir = os.path.join(eval_output_dir, f"difference_volumes")
        save_difference_volume(g_model_AtoB, eval_dataloader, device, difference_output_dir, mapping_type, num_samples=args.save_num_samples)

    print(f"\nEvaluation complete. Results saved in subdirectories under: {eval_output_dir}")

if __name__ == "__main__":
    # Set multiprocessing start method for Windows/macOS compatibility if needed
    if sys.platform in ['win32', 'cygwin', 'darwin']:
        if multiprocessing.get_start_method(allow_none=True) is None:
             try: multiprocessing.set_start_method('spawn', force=True); print("Set multiprocessing start method to 'spawn'.")
             except: multiprocessing.set_start_method('spawn'); print("Set multiprocessing start method to 'spawn'.")

    main() # Call main execution logic
