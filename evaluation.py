# evaluation.py
"""
Enhanced Evaluation Script for 3D CycleGAN Models (Using Padding/Cropping)

Performs various evaluation tasks based on a trained checkpoint:
1.  Visualize sample translations (A->B->A and B->A->B).
2.  Plot training losses and validation metrics from log files.
3.  Calculate Fréchet Inception Distance (FID) scores.

Relies on checkpoints, logs, and model definitions from train_cycleGAN_v3.py.
Assumes data is preprocessed to [0, 1] range initially and handled as [-1, 1] by models.
Uses padding/cropping consistent with training to handle generator output size.
"""

import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import json
import sys
from tqdm import tqdm
import torch.nn.functional as F
import glob # Added for finding log file

# --- Try importing necessary components ---
try:
    from utils import (PairedPreprocessedDataset, Generator3D,
                                   load_checkpoint as load_train_checkpoint, # Use the specific loader
                                   Logger) # Assuming Logger is needed for log path logic
except ImportError as e:
    print(f"Error importing from train_cycleGAN_v3.py: {e}", file=sys.stderr)
    print("Please ensure train_cycleGAN_v3.py is in the same directory or accessible via PYTHONPATH.", file=sys.stderr)
    sys.exit(1)

try:
    from pytorch_fid import fid_score
except ImportError:
    print("Error: pytorch-fid not found. Please install it: pip install pytorch-fid", file=sys.stderr)
    # Allow script to continue for other tasks, but FID will fail if requested.
    fid_score = None

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"> Using device: {device}")

# ------------------------------
# Model Initialization (Consistent with Training)
# ------------------------------
def init_models(mapping_type, n_resnet, device='cpu'):
    """
    Creates and initializes the generator pair based on mapping type and ResNet blocks.
    Args:
        mapping_type (str): 't2_flair' or 't1_contrast'.
        n_resnet (int): Number of ResNet blocks used during training.
        device (str): Device to load models onto.
    Returns:
        tuple: (g_model_AtoB, g_model_BtoA)
    """
    if mapping_type in ['t2_flair', 't1_contrast']:
        in_channels = 1
        out_channels = 1
    else:
        raise ValueError(f"Unknown mapping_type: {mapping_type}")

    g_model_AtoB = Generator3D(in_channels, out_channels, n_resnet=n_resnet).to(device)
    g_model_BtoA = Generator3D(out_channels, in_channels, n_resnet=n_resnet).to(device)
    return g_model_AtoB, g_model_BtoA

# ------------------------------
# Visualization Function (Using Padding/Cropping)
# ------------------------------
def visualize_sample_translations(epoch_label, g_model_AtoB, g_model_BtoA, dataset,
                                  mapping_type, device='cpu', n_samples=5, save_dir=None):
    """
    Visualizes sample translations for both directions (A->B->A and B->A->B).
    Handles data in the [-1, 1] range. Uses padding/cropping for size alignment.

    Args:
        epoch_label (str): Label for the plot title/filename (e.g., "epoch050").
        g_model_AtoB (nn.Module): Generator model for A to B translation.
        g_model_BtoA (nn.Module): Generator model for B to A translation.
        dataset (Dataset): Dataset compatible with PairedPreprocessedDataset.
        mapping_type (str): Specifies the domain mapping ('t2_flair', 't1_contrast').
        device (str): Device to run inference on.
        n_samples (int): Number of random samples to visualize.
        save_dir (str, optional): Directory to save the plots. Defaults to current dir.
    """
    print(f"Generating sample translation visualizations for {mapping_type} at {epoch_label}...")

    # --- AtoB Visualization ---
    g_model_AtoB.eval()
    g_model_BtoA.eval()

    if hasattr(dataset, 'augment'): # Temporarily disable augmentation
        old_augment_state = dataset.augment
        dataset.augment = False

    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)
    fig_AtoB, axs_AtoB = plt.subplots(4, len(indices), figsize=(4 * len(indices), 16))
    if len(indices) == 1: # Handle single sample case for axs indexing
        axs_AtoB = np.expand_dims(axs_AtoB, axis=1)

    # Define domain names for titles
    if mapping_type == 't2_flair':
        domain_A, domain_B = 'T2', 'FLAIR'
    elif mapping_type == 't1_contrast':
        domain_A, domain_B = 'T1', 'T1CE'
    else:
        domain_A, domain_B = 'A', 'B'

    for i, idx in enumerate(indices):
        img_A, img_B = dataset[idx] # These are already scaled to [-1, 1]
        real_A = img_A.unsqueeze(0).to(device) # Add batch dim: [1, C, D, H, W]

        with torch.no_grad():
            # Pad, generate, crop (consistent with training loop logic)
            real_A_padded = F.pad(real_A, (0, 0, 0, 0, 0, 1))
            fake_B = g_model_AtoB(real_A_padded)
            fake_B = fake_B[:, :, :-1, :, :] # Crop padding

            # Pad fake_B before feeding to BtoA if BtoA expects padded input
            # However, the training loop feeds the *cropped* fake_B/fake_A to the next generator
            # So we use the cropped fake_B directly here.
            rec_A = g_model_BtoA(fake_B) # Use cropped fake_B

            # We expect rec_A to have the padded depth, so crop it
            rec_A = rec_A[:, :, :-1, :, :] # Crop padding

        # Convert outputs from [-1, 1] back to [0, 1] for visualization
        real_A_np = (real_A.cpu().numpy()[0] + 1) / 2.0
        fake_B_np = (fake_B.cpu().numpy()[0] + 1) / 2.0
        rec_A_np  = (rec_A.cpu().numpy()[0] + 1) / 2.0

        # Compute absolute error map
        diff_A = np.abs(real_A_np - rec_A_np)
        d_mid = real_A_np.shape[1] // 2 # Middle slice index along depth

        # Plot real input (A)
        axs_AtoB[0, i].imshow(real_A_np[0, d_mid, :, :], cmap='gray')
        axs_AtoB[0, i].set_title(f"Real {domain_A}")
        axs_AtoB[0, i].axis('off')

        # Plot synthetic output (B)
        axs_AtoB[1, i].imshow(fake_B_np[0, d_mid, :, :], cmap='gray')
        axs_AtoB[1, i].set_title(f"Generated {domain_B}")
        axs_AtoB[1, i].axis('off')

        # Plot reconstructed input (A)
        axs_AtoB[2, i].imshow(rec_A_np[0, d_mid, :, :], cmap='gray')
        axs_AtoB[2, i].set_title(f"Reconstructed {domain_A}")
        axs_AtoB[2, i].axis('off')

        # Plot error heatmap
        im = axs_AtoB[3, i].imshow(diff_A[0, d_mid, :, :], cmap='hot', vmin=0, vmax=diff_A.max())
        axs_AtoB[3, i].set_title("Abs Difference")
        axs_AtoB[3, i].axis('off')
        # Optional: Add colorbar
        # fig_AtoB.colorbar(im, ax=axs_AtoB[3, i])


    fig_AtoB.suptitle(f"{mapping_type.upper()}: {domain_A} -> {domain_B} -> {domain_A} Translation ({epoch_label})", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename_AtoB = os.path.join(save_dir, f'{mapping_type}_AtoB_samples_{epoch_label}.png')
        plt.savefig(filename_AtoB)
        print(f"  Saved AtoB visualization: {filename_AtoB}")
    else:
        plt.show()
    plt.close(fig_AtoB)

    # --- BtoA Visualization ---
    fig_BtoA, axs_BtoA = plt.subplots(4, len(indices), figsize=(4 * len(indices), 16))
    if len(indices) == 1: # Handle single sample case for axs indexing
        axs_BtoA = np.expand_dims(axs_BtoA, axis=1)


    for i, idx in enumerate(indices):
        img_A, img_B = dataset[idx] # [-1, 1] range
        real_B = img_B.unsqueeze(0).to(device) # Add batch dim

        with torch.no_grad():
             # Pad, generate, crop
             real_B_padded = F.pad(real_B, (0, 0, 0, 0, 0, 1))
             fake_A = g_model_BtoA(real_B_padded)
             fake_A = fake_A[:, :, :-1, :, :] # Crop

             # Use cropped fake_A directly as input for AtoB
             rec_B = g_model_AtoB(fake_A) # Use cropped fake_A

             # We expect rec_B to have padded depth, so crop it
             rec_B = rec_B[:, :, :-1, :, :] # Crop

        # Convert outputs from [-1, 1] back to [0, 1] for visualization
        real_B_np = (real_B.cpu().numpy()[0] + 1) / 2.0
        fake_A_np = (fake_A.cpu().numpy()[0] + 1) / 2.0
        rec_B_np  = (rec_B.cpu().numpy()[0] + 1) / 2.0

        # Compute absolute error map
        diff_B = np.abs(real_B_np - rec_B_np)
        d_mid = real_B_np.shape[1] // 2 # Middle slice index

        # Plot real input (B)
        axs_BtoA[0, i].imshow(real_B_np[0, d_mid, :, :], cmap='gray')
        axs_BtoA[0, i].set_title(f"Real {domain_B}")
        axs_BtoA[0, i].axis('off')

        # Plot synthetic output (A)
        axs_BtoA[1, i].imshow(fake_A_np[0, d_mid, :, :], cmap='gray')
        axs_BtoA[1, i].set_title(f"Generated {domain_A}")
        axs_BtoA[1, i].axis('off')

        # Plot reconstructed input (B)
        axs_BtoA[2, i].imshow(rec_B_np[0, d_mid, :, :], cmap='gray')
        axs_BtoA[2, i].set_title(f"Reconstructed {domain_B}")
        axs_BtoA[2, i].axis('off')

        # Plot error heatmap
        im = axs_BtoA[3, i].imshow(diff_B[0, d_mid, :, :], cmap='hot', vmin=0, vmax=diff_B.max())
        axs_BtoA[3, i].set_title("Abs Difference")
        axs_BtoA[3, i].axis('off')
        # Optional: Add colorbar
        # fig_BtoA.colorbar(im, ax=axs_BtoA[3, i])

    fig_BtoA.suptitle(f"{mapping_type.upper()}: {domain_B} -> {domain_A} -> {domain_B} Translation ({epoch_label})", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_dir:
        filename_BtoA = os.path.join(save_dir, f'{mapping_type}_BtoA_samples_{epoch_label}.png')
        plt.savefig(filename_BtoA)
        print(f"  Saved BtoA visualization: {filename_BtoA}")
    else:
        plt.show()
    plt.close(fig_BtoA)

    # Restore model and dataset states
    g_model_AtoB.train()
    g_model_BtoA.train()
    if hasattr(dataset, 'augment'):
        dataset.augment = old_augment_state

# ------------------------------
# Plotting from Logs
# ------------------------------
def load_log_data(log_file):
    """Loads log data from a JSON file."""
    if not log_file or not os.path.exists(log_file):
        print(f"Error: Log file not found at {log_file}", file=sys.stderr)
        return None
    try:
        with open(log_file, 'r') as f:
            log_data = json.load(f)
        # Sort data by epoch and iteration just in case
        log_data.sort(key=lambda x: (x.get('epoch', 0), x.get('iteration', 0)))
        return log_data
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from log file: {log_file}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error loading log file {log_file}: {e}", file=sys.stderr)
        return None

def plot_losses(log_data, mapping_type, epoch_label, save_dir):
    """Plots training losses from log data."""
    print("Plotting training losses...")
    epochs = sorted(list(set(entry.get('epoch', 0) for entry in log_data)))
    # Check if any entry has 'iteration', otherwise assume epoch-level logging only
    has_iterations = any('iteration' in entry for entry in log_data)

    # Aggregate average losses per epoch for easier plotting
    epoch_avg_losses = {}
    epoch_counts = {}

    loss_keys = ['loss_G_total', 'loss_D_A', 'loss_D_B', 'loss_G_gan',
                 'loss_G_cycle', 'loss_G_identity', 'loss_G_intensity']

    for entry in log_data:
        epoch = entry.get('epoch')
        if epoch is None: continue

        # Initialize epoch entry if not present
        if epoch not in epoch_avg_losses:
            epoch_avg_losses[epoch] = {key: 0.0 for key in loss_keys}
            epoch_counts[epoch] = {key: 0 for key in loss_keys}

        # Accumulate losses present in the entry
        for key in loss_keys:
            if key in entry and entry[key] is not None: # Check for None values
                # Only accumulate if it's an iteration-level entry or if no iterations exist
                if has_iterations and 'iteration' in entry:
                    epoch_avg_losses[epoch][key] += entry[key]
                    epoch_counts[epoch][key] += 1
                elif not has_iterations and 'val_psnr_avg' not in entry: # Assume it's epoch avg if no iterations and not a val entry
                    epoch_avg_losses[epoch][key] = entry[key] # Directly assign epoch average
                    epoch_counts[epoch][key] = 1 # Mark as having data


    # Calculate averages for iteration-based logs
    if has_iterations:
        for epoch in epoch_avg_losses:
            for key in loss_keys:
                if epoch_counts[epoch][key] > 0:
                    epoch_avg_losses[epoch][key] /= epoch_counts[epoch][key]
                else:
                    # If no iteration data found for this key in this epoch, set to None
                    epoch_avg_losses[epoch][key] = None

    # Extract data for plotting, handling potential None values
    plot_epochs = sorted(epoch_avg_losses.keys())
    losses_to_plot = {}
    valid_epochs_for_key = {}
    for key in loss_keys:
        data = [epoch_avg_losses[e].get(key) for e in plot_epochs]
        # Filter out None values for plotting
        valid_epochs = [e for e, d in zip(plot_epochs, data) if d is not None]
        valid_data = [d for d in data if d is not None]
        if not valid_data: # Skip plotting if no valid data for this key
            print(f"  Skipping plot for loss key '{key}' - no data found.")
            continue
        losses_to_plot[key] = valid_data
        valid_epochs_for_key[key] = valid_epochs


    # --- Plotting ---
    # Check if there's anything to plot
    if not losses_to_plot:
        print("  No valid loss data found to plot.")
        return

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{mapping_type.upper()} Training Losses ({epoch_label})', fontsize=16)

    # Plot Total G Loss
    if 'loss_G_total' in losses_to_plot:
        axs[0, 0].plot(valid_epochs_for_key['loss_G_total'], losses_to_plot['loss_G_total'], label='Total G Loss', marker='.')
        axs[0, 0].set_title('Generator Loss')
        axs[0, 0].set_xlabel('Epoch')
        axs[0, 0].set_ylabel('Loss')
        axs[0, 0].legend()
        axs[0, 0].grid(True)
    else:
         axs[0, 0].set_title('Generator Loss (No Data)')
         axs[0, 0].axis('off')


    # Plot D Losses
    plot_d_loss = False
    if 'loss_D_A' in losses_to_plot:
        axs[0, 1].plot(valid_epochs_for_key['loss_D_A'], losses_to_plot['loss_D_A'], label='Discriminator A Loss', marker='.')
        plot_d_loss = True
    if 'loss_D_B' in losses_to_plot:
        axs[0, 1].plot(valid_epochs_for_key['loss_D_B'], losses_to_plot['loss_D_B'], label='Discriminator B Loss', marker='.')
        plot_d_loss = True

    if plot_d_loss:
        axs[0, 1].set_title('Discriminator Losses')
        axs[0, 1].set_xlabel('Epoch')
        axs[0, 1].set_ylabel('Loss')
        axs[0, 1].legend()
        axs[0, 1].grid(True)
    else:
         axs[0, 1].set_title('Discriminator Losses (No Data)')
         axs[0, 1].axis('off')


    # Plot G Components
    plot_g_comp = False
    if 'loss_G_gan' in losses_to_plot:
        axs[1, 0].plot(valid_epochs_for_key['loss_G_gan'], losses_to_plot['loss_G_gan'], label='GAN Loss', marker='.')
        plot_g_comp = True
    if 'loss_G_cycle' in losses_to_plot:
        axs[1, 0].plot(valid_epochs_for_key['loss_G_cycle'], losses_to_plot['loss_G_cycle'], label='Cycle Loss', marker='.')
        plot_g_comp = True
    if 'loss_G_identity' in losses_to_plot:
        axs[1, 0].plot(valid_epochs_for_key['loss_G_identity'], losses_to_plot['loss_G_identity'], label='Identity Loss', marker='.')
        plot_g_comp = True
    if 'loss_G_intensity' in losses_to_plot:
        axs[1, 0].plot(valid_epochs_for_key['loss_G_intensity'], losses_to_plot['loss_G_intensity'], label='Intensity Loss', marker='.')
        plot_g_comp = True

    if plot_g_comp:
        axs[1, 0].set_title('Generator Loss Components')
        axs[1, 0].set_xlabel('Epoch')
        axs[1, 0].set_ylabel('Loss')
        axs[1, 0].legend()
        axs[1, 0].grid(True)
    else:
        axs[1, 0].set_title('Generator Loss Components (No Data)')
        axs[1, 0].axis('off')


    # Empty subplot for potential future use
    axs[1, 1].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f'{mapping_type}_training_losses_{epoch_label}.png')
        plt.savefig(filename)
        print(f"  Saved loss plot: {filename}")
    else:
        plt.show()
    plt.close(fig)

def plot_metrics(log_data, mapping_type, epoch_label, save_dir):
    """Plots validation metrics from log data."""
    print("Plotting validation metrics...")
    # Filter entries that contain validation metrics (use a reliable key)
    val_entries = [entry for entry in log_data if 'val_psnr_avg' in entry and entry['val_psnr_avg'] is not None]
    if not val_entries:
        print("  No validation metrics found in log file.")
        return

    # Sort by epoch to ensure correct plotting order
    val_entries.sort(key=lambda x: x.get('epoch', 0))
    epochs = [entry.get('epoch', 0) for entry in val_entries] # Get epoch for each valid entry

    metric_keys = ['val_psnr_avg', 'val_ssim_avg', 'val_loss_cycle', 'val_loss_id']
    metrics_to_plot = {}
    valid_epochs_for_key = {}

    for key in metric_keys:
        data = [entry.get(key) for entry in val_entries]
        # Filter out None values
        valid_indices = [i for i, d in enumerate(data) if d is not None]
        if not valid_indices:
             print(f"  Skipping plot for metric key '{key}' - no data found.")
             continue

        valid_epochs = [epochs[i] for i in valid_indices]
        valid_data = [data[i] for i in valid_indices]
        metrics_to_plot[key] = valid_data
        valid_epochs_for_key[key] = valid_epochs


    # --- Plotting ---
    if not metrics_to_plot:
        print("  No valid metric data found to plot.")
        return

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{mapping_type.upper()} Validation Metrics ({epoch_label})', fontsize=16)

    # PSNR
    if 'val_psnr_avg' in metrics_to_plot:
        axs[0, 0].plot(valid_epochs_for_key['val_psnr_avg'], metrics_to_plot['val_psnr_avg'], label='Avg PSNR', marker='.')
        axs[0, 0].set_title('Validation Peak Signal-to-Noise Ratio')
        axs[0, 0].set_xlabel('Epoch')
        axs[0, 0].set_ylabel('PSNR (dB)')
        axs[0, 0].legend()
        axs[0, 0].grid(True)
    else:
         axs[0, 0].set_title('Validation PSNR (No Data)')
         axs[0, 0].axis('off')


    # SSIM
    if 'val_ssim_avg' in metrics_to_plot:
        axs[0, 1].plot(valid_epochs_for_key['val_ssim_avg'], metrics_to_plot['val_ssim_avg'], label='Avg SSIM', marker='.')
        axs[0, 1].set_title('Validation Structural Similarity Index')
        axs[0, 1].set_xlabel('Epoch')
        axs[0, 1].set_ylabel('SSIM')
        axs[0, 1].legend()
        axs[0, 1].grid(True)
    else:
        axs[0, 1].set_title('Validation SSIM (No Data)')
        axs[0, 1].axis('off')


    # Cycle Loss
    if 'val_loss_cycle' in metrics_to_plot:
        axs[1, 0].plot(valid_epochs_for_key['val_loss_cycle'], metrics_to_plot['val_loss_cycle'], label='Cycle Loss', marker='.')
        axs[1, 0].set_title('Validation Cycle Consistency Loss')
        axs[1, 0].set_xlabel('Epoch')
        axs[1, 0].set_ylabel('L1 Loss')
        axs[1, 0].legend()
        axs[1, 0].grid(True)
    else:
         axs[1, 0].set_title('Validation Cycle Loss (No Data)')
         axs[1, 0].axis('off')


    # Identity Loss
    if 'val_loss_id' in metrics_to_plot:
        axs[1, 1].plot(valid_epochs_for_key['val_loss_id'], metrics_to_plot['val_loss_id'], label='Identity Loss', marker='.')
        axs[1, 1].set_title('Validation Identity Loss')
        axs[1, 1].set_xlabel('Epoch')
        axs[1, 1].set_ylabel('L1 Loss')
        axs[1, 1].legend()
        axs[1, 1].grid(True)
    else:
         axs[1, 1].set_title('Validation Identity Loss (No Data)')
         axs[1, 1].axis('off')


    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f'{mapping_type}_validation_metrics_{epoch_label}.png')
        plt.savefig(filename)
        print(f"  Saved metrics plot: {filename}")
    else:
        plt.show()
    plt.close(fig)

# ------------------------------
# FID Calculation
# ------------------------------
def generate_images_for_fid(generator, dataset, num_images, output_dir, device, input_domain='A'):
    """
    Generates images using the generator and saves them as PNGs for FID calculation.
    Saves middle slice of the 3D volume.
    Handles data in [-1, 1] range, saves as [0, 255] uint8 PNG.

    Args:
        generator (nn.Module): The generator model (e.g., g_model_AtoB or g_model_BtoA).
        dataset (Dataset): Dataset providing input images.
        num_images (int): Maximum number of images to generate.
        output_dir (str): Directory to save generated PNGs.
        device (torch.device): Device to run inference on.
        input_domain (str): 'A' or 'B'. Specifies which image from the dataset pair
                           to use as input to the generator. 'A' means use img_A,
                           'B' means use img_B.
    """
    print(f"Generating {num_images} images for FID calculation (input: {input_domain}), saving to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    generator.eval()
    count = 0
    # Use a DataLoader for efficient iteration, batch_size=1 for individual saving
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0) # Use 0 workers for simplicity here

    with torch.no_grad():
        for i, batch_data in enumerate(tqdm(loader, desc=f"Generating FID images (Input {input_domain})", file=sys.stdout)):
            if count >= num_images:
                break

            # Select the correct input image based on input_domain
            img_A, img_B = batch_data
            if input_domain == 'A':
                real_input = img_A.to(device) # [-1, 1] range
            elif input_domain == 'B':
                real_input = img_B.to(device) # [-1, 1] range
            else:
                raise ValueError("input_domain must be 'A' or 'B'")

            # Pad, generate, crop
            real_input_padded = F.pad(real_input, (0, 0, 0, 0, 0, 1))
            fake_output = generator(real_input_padded)
            fake_output = fake_output[:, :, :-1, :, :] # Crop padding [-1, 1]

            # Convert to numpy [0, 1] -> [0, 255] uint8
            fake_output_np = (fake_output.cpu().numpy()[0] + 1) / 2.0 # [C, D, H, W], range [0, 1]

            # Ensure depth dimension is valid before slicing
            if fake_output_np.shape[1] <= 0:
                print(f"Warning: Generated image {i} has non-positive depth ({fake_output_np.shape[1]}). Skipping.", file=sys.stderr)
                continue

            d_mid = fake_output_np.shape[1] // 2
            slice_img = fake_output_np[0, d_mid, :, :] # Get middle slice [H, W]

            # Scale to 0-255 and convert to uint8
            slice_img_uint8 = (slice_img * 255).clip(0, 255).astype(np.uint8)

            # Save as PNG
            try:
                from PIL import Image # Use PIL for saving PNG
                pil_img = Image.fromarray(slice_img_uint8, mode='L') # 'L' for grayscale
                # Ensure unique filenames if num_images > len(dataset)
                save_path = os.path.join(output_dir, f'generated_{count:05d}.png')
                pil_img.save(save_path)
                count += 1
            except ImportError:
                 print("Error: PIL (Pillow) library not found. Cannot save PNG images for FID.", file=sys.stderr)
                 print("Please install Pillow: pip install Pillow")
                 generator.train() # Restore train mode before exiting
                 return None # Indicate failure
            except Exception as e:
                 print(f"Error saving image {count}: {e}", file=sys.stderr)


    generator.train() # Restore train mode
    print(f"Finished generating {count} images.")
    if count == 0:
        print("Warning: No images were generated for FID.", file=sys.stderr)
        return None # Indicate failure if no images generated
    return output_dir # Return path to generated images

def calculate_fid_scores(real_images_dir_a, real_images_dir_b, generated_images_dir_a, generated_images_dir_b, epoch_label, mapping_type, save_dir, batch_size=16):
    """Calculates FID scores between generated and real images."""
    if fid_score is None:
        print("Error: FID calculation skipped because pytorch-fid is not installed.", file=sys.stderr)
        return

    print("Calculating FID scores...")
    fid_results = {}

    # --- Helper function to check directory ---
    def check_fid_dir(name, path):
        if not path or not os.path.isdir(path):
            print(f"Error: {name} directory not found or invalid: {path}", file=sys.stderr)
            return False
        image_files = [fname for fname in os.listdir(path) if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))] # FID uses image files
        if not image_files:
             print(f"Warning: {name} directory '{path}' contains no standard image files (.png, .jpg, etc.). FID calculation might fail.", file=sys.stderr)
             # Allow continuing, FID tool might handle other types or error out itself
        elif len(image_files) < 2: # FID needs at least 2 images per directory
             print(f"Warning: {name} directory '{path}' contains only {len(image_files)} image(s). FID requires at least 2.", file=sys.stderr)
             # Allow continuing, FID tool will likely error
        return True

    # --- Check directories ---
    dirs_to_check = {
        "Real A": real_images_dir_a,
        "Real B": real_images_dir_b,
        "Generated A": generated_images_dir_a,
        "Generated B": generated_images_dir_b
    }
    all_dirs_valid = True
    for name, path in dirs_to_check.items():
        if not check_fid_dir(name, path):
            all_dirs_valid = False # Mark as invalid but check all dirs

    if not all_dirs_valid:
        print("FID calculation aborted due to missing or invalid directories.", file=sys.stderr)
        return

    # --- Calculate FID: Generated A vs Real A ---
    try:
        print(f"  Calculating FID: Gen A ({generated_images_dir_a}) vs Real A ({real_images_dir_a})")
        fid_genA_realA = fid_score.calculate_fid_given_paths(
            [real_images_dir_a, generated_images_dir_a],
            batch_size=batch_size,
            device=device,
            dims=2048 # Standard InceptionV3 feature dimension
        )
        fid_results['Generated_A_vs_Real_A'] = fid_genA_realA
        print(f"  FID (Generated A vs Real A): {fid_genA_realA:.3f}")
    except Exception as e:
        print(f"  Error calculating FID (Generated A vs Real A): {e}", file=sys.stderr)
        fid_results['Generated_A_vs_Real_A'] = f'Error: {e}'

    # --- Calculate FID: Generated A vs Real B (Cross-domain realism check) ---
    try:
        print(f"  Calculating FID: Gen A ({generated_images_dir_a}) vs Real B ({real_images_dir_b})")
        fid_genA_realB = fid_score.calculate_fid_given_paths(
            [real_images_dir_b, generated_images_dir_a],
            batch_size=batch_size, device=device, dims=2048
        )
        fid_results['Generated_A_vs_Real_B'] = fid_genA_realB
        print(f"  FID (Generated A vs Real B): {fid_genA_realB:.3f}")
    except Exception as e:
        print(f"  Error calculating FID (Generated A vs Real B): {e}", file=sys.stderr)
        fid_results['Generated_A_vs_Real_B'] = f'Error: {e}'

    # --- Calculate FID: Generated B vs Real B ---
    try:
        print(f"  Calculating FID: Gen B ({generated_images_dir_b}) vs Real B ({real_images_dir_b})")
        fid_genB_realB = fid_score.calculate_fid_given_paths(
            [real_images_dir_b, generated_images_dir_b],
            batch_size=batch_size, device=device, dims=2048
        )
        fid_results['Generated_B_vs_Real_B'] = fid_genB_realB
        print(f"  FID (Generated B vs Real B): {fid_genB_realB:.3f}")
    except Exception as e:
        print(f"  Error calculating FID (Generated B vs Real B): {e}", file=sys.stderr)
        fid_results['Generated_B_vs_Real_B'] = f'Error: {e}'

    # --- Calculate FID: Generated B vs Real A (Cross-domain realism check) ---
    try:
        print(f"  Calculating FID: Gen B ({generated_images_dir_b}) vs Real A ({real_images_dir_a})")
        fid_genB_realA = fid_score.calculate_fid_given_paths(
            [real_images_dir_a, generated_images_dir_b],
            batch_size=batch_size, device=device, dims=2048
        )
        fid_results['Generated_B_vs_Real_A'] = fid_genB_realA
        print(f"  FID (Generated B vs Real A): {fid_genB_realA:.3f}")
    except Exception as e:
        print(f"  Error calculating FID (Generated B vs Real A): {e}", file=sys.stderr)
        fid_results['Generated_B_vs_Real_A'] = f'Error: {e}'


    # Save results
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f'{mapping_type}_fid_scores_{epoch_label}.json')
        try:
            # Convert potential errors to strings for JSON serialization
            serializable_results = {k: (str(v) if isinstance(v, Exception) else v) for k, v in fid_results.items()}
            with open(filename, 'w') as f:
                json.dump(serializable_results, f, indent=4)
            print(f"  Saved FID scores: {filename}")
        except Exception as e:
            print(f"  Error saving FID scores: {e}", file=sys.stderr)

# ------------------------------
# Main Execution Logic
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate 3D CycleGAN Model")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the .pth checkpoint file from train_cycleGAN_v3.py")
    # Removed mapping_type from args, will read from checkpoint
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to evaluation data directory (e.g., validation split .npy files)")
    parser.add_argument("--output_dir", type=str, default="./evaluation_output",
                        help="Base directory to save evaluation results (plots, FID scores, etc.)")
    parser.add_argument("--tasks", nargs='+', default=['visualize', 'plot_logs'],
                        choices=['visualize', 'plot_logs', 'calculate_fid'],
                        help="List of evaluation tasks to perform.")
    parser.add_argument("--samples", type=int, default=5,
                        help="Number of random samples for visualization task.")
    parser.add_argument("--log_file", type=str, default=None,
                        help="Path to the training log JSON file (required for 'plot_logs'). "
                             "If None, attempts to find latest log in checkpoint dir parent.")
    # FID specific arguments
    parser.add_argument("--fid_real_a_dir", type=str, default=None,
                        help="Path to DIRECTORY containing REAL domain A images (PNGs) for FID.")
    parser.add_argument("--fid_real_b_dir", type=str, default=None,
                        help="Path to DIRECTORY containing REAL domain B images (PNGs) for FID.")
    parser.add_argument("--fid_num_images", type=int, default=500,
                        help="Number of images to generate for FID calculation.")
    parser.add_argument("--fid_batch_size", type=int, default=16,
                        help="Batch size for FID calculation.")
    parser.add_argument("--fid_regenerate", action='store_true',
                        help="Force regeneration of images for FID even if directories exist.")


    args = parser.parse_args()

    # --- Load Checkpoint and Extract Info ---
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}", file=sys.stderr)
        sys.exit(1)

    try:
        # Load only necessary info first without initializing models
        checkpoint = torch.load(args.checkpoint, map_location='cpu') # Load to CPU first
        start_epoch = checkpoint.get('epoch', -1) # Epoch number completed
        epoch_label = f"epoch{start_epoch+1:03d}" # Label for outputs
        mapping_type = checkpoint.get('mapping_type')
        n_resnet = checkpoint.get('n_resnet') # Crucial: Get n_resnet from checkpoint

        if not mapping_type or n_resnet is None:
             raise KeyError("Checkpoint missing 'mapping_type' or 'n_resnet'.")

        print(f"> Loaded checkpoint: {os.path.basename(args.checkpoint)}")
        print(f"  - Completed Epoch: {start_epoch}")
        print(f"  - Mapping Type: {mapping_type}")
        print(f"  - ResNet Blocks: {n_resnet}")

    except Exception as e:
        print(f"Error loading checkpoint info from {args.checkpoint}: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Prepare Output Directory ---
    # Structure: output_dir / mapping_type / epoch_label / task_outputs
    eval_output_dir = os.path.join(args.output_dir, mapping_type, epoch_label)
    os.makedirs(eval_output_dir, exist_ok=True)
    print(f"> Saving evaluation results to: {eval_output_dir}")

    # --- Initialize Models ---
    try:
        g_model_AtoB, g_model_BtoA = init_models(mapping_type, n_resnet, device)
        # Load state dicts from the loaded checkpoint
        g_model_AtoB.load_state_dict(checkpoint['g_model_AtoB_state'])
        g_model_BtoA.load_state_dict(checkpoint['g_model_BtoA_state'])
        print("> Generator models initialized and loaded from checkpoint.")
    except Exception as e:
        print(f"Error initializing or loading models: {e}", file=sys.stderr)
        sys.exit(1)
    # No need to load optimizers/schedulers/discriminators for evaluation

    # --- Prepare Dataset ---
    try:
        # Use the data_dir provided for evaluation tasks (e.g., visualization, FID generation)
        # Ensure augment=False for consistent evaluation/visualization
        eval_dataset = PairedPreprocessedDataset(args.data_dir, mapping_type, augment=False)
        if len(eval_dataset) == 0:
            print(f"Warning: No data found in evaluation data directory: {args.data_dir}", file=sys.stderr)
            # Allow continuing for tasks like plot_logs, but others might fail.
        else:
             print(f"> Evaluation dataset loaded: {len(eval_dataset)} samples from {args.data_dir}")
    except Exception as e:
        print(f"Error creating evaluation dataset from {args.data_dir}: {e}", file=sys.stderr)
        # Exit if dataset is needed for requested tasks
        if 'visualize' in args.tasks or 'calculate_fid' in args.tasks:
             sys.exit(1)

    # ==========================
    # --- Execute Tasks ---
    # ==========================

    # --- Task: Visualize Sample Translations ---
    if 'visualize' in args.tasks:
        if len(eval_dataset) > 0:
            viz_save_dir = os.path.join(eval_output_dir, 'visualizations')
            visualize_sample_translations(
                epoch_label=epoch_label,
                g_model_AtoB=g_model_AtoB,
                g_model_BtoA=g_model_BtoA,
                dataset=eval_dataset,
                mapping_type=mapping_type,
                device=device,
                n_samples=args.samples,
                save_dir=viz_save_dir
            )
        else:
            print("Skipping visualization task: Evaluation dataset is empty.")

    # --- Task: Plot Logs ---
    if 'plot_logs' in args.tasks:
        log_file_to_use = args.log_file
        # Attempt to find log file automatically if not provided
        if not log_file_to_use:
             # Look in the parent directory of the checkpoint's directory
             checkpoint_parent_dir = os.path.dirname(os.path.dirname(args.checkpoint)) # Go up two levels (e.g., from ./output/t2_flair/checkpoints -> ./output/t2_flair)
             log_dir_guess = os.path.join(checkpoint_parent_dir, 'logs')
             if os.path.isdir(log_dir_guess):
                 # Find the latest log file for this mapping type
                 potential_logs = sorted(glob.glob(os.path.join(log_dir_guess, f'{mapping_type}_log_*.json')))
                 if potential_logs:
                     log_file_to_use = potential_logs[-1]
                     print(f"  Automatically found log file: {log_file_to_use}")
                 else:
                      print(f"  Could not automatically find log file in {log_dir_guess}")
             else:
                 print(f"  Could not automatically find log directory at {log_dir_guess}")


        if log_file_to_use:
            log_data = load_log_data(log_file_to_use)
            if log_data:
                plot_save_dir = os.path.join(eval_output_dir, 'plots')
                plot_losses(log_data, mapping_type, epoch_label, plot_save_dir)
                plot_metrics(log_data, mapping_type, epoch_label, plot_save_dir)
            else:
                 print("Skipping plot_logs task: Failed to load log data.")
        else:
            print("Skipping plot_logs task: No log file specified or found.")

    # --- Task: Calculate FID ---
    if 'calculate_fid' in args.tasks:
        if fid_score is None:
             print("Skipping calculate_fid task: pytorch-fid package not installed.", file=sys.stderr)
        elif not args.fid_real_a_dir or not args.fid_real_b_dir:
             print("Skipping calculate_fid task: Paths to real image directories (--fid_real_a_dir, --fid_real_b_dir) are required.", file=sys.stderr)
        elif len(eval_dataset) == 0:
             print("Skipping calculate_fid task: Evaluation dataset is empty, cannot generate images.", file=sys.stderr)
        else:
             # Define directories for generated images
             fid_gen_base_dir = os.path.join(eval_output_dir, 'fid_generated_images')
             gen_a_dir = os.path.join(fid_gen_base_dir, 'generated_A') # Images generated by BtoA generator
             gen_b_dir = os.path.join(fid_gen_base_dir, 'generated_B') # Images generated by AtoB generator

             # Generate images only if needed
             gen_a_needed = args.fid_regenerate or not os.path.isdir(gen_a_dir) or not os.listdir(gen_a_dir)
             gen_b_needed = args.fid_regenerate or not os.path.isdir(gen_b_dir) or not os.listdir(gen_b_dir)
             gen_a_success = True
             gen_b_success = True

             if gen_a_needed or gen_b_needed:
                 print("\n--- Generating images for FID ---")
                 if gen_a_needed:
                     print("Generating Domain A images (from B)...")
                     gen_a_path = generate_images_for_fid(
                         generator=g_model_BtoA, # Generate A from B
                         dataset=eval_dataset,   # Needs input B
                         num_images=args.fid_num_images,
                         output_dir=gen_a_dir,
                         device=device,
                         input_domain='B' # Specify input domain
                     )
                     gen_a_success = gen_a_path is not None
                 else:
                      print(f"Skipping generation for Domain A, directory exists: {gen_a_dir}")

                 if gen_b_needed:
                     print("Generating Domain B images (from A)...")
                     gen_b_path = generate_images_for_fid(
                         generator=g_model_AtoB, # Generate B from A
                         dataset=eval_dataset,   # Needs input A
                         num_images=args.fid_num_images,
                         output_dir=gen_b_dir,
                         device=device,
                         input_domain='A' # Specify input domain
                     )
                     gen_b_success = gen_b_path is not None
                 else:
                      print(f"Skipping generation for Domain B, directory exists: {gen_b_dir}")

                 print("---------------------------------\n")
             else:
                  print("Skipping FID image generation, directories already exist.")


             # Calculate FID only if generation was successful (or skipped) and dirs exist
             if gen_a_success and gen_b_success:
                  fid_save_dir = os.path.join(eval_output_dir, 'fid_scores')
                  calculate_fid_scores(
                      real_images_dir_a=args.fid_real_a_dir,
                      real_images_dir_b=args.fid_real_b_dir,
                      generated_images_dir_a=gen_a_dir,
                      generated_images_dir_b=gen_b_dir,
                      epoch_label=epoch_label,
                      mapping_type=mapping_type,
                      save_dir=fid_save_dir,
                      batch_size=args.fid_batch_size
                  )
             else:
                  print("Skipping FID calculation due to errors during image generation.")


    print(f"\nEvaluation complete. Results saved in: {eval_output_dir}")

