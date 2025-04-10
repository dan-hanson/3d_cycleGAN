# utils.py
"""
Utility functions and class definitions shared across training and evaluation scripts.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import torchio as tio
from glob import glob
import json
import datetime
import sys
import random # Needed for update_image_pool

# ------------------------------
# Model Definitions (Copied from train_cycleGAN_v3.py)
# ------------------------------
class Discriminator3D(nn.Module):
    def __init__(self, in_channels):
        super(Discriminator3D, self).__init__()
        self.model = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1), nn.InstanceNorm3d(64, affine=True), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1), nn.InstanceNorm3d(128, affine=True), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1), nn.InstanceNorm3d(256, affine=True), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(256, 256, kernel_size=4, stride=1, padding=1), nn.InstanceNorm3d(256, affine=True), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(256, 1, kernel_size=4, stride=1, padding=1) # Output is a patch map
        )
    def forward(self, x):
        return self.model(x)

class ResNetBlock3D(nn.Module):
    def __init__(self, channels):
        super(ResNetBlock3D, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False), nn.InstanceNorm3d(channels, affine=True), nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False), nn.InstanceNorm3d(channels, affine=True)
        )
    def forward(self, x):
        return x + self.conv_block(x) # Residual connection

class Generator3D(nn.Module):
    def __init__(self, in_channels, out_channels, n_resnet=9):
        super(Generator3D, self).__init__()
        ngf = 32 # Number of generator filters in the first conv layer
        model = [
            nn.Conv3d(in_channels, ngf, kernel_size=7, padding=3, bias=False), nn.InstanceNorm3d(ngf, affine=True), nn.ReLU(inplace=True)
        ]
        # Downsampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [
                nn.Conv3d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.InstanceNorm3d(ngf * mult * 2, affine=True), nn.ReLU(inplace=True)
            ]
        # ResNet blocks
        mult = 2**n_downsampling
        for i in range(n_resnet):
            model += [ResNetBlock3D(ngf * mult)]
        # Upsampling
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [
                nn.ConvTranspose3d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                nn.InstanceNorm3d(int(ngf * mult / 2), affine=True), nn.ReLU(inplace=True)
            ]
        # Final layer
        model += [
            nn.Conv3d(ngf, out_channels, kernel_size=7, padding=3),
            nn.Tanh() # Output range [-1, 1]
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

# ------------------------------
# Dataset Class (Copied from train_cycleGAN_v3.py)
# ------------------------------
class PairedPreprocessedDataset(Dataset):
    def __init__(self, dir_path, mapping_type, augment=False, torchio_transform=None):
        """
        Loads preprocessed .npy files, selects channels, rescales, and applies transforms.

        Args:
            dir_path (str): Directory with .npy files (shape H,W,D,C).
                              **ASSUMES PREPROCESSING SAVED AS [FLAIR(0), T1CE(1), T2(2), T1(3)] channels.**
                              **ASSUMES PREPROCESSING NORMALIZED TO [0, 1].**
            mapping_type (str): Either "t2_flair" or "t1_contrast".
            augment (bool): If True, apply transformations defined in torchio_transform.
            torchio_transform (torchio.Compose, optional): Torchio transform pipeline.
        """
        # Use glob.escape for robustness with paths containing special characters
        search_path = os.path.join(dir_path, '*.npy')
        self.files = sorted(glob(search_path)) # Call glob function directly
        if not self.files:
             print(f"Warning: No .npy files found matching pattern: {search_path}")
        self.mapping_type = mapping_type
        self.augment = augment
        self.transform = torchio_transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        try:
            # Load data assuming (H, W, D, C) channel order from preprocessing
            # Assumes data range is [0, 1] after loading
            volume = np.load(file_path).astype(np.float32)
            if volume.ndim == 5 and volume.shape[0] == 1:
                volume = volume[0] # Remove leading singleton dimension if present

            # Transpose from (H,W,D,C) to (C,D,H,W) - PyTorch/Torchio convention
            # Resulting Channel Order: [FLAIR(0), T1CE(1), T2(2), T1(3)]
            volume = np.transpose(volume, (3, 2, 0, 1))

            # --- Step 1: Fix Channel Indices ---
            if self.mapping_type == "t2_flair":
                # A = T2 (Index 2), B = FLAIR (Index 0)
                img_A_raw = volume[2:3, ...]
                img_B_raw = volume[0:1, ...]
            elif self.mapping_type == "t1_contrast":
                # A = T1 (Index 3), B = T1CE (Index 1)
                img_A_raw = volume[3:4, ...]
                img_B_raw = volume[1:2, ...]
            else:
                raise ValueError(f"Unknown mapping_type: {self.mapping_type}")

            # --- Step 2: Add Normalization Rescaling ---
            # Rescale from [0, 1] (output of preprocessing) to [-1, 1] (input for CycleGAN Tanh)
            img_A = img_A_raw * 2.0 - 1.0
            img_B = img_B_raw * 2.0 - 1.0

            # --- Combine, Augment (Optional), Split Back ---
            # Combine modalities for consistent spatial augmentation
            # Use np.stack which handles the new axis, then squeeze if needed
            combined_modalities = np.stack((img_A.squeeze(0), img_B.squeeze(0)), axis=0) # Shape (2, D, H, W)
            combined_tensor = torch.from_numpy(combined_modalities)

            # Apply Torchio augmentations if enabled
            if self.augment and self.transform:
                subject = tio.Subject(modalities=tio.ScalarImage(tensor=combined_tensor))
                augmented_subject = self.transform(subject)
                augmented_tensor = augmented_subject['modalities'].data
            else:
                augmented_tensor = combined_tensor # No augmentation

            # Split back into individual modalities [1, D, H, W]
            final_img_A = augmented_tensor[0:1, ...]
            final_img_B = augmented_tensor[1:2, ...]

            return final_img_A, final_img_B

        except FileNotFoundError:
            print(f"Error: File not found {file_path}", file=sys.stderr)
            raise
        except Exception as e:
            print(f"Error processing file {file_path}: {e}", file=sys.stderr)
            # Return None or dummy data might be better than raising here
            # depending on how DataLoader handles errors. Raising is safer.
            raise

# ------------------------------
# Checkpoint Loading (Copied from train_cycleGAN_v3.py)
# ------------------------------
def load_checkpoint(checkpoint_path, g_model_AtoB, g_model_BtoA, d_model_A, d_model_B,
                    optimizer_G, optimizer_D_A, optimizer_D_B,
                    scheduler_G, scheduler_D_A, scheduler_D_B, device='cpu'):
    """
    Loads training state from a checkpoint. Returns start_epoch.
    Modified to accept device explicitly.
    """
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at {checkpoint_path}", file=sys.stderr)
        return 0 # Return start epoch 0 if checkpoint not found
    try:
        # Ensure checkpoint is loaded to the correct device
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Load model states
        g_model_AtoB.load_state_dict(checkpoint['g_model_AtoB_state'])
        g_model_BtoA.load_state_dict(checkpoint['g_model_BtoA_state'])
        d_model_A.load_state_dict(checkpoint['d_model_A_state'])
        d_model_B.load_state_dict(checkpoint['d_model_B_state'])

        # Load optimizer states
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state'])
        optimizer_D_A.load_state_dict(checkpoint['optimizer_D_A_state'])
        optimizer_D_B.load_state_dict(checkpoint['optimizer_D_B_state'])

        # Load scheduler states carefully
        if 'scheduler_G_state' in checkpoint: scheduler_G.load_state_dict(checkpoint['scheduler_G_state'])
        else: print("Warning: Scheduler G state not found in checkpoint.")
        if 'scheduler_D_A_state' in checkpoint: scheduler_D_A.load_state_dict(checkpoint['scheduler_D_A_state'])
        else: print("Warning: Scheduler D_A state not found in checkpoint.")
        if 'scheduler_D_B_state' in checkpoint: scheduler_D_B.load_state_dict(checkpoint['scheduler_D_B_state'])
        else: print("Warning: Scheduler D_B state not found in checkpoint.")

        start_epoch = checkpoint.get('epoch', -1) + 1
        loaded_mapping = checkpoint.get('mapping_type', 'N/A')
        loaded_n_resnet = checkpoint.get('n_resnet', 'N/A') # Load n_resnet if saved
        print(f'>Loaded checkpoint from epoch {start_epoch - 1} (Mapping: {loaded_mapping}, ResNet blocks: {loaded_n_resnet})')

        # --- Crucial: Move optimizer states to the correct device ---
        # This is necessary if the checkpoint was saved on GPU and loaded on CPU or vice-versa
        for state in optimizer_G.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        for state in optimizer_D_A.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        for state in optimizer_D_B.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        # --- End Optimizer State Device Correction ---


        return start_epoch
    except Exception as e:
        print(f"Error loading checkpoint {checkpoint_path}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc() # Print detailed traceback
        print("Starting training from epoch 0.")
        return 0

# ------------------------------
# Logger Class (Copied from train_cycleGAN_v3.py)
# ------------------------------
class Logger:
    def __init__(self, log_dir, mapping_type):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_path = os.path.join(log_dir, f'{mapping_type}_log_{timestamp}.json')
        self.log_data = []
        print(f"Logging to: {self.log_path}")

    def log(self, epoch, i, metrics):
        """Log metrics at a given training step/epoch."""
        serializable_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                serializable_metrics[k] = v.item()
            elif isinstance(v, np.ndarray):
                 serializable_metrics[k] = v.tolist() # Or item() if scalar
            elif isinstance(v, (int, float, str, bool, list, dict, type(None))):
                 serializable_metrics[k] = v
            else:
                 serializable_metrics[k] = str(v) # Fallback

        record = {'epoch': epoch, 'iteration': i} # Use 'iteration' for step, or 'n_steps' for epoch end
        record.update(serializable_metrics)
        self.log_data.append(record)

    def save(self):
        """Write log data to disk as JSON."""
        try:
            with open(self.log_path, 'w') as f:
                json.dump(self.log_data, f, indent=2)
        except Exception as e:
            print(f"Error saving log file {self.log_path}: {e}", file=sys.stderr)

    def append_from_file(self, previous_log):
        """Load previous training log to continue appending."""
        if previous_log and os.path.exists(previous_log):
            try:
                with open(previous_log, 'r') as f:
                    self.log_data = json.load(f)
                print(f"Appended previous log data from: {previous_log}")
            except Exception as e:
                 print(f"Error loading previous log file {previous_log}: {e}", file=sys.stderr)
        else:
             print(f"No valid previous log file found at {previous_log}. Starting new log.")

# ------------------------------
# Image Pool (Copied from train_cycleGAN_v3.py)
# ------------------------------
def update_image_pool(pool, images, max_size=50):
    """Keeps a buffer of generated images for stabilizing discriminator training."""
    selected = []
    for image in images:
        image = image.detach() # Detach from computation graph
        if image.dim() == 4:
             # If input is [C, D, H, W], add batch dim -> [1, C, D, H, W]
             image = image.unsqueeze(0)
        elif image.dim() != 5:
             print(f"Warning: Unexpected image dimension in pool: {image.dim()}", file=sys.stderr)
             continue # Skip this image

        if len(pool) < max_size:
            pool.append(image)
            selected.append(image)
        elif random.random() < 0.5:
            # Use current image and don't update pool
            selected.append(image)
        else:
            # Use image from pool and replace it with current image
            ix = random.randint(0, max_size - 1)
            selected.append(pool[ix].clone())
            pool[ix] = image
    if not selected: # Handle case where input 'images' was empty or invalid
        return torch.empty(0) # Return an empty tensor
    return torch.cat(selected, dim=0)


