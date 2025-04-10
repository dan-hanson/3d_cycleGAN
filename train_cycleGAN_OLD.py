"""
3D CycleGAN for Synthetic MRI Data Augmentation (Paired Mapping Version)

Objective:
Generate synthetic missing modality data from 3D MRI volumes.
For preprocessed data (e.g., cropped to 128x128x128 or your current size) with 4 channels,
we implement two specific mappings:
    - "t2_flair": Generate FLAIR from T2.
    - "t1_contrast": Generate contrast (T1CE) from T1.
Domain A: T1 or T2 images.
Domain B: Either FLAIR or T1CE images, depending on mapping_type.
The CycleGAN learns bidirectional translation between these two domains.

Usage:
    python train_cycleGAN_v2.py --mapping_type t2_flair --epochs 50
    # To resume from a checkpoint:
    python train_cycleGAN_v2.py --mapping_type t2_flair --epochs 50 --resume checkpoints/checkpoint_t2_flair_epoch000.pth

Environment Setup:
    conda activate cyclegan3d_1
"""

import os
import datetime
import random
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

# Import our augmentation functions; note that we now pass visualize=False during training.
from augmentation import random_rotation_3d, display_random_slices

# ------------------------------
# Check Environment and GPU
print("Checking environment...")
print("CUDA available:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    x = torch.randn(1).to(device)
    print("GPU is working. Device:", device)
except Exception as e:
    print("GPU test failed:", e)

# ------------------------------
# Utility: Save training log to disk
class Logger:
    def __init__(self, log_dir, mapping_type):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_path = os.path.join(log_dir, f'{mapping_type}_log_{timestamp}.json')
        self.log_data = []

    def log(self, epoch, i, metrics):
        """Log metrics at a given training step."""
        record = {'epoch': epoch, 'iteration': i}
        record.update(metrics)
        self.log_data.append(record)

    def save(self):
        """Write log data to disk as JSON."""
        with open(self.log_path, 'w') as f:
            json.dump(self.log_data, f, indent=2)

    def append_from_file(self, previous_log):
        """Load previous training log to continue appending."""
        if os.path.exists(previous_log):
            with open(previous_log, 'r') as f:
                self.log_data = json.load(f)

# ------------------------------
# Utility: Match depth of two tensors
def match_depth(tensor_in, tensor_ref):
    """
    Adjusts the depth dimension of tensor_in to match tensor_ref.
    Both tensors are expected to have shape (B, C, D, H, W).
    If tensor_in has extra depth, simply crop it.
    """
    d_in = tensor_in.shape[2]
    d_ref = tensor_ref.shape[2]
    if d_in > d_ref:
        return tensor_in[:, :, :d_ref, :, :]
    elif d_in < d_ref:
        pad = d_ref - d_in
        return F.pad(tensor_in, (0, 0, 0, 0, 0, pad))
    return tensor_in

# ------------------------------
# Dataset for Specific Modality Mapping
class PairedPreprocessedDataset(Dataset):
    def __init__(self, dir_path, mapping_type, augment=False):
        """
        Args:
            dir_path (str): Directory with .npy files (each volume is shape (H,W,D,C)).
            mapping_type (str): Either "t2_flair" or "t1_contrast".
                For "t2_flair": Domain A = T2 (channel index 1), Domain B = FLAIR (channel index 2).
                For "t1_contrast": Domain A = T1 (channel index 0), Domain B = T1CE (channel index 3).
            augment (bool): If True, apply random 3D rotations.
        """
        self.files = sorted(glob(os.path.join(dir_path, '*.npy')))
        self.mapping_type = mapping_type
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        volume = np.load(file_path).astype(np.float32)  # Expected shape: (H,W,D,C) or (1,H,W,D,C)
        if volume.ndim == 5 and volume.shape[0] == 1:
            volume = volume[0]
        # Transpose from (H,W,D,C) to (C,D,H,W)
        volume = np.transpose(volume, (3, 2, 0, 1))
        # Extract modalities according to mapping_type
        if self.mapping_type == "t2_flair":
            img_A = volume[1:2, ...]  # T2
            img_B = volume[2:3, ...]  # FLAIR
        elif self.mapping_type == "t1_contrast":
            img_A = volume[0:1, ...]  # T1
            img_B = volume[3:4, ...]  # T1CE
        else:
            raise ValueError("Unknown mapping_type: {}".format(self.mapping_type))
        
        if self.augment:
            # Pass visualize=False so training isn't interrupted by plots
            img_A = random_rotation_3d(img_A, mean=0, std=10, pad_margin=20, visualize=False)
            img_B = random_rotation_3d(img_B, mean=0, std=10, pad_margin=20, visualize=False)
        
        return torch.from_numpy(img_A), torch.from_numpy(img_B)

# ------------------------------
# 3D Discriminator: PatchGAN (Using 32 initial filters)
class Discriminator3D(nn.Module):
    def __init__(self, in_channels):
        super(Discriminator3D, self).__init__()
        self.model = nn.Sequential(

            # Initial convolutional layer
            nn.Conv3d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm3d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm3d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm3d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(256, 256, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm3d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            ## Final output layer
            nn.Conv3d(256, 1, kernel_size=4, stride=1, padding=1)
        )
    
    def forward(self, x):
        return self.model(x)

# ------------------------------
# 3D ResNet Block for Generator
class ResNetBlock3D(nn.Module):
    def __init__(self, channels):
        super(ResNetBlock3D, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(channels, affine=True)
        )
    
    def forward(self, x):
        return x + self.conv_block(x)

# ------------------------------
# 3D Generator: Encoder-ResNet-Decoder
class Generator3D(nn.Module):
    def __init__(self, in_channels, out_channels, n_resnet=9):
        super(Generator3D, self).__init__()
        init_channels = 32
        model = [
            nn.Conv3d(in_channels, init_channels, kernel_size=7, padding=3, bias=False),
            nn.InstanceNorm3d(init_channels, affine=True),
            nn.ReLU(inplace=True)
        ]
        curr_dim = init_channels
        # Downsampling layers
        for i in range(2):
            out_dim = curr_dim * 2
            model += [
                nn.Conv3d(curr_dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False),
                nn.InstanceNorm3d(out_dim, affine=True),
                nn.ReLU(inplace=True)
            ]
            curr_dim = out_dim
        # ResNet blocks
        for i in range(n_resnet):
            model += [ResNetBlock3D(curr_dim)]
        # Upsampling layers
        for i in range(2):
            out_dim = curr_dim // 2
            model += [
                nn.ConvTranspose3d(curr_dim, out_dim, kernel_size=3, stride=2,
                                   padding=1, output_padding=1, bias=False),
                nn.InstanceNorm3d(out_dim, affine=True),
                nn.ReLU(inplace=True)
            ]
            curr_dim = out_dim
        # Final layer
        model += [
            nn.Conv3d(curr_dim, out_channels, kernel_size=7, padding=3),
            nn.Tanh()
        ]
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)

# ------------------------------
# Utility: Update image pool (for discriminator training)
def update_image_pool(pool, images, max_size=50):
    selected = []
    for image in images:
        if image.dim() == 4:  # add batch dimension if missing
            image = image.unsqueeze(0)
        if len(pool) < max_size:
            pool.append(image)
            selected.append(image)
        elif random.random() < 0.5:
            selected.append(image)
        else:
            ix = random.randint(0, len(pool) - 1)
            selected.append(pool[ix])
            pool[ix] = image
    return torch.cat(selected, dim=0)

# ------------------------------
# Utility: Save models with mapping type
def save_models(step, g_model_AtoB, g_model_BtoA, mapping_type, save_dir='./models'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename1 = os.path.join(save_dir, f'g_model_AtoB_{mapping_type}_{step+1:06d}.pth')
    filename2 = os.path.join(save_dir, f'g_model_BtoA_{mapping_type}_{step+1:06d}.pth')
    torch.save(g_model_AtoB.state_dict(), filename1)
    torch.save(g_model_BtoA.state_dict(), filename2)
    print('>Saved models:', filename1, filename2)

# ------------------------------
# Utility: Checkpoint saving and loading
def save_checkpoint(epoch, g_model_AtoB, g_model_BtoA, d_model_A, d_model_B,
                    optimizer_G, optimizer_D_A, optimizer_D_B, mapping_type, checkpoint_dir='./checkpoints'):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint = {
        'epoch': epoch,
        'g_model_AtoB_state': g_model_AtoB.state_dict(),
        'g_model_BtoA_state': g_model_BtoA.state_dict(),
        'd_model_A_state': d_model_A.state_dict(),
        'd_model_B_state': d_model_B.state_dict(),
        'optimizer_G_state': optimizer_G.state_dict(),
        'optimizer_D_A_state': optimizer_D_A.state_dict(),
        'optimizer_D_B_state': optimizer_D_B.state_dict(),
        'mapping_type': mapping_type
    }
    filename = os.path.join(checkpoint_dir, f'checkpoint_{mapping_type}_epoch{epoch:03d}_{timestamp}.pth')
    torch.save(checkpoint, filename)
    print(f'>Checkpoint saved: {filename}')

def load_checkpoint(checkpoint_path, g_model_AtoB, g_model_BtoA, d_model_A, d_model_B,
                    optimizer_G, optimizer_D_A, optimizer_D_B):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    g_model_AtoB.load_state_dict(checkpoint['g_model_AtoB_state'])
    g_model_BtoA.load_state_dict(checkpoint['g_model_BtoA_state'])
    d_model_A.load_state_dict(checkpoint['d_model_A_state'])
    d_model_B.load_state_dict(checkpoint['d_model_B_state'])
    optimizer_G.load_state_dict(checkpoint['optimizer_G_state'])
    optimizer_D_A.load_state_dict(checkpoint['optimizer_D_A_state'])
    optimizer_D_B.load_state_dict(checkpoint['optimizer_D_B_state'])
    start_epoch = checkpoint['epoch'] + 1
    print(f'>Loaded checkpoint from epoch {checkpoint["epoch"]}')
    return start_epoch

# ------------------------------
# Utility: Visualize generated outputs (show middle slice)
def summarize_performance(epoch, g_model_AtoB, g_model_BtoA, dataset, name, n_samples=5, device='cpu', save_dir=None):
    """
    Displays four rows:
      Row 1: Real input (e.g., T2 or T1)
      Row 2: Generated output (e.g., synthetic FLAIR or T1CE)
      Row 3: Cycle reconstruction (e.g., mapping the generated output back to T2 or T1)
      Row 4: Absolute difference between Real input and Cycle reconstruction (heatmap)
    """

    # Set both generators to evaluation mode to disable dropout, etc.
    g_model_AtoB.eval()
    g_model_BtoA.eval()

    # Temporarily disable dataset augmentation for clean visualizations
    if hasattr(dataset, 'augment'):
        old_augment_state = dataset.augment
        dataset.augment = False

    # Choose random samples from dataset
    indices = np.random.choice(len(dataset), n_samples, replace=False)
    fig, axs = plt.subplots(4, n_samples, figsize=(4 * n_samples, 16))

    for i, idx in enumerate(indices):
        img_A, img_B = dataset[idx]  # real_A and real_B
        real_A = img_A.unsqueeze(0).to(device)

        # Generate synthetic output and cycle reconstruction
        with torch.no_grad():
            fake_B = g_model_AtoB(real_A)
            # Use match_depth so that fake_B matches real_A's depth
            fake_B = match_depth(fake_B, real_A)
            rec_A = g_model_BtoA(fake_B)
            rec_A = match_depth(rec_A, real_A)

        # Convert outputs from [-1, 1] to [0, 1]
        real_A_np = (real_A.cpu().numpy()[0] + 1) / 2.0
        fake_B_np = (fake_B.cpu().numpy()[0] + 1) / 2.0
        rec_A_np  = (rec_A.cpu().numpy()[0] + 1) / 2.0

        # Compute absolute error map
        diff_A = np.abs(real_A_np - rec_A_np)
        d_mid = real_A_np.shape[1] // 2

        # Plot real input
        axs[0, i].imshow(real_A_np[0, d_mid, :, :], cmap='gray')
        axs[0, i].set_title("Real Input")
        axs[0, i].axis('off')

        # Plot synthetic output
        axs[1, i].imshow(fake_B_np[0, d_mid, :, :], cmap='gray')
        axs[1, i].set_title("Generated Output")
        axs[1, i].axis('off')

        # Plot reconstructed input
        axs[2, i].imshow(rec_A_np[0, d_mid, :, :], cmap='gray')
        axs[2, i].set_title("Cycle Reconstruction")
        axs[2, i].axis('off')

        # Plot error heatmap
        im = axs[3, i].imshow(diff_A[0, d_mid, :, :], cmap='hot')
        axs[3, i].set_title("Absolute Difference")
        axs[3, i].axis('off')

    # Finalize and save the plot
    fig.suptitle(f"Epoch {epoch+1} - {name} Translation", fontsize=16)
    plt.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f'{name}_generated_plot_epoch{epoch+1:03d}.png')
    else:
        filename = f'{name}_generated_plot_epoch{epoch+1:03d}.png'

    plt.savefig(filename)
    plt.close()

    # Restore models and dataset to training state
    g_model_AtoB.train()
    g_model_BtoA.train()
    if hasattr(dataset, 'augment'):
        dataset.augment = old_augment_state

# ------------------------------
# Training Loop for CycleGAN (Paired version)
def train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA,
          dataloader, epochs=1, device='cpu', mapping_type='t2_flair', start_epoch=0,
          log_dir='output/logs', save_viz=True, viz_dir='output/visualization',
          use_tb=False, tb_dir='output/tensorboard'):

    criterion_GAN = nn.MSELoss()            # adversarial loss
    criterion_cycle = nn.L1Loss()           # cycle-consistency loss
    criterion_identity = nn.L1Loss()        # identity loss
    # Note: The intensity loss will be implemented with a masked L1 loss below.
    criterion_intensity = nn.L1Loss()       # defined for consistency but not used directly

    optimizer_G = optim.Adam(list(g_model_AtoB.parameters()) + list(g_model_BtoA.parameters()),
                             lr=0.0002, betas=(0.5, 0.999))
    optimizer_D_A = optim.Adam(d_model_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D_B = optim.Adam(d_model_B.parameters(), lr=0.0002, betas=(0.5, 0.999))

    pool_A = []
    pool_B = []

    # Loss weights
    lambda_id = 5
    lambda_cycle = 10
    lambda_intensity = 2

    logger = Logger(log_dir, mapping_type)
    writer = SummaryWriter(log_dir=tb_dir) if use_tb else None

    g_model_AtoB.train()
    g_model_BtoA.train()
    d_model_A.train()
    d_model_B.train()

    n_steps = len(dataloader)

    for epoch in range(start_epoch, epochs):
        print("Epoch {}/{}".format(epoch+1, epochs))
        progress_bar = tqdm(dataloader, total=n_steps)
        for i, (real_A, real_B) in enumerate(progress_bar):
            real_A = real_A.to(device)
            real_B = real_B.to(device)

            # Pad both real_A and real_B along the depth axis so that their depth becomes even.
            # (E.g., from 135 to 136 slices.) This ensures that the generator's operations are consistent.
            real_A = F.pad(real_A, (0, 0, 0, 0, 0, 1))  # Shape: (B, C, original_depth+1, H, W)
            real_B = F.pad(real_B, (0, 0, 0, 0, 0, 1))

            # Define the unpadded targets (original volumes) for loss calculations.
            real_A_unpadded = real_A[:, :, :-1, :, :]  # Shape: (B, C, original_depth, H, W)
            real_B_unpadded = real_B[:, :, :-1, :, :]

            with torch.no_grad():
                pred_shape_A = d_model_A(real_A_unpadded).shape
                pred_shape_B = d_model_B(real_B_unpadded).shape

            valid_A = torch.ones(pred_shape_A, device=device)
            fake_A_target = torch.zeros(pred_shape_A, device=device)
            valid_B = torch.ones(pred_shape_B, device=device)
            fake_B_target = torch.zeros(pred_shape_B, device=device)

            # === GENERATOR FORWARD PASS ===
            optimizer_G.zero_grad()

            # Identity Loss: generators should reproduce the input when fed the target domain.
            identity_A = g_model_BtoA(real_A)
            identity_A = identity_A[:, :, :-1, :, :]  # Crop to recover original depth
            loss_id_A = criterion_identity(identity_A, real_A_unpadded)

            identity_B = g_model_AtoB(real_B)
            identity_B = identity_B[:, :, :-1, :, :]
            loss_id_B = criterion_identity(identity_B, real_B_unpadded)
            loss_identity = (loss_id_A + loss_id_B) * lambda_id

            # GAN Loss: translate images between domains.
            fake_B = g_model_AtoB(real_A)
            fake_B = fake_B[:, :, :-1, :, :]  # Crop to original depth
            loss_GAN_A2B = criterion_GAN(d_model_B(fake_B), valid_B)

            fake_A = g_model_BtoA(real_B)
            fake_A = fake_A[:, :, :-1, :, :]
            loss_GAN_B2A = criterion_GAN(d_model_A(fake_A), valid_A)
            loss_GAN = loss_GAN_A2B + loss_GAN_B2A

            # Cycle Consistency Loss: translating back should recover the original image.
            recov_A = g_model_BtoA(fake_B)
            recov_A = recov_A[:, :, :-1, :, :]
            loss_cycle_A = criterion_cycle(recov_A, real_A_unpadded)

            recov_B = g_model_AtoB(fake_A)
            recov_B = recov_B[:, :, :-1, :, :]
            loss_cycle_B = criterion_cycle(recov_B, real_B_unpadded)
            loss_cycle = (loss_cycle_A + loss_cycle_B) * lambda_cycle

            # Intensity Loss: enforce matching of intensity distributions.
            # To avoid over-penalizing the background, mask out regions with insignificant signal.
            mask_thresh = 0.05  # Adjust this threshold based on your data's intensity scale.
            mask_A = (real_A_unpadded.abs() > mask_thresh).float()
            mask_B = (real_B_unpadded.abs() > mask_thresh).float()

            loss_intensity_A = torch.sum(torch.abs(fake_B - real_B_unpadded) * mask_B) / (torch.sum(mask_B) + 1e-8)
            loss_intensity_B = torch.sum(torch.abs(fake_A - real_A_unpadded) * mask_A) / (torch.sum(mask_A) + 1e-8)
            loss_intensity = (loss_intensity_A + loss_intensity_B) * lambda_intensity

            loss_G = loss_GAN + loss_cycle + loss_identity + loss_intensity
            loss_G.backward()
            optimizer_G.step()

            # === DISCRIMINATOR FORWARD PASS ===
            optimizer_D_A.zero_grad()
            loss_D_A_real = criterion_GAN(d_model_A(real_A_unpadded), valid_A)
            fake_A_pool = update_image_pool(pool_A, [fake_A.detach()])
            # Use fake_A_target (zeros with patch shape) as the target instead of fake_A.
            loss_D_A_fake = criterion_GAN(d_model_A(fake_A_pool), fake_A_target)
            loss_D_A = 0.5 * (loss_D_A_real + loss_D_A_fake)
            loss_D_A.backward()
            optimizer_D_A.step()

            optimizer_D_B.zero_grad()
            loss_D_B_real = criterion_GAN(d_model_B(real_B_unpadded), valid_B)
            fake_B_pool = update_image_pool(pool_B, [fake_B.detach()])
            # Similarly, use fake_B_target here.
            loss_D_B_fake = criterion_GAN(d_model_B(fake_B_pool), fake_B_target)
            loss_D_B = 0.5 * (loss_D_B_real + loss_D_B_fake)
            loss_D_B.backward()
            optimizer_D_B.step()


            metrics = {
                'epoch': epoch,
                'iteration': i,
                'loss_D_A': loss_D_A.item(),
                'loss_D_B': loss_D_B.item(),
                'loss_G': loss_G.item(),
                'loss_id': (loss_id_A + loss_id_B).item(),
                'loss_cycle': (loss_cycle_A + loss_cycle_B).item(),
                'loss_intensity': loss_intensity.item(),
                'fake_A_mean': fake_A.mean().item(),
                'fake_A_std': fake_A.std().item(),
                'fake_B_mean': fake_B.mean().item(),
                'fake_B_std': fake_B.std().item()
            }
            logger.log(epoch, i, metrics)
            if writer:
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        writer.add_scalar(f"{mapping_type}/{k}", v, epoch * n_steps + i)

            if (i+1) % 10 == 0:
                progress_bar.set_description("Iter {}/{}: D_A {:.3f}, D_B {:.3f}, G {:.3f}".format(
                    i+1, n_steps, loss_D_A.item(), loss_D_B.item(), loss_G.item()))

            if (i+1) == n_steps and save_viz:
                summarize_performance(epoch, g_model_AtoB, g_model_BtoA, dataloader.dataset, 'AtoB', device=device, save_dir=viz_dir)
                summarize_performance(epoch, g_model_BtoA, g_model_AtoB, dataloader.dataset, 'BtoA', device=device, save_dir=viz_dir)
            if (i+1) % (n_steps * 5) == 0:
                save_models(i, g_model_AtoB, g_model_BtoA, mapping_type)

        # Save checkpoint at the end of each epoch
        save_checkpoint(epoch, g_model_AtoB, g_model_BtoA, d_model_A, d_model_B,
                        optimizer_G, optimizer_D_A, optimizer_D_B, mapping_type)
        logger.save()
        if writer:
            writer.flush()
    if writer:
        writer.close()

# ------------------------------
# Main Execution with argument parsing
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train 3D CycleGAN for MRI Data Augmentation')
    parser.add_argument('--mapping_type', type=str, default='t2_flair', choices=['t2_flair', 't1_contrast'],
                        help='Mapping type to train: t2_flair or t1_contrast')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--data_dir', type=str, default=r'./processed_data/brats128_cyclegan/images',
                        help='Directory containing preprocessed .npy volumes')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--log_dir', type=str, default='./output/logs', help='Directory to save training logs')
    parser.add_argument('--viz_dir', type=str, default='./output/visualization', help='Directory to save visualizations')
    parser.add_argument('--no_viz', action='store_true', help='Disable saving visualizations')
    parser.add_argument('--use_tb', action='store_true', help='Enable TensorBoard logging')
    parser.add_argument('--tb_dir', type=str, default='output/tensorboard', help='TensorBoard log directory')
    args = parser.parse_args()
    
    # Convert paths to absolute paths to avoid relative path issues on Windows.
    args.log_dir = os.path.abspath(args.log_dir)
    args.viz_dir = os.path.abspath(args.viz_dir)
    args.tb_dir = os.path.abspath(args.tb_dir)
    args.data_dir = os.path.abspath(args.data_dir)

    # Optionally print the absolute paths for verification:
    print("Log Directory: ", args.log_dir)
    print("Visualization Directory: ", args.viz_dir)
    print("TensorBoard Directory: ", args.tb_dir)
    print("Data Directory: ", args.data_dir)

    mapping_type = args.mapping_type
    epochs = args.epochs
    complete_data_dir = args.data_dir
    save_viz = not args.no_viz

    # Create dataset and dataloader
    augment = True
    dataset = PairedPreprocessedDataset(complete_data_dir, mapping_type, augment=augment)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Optionally display some augmented slices for debugging
    sample_A, sample_B = dataset[0]
    print("Displaying augmented Domain A image:")
    display_random_slices(sample_A.numpy(), n_slices=5)
    print("Displaying augmented Domain B image:")
    display_random_slices(sample_B.numpy(), n_slices=5)

    # Define input and output channels (assumes single channel per modality)
    in_channels = 1
    out_channels = 1

    # Instantiate models.
    d_model_A = Discriminator3D(in_channels).to(device)
    d_model_B = Discriminator3D(out_channels).to(device)
    g_model_AtoB = Generator3D(in_channels, out_channels, n_resnet=6).to(device)
    g_model_BtoA = Generator3D(out_channels, in_channels, n_resnet=6).to(device)

    # Print model summary using sample dimensions
    sample_A, _ = dataset[0]
    _, d, h, w = sample_A.shape
    print("Generator AtoB Model Summary:")
    summary(g_model_AtoB, input_size=(1, d, h, w))

    # If resume flag is provided, load checkpoint
    start_epoch = 0
    if args.resume is not None:
        optimizer_G = optim.Adam(list(g_model_AtoB.parameters()) + list(g_model_BtoA.parameters()),
                                 lr=0.0002, betas=(0.5, 0.999))
        optimizer_D_A = optim.Adam(d_model_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
        optimizer_D_B = optim.Adam(d_model_B.parameters(), lr=0.0002, betas=(0.5, 0.999))
        start_epoch = load_checkpoint(args.resume, g_model_AtoB, g_model_BtoA, d_model_A, d_model_B,
                                      optimizer_G, optimizer_D_A, optimizer_D_B)

    # Begin training
    train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, dataloader,
          epochs=epochs, device=device, mapping_type=mapping_type, start_epoch=start_epoch,
          log_dir=args.log_dir, save_viz=save_viz, viz_dir=args.viz_dir,
          use_tb=args.use_tb, tb_dir=args.tb_dir)
