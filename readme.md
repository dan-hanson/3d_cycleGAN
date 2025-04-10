

## Environment Setup (`environment.yml`)

This file lists the dependencies required to run the code. You can create a Conda environment using this file.

```yaml
# environment.yml
name: cyclegan3d_mri
channels:
  - pytorch # Add pytorch channel for PyTorch builds
  - conda-forge # Add conda-forge for broader package availability
  - defaults
dependencies:
  # Core Python
  - python=3.9 # Specify Python version
  - pip

  # Numerical and Data Handling
  - numpy
  - scipy # Often a dependency and useful for scientific computing

  # PyTorch and related libraries
  - pytorch::pytorch=*=*cuda* # Prioritize GPU build from pytorch channel (adjust *cuda* based on your CUDA version if needed, e.g., pytorch::pytorch=*=*cu118*)
  - pytorch::torchvision=*=*cuda*
  - pytorch::torchaudio=*=*cuda*
  - cpuonly # Add cpuonly as fallback if no GPU build matches
  - torchmetrics
  - torchsummary

  # Medical Imaging & Augmentation
  - conda-forge::torchio
  - conda-forge::nibabel # For loading NIfTI files in preprocessing

  # Plotting & Utilities
  - matplotlib
  - conda-forge::tqdm # Progress bars
  - conda-forge::pillow # For image saving (PIL)
  - conda-forge::tensorboard # For logging

  # Dataset Splitting (used in preprocessing)
  - conda-forge::split-folders

  # FID Calculation (install via pip if conda package unavailable/outdated)
  # - conda-forge::pytorch-fid # Check if available, otherwise use pip

  # Pip dependencies (for packages not easily found on conda channels)
  - pip:
      - pytorch-fid # Install FID calculation library

# To create the environment: conda env create -f environment.yml
# To activate: conda activate cyclegan3d_mri
README3D MRI CycleGAN for Synthetic Modality TranslationOverviewThis repository contains the code for training and evaluating a 3D CycleGAN model designed to translate between different MRI modalities. The primary goal is to generate synthetic MRI data, specifically for mapping:T2-weighted images (T2) <-> Fluid-Attenuated Inversion Recovery (FLAIR) imagesT1-weighted images (T1) <-> T1-weighted Contrast-Enhanced (T1CE) imagesThe models are trained using preprocessed data, ideally from datasets like BraTS 2020.Features3D CycleGAN Architecture: Implements 3D convolutional layers suitable for volumetric medical imaging data.Modality Translation: Supports T2 <-> FLAIR and T1 <-> T1CE mappings.Data Preprocessing: Includes a script (data_preprocessing.py) to normalize, crop, and format NIfTI files into .npy volumes.Data Augmentation: Leverages torchio for volumetric data augmentation during training (affine, flip, noise, bias field, gamma).Comprehensive Training: Incorporates Adversarial Loss (MSE), Cycle-Consistency Loss (L1), Identity Loss (L1), and a masked Intensity Loss (L1) to improve realism.Validation & Metrics: Performs validation during training using Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index Measure (SSIM).Logging: Saves detailed training progress (losses, metrics, learning rate) to JSON files and optionally TensorBoard.Checkpointing: Saves model weights, optimizer states, and scheduler states periodically to allow resuming training.Evaluation Suite: Provides tools (evaluation.py) to:Visualize sample translations from a checkpoint.Plot training/validation curves from log files.Calculate Fréchet Inception Distance (FID) scores (requires pytorch-fid and prepared image directories).Prerequisites & SetupConda Environment: It's recommended to use a conda environment. You can create one using the provided (updated) environment.yml file:conda env create -f environment.yml
conda activate cyclegan3d_mri # Or your chosen environment name
PyTorch: Ensure you have a compatible version of PyTorch installed, preferably with GPU support (CUDA). The environment file attempts to install a GPU version. See pytorch.org for details if needed.pytorch-fid (Optional): If you plan to calculate FID scores, the environment file includes installation via pip.Data Preparation (data_preprocessing.py)This script prepares your raw NIfTI data (e.g., from BraTS) for training.Functionality:Loads Flair, T1, T1CE, T2 modalities (and optionally segmentation masks).Performs cropping to a standard size (e.g., 160x160x135).Normalizes each modality independently to the [0, 1] range.Stacks modalities into a 4-channel .npy file (H, W, D, C) with order: [FLAIR, T1CE, T2, T1].(Optional) Splits the processed training data into train/val sets using splitfolders.(Optional) Creates a combined dataset suitable for CycleGAN training (e.g., merging training split and validation data).Usage:The script can be run directly, often configured to process specific input directories and produce outputs in designated folders. See the script's if __name__ == "__main__": block for argument parsing and default paths (which may need adjustment based on your data location).# Example: Run the full preprocessing pipeline (adjust paths as needed)
python data_preprocessing.py \
    --input_train /path/to/raw/BraTS2020_TrainingData \
    --input_val /path/to/raw/BraTS2020_ValidationData \
    --output_base /path/to/processed_data
Output: The key output for training the CycleGAN is typically a directory (e.g., /path/to/processed_data/brats128_cyclegan/images) containing the 4-channel .npy files.Training (train_cycleGAN.py)This script trains the 3D CycleGAN model. (Note: actual filename might be train_cycleGAN_v3.py)Usage Example:# Train T2 <-> FLAIR mapping with TensorBoard logging
python train_cycleGAN.py \
    --mapping_type t2_flair \
    --epochs 150 \
    --data_dir /path/to/processed_data/brats128_cyclegan/images \
    --val_data_dir /path/to/processed_data/brats128_split/val/images \
    --output_base_dir ./output \
    --use_tb \
    --n_resnet 9 \
    --num_workers 4

# Resume T1 <-> T1CE training from a checkpoint
python train_cycleGAN.py \
    --mapping_type t1_contrast \
    --epochs 200 \
    --data_dir /path/to/processed_data/brats128_cyclegan/images \
    --val_data_dir /path/to/processed_data/brats128_split/val/images \
    --output_base_dir ./output \
    --resume ./output/t1_contrast/checkpoints/checkpoint_t1_contrast_epoch099_....pth \
    --use_tb \
    --n_resnet 9
Command-Line Arguments:--mapping_type: Specifies the translation task.Choices: t2_flair, t1_contrastDefault: t2_flair--epochs: Number of training epochs.Default: 100--data_dir: Directory containing the 4-channel .npy training volumes (output of data_preprocessing.py). Data should be normalized to [0, 1] initially; the script handles rescaling to [-1, 1].Default: ./processed_data/brats128_training/images (likely needs adjustment)--val_data_dir: Directory containing .npy volumes for validation (e.g., from the validation split).Default: ./processed_data/brats128_validation/images (likely needs adjustment)--resume: Path to a .pth checkpoint file to resume training from.Default: None--output_base_dir: Base directory where all outputs (logs, visualizations, checkpoints, models) will be saved, organized by mapping_type.Default: ./output--log_dir: (Partially superseded by --output_base_dir) Specific directory for logs.Default: ./output/logs--viz_dir: (Partially superseded by --output_base_dir) Specific directory for visualizations.Default: ./output/visualization--no_viz: Disable saving visualizations during training.Default: False (visualizations are saved)--use_tb: Enable TensorBoard logging. Logs saved in <output_base_dir>/<mapping_type>/tensorboard.Default: False--lr: Initial learning rate for Adam optimizers.Default: 0.0002--n_resnet: Number of ResNet blocks in the generator architecture.Default: 6 (9 is also common)--num_workers: Number of worker processes for the DataLoader. Adjust based on CPU cores.Default: 4Evaluation (evaluation.py)This script evaluates a trained model using a saved checkpoint.Usage Example:# Visualize samples and plot logs from a checkpoint
python evaluation.py \
    --checkpoint ./output/t2_flair/checkpoints/checkpoint_t2_flair_epoch149_....pth \
    --data_dir /path/to/processed_data/brats128_split/val/images \
    --output_dir ./evaluation_output \
    --tasks visualize plot_logs \
    --samples 5

# Calculate FID scores (requires preparing real image PNG directories first!)
python evaluation.py \
    --checkpoint ./output/t1_contrast/checkpoints/checkpoint_t1_contrast_epoch099_....pth \
    --data_dir /path/to/processed_data/brats128_split/val/images \
    --output_dir ./evaluation_output \
    --tasks calculate_fid \
    --fid_real_a_dir /path/to/real_T1_pngs \
    --fid_real_b_dir /path/to/real_T1CE_pngs \
    --fid_num_images 1000
Command-Line Arguments:--checkpoint: Path to the trained .pth checkpoint file. (Required)--data_dir: Path to the evaluation data directory (containing .npy volumes, e.g., validation set). (Required)--output_dir: Base directory to save evaluation results. Outputs are organized as <output_dir>/<mapping_type>/<epoch_label>/.Default: ./evaluation_output--tasks: List of evaluation tasks to perform.Choices: visualize, plot_logs, calculate_fidDefault: ['visualize', 'plot_logs']--samples: Number of random samples for the visualize task.Default: 5--log_file: Path to the training log JSON file (needed for plot_logs). If None, it tries to find the latest log based on the checkpoint path.Default: None--fid_real_a_dir: Path to a directory containing real domain A images (e.g., PNGs of middle slices) for FID calculation. (Required for calculate_fid)Default: None--fid_real_b_dir: Path to a directory containing real domain B images (e.g., PNGs of middle slices) for FID calculation. (Required for calculate_fid)Default: None--fid_num_images: Number of synthetic images to generate for FID calculation.Default: 500--fid_batch_size: Batch size for calculating FID features (adjust based on GPU memory).Default: 16Other Scriptsaugmentation.py: Contains utility functions for displaying 3D volume slices, used for debugging and visualization.inspect_intensity.py: A helper script to check the intensity statistics and histograms of preprocessed .npy files.NotesThe training script assumes input .npy files have channels ordered [FLAIR, T1CE, T2, T1] and are normalized to [0, 1]. It internally selects the correct channels based on --mapping_type and rescales data to [-1, 1] for the GAN.Ensure sufficient disk space for saving checkpoints, logs, and potentially generated FID images.GPU memory requirements can be significant for 3D models