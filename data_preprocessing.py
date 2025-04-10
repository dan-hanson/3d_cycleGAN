import os
import numpy as np
import nibabel as nib
from pathlib import Path
import json
import glob
import shutil
import subprocess
from typing import Tuple, List, Dict
import splitfolders
import multiprocessing
from functools import partial
import time
import sys
import argparse
from tqdm import tqdm


def process_single_case(case_data, output_path, min_label_ratio=0.007, has_mask=True): 
    """
    Process a single case with optimized operations
    
    Args:
        case_data (tuple): Tuple containing (case_idx, flair_path, t1ce_path, t2_path, t1_path, mask_path)
                          If has_mask=False, mask_path can be None
        output_path (str): Path to save preprocessed data
        min_label_ratio (float): Minimum ratio of non-zero labels required
        has_mask (bool): Whether the case has a segmentation mask
        
    Returns:
        tuple: (status, case_id) where status is True if valid, False if error, None if skipped
    """
    if has_mask:
        case_idx, flair_path, t1ce_path, t2_path, t1_path, mask_path = case_data
    else:
        case_idx, flair_path, t1ce_path, t2_path, t1_path = case_data
        mask_path = None
    
    # Extract case_id from directory name
    case_id = os.path.basename(os.path.dirname(t2_path))
    
    try:
        # For better output from multiple processes
        sys.stdout.write(f"Starting to process {case_id}...\n")
        sys.stdout.flush()
        
        # Load modalities and explicitly convert to the right types
        temp_image_flair = nib.load(flair_path).get_fdata()
        temp_image_t1ce = nib.load(t1ce_path).get_fdata()
        temp_image_t2 = nib.load(t2_path).get_fdata()
        temp_image_t1 = nib.load(t1_path).get_fdata()
        
        # Load mask if available
        if has_mask:
            temp_mask = nib.load(mask_path).get_fdata()
            # Convert mask to uint8 for memory efficiency
            temp_mask = temp_mask.astype(np.uint8)
            # Remap label 4 to 3 (following BraTS convention)
            temp_mask[temp_mask == 4] = 3
        
        # Explicitly convert to float32 (important for in-place operations)
        temp_image_flair = temp_image_flair.astype(np.float32)
        temp_image_t1ce = temp_image_t1ce.astype(np.float32)
        temp_image_t2 = temp_image_t2.astype(np.float32)
        temp_image_t1 = temp_image_t1.astype(np.float32)
        
        # Pre-crop to reduce memory footprint before normalization - wider crop window
        temp_image_flair = temp_image_flair[40:200, 40:200, 10:145]
        temp_image_t1ce = temp_image_t1ce[40:200, 40:200, 10:145]
        temp_image_t2 = temp_image_t2[40:200, 40:200, 10:145]
        temp_image_t1 = temp_image_t1[40:200, 40:200, 10:145]

        if has_mask:
            temp_mask = temp_mask[40:200, 40:200, 10:145]
            
            # Check if case has enough non-zero labels early to avoid unnecessary processing
            val, counts = np.unique(temp_mask, return_counts=True)
            
            if (1 - (counts[0]/counts.sum())) <= min_label_ratio:
                sys.stdout.write(f"Case {case_id} skipped: insufficient non-zero labels\n")
                sys.stdout.flush()
                return None, case_id  # Not enough non-zero labels
        
        # Optimize normalization using vectorized operations
        # Use in-place operations to reduce memory usage
        for img in [temp_image_flair, temp_image_t1ce, temp_image_t2, temp_image_t1]:
            # Verify data type to prevent errors
            if not np.issubdtype(img.dtype, np.floating):
                img = img.astype(np.float32)
                
            min_val = np.min(img)
            max_val = np.max(img)
            if max_val > min_val:  # Avoid division by zero
                img -= min_val
                img /= (max_val - min_val)
        
        # Stack modalities (flair, t1ce, t2, t1) - more memory efficient than separate operations
        temp_combined_images = np.stack([temp_image_flair, temp_image_t1ce, temp_image_t2, temp_image_t1], axis=3)
        
        # Save the processed files
        np.save(
            Path(output_path) / 'images' / f'image_{case_id}.npy',
            temp_combined_images
        )
        
        if has_mask:
            np.save(
                Path(output_path) / 'masks' / f'mask_{case_id}.npy',
                temp_mask
            )
        
        sys.stdout.write(f"Case {case_id} processed successfully\n")
        sys.stdout.flush()
        return True, case_id
        
    except Exception as e:
        sys.stdout.write(f"Error processing {case_id}: {str(e)}\n")
        sys.stdout.flush()
        return False, case_id


def preprocess_brats2020(input_path: str, output_path: str, dataset_type: str = "training", has_mask: bool = True):
    """
    Preprocess BraTS2020 dataset with parallel processing for speed
    
    Args:
        input_path (str): Path to raw BraTS2020 dataset (BraTS20_Training_* or BraTS20_Validation_*)
        output_path (str): Path to save preprocessed data
        dataset_type (str): Either "training" or "validation"
        has_mask (bool): Whether the dataset has segmentation masks
    """
    print(f"Starting preprocessing of BraTS2020 {dataset_type} dataset...")
    print(f"Input path: {input_path}")
    
    # Determine number of workers (use 1 less than CPU count to avoid system freeze)
    num_workers = max(1, multiprocessing.cpu_count() - 1)
    
    # Create output directories
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / 'images').mkdir(exist_ok=True)
    
    if has_mask:
        (output_path / 'masks').mkdir(exist_ok=True)
    
    # Find all directories
    print("Scanning for input files...")
    
    # Determine directory pattern based on dataset type
    dir_pattern = f'BraTS20_{dataset_type.capitalize()}_*' if dataset_type.lower() in ['training', 'validation'] else 'BraTS20_*'
    
    patient_dirs = sorted(glob.glob(f'{input_path}/{dir_pattern}'))
    
    # If no directories found with this pattern, try searching subdirectories
    if not patient_dirs:
        print(f"No directories found with pattern '{dir_pattern}'. Trying to search in subdirectories...")
        patient_dirs = sorted(glob.glob(f'{input_path}/**/{dir_pattern}', recursive=True))
    
    patient_dirs = [d for d in patient_dirs if os.path.isdir(d)]
    
    if not patient_dirs:
        print(f"ERROR: No patient directories found matching pattern '{dir_pattern}'")
        return {"valid_cases": [], "skipped_cases": []}
    
    print(f"Found {len(patient_dirs)} patient directories")
    
    # Process all patient directories
    case_data = []
    
    for idx, patient_dir in enumerate(patient_dirs):
        patient_id = os.path.basename(patient_dir)
        
        # Define paths for each modality file
        flair_file = os.path.join(patient_dir, f"{patient_id}_flair.nii.gz")
        t1ce_file = os.path.join(patient_dir, f"{patient_id}_t1ce.nii.gz")
        t2_file = os.path.join(patient_dir, f"{patient_id}_t2.nii.gz")
        t1_file = os.path.join(patient_dir, f"{patient_id}_t1.nii.gz")
        
        if has_mask:
            mask_file = os.path.join(patient_dir, f"{patient_id}_seg.nii.gz")
            
            # Check if all necessary files exist
            if all(os.path.exists(f) for f in [flair_file, t1ce_file, t2_file, t1_file, mask_file]):
                case_data.append((idx, flair_file, t1ce_file, t2_file, t1_file, mask_file))
            else:
                # Try alternative file extensions (.nii instead of .nii.gz)
                flair_file = os.path.join(patient_dir, f"{patient_id}_flair.nii")
                t1ce_file = os.path.join(patient_dir, f"{patient_id}_t1ce.nii")
                t2_file = os.path.join(patient_dir, f"{patient_id}_t2.nii")
                t1_file = os.path.join(patient_dir, f"{patient_id}_t1.nii")
                mask_file = os.path.join(patient_dir, f"{patient_id}_seg.nii")
                
                if all(os.path.exists(f) for f in [flair_file, t1ce_file, t2_file, t1_file, mask_file]):
                    case_data.append((idx, flair_file, t1ce_file, t2_file, t1_file, mask_file))
                else:
                    print(f"Warning: Missing files for {patient_id}. Skipping.")
        else:
            # For validation data without masks
            if all(os.path.exists(f) for f in [flair_file, t1ce_file, t2_file, t1_file]):
                case_data.append((idx, flair_file, t1ce_file, t2_file, t1_file))
            else:
                # Try alternative file extensions (.nii instead of .nii.gz)
                flair_file = os.path.join(patient_dir, f"{patient_id}_flair.nii")
                t1ce_file = os.path.join(patient_dir, f"{patient_id}_t1ce.nii")
                t2_file = os.path.join(patient_dir, f"{patient_id}_t2.nii")
                t1_file = os.path.join(patient_dir, f"{patient_id}_t1.nii")
                
                if all(os.path.exists(f) for f in [flair_file, t1ce_file, t2_file, t1_file]):
                    case_data.append((idx, flair_file, t1ce_file, t2_file, t1_file))
                else:
                    print(f"Warning: Missing files for {patient_id}. Skipping.")
    
    print(f"Found {len(case_data)} complete cases out of {len(patient_dirs)} directories")
    
    if len(case_data) == 0:
        print("ERROR: No valid cases found for processing!")
        return {"valid_cases": [], "skipped_cases": []}
    
    processed_files = {'valid_cases': [], 'skipped_cases': []}
    
    # Process cases
    print(f"Processing {len(case_data)} cases...")
    print(f"Using parallel processing with {num_workers} workers...")
    
    # Process first case separately to catch any setup issues early
    if case_data:
        print("Processing first case to check for issues...")
        first_status, first_case_id = process_single_case(case_data[0], output_path, has_mask=has_mask)
        
        if first_status is True:
            processed_files['valid_cases'].append(first_case_id)
        elif first_status is False:
            processed_files['skipped_cases'].append(first_case_id)
        
        # Process remaining cases in parallel
        remaining_case_data = case_data[1:]
        print(f"Processing remaining {len(remaining_case_data)} cases in parallel...")
        
        if remaining_case_data:
            # Set up the parallel processing function
            process_func = partial(process_single_case, output_path=output_path, has_mask=has_mask)
            
            # Process in smaller batches to avoid memory issues
            batch_size = min(32, len(remaining_case_data))
            batches = [remaining_case_data[i:i+batch_size] for i in range(0, len(remaining_case_data), batch_size)]
            
            total_processed = len(processed_files['valid_cases']) + len(processed_files['skipped_cases'])
            
            for batch_idx, batch in enumerate(batches):
                # Use a context manager to ensure proper cleanup of resources
                with multiprocessing.Pool(processes=num_workers) as pool:
                    batch_results = pool.map(process_func, batch)
                
                # Process batch results
                valid_in_batch = 0
                for status, case_id in batch_results:
                    if status is True:
                        processed_files['valid_cases'].append(case_id)
                        valid_in_batch += 1
                    elif status is False:
                        processed_files['skipped_cases'].append(case_id)
                
                total_processed = len(processed_files['valid_cases']) + len(processed_files['skipped_cases'])
                progress_percent = (total_processed / len(case_data)) * 100
                
                print(f"Batch {batch_idx+1}/{len(batches)} ({valid_in_batch}/{len(batch)} valid cases) - Overall: {total_processed}/{len(case_data)} ({progress_percent:.1f}%) processed")
    else:
        print("No cases found to process!")
    
    # Save processing results
    with open(output_path / f'processing_results_{dataset_type}.json', 'w') as f:
        json.dump(processed_files, f, indent=2)
    
    print(f"Preprocessing complete. Processed {len(processed_files['valid_cases'])} valid cases.")
    print(f"Skipped {len(processed_files['skipped_cases'])} cases.")
    
    return processed_files


def split_dataset(input_folder: str, output_folder: str):
    """
    Split preprocessed dataset into training and validation sets with fixed 0.9 train ratio
    
    Args:
        input_folder (str): Path to preprocessed data
        output_folder (str): Path to save split data
    """
    # Hardcoded train ratio to 0.9
    train_ratio = 0.9
    print(f"Splitting dataset with fixed train ratio: {train_ratio}")
    
    # Check if there are actually files to split
    image_count = len(glob.glob(f"{input_folder}/images/*.npy"))
    mask_count = len(glob.glob(f"{input_folder}/masks/*.npy"))
    
    if image_count == 0 or mask_count == 0:
        print(f"WARNING: No files found to split! Images: {image_count}, Masks: {mask_count}")
        return
    
    # Split with a ratio
    splitfolders.ratio(
        input_folder, 
        output=output_folder, 
        seed=42, 
        ratio=(train_ratio, 1-train_ratio), 
        group_prefix=None
    )
    
    # Count files in each split
    train_images = len(glob.glob(f"{output_folder}/train/images/*.npy"))
    val_images = len(glob.glob(f"{output_folder}/val/images/*.npy"))
    
    print(f"Dataset split complete:")
    print(f"  Training: {train_images} images")
    print(f"  Validation: {val_images} images")


def create_cyclegan_dataset(train_split_path: str, validation_data_path: str, cyclegan_output_path: str):
    """
    Create a dataset for CycleGAN training by combining 90% training split (no masks) with validation data
    Using rsync for fast bulk file copying
    
    Args:
        train_split_path (str): Path to the train split from the training data
        validation_data_path (str): Path to the processed validation data
        cyclegan_output_path (str): Path to save the combined CycleGAN dataset
    """
    print(f"Creating CycleGAN dataset...")
    
    # Create output directory
    cyclegan_path = Path(cyclegan_output_path)
    cyclegan_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for CycleGAN
    (cyclegan_path / 'images').mkdir(exist_ok=True)
    
    # Get file lists for checking
    train_images = glob.glob(f"{train_split_path}/train/images/*.npy")
    val_images = glob.glob(f"{validation_data_path}/images/*.npy")
    
    print(f"Found {len(train_images)} training images and {len(val_images)} validation images")
    
    # Use rsync for bulk copying instead of file-by-file copying
    
    # Copy training images (90% of original training)
    if train_images:
        print(f"Bulk copying training images using rsync...")
        train_src_dir = f"{train_split_path}/train/images/"
        dest_dir = f"{cyclegan_path}/images/"
        
        try:
            # Check if rsync is available
            result = subprocess.run(["which", "rsync"], capture_output=True, text=True)
            rsync_available = result.returncode == 0
            
            if rsync_available:
                # Use rsync for fast copying
                subprocess.run([
                    "rsync", "-av", "--progress",
                    train_src_dir,
                    dest_dir
                ], check=True)
            else:
                print("rsync not found, falling back to traditional copy methods...")
                # Fallback to using cp command, which is still faster than Python's file-by-file copy
                subprocess.run([
                    "cp", "-r",
                    f"{train_src_dir}*",
                    dest_dir
                ], check=True)
                
        except subprocess.SubprocessError as e:
            print(f"Error during bulk copy of training images: {str(e)}")
            print("Falling back to regular copy...")
            
            # Fallback to regular copying if subprocess fails
            for src in train_images:
                dest = os.path.join(dest_dir, os.path.basename(src))
                shutil.copy2(src, dest)
    
    # Copy validation images
    if val_images:
        print(f"Bulk copying validation images using rsync...")
        val_src_dir = f"{validation_data_path}/images/"
        dest_dir = f"{cyclegan_path}/images/"
        
        try:
            # Check if rsync is available
            result = subprocess.run(["which", "rsync"], capture_output=True, text=True)
            rsync_available = result.returncode == 0
            
            if rsync_available:
                # Use rsync for fast copying
                subprocess.run([
                    "rsync", "-av", "--progress",
                    val_src_dir,
                    dest_dir
                ], check=True)
            else:
                print("rsync not found, falling back to traditional copy methods...")
                # Fallback to using cp command
                subprocess.run([
                    "cp", "-r",
                    f"{val_src_dir}*",
                    dest_dir
                ], check=True)
                
        except subprocess.SubprocessError as e:
            print(f"Error during bulk copy of validation images: {str(e)}")
            print("Falling back to regular copy...")
            
            # Fallback to regular copying if subprocess fails
            for src in val_images:
                dest = os.path.join(dest_dir, os.path.basename(src))
                shutil.copy2(src, dest)
    
    # Count total files
    total_images = len(glob.glob(f"{cyclegan_output_path}/images/*.npy"))
    
    print(f"CycleGAN dataset creation complete:")
    print(f"  Total images: {total_images}")


def create_complete_dataset(
    training_data_path: str,
    validation_data_path: str,
    processed_training_path: str,
    processed_validation_path: str,
    split_data_path: str,
    cyclegan_data_path: str
):
    """
    Complete pipeline to prepare BraTS2020 dataset for both segmentation and CycleGAN training
    
    Args:
        training_data_path (str): Path to raw training data
        validation_data_path (str): Path to raw validation data
        processed_training_path (str): Path to save preprocessed training data
        processed_validation_path (str): Path to save preprocessed validation data
        split_data_path (str): Path to save split training data
        cyclegan_data_path (str): Path to save combined data for CycleGAN
    """
    print(f"=== STARTING COMPLETE DATASET PREPARATION ===")
    
    # Step 1: Preprocess the training dataset (with masks)
    print("\n=== STEP 1: PREPROCESSING TRAINING DATA ===")
    # Search for training directories
    training_dirs = sorted(glob.glob(f'{training_data_path}/BraTS20_Training_*'))
    if training_dirs:
        print(f"Found {len(training_dirs)} training directories")
        training_processed = preprocess_brats2020(
            training_data_path, 
            processed_training_path,
            dataset_type="training",
            has_mask=True
        )
        print(f"Training data processed: {len(training_processed['valid_cases'])} valid cases")
    else:
        print(f"WARNING: No training directories found. Skipping training data preprocessing.")
    
    # Step 2: Preprocess the validation dataset
    print("\n=== STEP 2: PREPROCESSING VALIDATION DATA ===")
    # Search for validation directories
    validation_dirs = sorted(glob.glob(f'{validation_data_path}/BraTS20_Validation_*'))
    
    if validation_dirs:
        print(f"Found {len(validation_dirs)} validation directories")
        # Check for the presence of mask files in the first validation directory
        first_dir = validation_dirs[0]
        dir_basename = os.path.basename(first_dir)
        mask_path_gz = os.path.join(first_dir, f"{dir_basename}_seg.nii.gz")
        mask_path = os.path.join(first_dir, f"{dir_basename}_seg.nii")
        
        has_mask_validation = os.path.exists(mask_path_gz) or os.path.exists(mask_path)
        print(f"Validation data has masks: {has_mask_validation}")
        
        validation_processed = preprocess_brats2020(
            validation_data_path, 
            processed_validation_path,
            dataset_type="validation",
            has_mask=has_mask_validation
        )
        print(f"Validation data processed: {len(validation_processed['valid_cases'])} valid cases")
    else:
        print(f"WARNING: No validation directories found. Skipping validation data preprocessing.")
    
    # Step 3: Split training data into train/val with fixed 0.9 ratio
    print("\n=== STEP 3: SPLITTING TRAINING DATA (90/10 SPLIT) ===")
    # Check if there's processed training data to split
    train_images = len(glob.glob(f"{processed_training_path}/images/*.npy"))
    train_masks = len(glob.glob(f"{processed_training_path}/masks/*.npy"))
    
    if train_images > 0 and train_masks > 0:
        print(f"Found {train_images} training images and {train_masks} masks to split")
        split_dataset(processed_training_path, split_data_path)
    else:
        print(f"WARNING: No training data found to split. Skipping split step.")
    
    # Step 4: Create CycleGAN dataset (90% training + validation) with rsync
    print("\n=== STEP 4: CREATING CYCLEGAN DATASET ===")
    # Check if there's training split and validation data
    train_split_images = len(glob.glob(f"{split_data_path}/train/images/*.npy")) if os.path.exists(f"{split_data_path}/train/images") else 0
    val_images = len(glob.glob(f"{processed_validation_path}/images/*.npy"))
    
    if train_split_images > 0 or val_images > 0:
        print(f"Found {train_split_images} training split images and {val_images} validation images")
        create_cyclegan_dataset(
            split_data_path,
            processed_validation_path,
            cyclegan_data_path
        )
    else:
        print(f"WARNING: No data found for CycleGAN dataset. Skipping CycleGAN dataset creation.")
    
    print(f"\n=== COMPLETE DATASET PREPARATION FINISHED ===")

    print(f"1. Segmentation training data: {split_data_path}")
    print(f"2. CycleGAN training data: {cyclegan_data_path}")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process BraTS2020 dataset for segmentation and CycleGAN')
    
    # Input paths with defaults pointing to /data directories
    parser.add_argument('--input_train', type=str, default='/data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData', 
                        help='Path to raw training data (default: /data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData)')
    parser.add_argument('--input_val', type=str, default='/data/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData', 
                        help='Path to raw validation data (default: /data/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData)')
    
    # Output paths
    parser.add_argument('--output_base', type=str, default='/data/processed',
                        help='Base directory for all output folders (default: /data/processed)')
    
    args = parser.parse_args()
    
    # Create derived output paths based on the base output directory
    output_base = Path(args.output_base)
    output_base.mkdir(parents=True, exist_ok=True)
    
    PROCESSED_TRAINING_PATH = str(output_base / 'brats128_training')
    PROCESSED_VALIDATION_PATH = str(output_base / 'brats128_validation')
    SPLIT_DATA_PATH = str(output_base / 'brats128_split')
    CYCLEGAN_DATA_PATH = str(output_base / 'brats128_cyclegan')
    
    start_time = time.time()
    
    create_complete_dataset(
        training_data_path=args.input_train,
        validation_data_path=args.input_val,
        processed_training_path=PROCESSED_TRAINING_PATH,
        processed_validation_path=PROCESSED_VALIDATION_PATH,
        split_data_path=SPLIT_DATA_PATH,
        cyclegan_data_path=CYCLEGAN_DATA_PATH
    )
    
    print(f"Total processing time: {time.time() - start_time:.2f} seconds")