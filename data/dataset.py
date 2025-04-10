import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Callable
import random


class BraTS20Dataset(Dataset):
    """PyTorch Dataset for BraTS20 preprocessed data"""
    
    def __init__(self, 
                 data_path: str, 
                 transform: Optional[Callable] = None,
                 return_path: bool = False):
        """
        Initialize the dataset
        
        Args:
            data_path (str): Path to preprocessed data directory
            transform (Optional[Callable]): Optional transform to be applied to images
            return_path (bool): If True, __getitem__ returns file paths along with data
        """
        self.data_path = Path(data_path)
        self.transform = transform
        self.return_path = return_path
        
        # Get all processed files
        self.image_files = sorted(list((self.data_path / 'images').glob('*.npy')))
        self.mask_files = sorted(list((self.data_path / 'masks').glob('*.npy')))
        
        # Validation check
        assert len(self.image_files) == len(self.mask_files), \
            f"Number of images ({len(self.image_files)}) and masks ({len(self.mask_files)}) must match"
            
        print(f"Found {len(self.image_files)} image-mask pairs")
    
    def __len__(self) -> int:
        """Return the total number of samples"""
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """
        Get a sample from the dataset
        
        Args:
            idx (int): Index of the sample to get
            
        Returns:
            Tuple containing:
                - image: torch.Tensor of shape (C, D, H, W)
                - mask: torch.Tensor of shape (D, H, W)
                - (optional) paths: Tuple[str, str] if return_path is True
        """
        # Load image and mask
        image = np.load(self.image_files[idx])
        mask = np.load(self.mask_files[idx])
        
        # Convert to torch tensors
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).long()
        
        # Move channels to PyTorch format (N, C, D, H, W)
        image = image.permute(3, 0, 1, 2)
        
        if self.transform:
            image = self.transform(image)
        
        if self.return_path:
            return image, mask, (str(self.image_files[idx]), str(self.mask_files[idx]))
        return image, mask


def get_data_loaders(
    data_path: str,
    batch_size: int = 1,
    train_val_split: float = 0.8,
    num_workers: int = 4,
    transform: Optional[Callable] = None,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation data loaders
    
    Args:
        data_path (str): Path to preprocessed data directory
        batch_size (int): Batch size for data loaders
        train_val_split (float): Fraction of data to use for training
        num_workers (int): Number of worker processes for data loading
        transform (Optional[Callable]): Optional transform to be applied to images
        random_seed (int): Random seed for reproducibility
        
    Returns:
        Tuple[DataLoader, DataLoader]: Train and validation data loaders
    """
    # Set random seed for reproducibility
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    # Create full dataset
    dataset = BraTS20Dataset(data_path, transform=transform)

    # print size
    print(len(dataset))

    # Calculate split sizes
    train_size = int(train_val_split * len(dataset))
    val_size = len(dataset) - train_size
    
    # Split dataset
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Example usage
    DATA_PATH = "processed_data"
    
    # Create datasets and dataloaders
    train_loader, val_loader = get_data_loaders(
        DATA_PATH,
        batch_size=1,  # 3D volumes are large, start with batch_size=1
        train_val_split=0.8,
        num_workers=4
    )
    
    # Test loading a batch
    for images, masks in train_loader:
        print(f"Image batch shape: {images.shape}")
        print(f"Mask batch shape: {masks.shape}")
        break  # Just test one batch