import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

def compute_intensity_stats(volume):
    """
    Computes basic statistics for a 3D or 4D volume.
    If the volume is 4D (C, D, H, W), it computes stats over all voxels.
    """
    intensities = volume.flatten()
    stats = {
        'min': np.min(intensities),
        'max': np.max(intensities),
        'mean': np.mean(intensities),
        'std': np.std(intensities),
        'median': np.median(intensities),
        '90th_percentile': np.percentile(intensities, 90),
        '95th_percentile': np.percentile(intensities, 95)
    }
    return stats

def plot_intensity_histogram(volume, bins=50, title="Intensity Histogram"):
    """
    Plots a histogram of intensity values from a volume.
    """
    intensities = volume.flatten()
    plt.figure(figsize=(6,4))
    plt.hist(intensities, bins=bins, edgecolor='k')
    plt.title(title)
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

# Set the path to your preprocessed volume directory.
data_dir = r'./processed_data/brats128_cyclegan/images'
files = sorted(glob(os.path.join(data_dir, '*.npy')))

n_samples = min(5, len(files))
print(f"Inspecting {n_samples} volumes out of {len(files)} total files.")

for i in range(n_samples):
    # Load volume from a .npy file.
    volume = np.load(files[i]).astype(np.float32)
    # Handle volumes saved with an extra singleton dimension if necessary.
    if volume.ndim == 5 and volume.shape[0] == 1:
        volume = volume[0]
    # Transpose so that volume shape is (C, D, H, W)
    volume = np.transpose(volume, (3, 2, 0, 1))

    stats = compute_intensity_stats(volume)
    print(f"Stats for volume {i+1}:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    plot_intensity_histogram(volume, title=f"Volume {i+1} Intensity Histogram")
