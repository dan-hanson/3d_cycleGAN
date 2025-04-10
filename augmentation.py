import numpy as np
from scipy.ndimage import rotate
import matplotlib.pyplot as plt

def random_rotation_3d(volume, mean=0, std=10, pad_margin=20, visualize=True):
    """
    Applies a random rotation to a 3D volume (with shape (C, D, H, W)) using pre-padding.
    All channels are rotated identically.
    
    Steps:
      1. Pad the volume along D, H, and W by pad_margin voxels.
      2. Randomly choose one axis and a rotation angle from a normal distribution.
      3. Rotate each channel of the padded volume.
      4. Crop the center back to the original dimensions.
      5. (Optionally) Display orthogonal slices from the first channel.
    
    Args:
        volume (np.ndarray): Input volume with shape (C, D, H, W).
        mean (float): Mean rotation angle (degrees).
        std (float): Standard deviation of the rotation angle.
        pad_margin (int): Number of voxels to pad on each side.
        visualize (bool): If True, display the rotated volume slices.
    
    Returns:
        np.ndarray: The rotated volume, same shape as input.
    """
    assert volume.ndim == 4, "Input volume must have shape (C, D, H, W)"
    C, D, H, W = volume.shape

    # 1. Pad the volume
    padded = np.pad(volume, 
                    pad_width=((0, 0), (pad_margin, pad_margin),
                               (pad_margin, pad_margin), (pad_margin, pad_margin)),
                    mode='constant', constant_values=0)
    _, Dp, Hp, Wp = padded.shape

    # 2. Random rotation parameters
    axis = np.random.choice([0, 1, 2])
    angle = np.random.normal(loc=mean, scale=std)

    # For a single channel (3D array of shape (D, H, W)), valid rotation axes are:
    #   axis 0: rotate in (H, W) plane → axes (1, 2)
    #   axis 1: rotate in (D, W) plane → axes (0, 2)
    #   axis 2: rotate in (D, H) plane → axes (0, 1)
    rotation_planes = {
        0: (1, 2),
        1: (0, 2),
        2: (0, 1)
    }
    axes = rotation_planes[axis]

    # 3. Rotate each channel of the padded volume using the same rotation
    rotated_padded = np.zeros_like(padded)
    for c in range(C):
        rotated_padded[c] = rotate(
            padded[c], angle=angle, axes=axes, reshape=False,
            order=1, mode='constant', cval=0.0, prefilter=False
        )

    # 4. Crop the center back to original dimensions
    start_D = (Dp - D) // 2
    start_H = (Hp - H) // 2
    start_W = (Wp - W) // 2
    cropped = rotated_padded[:, start_D:start_D + D, start_H:start_H + H, start_W:start_W + W]

    # 5. Visualization (if enabled)
    if visualize:
        vol_vis = cropped[0]
        d, h, w = vol_vis.shape
        mid_d, mid_h, mid_w = d // 2, h // 2, w // 2

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(vol_vis[mid_d, :, :], cmap='gray')
        axs[0].set_title(f'Axial (Z) Slice {mid_d}')
        axs[1].imshow(vol_vis[:, mid_h, :], cmap='gray')
        axs[1].set_title(f'Coronal (Y) Slice {mid_h}')
        axs[2].imshow(vol_vis[:, :, mid_w], cmap='gray')
        axs[2].set_title(f'Sagittal (X) Slice {mid_w}')
        for ax in axs:
            ax.axis('off')
        plt.suptitle(f"Rotation axis: {axis} | Angle: {angle:.2f}° | Pad: {pad_margin}", fontsize=14)
        plt.tight_layout()
        plt.show()

    return cropped

def display_random_slices(volume, n_slices=5):
    """
    Displays n evenly spaced slices from the first channel of a 3D or 4D volume.
    
    Args:
        volume (np.ndarray): Volume with shape (C, D, H, W) or (D, H, W).
        n_slices (int): Number of slices to display.
    """
    if volume.ndim == 4:
        volume = volume[0]  # Use first channel
    assert volume.ndim == 3, "Input must be 3D after channel selection"

    depth = volume.shape[0]
    step = max(depth // n_slices, 1)
    fig, axs = plt.subplots(1, n_slices, figsize=(3 * n_slices, 3))
    for i in range(n_slices):
        idx = i * step
        axs[i].imshow(volume[idx], cmap='gray')
        axs[i].set_title(f"Slice {idx}")
        axs[i].axis('off')
    plt.tight_layout()
    plt.show()
