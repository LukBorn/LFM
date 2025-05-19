import numpy as np
import cupy as cp
import imageio
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from tqdm.auto import tqdm

def volume_to_gif(volume, gif_path="output.gif", cmap="gray", fps=10, vmin=None, vmax=None):
    images = []

    for i in tqdm(range(volume.shape[0])):
        # Normalize the image data
        frame = volume[i]
        if vmin is None:
            vmin = frame.min()
        if vmax is None:
            vmax = frame.max()
        normalized_frame = (frame - vmin) / (vmax - vmin)
        normalized_frame = (normalized_frame * 255).astype(np.uint8)

        # Apply colormap
        colormap = plt.get_cmap(cmap)
        colored_frame = colormap(normalized_frame)
        colored_frame = (colored_frame[:, :, :3] * 255).astype(np.uint8)  # Drop alpha channel

        images.append(colored_frame)

    imageio.mimsave(gif_path, images, fps=fps, loop = 0)
    print(f"GIF saved to: {gif_path}")


def create_projection_image(volume, projection_func=None, pad=10):
    """
    Creates a 2D image showing projections of a 3D volume along all three axes.

    Parameters:
    -----------
    volume : cp.ndarray or np.ndarray
        3D input volume of shape (depth, height, width)
    projection_func : callable, optional
        Function to use for projection (e.g., np.max, np.mean, np.sum)
        Defaults to max if None is provided
    pad : int
        Padding between the projections

    Returns:
    --------
    projection_image : same type as volume
        2D image showing all three projections arranged with:
        - xy (axial) in the center
        - xz at the bottom
        - yz on the right side
    """
    # Get dimensions
    depth, height, width = volume.shape

    # Calculate output dimensions with padding
    output_height = height + depth + 3 * pad
    output_width = width + depth + 3 * pad

    # Create empty output image (using same array type as input)
    if isinstance(volume, np.ndarray):
        output = np.zeros((output_height, output_width), dtype=volume.dtype)
        if projection_func is None:
            projection_func = np.max
    elif isinstance(volume, cp.ndarray):
        output = cp.zeros((output_height, output_width), dtype=volume.dtype)
        if projection_func is None:
            projection_func = cp.max
    else:
        raise TypeError("Volume must be of type cp.ndarray or cp.ndarray")

    # XY projection (center)
    output[pad:pad + height, pad:pad + width] = projection_func(volume, axis=0)

    # XZ projection (bottom)
    output[pad + height + pad:pad + height + pad + depth, pad:pad + width] = projection_func(volume, axis=1)

    # YZ projection (right side) - needs transpose to align correctly
    output[pad:pad + height, pad + width + pad:pad + width + pad + depth] = projection_func(volume, axis=2).T

    return output