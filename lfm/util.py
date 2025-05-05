import numpy as np
import imageio
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from tqdm.auto import tqdm

def volume_to_gif(volume, gif_path="output.gif", cmap="gray", fps=10, vmin=None, vmax=None):
    images = []

    for i in tqdm(range(volume.shape[0])):
        fig, ax = plt.subplots()
        canvas = FigureCanvas(fig)
        ax.axis("off")
        im = ax.imshow(volume[i], cmap=cmap, vmin=vmin, vmax=vmax)
        canvas.draw()

        # Convert canvas to image
        image = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
        image = image.reshape(canvas.get_width_height()[::-1] + (4,))
        images.append(image)
        plt.close(fig)

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
    # Default to max projection if no function provided
    if projection_func is None:
        projection_func = np.max

    # Get dimensions
    depth, height, width = volume.shape

    # Calculate output dimensions with padding
    output_height = height + depth + 2 * pad
    output_width = width + depth + 2 * pad

    # Create empty output image (using same array type as input)
    output = np.zeros((output_height, output_width), dtype=volume.dtype)

    # Create projections
    xy_proj = projection_func(volume, axis=0)  # Top-down view
    xz_proj = projection_func(volume, axis=1)  # Front view
    yz_proj = projection_func(volume, axis=2)  # Side view

    # Place projections in output image
    # XY projection (center)
    output[pad:pad + height, pad:pad + width] = xy_proj

    # XZ projection (bottom)
    output[pad + height + pad:pad + height + pad + depth, pad:pad + width] = xz_proj

    # YZ projection (right side) - needs transpose to align correctly
    output[pad:pad + height, pad + width + pad:pad + width + pad + depth] = yz_proj.T

    return output