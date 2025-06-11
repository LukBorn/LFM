import numpy as np
import cupy as cp
import imageio
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from tqdm.auto import tqdm
import glob
import os
import cv2
import av

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

def array_to_video(array_3d, filename=None, fps=10, cmap='viridis', title=None, vmin=None, vmax=None):
    """Create animation from 3D array and optionally save it to file."""
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from IPython.display import HTML

    fig, ax = plt.subplots(figsize=(8, 8))
    if vmin is None:
        vmin = 0
    if vmax is None:
        vmax = array_3d.max()
    img = ax.imshow(array_3d[0], cmap=cmap, vmin=vmin, vmax=vmax)
    title_obj = ax.set_title(f"Frame 0/{array_3d.shape[0]-1}")
    plt.tight_layout()

    def update(frame):
        img.set_array(array_3d[frame])
        title_obj.set_text(f"{title + ' - ' if title else ''}Frame {frame}/{array_3d.shape[0]-1}")
        return [img, title_obj]

    anim = animation.FuncAnimation(
        fig, update, frames=array_3d.shape[0],
        interval=1000/fps, blit=True
    )

    if filename:
        writer = animation.FFMpegWriter(fps=fps)
        anim.save(filename, writer=writer)

    plt.close()
    return HTML(anim.to_jshtml())






def generate_random_gaussians_3d(shape,
                                 sparseness=0.01,  # fraction of voxels that contain Gaussians
                                 intensity_dist=(50, 200),  # (min, max) for uniform distribution
                                 sigma_dist=(2, 5),  # (min, max) for standard deviations
                                 seed=None):
    """
    Generate a 3D volume with randomly placed 3D Gaussian kernels.

    Parameters:
    -----------
    shape: tuple
        Shape of the output volume (depth, height, width)
    sparseness: float
        Fraction of voxels that contain Gaussian kernels (0-1)
    intensity_dist: tuple
        (min, max) for uniform intensity distribution
    sigma_dist: tuple
        (min, max) for standard deviation of Gaussians
    seed: int
        Random seed for reproducibility

    Returns:
    --------
    volume: cp.ndarray
        3D volume with Gaussian kernels
    """
    if seed is not None:
        np.random.seed(seed)
        cp.random.seed(seed)

    # Create empty volume
    volume = cp.zeros(shape, dtype=cp.float32)
    depth, height, width = shape

    # Calculate number of Gaussians based on sparseness
    total_voxels = depth * height * width
    n_gaussians = int(total_voxels * sparseness)

    # Generate random positions
    positions = np.random.randint(n_gaussians, 3) * np.array([depth, height, width])

    # Generate random intensities and sigmas
    intensities = np.random.uniform(intensity_dist[0], intensity_dist[1], n_gaussians)
    sigmas = np.random.uniform(sigma_dist[0], sigma_dist[1], (n_gaussians, 3))

    # Create coordinate grids
    z_indices, y_indices, x_indices = cp.mgrid[:depth, :height, :width]

    # Add each Gaussian to the volume
    for i in range(n_gaussians):
        z_pos, y_pos, x_pos = positions[i]
        sigma_z, sigma_y, sigma_x = sigmas[i]
        intensity = intensities[i]

        # Calculate 3D Gaussian
        gaussian = intensity * cp.exp(
            -((z_indices - z_pos) ** 2 / (2 * sigma_z ** 2) +
              (y_indices - y_pos) ** 2 / (2 * sigma_y ** 2) +
              (x_indices - x_pos) ** 2 / (2 * sigma_x ** 2))
        )

        # Add to volume
        volume += gaussian

    return volume
