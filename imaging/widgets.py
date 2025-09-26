import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import polygon as skpolygon
from ipywidgets import IntSlider, VBox, Button, HBox, Label, Output
from scipy.ndimage import distance_transform_edt
import cupy as cp
from cupyx.scipy.ndimage import gaussian_filter
import h5py
import os
import warnings
import ast
import IPython

def get_mask_widget(volume, filepath, vmin=None, vmax=None, figsize=(8,8), mask_every=5, sigma=2.0, transpose=False):
    """
    For manually defining a mask for the reference volume
    volume: volume that is displayed while
    filepath: path to save the mask to
    vmin, vmax, transpose: parameters for displaying the underlying volume
    figsize: size of the widget
    mask_every: interval of zslices to define the 2d masks for
    sigma: size of kernel for smoothing the edges of the mask

    Controls:
    double left click: define new point
    single left click and drag: moves nearest point. 
        Sometimes you have to click near the point you want to move for it to register that you want to move
    right click: delete closest point
    """
    
    if transpose:
        volume = volume.transpose(0,2,1)
    
    nz, ny, nx = volume.shape
    if vmin is None:
        vmin = float(np.min(volume))
    if vmax is None:
        vmax = float(np.max(volume))

    fig, ax_orig = plt.subplots(1, 1, figsize=figsize)
    maskable_slices = list(range(0, nz, mask_every))
    current_idx = 0
    current_z = maskable_slices[current_idx]
    im_orig = ax_orig.imshow(volume[current_z], cmap='gray', vmin=vmin, vmax=vmax)
    ax_orig.set_title("Draw 2D Mask (double-click to add, drag to move, right-click to delete)")

    # Try to load masks_2d and mask_every from file if it exists
    masks_2d = {}
    loaded_mask_every = None
    if os.path.exists(filepath):
        try:
            with h5py.File(filepath, "r") as f:
                if "mask_every" in f.attrs:
                    loaded_mask_every = int(f.attrs["mask_every"])
                if loaded_mask_every == mask_every and "masks_2d" in f:
                    masks_2d = {}
                    for k, v in f["masks_2d"].items():
                        arr = np.array(v)
                        if arr.size == 0:
                            masks_2d[int(k)] = []
                        else:
                            masks_2d[int(k)] = [tuple(pt) for pt in arr]
                elif loaded_mask_every is not None and loaded_mask_every != mask_every:
                    warnings.warn(f"mask_every in file ({loaded_mask_every}) does not match current ({mask_every}). Starting with empty masks.")
        except Exception as e:
            warnings.warn(f"Could not load masks_2d from file: {e}")

    points_2d = []  # Current slice points [(y, x), ...]
    point_artists = []
    polygon_artist = None
    dragging_idx = None
    last_event_was_double = False
    status_label = Label("")

    def redraw():
        nonlocal polygon_artist
        for artist in point_artists:
            artist.remove()
        point_artists.clear()
        if polygon_artist is not None:
            polygon_artist.remove()
            polygon_artist = None
        for y, x in points_2d:
            artist, = ax_orig.plot(x, y, 'ro', picker=5)
            point_artists.append(artist)
        if len(points_2d) > 2:
            poly = np.array(points_2d)
            polygon_artist = plt.Polygon(np.column_stack([poly[:,1], poly[:,0]]), closed=True, fill=False, edgecolor='lime', linewidth=2)
            ax_orig.add_patch(polygon_artist)
        im_orig.set_data(volume[current_z])
        fig.canvas.draw_idle()

    def get_closest_point(y, x, threshold=10):
        if not points_2d:
            return None
        arr = np.array(points_2d)
        dists = np.hypot(arr[:,0] - y, arr[:,1] - x)
        idx = np.argmin(dists)
        if dists[idx] < threshold:
            return idx
        return None

    def on_press(event):
        nonlocal dragging_idx, last_event_was_double
        if event.inaxes != ax_orig:
            return
        x, y = event.xdata, event.ydata
        if event.button == 1:
            if last_event_was_double:
                last_event_was_double = False
                return
            idx = get_closest_point(y, x)
            if idx is not None:
                dragging_idx = idx
        elif event.button == 3:
            idx = get_closest_point(y, x)
            if idx is not None:
                points_2d.pop(idx)
                redraw()

    def on_release(event):
        nonlocal dragging_idx
        dragging_idx = None

    def on_motion(event):
        if dragging_idx is None or event.inaxes != ax_orig:
            return
        x, y = event.xdata, event.ydata
        points_2d[dragging_idx] = (y, x)
        redraw()

    def on_double_click(event):
        nonlocal last_event_was_double
        if event.inaxes != ax_orig or event.button != 1:
            return
        x, y = event.xdata, event.ydata
        points_2d.append((y, x))
        last_event_was_double = True
        redraw()

    def on_slice_change(change):
        nonlocal current_idx, current_z, points_2d
        # Save current points
        masks_2d[current_z] = points_2d.copy()
        # Change slice
        current_idx = change['new']
        current_z = maskable_slices[current_idx]
        points_2d[:] = masks_2d.get(current_z, []).copy()
        redraw()

    def export_mask_callback(btn):
        # Save current points
        masks_2d[current_z] = points_2d.copy()
        # Build 3D mask using distance transform interpolation
        mask_3d = np.zeros_like(volume, dtype=bool)
        mask_slices = sorted([z for z in masks_2d if len(masks_2d[z]) > 0])
        if mask_slices:
            binary_masks = {}
            for z in mask_slices:
                pts = masks_2d[z]
                mask = np.zeros((ny, nx), dtype=bool)
                if len(pts) == 1:
                    y, x = np.round(pts[0]).astype(int)
                    if 0 <= y < ny and 0 <= x < nx:
                        mask[y, x] = True
                elif len(pts) > 2:
                    poly = np.array(pts)
                    rr, cc = skpolygon(poly[:,0], poly[:,1], (ny, nx))
                    mask[rr, cc] = True
                binary_masks[z] = mask
            dt = {}
            for z in mask_slices:
                mask = binary_masks[z]
                dt[z] = distance_transform_edt(mask) - distance_transform_edt(~mask)
            dt_stack = np.zeros((nz, ny, nx), dtype=np.float32)
            print(dt_stack.shape)
            for z in range(nz):
                if z <= mask_slices[0]:
                    dt_stack[z] = dt[mask_slices[0]]
                elif z >= mask_slices[-1]:
                    dt_stack[z] = dt[mask_slices[-1]]
                else:
                    for i in range(len(mask_slices)-1):
                        z0, z1 = mask_slices[i], mask_slices[i+1]
                        if z0 <= z <= z1:
                            alpha = (z - z0) / (z1 - z0) if z1 != z0 else 0
                            dt_stack[z] = (1-alpha)*dt[z0] + alpha*dt[z1]
                            break
            mask_3d = dt_stack
            # Smooth with cupy
            mask_3d_cp = cp.asarray(mask_3d)
            mask_3d_smooth = gaussian_filter(mask_3d_cp, sigma=sigma).clip(0,1)
            if transpose:
                mask_3d_smooth = mask_3d_smooth.transpose(0,2,1)
            # Save both mask_3d_smooth and masks_2d to h5
            with h5py.File(filepath, "w") as f:
                f.create_dataset("mask_3d", data=mask_3d_smooth.get())
                g = f.create_group("masks_2d")
                for k, v in masks_2d.items():
                    arr = np.array(v, dtype=np.float32).reshape(-1, 2)  # shape (N, 2) or (0, 2)
                    g.create_dataset(str(k), data=arr)
                f.attrs["mask_every"] = mask_every
            status_label.value = f"Mask and masks_2d saved to {filepath}"
        else:
            status_label.value = "No masks defined!"

    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('button_press_event', lambda e: on_double_click(e) if e.dblclick else None)

    slider = IntSlider(value=0, min=0, max=len(maskable_slices)-1, step=1, description='z (mask)')
    slider.observe(on_slice_change, names='value')

    # Initialize first slice
    points_2d[:] = masks_2d.get(current_z, []).copy()
    redraw()

    export_button = Button(description="Export 3D Mask", button_style='success')
    export_button.on_click(export_mask_callback)
    
    out = Output()
    with out:
        plt.show()  # This will display the figure in the output widget
    
    return VBox([HBox([slider, export_button, status_label]), out])


from opm_img.orthoviews import OrthoViewsWidget
from video import get_clipped_array
def ortho_views_widget(vol,
                       vmin=0, 
                       vmax=100,
                       absolute_limits=False,
                       transpose=False,
                       gpu = True
                      ):
    vol = get_clipped_array(vol, vmin,vmax,absolute_limits, gpu=gpu)
    vol = vol.get() if gpu else vol
    if transpose:
        vol = vol.transpose(0,2,1)
    return OrthoViewsWidget(vol)

def play_video_widget(filename, width= 800, html_attributes = "controls loop"):
    return IPython.display.display(IPython.display.Video(filename, embed=True, width=width, html_attributes=html_attributes))

