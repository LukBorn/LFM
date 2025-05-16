import numpy as np
from scipy.optimize import curve_fit


def weighted_linear_fit(series, z=None, weight_sigma=5):
    """
    Fit a weighted linear model to `x_col` in `df`, with weights centered at index `z`.
    Returns the DataFrame with a new column 'x_fit'.
    """
    # Extract x values and index positions
    x_vals = series.values
    idx = np.arange(len(series))

    # Default to center if z not given
    if z is None:
        z = len(series) // 2

    # Create Gaussian weights centered at z
    weights = np.exp(-0.5 * ((idx - z) / weight_sigma) ** 2)

    # Define linear model
    def linear(x, m, b):
        return m * x + b

    # Perform weighted curve fit
    popt, _ = curve_fit(
        linear,
        idx,
        x_vals,
        sigma=1 / (weights + 1e-8),  # inverse of weights as error
        absolute_sigma=False
    )

    # Compute fitted values
    return linear(idx, *popt)



def gaussian_2d(coords, x0, y0, sigma, amplitude, offset):
    x, y = coords
    return amplitude * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2)) + offset

def fit_gaussian(image, x, y, r, search_radius=5, search_r_range=5):
    h, w = image.shape

    # Define ROI
    pad = int(search_radius)
    x_min = max(0, int(x - pad))
    x_max = min(w, int(x + pad))
    y_min = max(0, int(y - pad))
    y_max = min(h, int(y + pad))

    subimg = image[y_min:y_max, x_min:x_max]
    if subimg.size == 0:
        raise ValueError("Empty subimage, adjust search_radius.")

    # Grid
    y_grid, x_grid = np.mgrid[y_min:y_max, x_min:x_max]
    x_data = np.vstack((x_grid.ravel(), y_grid.ravel()))
    y_data = subimg.ravel()

    # Initial guess
    guess = (
        x,                # x0
        y,                # y0
        r,                # sigma
        subimg.max(),     # amplitude
        np.median(subimg) # offset
    )

    # Constrain sigma within search_r_range around r
    sigma_min = r
    sigma_max = r + search_r_range

    bounds = (
        (x_min, y_min, sigma_min,      0,                0),
        (x_max, y_max, sigma_max, 2 * subimg.max(), subimg.max())
    )

    popt, _ = curve_fit(gaussian_2d, x_data, y_data, p0=guess, bounds=bounds)
    x_fit, y_fit, sigma_fit, amp_fit, offset = popt

    # Correlation
    fit = gaussian_2d(x_data, *popt)
    corr = np.corrcoef(fit, y_data)[0, 1]

    return x_fit, y_fit, sigma_fit, corr