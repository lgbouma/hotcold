"""
Semi-working 3D Gaussian blob orbiting a star.

(by "semi" this means that the light curve and RV curve calculation seems ok.  but the viz is out of whack, and this approach is probably WAY too complex.)
"""
import numpy as np
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
from astropy import constants as const
from astropy import units as u
import math

# Global constants
T_SEC = 3.93 * 3600  # period in seconds
OMEGA = 2 * np.pi / T_SEC  # angular velocity (rad/s)
R_STAR = const.R_sun  # assume star radius = 1 R_star = 1 R_sun


def compute_blob_rotation(blob_center: np.ndarray,
                          tilt: float) -> np.ndarray:
    """Compute rotation matrix for blob's local frame.

    The blob's local frame is defined so that:
      - The x' axis points radially from the star to the blob center.
      - The z' axis is aligned with the orbital angular momentum.
      - The y' axis = cross(z', x').

    Generated by ChatGPT o3-mini-high on March 31 2025.

    Args:
        blob_center (np.ndarray): 3-element array of blob center coordinates.
        tilt (float): Tilt angle in radians (0 means orbit in xy plane).

    Returns:
        np.ndarray: 3x3 rotation matrix with columns [x', y', z'].
    """
    # x' axis: unit vector from star to blob center.
    r_norm = np.linalg.norm(blob_center)
    if r_norm == 0:
        x_prime = np.array([1, 0, 0])
    else:
        x_prime = blob_center / r_norm
    # Orbital angular momentum vector.
    # For an orbit tilted by 'tilt' about the y-axis:
    z_prime = np.array([np.sin(tilt), 0, np.cos(tilt)])
    # y' axis:
    y_prime = np.cross(z_prime, x_prime)
    y_norm = np.linalg.norm(y_prime)
    if y_norm == 0:
        y_prime = np.array([0, 1, 0])
    else:
        y_prime = y_prime / y_norm
    # Recompute z_prime for orthonormality.
    z_prime = np.cross(x_prime, y_prime)
    R = np.column_stack((x_prime, y_prime, z_prime))
    return R


def compute_blob_density(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                         blob_center: np.ndarray, sigma_x: float,
                         sigma_y: float, sigma_z: float,
                         R: np.ndarray) -> np.ndarray:
    """Compute Gaussian blob density at given points.

    The density in the blob's local frame is defined as:
      exp[-0.5*((x')^2/sigma_x^2 + (y')^2/sigma_y^2 +
                (z')^2/sigma_z^2)].

    Generated by ChatGPT o3-mini-high on March 31 2025.

    Args:
        x (np.ndarray): x coordinates (in R_star units).
        y (np.ndarray): y coordinates.
        z (np.ndarray): z coordinates.
        blob_center (np.ndarray): Blob center (in R_star units).
        sigma_x (float): Gaussian width along blob x' axis.
        sigma_y (float): Gaussian width along blob y' axis.
        sigma_z (float): Gaussian width along blob z' axis.
        R (np.ndarray): 3x3 rotation matrix for blob's local frame.

    Returns:
        np.ndarray: Density values at the given points.
    """
    dx = x - blob_center[0]
    dy = y - blob_center[1]
    dz = z - blob_center[2]
    coords = np.vstack((dx.flatten(), dy.flatten(), dz.flatten()))
    # Transform to blob-local coordinates.
    local = np.linalg.solve(R, coords)
    local_x = local[0].reshape(x.shape)
    local_y = local[1].reshape(x.shape)
    local_z = local[2].reshape(x.shape)
    exponent = -0.5 * (((local_x / sigma_x) ** 2) +
                       ((local_y / sigma_y) ** 2) +
                       ((local_z / sigma_z) ** 2))
    density = np.exp(exponent)
    return density


def compute_observer_view(a_blob: float, sigma_x: float, sigma_y: float,
                          sigma_z: float, phi: float, tilt: float,
                          n_x: int = 50, n_y: int = 50,
                          n_z: int = 50) -> tuple:
    """Compute observer view image and overall flux and RV.

    The observer is at infinity along +x. The image is a 2D map on the
    sky plane (y–z) of the integrated emission (density²) along x,
    applying obscuration by the star (radius = 1). Also computes the total
    flux and intensity-weighted radial velocity.

    Generated by ChatGPT o3-mini-high on March 31 2025.

    Args:
        a_blob (float): Blob orbital radius (in R_star).
        sigma_x (float): Gaussian width along blob x' axis.
        sigma_y (float): Gaussian width along blob y' axis.
        sigma_z (float): Gaussian width along blob z' axis.
        phi (float): Blob central orbital angle in radians.
        tilt (float): Tilt angle in radians.
        n_x (int, optional): Number of integration points along x.
            Defaults to 50.
        n_y (int, optional): Number of grid points along y.
            Defaults to 50.
        n_z (int, optional): Number of grid points along z.
            Defaults to 50.

    Returns:
        tuple: (I_image, v_image, y_grid, z_grid, total_flux,
                flux_weighted_rv)
    """
    L = a_blob + 3 * max(sigma_x, sigma_y, sigma_z) + 1
    y_vals = np.linspace(-L, L, n_y)
    z_vals = np.linspace(-L, L, n_z)
    y_grid, z_grid = np.meshgrid(y_vals, z_vals, indexing='ij')
    x_min = -L
    x_max = L
    # Common x grid (will adjust lower limit per pixel)
    x_lin = np.linspace(x_min, x_max, n_x)
    # Compute blob center with tilt.
    x_center = a_blob * np.cos(phi) * np.cos(tilt)
    y_center = a_blob * np.sin(phi)
    z_center = -a_blob * np.cos(phi) * np.sin(tilt)
    blob_center = np.array([x_center, y_center, z_center])
    R = compute_blob_rotation(blob_center, tilt)
    I_image = np.zeros((n_y, n_z))
    v_image = np.zeros((n_y, n_z))
    total_flux = 0.0
    flux_v_sum = 0.0
    dx = (x_max - x_min) / (n_x - 1)
    for i in range(n_y):
        for j in range(n_z):
            y_pt = y_grid[i, j]
            z_pt = z_grid[i, j]
            if y_pt**2 + z_pt**2 <= 1:
                x_lower = np.sqrt(1 - y_pt**2 - z_pt**2)
            else:
                x_lower = x_min
            x_vals = np.linspace(x_lower, x_max, n_x)
            y_arr = np.full_like(x_vals, y_pt)
            z_arr = np.full_like(x_vals, z_pt)
            density = compute_blob_density(x_vals, y_arr, z_arr, blob_center,
                                           sigma_x, sigma_y, sigma_z, R)
            emission = density**2
            I_pix = np.trapz(emission, x=x_vals)
            if I_pix > 0:
                v_pix = (np.trapz(emission * 
                          compute_line_of_sight_v(x_vals, y_arr, z_arr, tilt),
                          x=x_vals) / I_pix)
            else:
                v_pix = 0.0
            I_image[i, j] = I_pix
            v_image[i, j] = v_pix
            total_flux += I_pix * dx
            flux_v_sum += I_pix * v_pix * dx
    if total_flux > 0:
        flux_weighted_rv = flux_v_sum / total_flux
    else:
        flux_weighted_rv = 0.0
    return I_image, v_image, y_grid, z_grid, total_flux, flux_weighted_rv


def compute_line_of_sight_v(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                            tilt: float) -> np.ndarray:
    """Compute line-of-sight velocity (x component) at given points.

    The velocity is given by v = Omega x r (with Omega determined by tilt)
    and the observer is along +x.

    Generated by ChatGPT o3-mini-high on March 31 2025.

    Args:
        x (np.ndarray): x coordinates (in R_star units).
        y (np.ndarray): y coordinates.
        z (np.ndarray): z coordinates.
        tilt (float): Tilt angle in radians.

    Returns:
        np.ndarray: Radial velocity in km/s.
    """
    x_phys = x * R_STAR.value
    y_phys = y * R_STAR.value
    z_phys = z * R_STAR.value
    Omega_vec = np.array([OMEGA * np.sin(tilt), 0,
                          OMEGA * np.cos(tilt)])
    r_phys = np.vstack((x_phys, y_phys, z_phys))
    # Compute cross product; v = Omega x r.
    v_field = np.cross(Omega_vec, r_phys, axisa=0, axisb=0).T
    v_obs = v_field[:, 0]  # x component
    return v_obs / 1000.0  # convert m/s to km/s


def compute_topdown_view(a_blob: float, sigma_x: float, sigma_y: float,
                         sigma_z: float, phi: float, tilt: float,
                         n_x: int = 50, n_y: int = 50,
                         n_z: int = 50) -> tuple:
    """Compute top-down view image (projection onto the xy plane).

    The top-down view is the emission (density²) integrated along z,
    with no obscuration applied.

    Generated by ChatGPT o3-mini-high on March 31 2025.

    Args:
        a_blob (float): Blob orbital radius (in R_star).
        sigma_x (float): Gaussian width along blob x' axis.
        sigma_y (float): Gaussian width along blob y' axis.
        sigma_z (float): Gaussian width along blob z' axis.
        phi (float): Blob central orbital angle in radians.
        tilt (float): Tilt angle in radians.
        n_x (int, optional): Number of grid points along x.
            Defaults to 50.
        n_y (int, optional): Number of grid points along y.
            Defaults to 50.
        n_z (int, optional): Number of integration points along z.
            Defaults to 50.

    Returns:
        tuple: (I_top, x_grid, y_grid) where I_top is the integrated
               emission.
    """
    L = a_blob + 3 * max(sigma_x, sigma_y, sigma_z) + 1
    x_vals = np.linspace(-L, L, n_x)
    y_vals = np.linspace(-L, L, n_y)
    x_grid, y_grid = np.meshgrid(x_vals, y_vals, indexing='ij')
    z_min = -L
    z_max = L
    z_lin = np.linspace(z_min, z_max, n_z)
    dz = (z_max - z_min) / (n_z - 1)
    x_center = a_blob * np.cos(phi) * np.cos(tilt)
    y_center = a_blob * np.sin(phi)
    z_center = -a_blob * np.cos(phi) * np.sin(tilt)
    blob_center = np.array([x_center, y_center, z_center])
    R = compute_blob_rotation(blob_center, tilt)
    I_top = np.zeros_like(x_grid)
    for i in range(x_grid.shape[0]):
        for j in range(x_grid.shape[1]):
            x_pt = x_grid[i, j]
            y_pt = y_grid[i, j]
            x_arr = np.full_like(z_lin, x_pt)
            y_arr = np.full_like(z_lin, y_pt)
            density = compute_blob_density(x_arr, y_arr, z_lin, blob_center,
                                           sigma_x, sigma_y, sigma_z, R)
            emission = density**2
            I_top[i, j] = np.trapz(emission, x=z_lin)
    return I_top, x_grid, y_grid


def compute_lightcurve_rv(a_blob: float, sigma_x: float, sigma_y: float,
                          sigma_z: float, tilt: float,
                          num_phi: int = 180, n_x: int = 30,
                          n_y: int = 30, n_z: int = 30) -> tuple:
    """Compute light and radial velocity curves over a full orbit.

    For each orbital phase (blob central angle), computes the total flux and
    intensity-weighted radial velocity from the observer view integration.

    Generated by ChatGPT o3-mini-high on March 31 2025.

    Args:
        a_blob (float): Blob orbital radius (in R_star).
        sigma_x (float): Gaussian width along blob x' axis.
        sigma_y (float): Gaussian width along blob y' axis.
        sigma_z (float): Gaussian width along blob z' axis.
        tilt (float): Tilt angle in radians.
        num_phi (int, optional): Number of phase points (0 to 2π).
            Defaults to 180.
        n_x (int, optional): Number of integration points along x.
            Defaults to 30.
        n_y (int, optional): Number of grid points along y.
            Defaults to 30.
        n_z (int, optional): Number of grid points along z.
            Defaults to 30.

    Returns:
        tuple: (phi_vals_deg, flux_vals, rv_vals) where phi_vals_deg is in
               degrees and rv_vals in km/s.
    """
    phi_vals = np.linspace(0, 2 * np.pi, num_phi)
    flux_vals = []
    rv_vals = []
    for phi in phi_vals:
        _, _, _, _, tot_flux, flux_rv = compute_observer_view(
            a_blob, sigma_x, sigma_y, sigma_z, phi, tilt,
            n_x=n_x, n_y=n_y, n_z=n_z)
        flux_vals.append(tot_flux)
        rv_vals.append(flux_rv)
    phi_vals_deg = np.rad2deg(phi_vals)
    return phi_vals_deg, flux_vals, rv_vals


app = dash.Dash(__name__)
app.layout = html.Div([
    html.Div([
        html.Div([
            dcc.Graph(id='observer-view')
        ], style={'width': '48%', 'display': 'inline-block'}),
        html.Div([
            dcc.Graph(id='topdown-view')
        ], style={'width': '48%', 'display': 'inline-block'})
    ]),
    html.Div([
        html.Div([
            dcc.Graph(id='lightcurve')
        ], style={'width': '48%', 'display': 'inline-block'}),
        html.Div([
            dcc.Graph(id='rvcurve')
        ], style={'width': '48%', 'display': 'inline-block'})
    ]),
    html.Div([
        html.Label("Blob Orbital Radius (a_blob in R_star)"),
        dcc.Slider(
            id='a_blob-slider', min=1, max=5, step=0.1, value=2.4,
            marks={i: str(i) for i in range(1, 6)}
        ),
        html.Label("Sigma X (in R_star)"),
        dcc.Slider(
            id='sigma_x-slider', min=0.1, max=1.0, step=0.05, value=0.3,
            marks={0.1: "0.1", 0.5: "0.5", 1.0: "1.0"}
        ),
        html.Label("Sigma Y (in R_star)"),
        dcc.Slider(
            id='sigma_y-slider', min=0.1, max=1.0, step=0.05, value=0.3,
            marks={0.1: "0.1", 0.5: "0.5", 1.0: "1.0"}
        ),
        html.Label("Sigma Z (in R_star)"),
        dcc.Slider(
            id='sigma_z-slider', min=0.1, max=1.0, step=0.05, value=0.3,
            marks={0.1: "0.1", 0.5: "0.5", 1.0: "1.0"}
        ),
        html.Label("Central Blob Angle (degrees)"),
        dcc.Slider(
            id='phi-slider', min=0, max=360, step=1, value=0,
            marks={0: "0", 180: "180", 360: "360"}
        ),
        html.Label("Tilt Angle (degrees)"),
        dcc.Slider(
            id='tilt-slider', min=-90, max=90, step=1, value=0,
            marks={-90: "-90", 0: "0", 90: "90"}
        )
    ], style={'width': '90%', 'padding': '20px', 'margin': 'auto'})
])


@app.callback(
    [Output('observer-view', 'figure'),
     Output('topdown-view', 'figure'),
     Output('lightcurve', 'figure'),
     Output('rvcurve', 'figure')],
    [Input('a_blob-slider', 'value'),
     Input('sigma_x-slider', 'value'),
     Input('sigma_y-slider', 'value'),
     Input('sigma_z-slider', 'value'),
     Input('phi-slider', 'value'),
     Input('tilt-slider', 'value')]
)
def update_figures(a_blob_val, sigma_x_val, sigma_y_val, sigma_z_val,
                   phi_deg, tilt_deg):
    """Update all four panels based on slider inputs.

    Computes the observer view, top-down view, light curve, and radial
    velocity curve from the blob parameters.

    Generated by ChatGPT o3-mini-high on March 31 2025.

    Args:
        a_blob_val (float): Blob orbital radius in R_star.
        sigma_x_val (float): Sigma_x (in R_star).
        sigma_y_val (float): Sigma_y (in R_star).
        sigma_z_val (float): Sigma_z (in R_star).
        phi_deg (float): Central blob angle in degrees.
        tilt_deg (float): Tilt angle in degrees.

    Returns:
        tuple: Four Plotly figure dictionaries.
    """
    phi_rad = np.deg2rad(phi_deg)
    tilt_rad = np.deg2rad(tilt_deg)
    (I_image, _, y_grid, z_grid, _, _) = compute_observer_view(
        a_blob_val, sigma_x_val, sigma_y_val, sigma_z_val, phi_rad,
        tilt_rad, n_x=30, n_y=30, n_z=30)
    observer_fig = go.Figure(
        data=go.Heatmap(
            z=I_image,
            x=y_grid[0, :],
            y=z_grid[:, 0],
            colorscale='Viridis',
            colorbar=dict(title='Intensity')
        )
    )
    observer_fig.update_layout(
        title="Observer View (Sky Plane, y–z)",
        xaxis_title="y (R_star)",
        yaxis_title="z (R_star)",
        yaxis=dict(scaleanchor="x", scaleratio=1)
    )
    (I_top, x_top, y_top) = compute_topdown_view(
        a_blob_val, sigma_x_val, sigma_y_val, sigma_z_val, phi_rad,
        tilt_rad, n_x=30, n_y=30, n_z=30)
    topdown_fig = go.Figure(
        data=go.Heatmap(
            z=I_top,
            x=x_top[0, :],
            y=y_top[:, 0],
            colorscale='Viridis',
            colorbar=dict(title='Intensity')
        )
    )
    theta_circle = np.linspace(0, 2 * np.pi, 100)
    star_x = np.cos(theta_circle)
    star_y = np.sin(theta_circle)
    topdown_fig.add_trace(go.Scatter(
        x=star_x, y=star_y, mode='lines',
        line=dict(color='red'),
        name='Star'
    ))
    topdown_fig.update_layout(
        title="Top-Down View (xy Plane)",
        xaxis_title="x (R_star)",
        yaxis_title="y (R_star)",
        yaxis=dict(scaleanchor="x", scaleratio=1)
    )
    (phi_vals_deg, flux_vals, rv_vals) = compute_lightcurve_rv(
        a_blob_val, sigma_x_val, sigma_y_val, sigma_z_val, tilt_rad,
        num_phi=90, n_x=20, n_y=20, n_z=20)
    lightcurve_fig = go.Figure()
    lightcurve_fig.add_trace(go.Scatter(
        x=phi_vals_deg, y=flux_vals, mode='lines', name='Flux'
    ))
    lightcurve_fig.add_shape(
        type="line",
        x0=phi_deg, x1=phi_deg,
        y0=min(flux_vals), y1=max(flux_vals),
        line=dict(color="red", dash="dash")
    )
    lightcurve_fig.update_layout(
        title="Light Curve",
        xaxis_title="Orbital Phase (deg)",
        yaxis_title="Integrated Flux (arb. units)"
    )
    rvcurve_fig = go.Figure()
    rvcurve_fig.add_trace(go.Scatter(
        x=phi_vals_deg, y=rv_vals, mode='lines',
        name='Radial Velocity'
    ))
    rvcurve_fig.add_shape(
        type="line",
        x0=phi_deg, x1=phi_deg,
        y0=min(rv_vals), y1=max(rv_vals),
        line=dict(color="red", dash="dash")
    )
    rvcurve_fig.update_layout(
        title="Radial Velocity Curve",
        xaxis_title="Orbital Phase (deg)",
        yaxis_title="Radial Velocity (km/s)"
    )
    return observer_fig, topdown_fig, lightcurve_fig, rvcurve_fig


if __name__ == '__main__':
    app.run(debug=True)
