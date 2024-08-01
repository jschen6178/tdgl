"""
Note:
Not useful for current TDGL simulations as we are only working in 2D.

Please use fmfield_z.py
"""
import os
import tempfile
import pint

ureg = pint.UnitRegistry()
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from IPython.display import HTML, display
import h5py
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["figure.figsize"] = (5, 4)

import tdgl
from tdgl.geometry import box, circle
from tdgl.visualization.animate import create_animation


"""
Vector potential calcuations
"""
from scipy import interpolate
from tdgl import Parameter

# B_Y = np.reshape(B_Y, (np.shape(B_Y)[0] * np.shape(B_Y)[1], 1))

def textured_vector_potential(
    positions,
    By,
):
    """
    Calculates the magnetic vector potential [Ax, Ay, Az] at ``positions``
    due uniform magnetic field along the z-axis with strength ``Bz``.

    Args:
    positions: Shape (n, 3) array of (x, y, z) positions in meters at which to
        evaluate the vector potential.
    Bz: The strength of the the field with shape (m, m) with units of Tesla, where
    m is the size of the Mumax simulation

    Returns:
    Shape (n, 3) array of the vector potential [Ax, Ay, Az] at ``positions``
    in units of Tesla * meter.

    """
    # assert isinstance(Bz, (float, str, pint.Quantity)), type(Bz)
    # positions = np.atleast_2d(positions)
    # assert positions.shape[1] == 3, positions.shape
    # if not isinstance(positions, pint.Quantity):
    #     positions = positions * ureg("meter")
    # if isinstance(Bz, str):
    #     Bz = ureg(Bz)
    # if isinstance(Bz, float):
    #     Bz = Bz * ureg("tesla")


    # Assuming 'positions' is already defined as in the previous example
    # Extract the x and y values from the positions array

    xy_vals = positions[:, :2]
    
    # Calculate the range (peak-to-peak) of x and y values
    dx = np.ptp(xy_vals[:, 0])
    dy = np.ptp(xy_vals[:, 1])
    # Calculate the center point for x and y
    center_x = np.min(xy_vals[:, 0]) + dx / 2
    center_y = np.min(xy_vals[:, 1]) + dy / 2
    center = np.array([center_x, center_y])
    # Subtract the center point from all xz values to center the data
    xy_vals_centered = xy_vals - center
    centered_xs = xy_vals_centered[:, [0]]
    centered_ys = xy_vals_centered[:, [1]]
    # make a grid equally sized as the positions but with spacings equivalent to the Mumax mesh
    grid_xs = np.linspace(centered_xs.min(), centered_xs.max(), np.shape(By)[0])
    grid_ys = np.linspace(centered_ys.min(), centered_ys.max(), np.shape(By)[1])
    X,Y = np.meshgrid(grid_xs, grid_ys)
    By_points = np.vstack([X.ravel(), Y.ravel()]).T
    
    # reshape Bz from 128x128 in Mumax into 128^2 by 2
    flattened_By_values = np.reshape(By, (np.shape(By)[0] * np.shape(By)[1], 1))

    # interpolate to find By at positions
    interpolated_By = interpolate.griddata(By_points, flattened_By_values, xy_vals_centered)
    interpolated_By = interpolated_By*ureg("tesla")
    
    centered_xs = centered_xs*ureg("meter")

    # x-y component of vector potential
    Az = (interpolated_By * -1*centered_xs)
    
    A = np.hstack([np.zeros_like(Az), np.zeros_like(Az), Az])
    
    A = A.to("tesla * meter")
    
    return A

def FM_field_vector_potential(
    x,
    y,
    z,
    *,
    field_units: str = "T",
    length_units: str = "um",
):
    CURRENT_DIRECTORY = os.getcwd()
    DATA_AND_LAYER_NAME = "B_demag_75mT_0K_layer2"
    DEMAG_B_Y_FILEPATH = os.path.join(CURRENT_DIRECTORY, "mumax_fields", "%s_y.npy" % DATA_AND_LAYER_NAME)
    DEMAG_B_Y = np.load(DEMAG_B_Y_FILEPATH)
    
    #################################################
    # CHANGE THIS DEPENDING ON APPLIED FIELD IN RUN #
    #################################################
    APPLIED_B_Y = 0.0

    B_Y = DEMAG_B_Y + APPLIED_B_Y
    if z.ndim == 0:
        z = z * np.ones_like(x)
    positions = np.array([x.squeeze(), y.squeeze(), z.squeeze()]).T
    positions = (positions * ureg(length_units)).to("m").magnitude
    By = B_Y * ureg(field_units)
    A = textured_vector_potential(positions, By)
    return A.to(f"{field_units} * {length_units}").magnitude

def FMField(
    field_units: str = "T", length_units: str = "um"
) -> Parameter:
    """Returns a Parameter that computes a constant as a function of ``x, y, z``.
    Args:
        value: The constant value of the field.
    Returns:
        A Parameter that returns ``value`` at all ``x, y, z``.
    """
    return Parameter(
        FM_field_vector_potential,
        field_units=field_units,
        length_units=length_units,
    )
