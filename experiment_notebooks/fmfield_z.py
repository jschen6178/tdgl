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
from tdgl.sources import LinearRamp

"""
Vector potential calcuations
"""
from scipy import interpolate
from tdgl import Parameter

# B_Z = np.reshape(B_Z, (np.shape(B_Z)[0] * np.shape(B_Z)[1], 1))

def textured_vector_potential(
    positions,
    Bz,
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
    # Subtract the center point from all xy values to center the data
    xy_vals_centered = xy_vals - center
    centered_xs = xy_vals_centered[:, 0]
    centered_ys = xy_vals_centered[:, 1]
    # make a grid equally sized as the positions but with spacings equivalent to the Mumax mesh
    grid_xs = np.linspace(centered_xs.min(), centered_xs.max(), np.shape(Bz)[0])
    grid_ys = np.linspace(centered_ys.min(), centered_ys.max(), np.shape(Bz)[1])
    X,Y = np.meshgrid(grid_xs, grid_ys)
    Bz_points = np.vstack([X.ravel(), Y.ravel()]).T
    
    # reshape Bz from 128x128 in Mumax into 128^2 by 2
    flattened_Bz_values = np.reshape(Bz, (np.shape(Bz)[0] * np.shape(Bz)[1], 1))

    # interpolate to find Bz at positions
    interpolated_Bz = interpolate.griddata(Bz_points, flattened_Bz_values, xy_vals_centered)
    interpolated_Bz = interpolated_Bz*ureg("tesla")
    centered_ys = centered_ys*ureg("meter")
    centered_xs = centered_xs*ureg("meter")

    # x-y component of vector potential
    Axy = 1/2*interpolated_Bz * np.stack([-1*centered_ys, centered_xs], axis=1)
    # 1/2 to cancel out the double counting
    
    A = np.hstack([Axy, np.zeros_like(Axy[:,:1])])
    
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
    CURRENT_DIRECTORY = os.path.dirname(os.getcwd())
    DATA_AND_LAYER_NAME = "B_demag_150mT_10K_FeCo_5_5_7Stack_layer7"
    DEMAG_B_Z_FILEPATH = os.path.join(CURRENT_DIRECTORY, "mumax_fields", "%s_z.npy" % DATA_AND_LAYER_NAME)
    DEMAG_B_Z = np.load(DEMAG_B_Z_FILEPATH)
    
    #################################################
    # CHANGE THIS DEPENDING ON APPLIED FIELD IN RUN #
    #################################################
    APPLIED_B_Z = 0.150

    B_Z = DEMAG_B_Z + APPLIED_B_Z
    if z.ndim == 0:
        z = z * np.ones_like(x)
    positions = np.array([x.squeeze(), y.squeeze(), z.squeeze()]).T
    positions = (positions * ureg(length_units)).to("m").magnitude
    Bz = B_Z * ureg(field_units)
    A = textured_vector_potential(positions, Bz)
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
