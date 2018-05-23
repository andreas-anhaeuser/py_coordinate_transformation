#!/usr/bin/python2
"""Helper functions to Vector class."""

import numpy as np

###################################################
# CONSTANTS                                       #
###################################################
pi = np.pi

###################################################
# CONVERTERS (CART, CYL, SPHERIC)                 #
###################################################
def car2sph(x, y, z):
    """Convert cartesian to spherical coordinates.

        Parameters
        ----------
        x, y, z : float or np.arrays of such
            right-handed cartesian coordinates

        Returns
        -------
        (r, phi, theta) : float or np.arrays of such
            right-handed spherical coordinates as defined in this module.
            Angles in rad.

        History
        -------
        2017-11-22 (AA): Created
    """
    # helper
    rho = np.sqrt(x**2 + y**2)

    # spherical coordinates
    r = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arctan2(y, x)
    theta = np.arctan2(z, rho)

    # normalize phi
    phi = np.mod(phi + pi, 2*pi) - pi

    return (r, phi, theta)

def sph2car(r, phi, theta):
    """Convert spherical to cartesian coordinates.

        Parameters
        ----------
        (r, phi, theta) : float or np.arrays of such
            right-handed spherical coordinates as defined in this module
            Angles in rad.

        Returns
        -------
        x, y, z : float or np.arrays of such
            right-handed cartesian coordinates

        History
        -------
        2017-11-22 (AA): Created
    """
    x = r * np.cos(theta) * np.cos(phi)
    y = r * np.cos(theta) * np.sin(phi)
    z = r * np.sin(theta)
    return (x, y, z)

def car2cyl(x, y, z):
    """Convert cartesian to cylindrical coordinates."""
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)

    # normalize phi
    phi = np.mod(phi + pi, 2*pi) - pi

    return rho, phi, z

def sph2cyl(r, phi, theta):
    """Convert spherical to cylindrical coordinates."""
    z = r * np.sin(theta)
    rho = r * np.cos(theta)
    return rho, phi, z

def cyl2car(rho, phi, z):
    """Convert cylindrical to cartesian coordinates."""
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y, z

def cyl2sph(rho, phi, z):
    """Convert cylindrical to sphercial coordinates."""
    theta = np.arctan2(z, rho)
    r = np.sqrt(rho**2 + z**2)
    return r, phi, theta

###################################################
# NORMALIZERS                                     #
###################################################
def normalize_cyl(rho, phi, z):
    """Bring cylindrical coordinates into canonical format.

        See 'Coordinate Systems' Section of module docstring.

        The function works on both scalars and arrays.
    """
    array = is_array(rho, phi, z)
    scalar = not array              # for more clarity later in the code

    # ============ normalize rho  ======================== #
    # (make it non-negative)
    if scalar and rho < 0:
        rho = -rho
        phi = phi + np.pi
    elif array:
        idx = rho < 0
        rho[idx] *= -1
        phi[idx] += np.pi
    # ==================================================== #

    # ========== normalize phi  ========================== #
    # bring it to [-pi, pi)
    phi = np.mod(phi + pi, 2*pi) - pi

    # special case: rho == 0
    if scalar and rho == 0:
        phi = 0.
    elif array:
        phi[rho==0] = 0.
    # ==================================================== #

    return (rho, phi, z)

def normalize_sph(r, phi, theta):
    """Bring sphercial coordinates into canonical format.

        See 'Coordinate Systems' Section of module docstring.

        The function works on both scalars and arrays.
    """
    array = is_array(r, phi, theta)
    scalar = not array              # for more clarity later in the code

    # ========== normalize r  ============================ #
    # (make it non-negative)
    if scalar and r < 0:
        r = -r
        phi += pi
        theta = -theta
    elif array:
        idx = r < 0
        r[idx] *= -1
        phi[idx] += pi
        theta[idx] *= -1
    # ==================================================== #

    # ========== normalize theta  ======================== #
    # bring theta to [-pi, pi)
    theta = np.mod(theta + pi, 2*pi) - pi

    # bring theta to [-pi/2, pi/2]
    if scalar:
        if theta > pi/2:
            phi += pi
            theta = pi - theta
        if theta < -pi/2:
            phi += pi
            theta = -pi - theta
    elif array:
        idx = theta > pi/2
        phi[idx] += pi
        theta[idx] *= -1
        theta[idx] += pi

        idx = theta < -pi/2
        phi[idx] += pi
        theta[idx] *= -1
        theta[idx] -= pi

    # special case: r == 0
    if scalar and r == 0:
        theta = 0.
    elif array:
        theta[r==0] = 0.
    # ==================================================== #

    # ========== normalize phi  ========================== #
    # bring phi to [-pi, pi)
    phi = np.mod(phi + pi, 2*pi) - pi

    # special case: r == 0
    if scalar and r == 0:
        phi = 0.
    elif array:
        phi[r==0] = 0.
    # ==================================================== #

    return (r, phi, theta)

###################################################
# HELPERS                                         #
###################################################
def is_array(*args):
    """Check whether input arguments are scalar or array.

        Helper function.

        If one of the input arguments is an array and not all others are arrays
        of the same shape, and Error is thrown.
    """
    init = False
    for arg in args:
        if not init:
            init = True
            array = isinstance(arg, np.ndarray)
            shape = np.shape(arg)
            continue

        if array:
            assert isinstance(arg, np.ndarray)
            assert np.shape(arg) == shape
        else:
            assert not isinstance(arg, np.ndarray)

    return array
