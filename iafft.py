#! /usr/bin/env python3
"""
iaaft - Iterated amplitude adjusted Fourier transform surrogates

        This module implements the iAAFT

[1] Bendito, E., Carmona, A., Encinas, A. M., & Gesto, J. M. Estimation of
    Fekete points (2007), J Comp. Phys. 225, pp 2354--2376  
    https://doi.org/10.1016/j.jcp.2007.03.017

"""
# Created: Tue Jun 22, 2021  09:44am
# Last modified: Tue Jun 22, 2021  09:44am
#
# Copyright (C) 2021  Bedartha Goswami <bedartha.goswami@uni-tuebingen.de> This
# program is free software: you can redistribute it and/or modify it under the
# terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------


import numpy as np
from scipy.spatial.distance import pdist
from tqdm import tqdm
from numba import jit
from scipy.spatial import SphericalVoronoi
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

G = 6.67408 * 1E-11         # m^3 / kg / s^2


def bendito(N=100, a=1., X=None, maxiter=1000, verbose=True):
    """
    Return the Fekete points according to the Bendito et al. (2007) algorithm.

    Parameters
    ----------
    N : int
        Number of points to be distributed on the surface of the unit sphere.
        Default is `N = 100`.
    a : float
        Positive scalar that weights the advance direction in accordance with
        the kernel under consideration and the surface (cf. Eq. 4 and Table 1
        of Bendito et al., 2007). Default is `a = 1` which corresponds to the
        Newtonian kernel.
    X : numpy.nadarray, with shape (N, 3)
        Initial configuration of points. The array consists of N observations
        (rows) of 3-D (x, y, z) locations of the points. If provided, `N` is
        overriden and set to `X.shape[0]`. Default is `None`.
    maxiter : int
        Maximum number of iterations to carry out. Since the error of the
        configuration continues to decrease exponentially after a certain
        number of iterations, a saturation / convergence criterion is not
        implemented. Users are advised to check until the regime of exponential
        decreased is reach by trying out different high values of `maxiter`.
        Default is 1000.

    Returns
    -------
    X_new : numpy.ndarray, with shape (N, 3)
        Final configuration of `N` points on the surface of the sphere after
        `maxiter` iterations. Each row contains the (x, y, z) coordinates of
        the points. If `X` is provided, the `X_new` has the same shape as `X`.
    dq : numpy.ndarray, with shape (maxiter,)
        Maximum disequilibrium degree after each iteration. This is defined as
        the maximum of the modulus of the disequilibrium vectors at each point
        location. Intuitively, this can be understood as a quantity that is
        proportional to the total potential energy of the current configuration
        of points on the sphere's surface.

    """
     # parse inputs
    if len(X) == 0:
        print("Initial configuration not provided. Generating random one ...")
        X = points_on_sphere(N)         # initial random configuration
    else:
        N = X.shape[0]

    # core loop
    ## intializ parameters
    dq = []
    w = np.zeros(X.shape)
    ## set up progress bar
    pb_fmt = "{desc:<5.5}{percentage:3.0f}%|{bar:30}{r_bar}"
    pb_desc = "Estimating Fekete points ..."
    ## iterate
    for k in tqdm(range(maxiter), bar_format=pb_fmt, desc=pb_desc,
                  disable=not verbose):

        # Core steps from Bendito et al. (2007), pg 6 bottom
        ## 1.a. Advance direction
        for i in range(len(X)):
            w[i] = descent_direction_i(X, i)

        # 1.b. Error as max_i |w_i|
        mod_w = np.sqrt((w ** 2).sum(axis=1))
        dq.append(np.max(mod_w))

        ## 2.a. Minimum distance between all points
        d = np.min(pdist(X))
        ## 2.b. Calculate x^k_hat = x^k + a * d^{k-1} w^{k-1}
        Xhat = X + a * d * w

        ## 3. New configuration
        X_new = (Xhat.T / np.sqrt((Xhat ** 2).sum(axis=1))).T
        X = X_new

    return X_new, dq


@jit(nopython=True)
def descent_direction_i(X, i):
    """
    Returns the 3D vector for the direction of decreasing energy at point i.

    Parameters
    ----------
    X : numpy.nadarray, with shape (N, 3)
        Current configuration of points. Each row of `X` is the 3D position
        vector for the corresponding point in the current configuration.
    i : int
        Index of the point for which the descent direction is to be estimated.
        The position vector of point `i` is the i-th row of `X`.

    Returns
    -------
    wi : numpy.ndarray, with shape (3,)
         The vector along which the particle at point `i` has to be moved in
         order for the total potential energy of the overall configuration to
         decrease. The vector is estimated as the ratio of the tangential force
         experienced by the particle at `i` to the magnitude of the total force
         experienced by the particle at `i`. The tangential force is calculated
         as the difference between the total force and the component of the
         total force along the (surface) normal direction at `i`.

    """
    xi = X[i]

    # total force at i
    xi_arr = xi.repeat(X.shape[0]).reshape(xi.shape[0], X.shape[0]).T
    diff = xi_arr - X
    j = np.where(np.sum(diff, axis=1) != 0)[0]
    diff_j = diff[j]
    denom = (np.sqrt(np.square(diff_j).sum(axis=1))) ** 3
    numer = (G * diff_j)
    Fi_tot = np.sum((numer.T / denom).T, axis=0)    # gives 3D net force vector

    # direction of descent towards lower energy
    xi_n = xi / np.sqrt(np.square(xi).sum())
    Fi_n = (Fi_tot * xi_n).sum() * xi_n
    Fi_T = Fi_tot - Fi_n
    wi = Fi_T / np.sqrt(np.square(Fi_tot).sum())

    return wi


def points_on_sphere(N, r=1.):
    """
    Returns random points on the surface of a 3D sphere.

    Parameters
    ----------
    N : int
        Number of points to be distributed randomly on sphere's surface
    r : float
        Positive number denoting the radius of the sphere. Default is `r = 1`.

    Returns
    -------
    X : numpy.ndarray, with shape (N, 3)
        Locations of the `N` points on the surface of the sphere of radius `r`.
        The i-th row in `X` is a 3D vector that gives the location of the i-th
        point.
    """
    phi = np.arccos(1. - 2. * np.random.rand(N))
    theta = 2. * np.pi * np.random.rand(N)
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np. sin(phi) * np.sin(theta)
    z = r * np.cos(phi)

    return np.c_[x, y, z]


def cartesian_to_spherical(X):
    """
    Returns spherical coordinates for a given array of Cartesian coordinates.

    Parameters
    ----------
    X : numpy.ndarray, with shape (N, 3)
        Locations of the `N` points on the surface of the sphere of radius `r`.
        The i-th row in `X` is a 3D vector that gives the location of the i-th
        point.

    Returns
    -------
    theta : numpy.ndaaray, with shape (N,)
        Azimuthal angle of the different points on the sphere. Values are
        between (0, 2pi). In geographical terms, this corresponds to the
        longitude of each location.
    phi : numpy.ndaaray, with shape (N,)
        Polar angle (or inclination) of the different points on the sphere.
        Values are between (0, pi). In geographical terms, this corresponds to
        the latitude of each location.
    r : float
        Radial distance of the points to the center of the sphere. Always
        greater than or equal to zero.
    """
    r = np.sqrt(np.square(X).sum(axis=1))   # radius
    theta = np.arccos(X[:, 2] / r)          # azimuthal angle
    phi = np.arctan(X[:, 1] / X[:, 0])      # polar angle (inclination)

    return theta, phi, r


def spherical_to_cartesian(theta, phi, r=1.):
    """
    Returns Cartesian coordinates for a given array of spherical coordinates.


    Parameters
    ----------
    theta : numpy.ndaaray, with shape (N,)
        Azimuthal angle of the different points on the sphere. Values are
        between (0, 2pi). In geographical terms, this corresponds to the
        longitude of each location.
    phi : numpy.ndaaray, with shape (N,)
        Polar angle (or inclination) of the different points on the sphere.
        Values are between (0, pi). In geographical terms, this corresponds to
        the latitude of each location.
    r : float
        Radial distance of the points to the center of the sphere. Always
        greater than or equal to zero. Default is `r = 1`.

    Returns
    -------
    X : numpy.ndarray, with shape (N, 3)
        Locations of the `N` points on the surface of the sphere of radius `r`.
        The i-th row in `X` is a 3D vector that gives the location of the i-th
        point in `(x, y, z)` coordinates.

    """
    x = r * np.cos(phi) * np.sin(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(theta)
    X = np.c_[x, y, z]

    return X


def plot_spherical_voronoi(X, ax):
    """
    Plot scipy.spatial.SphericalVoronoi output on the surface of a unit sphere.

    Parameters
    ----------
    X : numpy.ndarray, with shape (N, 3)
        Locations of the `N` points on the surface of the sphere of radius `r`.
        The i-th row in `X` is a 3D vector that gives the location of the i-th
        point in `(x, y, z)` coordinates.
    ax : matplotlib.pyplot.Axes
        Axis in which the Voronoi tessellation output is to be plotted.

    Returns
    -------
    ax : matplotlib.pyplot.Axes
        The same axis object used for plotting is returned.

    """
    vor = SphericalVoronoi(X)
    vor.sort_vertices_of_regions()
    verts = vor.vertices
    regs = vor.regions
    for i in range(X.shape[0]):
        verts_reg = np.array([verts[k] for k in regs[i]])
        verts_reg = [list(zip(verts_reg[:, 0], verts_reg[:, 1], verts_reg[:, 2]))]
        ax.add_collection3d(Poly3DCollection(verts_reg,
                                             facecolors="w",
                                             edgecolors="steelblue"
                                             ),
                            )
    ax.set_xlim(-1.01, 1.01)
    ax.set_ylim(-1.01, 1.01)
    ax.set_zlim(-1.01, 1.01)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2],
               marker=".", color="indianred", depthshade=True, s=40)
    return ax

