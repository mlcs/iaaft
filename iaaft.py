#! /usr/bin/env python3
"""
iaaft - Iterative amplitude adjusted Fourier transform surrogates

        This module implements the IAAFT method [1] to generate time series
        surrogates (i.e. randomized copies of the original time series) which
        ensures that each randomised copy preserves the power spectrum of the
        original time series.

[1] Venema, V., Ament, F. & Simmer, C. A stochastic iterative amplitude
    adjusted Fourier Transform algorithm with improved accuracy (2006), Nonlin.
    Proc. Geophys. 13, pp. 321--328  
    https://doi.org/10.5194/npg-13-321-2006

"""
# Created: Tue Jun 22, 2021  09:44am
# Last modified: Tue Jun 22, 2021  10:47am
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
from tqdm import tqdm
from pyfftw.interfaces.scipy_fftpack import fft, ifft
from numba import jit


def surrogates(x, ns, verbose=False, TOL_PC=5.):
    """Returns NSURR number of iAAFT surrogates of given data"""
    # as per the steps given in Lancaster et al., Phys. Rep (2018)
    nx = x.shape[0]
    xs = np.zeros((ns, nx))
    MAX_ITER = 10000
    ii = np.arange(nx)

    # allocate functions to local variables to save function call overhead
    npexp = np.exp

    # get the fft of the original array
    x_amp = np.abs(fft(x))
    x_srt = np.sort(x)
    r_orig = np.argsort(x)

    pb_fmt = "{desc:<5.5}{percentage:3.0f}%|{bar:30}{r_bar}"
    pb_desc = "Estimating IAAFT surrogates ..."
    for k in tqdm(range(ns), bar_format=pb_fmt, desc=pb_desc,
                  disable=not verbose):

        # 1) Generate random shuffle of the data
        count = 0
        r_prev = np.random.permutation(ii)
        r_curr = r_orig
        z_n = x[r_prev]
        percent_unequal = 100.

        while (percent_unequal > TOL_PC) and (count < MAX_ITER):
            r_prev = r_curr

            # 2) FFT current iteration yk, and then invert it but while
            # replacing the amplitudes with the original amplitudes but
            # keeping the angles from the FFT-ed version of the random
            y_prev = z_n
            fft_prev = fft(y_prev)
            phi_prev = jitangle(fft_prev)
            e_i_phi = jitexp(phi_prev * 1j)
            z_n = ifft(x_amp * e_i_phi)

            # 3) rescale zk to the original distribution of x
            r_curr = np.argsort(z_n)
            z_n[r_curr] = x_srt.copy()
            percent_unequal = ((r_curr != r_prev).sum() * 100.) / nx

            # 4) repeat until number of unequal entries between r_curr and 
            # r_prev is less than TOL_PC percent
            count += 1

        if count >= (MAX_ITER - 1):
            print("maximum number of iterations reached!")

        xs[k] = np.real(z_n)

    return xs


@jit
def jitangle(arr):
    """angle"""
    return np.angle(arr)


@jit
def jitexp(arr):
    """exp"""
    return np.exp(arr)





