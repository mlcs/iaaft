# Iterated Amplitude Adjusted Fourier Transform time series surrogates

This repo implements that iterative amplitude adjusted Fourier transform
method to generate time series surrogates (i.e randomised copies of a
given time series) which preserve the power spectrum (and consequently
the autocorrelation) of the original series. For a detailed discussion
on the method, we refer to the paper by Venema, Ament & Simmer (2006)
below.

![iaaft-surrogates-example](/example.png)

## Installation & Usage

+   **Installation:** The algorithm is implemented as a Python module
    with the idea that    you can simply download
    [`iaaft.py`](https://github.com/mlcs/iaaft/blob/fc9c622d15829a5fafe95b48b14b8f3e4bda0655/iaaft.py)
    in your project directory and import is a Python module with `import
    iaaft`.

+   **Prequisite Python packages**
    - `numpy`
    - `tqdm` (progress bar display)

+   **Usage:** Call
    [`iaaft.surrogates()`](https://github.com/mlcs/fekete/blob/421796cb23da0022cc28871696bd3b55ff52b77c/fekete.py#L42) with appropriate arguments.

+   **Example**: The script [`example.py`](/example.py) contains a
    simple example with an autoregressive time series of order 1 is
    given as in put and a total of 1000 IAAFT surrogates are generated
    (results shown in figure above). To run this script, it takes around
    35 secs on a Intel® Core™ i9-9880H CPU @ 2.30GHz.


## References
Venema, V., Ament, F. & Simmer, C. A stochastic iterative amplitude
adjusted Fourier Transform algorithm with improved accuracy (2006),
_Nonlin.  Proc. Geophys._ **13**, pp. 321--328  
[https://doi.org/10.5194/npg-13-321-2006](https://doi.org/10.5194/npg-13-321-2006)


## TODO

- [x] First working implementation
- [x] Optimize code for faster performance
- [x] Documentation
- [x] Example
- [x] Improve README (installation, license, usage, etc.)

## License

[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg?style=flat-square)](/LICENSE)

- Copyright © [Bedartha Goswami](https://machineclimate.de/people/goswami/).

## Issues?

If you find any issues simply open a bug report, or send an email to
[bedartha.goswami@uni-tuebingen.de](mailto:bedartha.goswami@uni-tuebingen.de)

