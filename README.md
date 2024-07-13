# _swprocess_ - A Python Package for Surface Wave Processing

> Joseph P. Vantassel, [jpvantassel.com](https://www.jpvantassel.com/)

[![DOI](https://zenodo.org/badge/202217252.svg)](https://zenodo.org/badge/latestdoi/202217252)
[![PyPI - License](https://img.shields.io/pypi/l/swprocess)](https://github.com/jpvantassel/swprocess/blob/main/LICENSE.txt)
[![CircleCI](https://circleci.com/gh/jpvantassel/swprocess.svg?style=svg)](https://circleci.com/gh/jpvantassel/swprocess)
[![Documentation Status](https://readthedocs.org/projects/swprocess/badge/?version=latest)](https://swprocess.readthedocs.io/en/latest/?badge=latest)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/swprocess)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/8faa1913edd84e4b9ba77807ab5583fd)](https://www.codacy.com/gh/jpvantassel/swprocess/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=jpvantassel/swprocess&amp;utm_campaign=Badge_Grade)
[![codecov](https://codecov.io/gh/jpvantassel/swprocess/branch/main/graph/badge.svg?token=XCDW6HMGBR)](https://codecov.io/gh/jpvantassel/swprocess)

## Table of Contents

-   [About _swprocess_](#about-swprocess)
-   [Why use _swprocess_](#why-use-swprocess)
-   [Examples](#examples)
-   [Getting Started](#getting-started)

## About _swprocess_

_swprocess_ is a Python package for surface wave processing. _swprocess_ was
developed by [Joseph P. Vantassel](https://www.jpvantassel.com/) under the
supervision of Professor Brady R. Cox at The University of Texas at Austin.
_swprocess_ continues to be developed and maintained by
[Joseph P. Vantassel and his research group at Virginia Tech](https://geoimaging-research.org/).

If you use _swprocess_ in your research or consulting, we ask you please cite
the following:

> Vantassel, J. P. (2021). jpvantassel/swprocess: latest (Concept). Zenodo.
> [https://doi.org/10.5281/zenodo.4584128](https://doi.org/10.5281/zenodo.4584128)

> Vantassel, J. P. & Cox, B. R. (2022). "SWprocess: a workflow for developing robust
> estimates of surface wave dispersion uncertainty". Journal of Seismology.
> [https://doi.org/10.1007/s10950-021-10035-y](https://doi.org/10.1007/s10950-021-10035-y)

_Note: For software, version specific citations should be preferred to
general concept citations, such as that listed above. To generate a version
specific citation for _swprocess_, please use the citation tool on the _swprocess_
[archive](https://doi.org/10.5281/zenodo.4584128)._

## Why use _swprocess_

_swprocess_ contains features not currently available in any other open-source
software, including:

-   Multiple pre-processing workflows for active-source [i.e., Multichannel
Analysis of Surface Waves (MASW)] measurements including:
    -   time-domain muting,
    -   frequency-domain stacking, and
    -   time-domain stacking.
-   Multiple wavefield transformations for active-source (i.e., MASW) measurements
including:
    -   frequency-wavenumber (Nolet and Panza, 1976),
    -   phase-shift (Park, 1998),
    -   slant-stack (McMechan and Yedlin, 1981), and
    -   frequency domain beamformer (Zywicki 1999).
-   Post-processing of active-source and passive-wavefield [i.e., microtremor
array measurements (MAM)] data from _swprocess_ and _Geopsy_, respectively.
-   Interactive trimming to remove low quality dispersion data.
-   Rigorous calculation of dispersion statistics to quantify epistemic and
aleatory uncertainty in surface wave measurements.

## Examples

### Active-source processing

<img src="https://github.com/jpvantassel/swprocess/blob/main/figs/nz_wghs_rayleigh_-20.0m.png?raw=true" width="775">

### Interactive trimming

<img src="https://github.com/jpvantassel/swprocess/blob/main/figs/nz_wghs_rayleigh_masw_int-trim.gif?raw=true" width="775">

### Calculation of dispersion statistics

<img src="https://github.com/jpvantassel/swprocess/blob/main/figs/nz_wghs_rayleigh.png?raw=true" width="775">

## Getting Started

### Installing or Upgrading _swprocess_

1.  If you do not have Python 3.8 or later installed, you will need to do
so. A detailed set of instructions can be found
[here](https://jpvantassel.github.io/python3-course/#/intro/installing_python).

2.  If you have not installed _swprocess_ previously use `pip install swprocess`.
If you are not familiar with `pip`, a useful tutorial can be found
[here](https://jpvantassel.github.io/python3-course/#/intro/pip). If you have
an earlier version and would like to upgrade to the latest version of
_swprocess_ use `pip install swprocess --upgrade`.

3.  Confirm that _swprocess_ has installed/updated successfully by examining the
last few lines of the text displayed in the console.

### Using _swprocess_

1.  Download the contents of the
  [examples](https://github.com/jpvantassel/swprocess/tree/main/examples)
  directory to any location of your choice.

2.  Start by processing the provided active-source data using the
  Jupyter notebook (`masw.ipynb`). If you have not installed `Jupyter`,
  detailed instructions can be found
  [here](https://jpvantassel.github.io/python3-course/#/intro/installing_jupyter).

3.  Post-process the provided passive-wavefield data using the
  Jupyter notebook (`mam_fk.ipynb`).

4.  Perform interactive trimming and calculate dispersion statistics for the
  example data using the Jupyter notebook (`stats.ipynb`). Compare your results
  to those shown in the figure above.

5.  Enjoy!