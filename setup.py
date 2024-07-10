"""A setuptools based setup module."""

from setuptools import setup, find_packages

def parse_meta(path_to_meta):
    with open(path_to_meta) as f:
        meta = {}
        for line in f.readlines():
            if line.startswith("__version__"):
                meta["__version__"] = line.split('"')[1]
    return meta

meta = parse_meta("swprocess/meta.py")

with open('README.md', encoding="utf8") as f:
    long_description = f.read()
    long_description.replace("nz_wghs_rayleigh_masw_int-trim.gif",
                             "nz_wghs_rayleigh_masw_int-trim.png")

setup(
    name='swprocess',
    version=meta['__version__'],
    description='Package for Surface Wave Processing',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/jpvantassel/swprocess',
    author='Joseph P. Vantassel',
    author_email='joseph.p.vantassel@gmail.com',
    classifiers=[
        'Development Status :: 4 - Beta',

        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',

        'Topic :: Education',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics',

        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',

        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    keywords='surface-wave dispersion processing geopsy active passive masw mam',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=["numpy", "scipy", "matplotlib",
                      "obspy", "sigpropy>=1.0.0", "pandas",
                      "swprepost", "PyQt5"],
    extras_require={
        'dev': ['coverage', 'tox', 'sphinx', 'sphinx_rtd_theme', 'jupyterlab'],
    },
    package_data={
    },
    data_files=[
    ],
    entry_points={
    },
    project_urls={
        'Bug Reports': 'https://github.com/jpvantassel/swprocess/issues',
        'Source': 'https://github.com/jpvantassel/swprocess',
        'Docs': 'https://swprocess.readthedocs.io/en/latest/?badge=latest',
    },
)
