"""A setuptools based setup module."""

from swprocess.__init__ import __version__
from setuptools import setup, find_packages

with open('README.md', encoding="utf8") as f:
    long_description = f.read()

setup(
    name='swprocess',
    version=__version__,
    description='Package for Surface-Wave Processing',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/jpvantassel/swprocess',
    author='Joseph P. Vantassel',
    author_email='jvantassel@utexas.edu',
    classifiers=[
        'Development Status :: 3 - Alpha',

        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',

        'Topic :: Education',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics',

        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',

        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    keywords='surface-wave dispersion processing geopsy active passive masw mam',
    packages=find_packages(),
    python_requires = '>3.6',
    install_requires=["numpy", "scipy", "matplotlib", "obspy", "sigpropy", "pandas", "xlrd"],
    extras_require={
        'dev': ['sphinx', 'sphinx_rtd_theme'],
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
        # 'Docs': 'https://swprocess.readthedocs.io/en/latest/?badge=latest',
    },
)