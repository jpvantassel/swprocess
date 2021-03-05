"""A setuptools based setup module."""

from setuptools import setup, find_packages

meta = {}
with open("swprocess/meta.py") as f:
    exec(f.read(), meta)

with open('README.md', encoding="utf8") as f:
    long_description = f.read()

setup(
    name='swprocess',
    version=meta['__version__'],
    description='Package for Surface Wave Processing',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/jpvantassel/swprocess',
    author='Joseph P. Vantassel',
    author_email='jvantassel@utexas.edu',
    classifiers=[
        'Development Status :: 4 - Beta',

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
    python_requires='>=3.6, <3.9',
    install_requires=["numpy", "scipy", "matplotlib",
                      "obspy", "sigpropy>=0.3.0", "pandas", "PyQt5"],
    extras_require={
        'dev': ['coverage'],
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
