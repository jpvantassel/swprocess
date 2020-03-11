"""A setuptools based setup module.

See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

from setuptools import setup, find_packages

with open('README.md', encoding="utf8") as f:
    long_description = f.read()

setup(
    name='utprocess',
    version='0.1.0',
    description='Package for Surface-Wave Processing',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/jpvantassel/utprocess',
    author='Joseph P. Vantassel',
    author_email='jvantassel@utexas.edu',
    classifiers=[
        'Development Status :: 3 - Alpha',

        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',

        'Topic :: Education',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics',

        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',

        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    keywords='surface-wave processing geopsy active passive',
    packages=find_packages(),
    python_requires = '>3.6',
    install_requires=["numpy", "scipy", "matplotlib", "obspy", "sigpropy"],
    extras_require={
        'dev': ['unittest', 'hypothesis'],
    },
    package_data={
    },
    data_files=[
    ],
    entry_points={  
    },
    project_urls={
    },
)