from setuptools import setup, find_packages

setup(
    name='magneupy',
    version='0.3.0',
    packages=find_packages(),
    install_requires=[
        'pymatgen',
        'periodictable',
        'uncertainties',
        'numpy',
        'lmfit',
        'h5py',
        'xarray',
        'scipy',
        'sh',
        'pandas',
        'tabulate',
        'matplotlib'
    ],
    python_requires='>=3.8',
    url='https://github.com/guygma/magneupy',
    license='MIT',
    author='Guy Marcus',
    author_email='guygma@gmail.com',
    description='Library for magnetic neutron diffraction analysis'
)
