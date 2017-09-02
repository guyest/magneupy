from distutils.core import setup

setup(
    name='magneupy',
    version='0.1.1',
    packages=['magneupy', 'magneupy.rep', 'magneupy.data', 'magneupy.helper', 'magneupy.crystal'],
    install_requires=['pymatgen','periodictable','uncertainties','numpy','lmfit','h5py','xarray','scipy'],
    url='https://github.com/guygma/magneupy',
    license='MIT',
    author='Guy Marcus',
    author_email='guygma@gmail.com',
    description='Library for magnetic neutron diffraction analysis'
)
