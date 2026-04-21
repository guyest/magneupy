import pytest
import numpy as np
from magneupy.material import Atom

def test_atom_init():
    # Note: Atom expects elname as bytes in some parts of the original code,
    # but the code uses elname.decode() so it should probably be bytes.
    atom = Atom(elname=b"Fe", oxidation=b"3+")
    assert atom.element == b"Fe"
    assert atom.oxidation == b"3+"
    assert atom.label == b"Fe"

def test_atom_scattering_length():
    atom = Atom(elname=b"Fe", oxidation=b"3+")
    # Fe coherent scattering length is 9.45 fm.
    # getNeutronScatteringLength divides by 10.
    assert pytest.approx(atom.bc, 0.01) == 0.945
