import pytest
import numpy as np
from magneupy.material import Crystal, NuclearStructure
import os

def test_mno_nuclear_structure():
    cif_path = os.path.join(os.path.dirname(__file__), "data", "MnO.cif")
    crystal = Crystal(cif=cif_path)

    import periodictable
    b_Mn = periodictable.Mn.neutron.b_c / 10.
    b_O = periodictable.O.neutron.b_c / 10.

    Q = np.array([[1, 1, 1], [2, 0, 0]])
    Fn2 = crystal.nuclear.getNuclearStructureFactor(Q=Q, squared=True)

    expected_F111_sq = (4 * (b_Mn - b_O))**2
    expected_F200_sq = (4 * (b_Mn + b_O))**2

    assert pytest.approx(Fn2[0], 0.001) == expected_F111_sq
    assert pytest.approx(Fn2[1], 0.001) == expected_F200_sq

def test_magnetic_structure_factor_fm():
    from magneupy.magnetic import MagAtom, MagneticStructure, MagAtomGroup
    from magneupy.material import Crystal

    cif_content = """data_Fe_FM
_cell_length_a 2.866
_cell_length_b 2.866
_cell_length_c 2.866
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P m -3 m'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Fe Fe 0.0 0.0 0.0
"""
    if not os.path.exists("tests/data"):
        os.makedirs("tests/data")
    with open("tests/data/Fe_simple.cif", "w") as f:
        f.write(cif_content)

    crystal = Crystal(cif="tests/data/Fe_simple.cif")
    mag_struct = MagneticStructure(magnames=["Fe"], parents=[crystal], qms=[np.array([0,0,0])])

    mag_atom = mag_struct.magatoms["mFe1"]
    mag_atom.mu = 2.2
    mag_atom.moment = np.array([[0, 0, 2.2]])

    Q = np.array([[1, 0, 0]])
    Fm2 = mag_struct.getMagneticStructureFactor(Qm=Q, squared=True)

    expected_Fm2 = 0.2596431331199003
    assert pytest.approx(Fm2[0], 1e-6) == expected_Fm2

def test_magnetic_structure_factor_afm_nio():
    from magneupy.magnetic import MagAtom, MagneticStructure, MagAtomGroup
    from magneupy.material import Crystal

    cif_path = "tests/data/NiO.cif"
    crystal = Crystal(cif=cif_path)

    mag_struct = MagneticStructure(magnames=["Ni"], parents=[crystal], qms=[np.array([0,0,0])])

    for i, (label, ma) in enumerate(mag_struct.magatoms.items()):
        if i < 2:
            ma.moment = np.array([[0,0,2.0]])
            ma.mu = 2.0
        elif i < 4:
            ma.moment = np.array([[0,0,-2.0]])
            ma.mu = 2.0

    Q = np.array([[1,1,1], [0.5, 0.5, 0.5]])
    Fm2 = mag_struct.getMagneticStructureFactor(Qm=Q, squared=True)

    assert pytest.approx(Fm2[0], 1e-6) == 0.0
    assert Fm2[1] > 0.0

def test_determine_msg():
    from magneupy.util.magnetic_symmetry import determine_msg
    import os
    from pymatgen.core.structure import Structure

    # Create structure with site-specific magmoms as scalars for collinear analysis
    s = Structure.from_file("tests/data/NiO.cif")
    s.add_site_property('magmom', [2.0, 2.0, -2.0, -2.0, 0, 0, 0, 0])
    json_path = "tests/data/NiO_mag_collinear.json"
    s.to(filename=json_path)

    res = determine_msg(json_path)
    assert res["ordering"] == "AFM"
    assert res["exchange_group_symbol"] == "P4/mmm"

def test_magnetic_structure_factor_incommensurate():
    from magneupy.magnetic import MagAtom, MagneticStructure, MagAtomGroup
    from magneupy.material import Crystal

    # Simple cubic with incommensurate qm = (0.1, 0, 0)
    cif_content = """data_Incomm
_cell_length_a 4.0
_cell_length_b 4.0
_cell_length_c 4.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Mn Mn 0.0 0.0 0.0
"""
    with open("tests/data/incomm.cif", "w") as f:
        f.write(cif_content)

    crystal = Crystal(cif="tests/data/incomm.cif")
    qm = np.array([0.1, 0, 0])
    mag_struct = MagneticStructure(magnames=["Mn"], parents=[crystal], qms=[qm])

    mag_atom = mag_struct.magatoms["mMn1"]
    mag_atom.mu = 1.0
    mag_atom.moment = np.array([[0, 1.0, 0]]) # moment along y

    # Scattering at Q = G + qm. G=(1,0,0) -> Q=(1.1, 0, 0)
    Q = np.array([[1.1, 0, 0]])
    Fm2 = mag_struct.getMagneticStructureFactor(Qm=Q, squared=True)
    assert Fm2[0] > 0.0

    # Scattering at Q = G - qm. G=(1,0,0) -> Q=(0.9, 0, 0)
    # The current implementation of getMagneticStructureFactor seems to just
    # use the provided Qm and magatom.moment. It doesn't seem to automatically
    # handle the two satellites unless they are both in the Q list.
    Q2 = np.array([[0.9, 0, 0]])
    Fm2_sat2 = mag_struct.getMagneticStructureFactor(Qm=Q2, squared=True)
    assert Fm2_sat2[0] > 0.0

def test_ceau_sb2_basics():
    from magneupy.material import Crystal
    # CeAuSb2: P4/nmm, a=4.393, c=10.231 (approx)
    cif_content = """data_CeAuSb2
_cell_length_a 4.393
_cell_length_b 4.393
_cell_length_c 10.231
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P 4/n m m'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Ce Ce3+ 0.5 0.5 0.12
Au Au 0.5 0.5 0.5
Sb1 Sb 0.5 0.5 0.82
Sb2 Sb 0 0 0
"""
    with open("tests/data/CeAuSb2.cif", "w") as f:
        f.write(cif_content)
    crystal = Crystal(cif="tests/data/CeAuSb2.cif")
    assert crystal.nuclear.spacegroup == "P 4/n m m"

def test_chromium_basics():
    from magneupy.material import Crystal
    # Cr: BCC, Im-3m, a=2.88
    cif_content = """data_Cr
_cell_length_a 2.88
_cell_length_b 2.88
_cell_length_c 2.88
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'I m -3 m'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Cr Cr 0 0 0
"""
    with open("tests/data/Cr.cif", "w") as f:
        f.write(cif_content)
    crystal = Crystal(cif="tests/data/Cr.cif")
    assert crystal.nuclear.a == 2.88
