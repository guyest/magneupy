import numpy as np
from magneupy.material import Crystal
from magneupy.magnetic import MagneticStructure, MagAtomGroup, MagAtom
import periodictable

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
with open("tests/data/Fe_debug.cif", "w") as f:
    f.write(cif_content)

crystal = Crystal(cif="tests/data/Fe_debug.cif")
mag_struct = MagneticStructure(magnames=["Fe"],
                               nuclear=crystal.nuclear, qms=[np.array([0,0,0])])
mag_struct.crystal = crystal
mag_atom = mag_struct.magatoms["mFe1"]
mag_atom.mu = 2.2
mag_atom.moment = np.array([[0, 0, 2.2]])

Q = np.array([[1, 0, 0]])
Fm2 = mag_struct.getMagneticStructureFactor(Qm=Q, squared=True)
print(f"Calculated Fm2: {Fm2[0]}")

# Manual calculation
gn = -3.82608545
gamma = gn/2
r0 = np.sqrt(0.07941124)
factor = gamma * r0 / 2
print(f"Factor: {factor}")

ff = mag_atom.ff(Q)
print(f"Form factor: {ff}")

Fm0 = factor * ff * mag_atom.moment
print(f"Fm0: {Fm0}")

Q_ang = mag_struct.rlu2ang(Q)
Qnorm = np.linalg.norm(Q_ang, axis=1)
Qh = Q_ang / Qnorm.reshape(-1, 1)
print(f"Qh: {Qh}")

Fm_perp = np.cross(Qh, np.cross(Fm0, Qh))
print(f"Fm_perp: {Fm_perp}")

Fm2_manual = np.abs(np.sum(Fm_perp.conjugate() * Fm_perp, axis=1))
print(f"Manual Fm2: {Fm2_manual[0]}")
