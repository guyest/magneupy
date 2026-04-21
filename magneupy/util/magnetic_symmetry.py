"""
Utilities for magnetic symmetry analysis.
"""
import sys
import warnings
from pymatgen.io.cif import CifParser
from pymatgen.analysis.magnetism.analyzer import CollinearMagneticStructureAnalyzer
from pymatgen.core.structure import Structure

def determine_msg(file_path):
    """
    Determine the magnetic symmetry and ordering of a structure from a file.

    Args:
        file_path (str): Path to a CIF or JSON structure file.

    Returns:
        dict: A dictionary containing magnetic ordering, symmetry information,
              and magnetic site counts.
    """
    if file_path.endswith('.cif'):
        parser = CifParser(file_path)
        struct = parser.parse_structures(primitive=False)[0]
    else:
        struct = Structure.from_file(file_path)

    # Analyze collinear magnetic structure
    analyzer = CollinearMagneticStructureAnalyzer(struct)

    if not analyzer.is_magnetic:
        return {"ordering": "NM", "is_magnetic": False}

    ordering = analyzer.ordering
    mag_atoms = analyzer.number_of_magnetic_sites

    # Get exchange group info
    # This is a proxy for magnetic symmetry in pymatgen
    group_symbol, group_number = analyzer.get_exchange_group_info()

    return {
        "ordering": ordering.value,
        "is_magnetic": True,
        "magnetic_atoms_count": mag_atoms,
        "exchange_group_symbol": group_symbol,
        "exchange_group_number": group_number,
        "is_collinear": analyzer.is_collinear
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 -m magneupy.util.magnetic_symmetry <structure_file>")
        sys.exit(1)

    res = determine_msg(sys.argv[1])
    for k, v in res.items():
        print(f"{k}: {v}")
