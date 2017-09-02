from collections import OrderedDict
from itertools import combinations_with_replacement, product
import numpy as np
import periodictable as pt
from pymatgen.io.cif import CifFile, CifParser

from magneupy.helper.functions import *
from magneupy.rep.rep import BasisVectorCollection, NucRepGroup
from magneupy.crystal.crystal import Crystal


class Atom(object):
    """
    ...
    ----------
    Attributes:
    ...
    ----------
    TODO:
    * 
    """

    def __init__(self, elname, oxidation, label=None, parent=None):
        """
        TODO:
        * Long term, it would be nice to make this extensible for use with RIXS and/or ARPES spectral function modeling. This would mean including those cross-sections and/or scattering lengths.
        * Decide on what the parent of the Atom should be (general class passed initial test that changes to a reference continue to work -- as they should in pythonic fashion)
        """

        # Set initial values of...
        #...the coordinates
        self.d = None
        self.r = None
        #...the element name and label
        self.element = None
        self.label = None
        #...the neutron scattering length
        self.bc = None


        # Set the atom name
        self.element = elname
        self.oxidation = oxidation
        if label is not None:
            self.label = label
        else:
            self.label = elname

        # Set the neutron scattering length
        self.bc = self.getNeutronScatteringLength()

        # For structural distortions
        # assert(isinstance(irrep, (StructIrrep, StructCorep)))
        #self.irrep = irrep

        self.setParent(parent)

        return

    def getNeutronScatteringLength(self):
        """
        The neutron coherent scattering length is pulled from the 'pt' module <-- will we need to make sure the incident NEUTRON WAVELENGTH is incorporated?
        UNITS: The scattering lengths are given here in femtometers (fm). 
        UNITS: We will divide the calculated factors by 10 so that when they are squared, the units are in barn (since barn = 100 fm^2).
        """
        return pt.elements.isotope(self.element).neutron.b_c / 10.

    def setLocation(self, frac_coords, ang_coords, a, b, c, alpha, beta, gamma):
        """
        TODO:
        * The lattice part of this (a,b,c,alpha,beta,gamma) should eventually just come from the parent's lattice defintion.
        """
        # Set the position within a crystal structure
        self.d = frac_coords
        self.r = ang_coords
        self.a = a
        self.b = b
        self.c = c
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma

        return

    def rlu2ang(self, Q):
        """
        TODO:
        * Needs to be generalized!
        """
        Q = np.asanyarray(Q)
        N = len(Q.shape)
        if N == 1:
            Q = Q.reshape((1,3))
        vec = 2*np.pi/np.array([self.nuclear.a, self.nuclear.b, self.nuclear.c])
        Q = np.stack(list(map(np.diag, np.einsum('...i,j', Q, vec))))
        #Q[:,0] = np.reshape(Q[:,0]*2.*np.pi/self.nuclear.a, Q.shape)
        #print(Q[:,0]*2.*np.pi/self.nuclear.a)
        #Q[:,1] = Q[:,1]*2.*np.pi/self.nuclear.b
        #Q[:,2] = Q[:,2]*2.*np.pi/self.nuclear.c
        #print(Q)
        return Q

    def setParent(self, parent):
        """
        This function creates a reference to the parent NuclearStructure class.
        """
        if parent is None: pass
        elif not isinstance(parent, (NuclearStructure, BasisVectorCollection)): raise TypeError('Atoms expect a NuclearStructure or BasisVectorCollection type as their parent. Please provide a proper object to reference.')
        self.nuclear = parent
        return


class AtomGroup(OrderedDict):
    """
    ...
    ----------
    Attributes:
    ...
    ----------
    TODO:
    * Make the list only be able to contain Atoms
    """
    def __init__(self, atoms=None, name=None, parent=None):
        """"""
        super(AtomGroup, self).__init__()
        self.name = name
        #if atoms is not None:
        #atoms = self.checkAtoms(atoms)
        #for atom in atoms:
        #self[atom.label] = atom
        return

    def checkAtoms(self, atoms):
        """"""
        try:
            assert(hasattr(atoms, '__iter__'))
        except AssertionError:
            try:
                assert(isinstance(atoms, Atom))
                atoms = list(atoms)
            except AssertionError:
                print('Not a good input to atoms')
        return atoms


class NuclearStructure(object):
    """
    TODO:
    * Decide: Make something like a Structures class that the others are subclassed from?
    """
    def __init__(self, cifname=None, structure_info=None, Q=None, Qmax=7, parents=None, plane=None):
        """"""
        # Set up the NuclearStructure family
        self.familyname = 'nuclear'
        self.setParents(parents)

        # Set the initial state for various fields
        self.atoms = []
        self.names = []

        # ...

        # make the Q values at which to sample
        plane = 'hhl' if plane is None else plane
        self.Q = self.makeQ(Qmax=Qmax,plane=plane) if Q is None else Q
        self.setNuclearStructureFactor()#units?

        self.setStructure(cifname=cifname, structure_info=structure_info)
        self.getNuclearStructureFactor()
        self.setNames()



        self.claimChildren()


        return

    def setStructure(self, cifname=None, structure_info=None):
        """"""
        if cifname is not None:
            assert(isinstance(cifname, str))

            # Use pymatgen methods to parse the CIF file.
            cifparser = CifParser(cifname)
            ciffile   = CifFile.from_file(cifname)
            cfkeys    = list(ciffile.data.keys())[0]
            cifblock  = ciffile.data[cfkeys]

            # obtain the final object from which to pull the pertinent info
            struc     = cifparser.get_structures(cifblock)[0]

            # Set the lattice definition parameters
            self.setLattice(struc)

            # Place the atoms in their locations
            self.placeAtoms(struc)

        elif structure_info is not None:
            assert(isinstance(structure_info, dict))
            self.__dict__.update(structure_info)


        return

    def setLattice(self, struc):
        """
        TODO:
        * Do a more exhaustive search of quantities or methods from pymatgen subclasses that would be useful to retain.
        """
        # Setup the lattice parameters
        self.a = struc.lattice.a
        self.b = struc.lattice.b
        self.c = struc.lattice.c

        # Setup the lattice angles
        self.alpha = struc.lattice.alpha
        self.beta  = struc.lattice.beta
        self.gamma = struc.lattice.gamma

        # Add some convenience attributes
        self.abc = (self.a, self.b, self.c)
        self.volume = struc.lattice.volume
        self.angles = (self.alpha, self.beta, self.gamma)
        self.abc_angles = (self.a, self.b, self.c, self.alpha, self.beta, self.gamma)

        return

    def placeAtoms(self, struc):
        """"""

        # Place the atoms in their locations
        for site in struc.sites:
            atom = Atom(self.getElementName(site), self.getOxidationState(site), label=site.species_string)
            atom.setLocation(site.frac_coords, site.coords, *self.abc_angles)
            self.atoms.append(atom)

        return

    @staticmethod
    def getElementName(site):
        """
        For parsing the element name only (no isotope or charge info)
        Eventually need a way to include the isotope info. 
        Maybe just read first two characters (handle element lengths of only 1) and keep if they are numbers.
        """
        pstr = string.ascii_letters
        elname = site.species_string
        elname = elname.encode('ascii','ignore')
        elall = b"".maketrans(b"",b"")
        elnolet = elall.translate(elall, pstr.encode('utf-8'))
        elname = elname.translate(elall, elnolet)

        return elname

    @staticmethod
    def getOxidationState(site):
        """
        For parsing the ion charge info only (no isotope or element name)
        TODO:
        * Eventually need a way to include the isotope info. Maybe just read first two characters (handle element lengths of only 1) and keep if they are numbers.
        * Needs to be more robust to possible other numbers from isotope. Can use requirement of being adjacent to +/-
        * Needs to retain information about +/-
        """
        pstr = string.digits
        charge = site.species_string
        charge = charge.encode('ascii','ignore')
        elall = b"".maketrans(b"",b"")
        elnolet = elall.translate(elall, pstr.encode('utf-8'))
        charge = charge.translate(elall, elnolet)

        return charge

    def setNames(self):
        """
        Eventually make this point to the AtomGroup.names (or just be able to pull it as a Child)
        """
        for atom in self.atoms:
            self.names.append(atom.element)
        return

    def setLabels(self):
        """
        TODO:
        * Move to AtomGroup
        """
        count = 1
        unique_labels = []
        for unique_label in unique_labels:
            for atom in self.atoms:
                if unique_label == atom.label:
                    atom.label = atom.label + str(count)
                    count +=1
                else: pass

        return

    def rlu2ang(self, Q):
        """
        TODO:
        * Needs to be generalized! (Move it up to crystal level)
        * Be more careful about modifying the value. Give an option or change the name
        """

        Q[:,0] = Q[:,0]*2.*np.pi/self.a
        Q[:,1] = Q[:,1]*2.*np.pi/self.b
        Q[:,2] = Q[:,2]*2.*np.pi/self.c

        return Q

    @staticmethod
    def makeQ(Qmax=4, firstQuad=False, sym=True, plane='hhl'):
        """
        TODO: 
        * Needs to be considerably generalized. Shouldn't be hard. Include human intuitive input: 'hhl' -> 2 repeats, according arrangement of peaks
        * The logic needs to be fixed. firstQuad and sym don't really make sense.
        """
        print(plane)
        if plane == 'hhl':
            if firstQuad:
                qs = np.arange(Qmax+1.)
                if sym:
                    Qiter = product(qs, repeat=2)
                else:
                    Qiter = combinations_with_replacement(Q, 2)
            else:
                qs = -1.*np.hstack((np.arange(Qmax)+1., -np.arange(Qmax+1.)))
                if sym:
                    Qiter = product(qs, repeat=2)
                else:
                    Qiter = combinations_with_replacement(Q, 2)

            Q = []
            for q in Qiter:
                Q.append(np.asanyarray(q))
            Q = np.asanyarray(Q)
            Q = np.vstack((Q[:,0],Q[:,0],Q[:,1])).transpose()

        if plane == 'hk0':
            if firstQuad:
                qs = np.arange(Qmax+1.)
                if sym:
                    Qiter = product(qs, repeat=2)
                else:
                    Qiter = combinations_with_replacement(Q, 2)
            else:
                qs = -1.*np.hstack((np.arange(Qmax)+1., -np.arange(Qmax+1.)))
                if sym:
                    Qiter = product(qs, repeat=2)
                else:
                    Qiter = combinations_with_replacement(Q, 2)

            Q = []
            for q in Qiter:
                Q.append(np.asanyarray(q))
            Q = np.asanyarray(Q)
            Q = np.vstack((Q[:,0],Q[:,1],np.zeros(Q[:,1].shape))).transpose()

        return Q

    def setNuclearStructureFactor(self, Q=None, units=None):
        """
        *! This and similar operations should probably happen at the level of Crystal!!!
        """
        if Q is not None:
            self.Q = Q
        # Construct the MagneticStructureFactorModel with
        self.Fn = pyData.NuclearStructureFactorModel(self.Q, np.zeros(np.max(self.Q.shape), dtype=complex))#, units=units)
        return

    def getNuclearStructureFactor(self, useDebyeWaller=False, squared=True, scale_factor=1., x0=1, y0=0, Q=None):
        """
        Q is given in units of rlu
        TODO:
        * Implement Debye Waller correction.
        """
        if Q is None:
            Q = self.Fn.coords
            #print(Q)
            self.Fn.values = np.complex128(self.Fn.values+0j)
            for atom in self.atoms:
                # Determine the contribution to the NuclearStructure factor.
                Fa = atom.bc * np.exp(2. * np.pi * 1j * np.einsum('...i,i',Q, atom.d))


                if useDebyeWaller:
                    self.Fn[Q] += Fa #* DebyeWaller_factor(self.Q, u)# where is u from?
                else:
                    self.Fn.values += Fa # The StructureFactorModel classes need to be fixed to do assignment properly...

            if squared:
                self.Fn.values = np.abs(self.Fn.values)**2.
                # Units are in barn. Internally modifed by periodictable.
                #norm = self.Fn.values[(self.Fn.coords[:,0]==x0)&(self.Fn.coords[:,2]==y0)]
                #self.Fn.values /= norm
                self.Fn.values *= scale_factor
                return self.Fn # make sure numpy has implemented this correctly for complex numbers. see magnetic part for other implementation.
            else:
                return self.Fn
        else:
            Q = np.asanyarray(Q)
            Fn = 0.+0j
            for atom in self.atoms:
                # Determine the contribution to the NuclearStructure factor.
                Fa = atom.bc * np.exp(2. * np.pi * 1j * np.einsum('...i,i',Q, atom.d))

                if useDebyeWaller:
                    Fn += Fa #* DebyeWaller_factor(self.Q, u)# where is u from?
                else:
                    Fn += Fa

            if squared:
                Fn2 = scale_factor*(np.abs(Fn)**2.)
                #print("Square structure factor for "+str(Q)+" is:"+str(Fn2))
                return Fn2
                # Units are in barn/sr; Internally modifed by periodictable.

            else:
                #print("Structure factor for "+str(Q)+" is:"+str(Fn))
                return sqrt(scale_factor)*Fn

    def claimChildren(self, family=['atoms']):
        """
        This is performed in the init stage so that all consitituents of the Nuclear Structure may back-refernece it by name.
        TODO:
        * Perhaps this is modified to include AtomGroups and/or to replace atoms with an AtomGroup
        * Need to include checks here that say whether the Child is of the right type. (Somewhat redundant as it is handled by the Child as well.)
        """
        children = getFamilyAttributes(self, family)
        for child in children:
            if hasattr(child, '__iter__'):
                for each_child in child:
                    each_child.setParent(self)
            elif child is not None: child.setParent(self)
        return

    def setParents(self, parents, **kwargs):
        """
        This method sets the parent reference of NuclearStructures to a Cystal and returns a TypeError if the parent is of the wrong type
        """
        errmsg = 'NuclearStructures expect a Crystal or NucRepGroup type as their parent. Please provide a Crystal type object to reference'
        errmsg = kwargs['errmsg'] if 'errmsg' in kwargs else errmsg

        types  = (Crystal, NucRepGroup)
        types  = kwargs['types'] if 'types'  in kwargs else types

        if hasattr(parents, '__iter__'):
            for parent in parents:
                if parent is None: pass
                elif not isinstance(parent, types): raise TypeError(errmsg)
                else: setattr(self, parent.familyname, parent)
                parent.setChild(self)
                if getattr(parent, self.familyname) is self: print(str(parent.familyname) + ' is Parent to ' + str(self.familyname))
                if getattr(self, parent.familyname) is parent: print(str(self.familyname) + ' is Child to ' + str(parent.familyname))
        else:
            self.setParents([parents])
        return