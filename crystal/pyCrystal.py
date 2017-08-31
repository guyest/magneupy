"""
Update this to not use '*'
"""
#from pyRep import *# MagRepGroup, NucRepGroup, BasisVectorCollection, BasisVectorGroup, BasisVector
#from pyData import *
# Need weakref for child/parent?

# Personal modules:
from . import pyRep, pyData

import numpy as np
import periodictable as pt
import string, inspect, pdb #cmath?
from lmfit import minimize, Parameters, conf_interval, Minimizer
from lmfit.printfuncs import report_fit, report_ci
from pymatgen.io.cif import CifFile, CifParser
from itertools import combinations_with_replacement, product
from copy import deepcopy
from collections import OrderedDict
from cmath import polar
rec2pol = np.vectorize(polar)


"""
# -----
Now for the parts pertaining to the crystal model
pyCryst
TODO:
* Include the SpacegroupFactory and PointgroupFactory from Mantid.
    * See also GSAS-II for spacegroup lists in hard code (python)
 -----
"""
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
    #
    def getNeutronScatteringLength(self):
        """
        The neutron coherent scattering length is pulled from the 'pt' module <-- will we need to make sure the incident NEUTRON WAVELENGTH is incorporated?
        UNITS: The scattering lengths are given here in femtometers (fm). 
        UNITS: We will divide the calculated factors by 10 so that when they are squared, the units are in barn (since barn = 100 fm^2).
        """
        return pt.elements.isotope(self.element).neutron.b_c / 10. 
    #
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
    #   
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
        # 
    def setParent(self, parent):
        """
        This function creates a reference to the parent NuclearStructure class.
        """
        if parent is None: pass
        elif not isinstance(parent, (NuclearStructure, pyRep.BasisVectorCollection)): raise TypeError('Atoms expect a NuclearStructure or BasisVectorCollection type as their parent. Please provide a proper object to reference.') 
        self.nuclear = parent
        return
    #
#
class MagAtom(Atom):
    """
    ...
    ----------
    Additional Attributes:
    ...
    ----------
    TODO:
    <done> See getFormFactor to include the form factor generator for this species at a given Q.
    * This is being implemented in Mantid -- eventually make that switch over. 
    <done> Include the Lande g-factor
    * Eventually, it would be nice to have a lookup table for the gj.
    * Need better way of initializing that takes the positions as reference but can still edit other fields. Perhaps input line by line for each attribute needed?
    """
    def __init__(self, basisvectorcollection=None, gj=2., mu=1., atom=None, elname=None, label=None, copy=False):
        """
        This requires a BasisVectorGroup be provided, but it may be a default instance.
        """
        
        if atom is not None:
            # Copy over all fields of the atom
            self.__dict__.update(atom.__dict__)
            if not copy:
                self.label = label
        else:
            # Create a new atom from input and copy it to the magatom
            Atom(elname, label=label)
            self.__dict__.update(atom.__dict__)
        
        self.gj = gj
        self.mu  = mu 
        self.moment = 0#np.array([0, 0, 0])
        self.phi = 0
        self.t = None
        self.qms = []
        self.deth = 1
        
        # Set the BasisVectorGroup for this magnetic atom from the parent Irrep. 
        # This contains the coefficients used for fitting the structure.
        if not isinstance(basisvectorcollection, (type(None), pyRep.BasisVectorCollection)): raise TypeError('The basisvectorcollection variable expects input to be a BasisVectorCollection object.')
        self.bvc = basisvectorcollection
            
        return
    #
    def ff(self, Qm, **kwargs):
        """
        Convenience alias for getFormFactor
        """
        return self.getFormFactor(Qm, **kwargs)
    #
    def getFormFactor(self, Qm, rlu=True, S=1/2, L=3, 
                      orbital=True, return_Q=False):
        """
        TODO:
        <done> Qm is converted from rlu to ang within the routine.
        * Generalize to include the other options provided by periodictable
        """
        J = np.abs(L-S)
        if rlu:
            # Convert Q into the proper units            
            Qm = self.rlu2ang(Qm)
    
        # Due to a bug in the code, we need to pass a 'charge key' that is not the charge for cerium
        # Or it may be that 3+, and 4+ are only there for j2_Q and greater?
        elname = self.element
        elall = b"".maketrans(b"",b"")
        pstr = string.ascii_letters          
        elnolet = elall.translate(elall, pstr.encode('utf-8'))
        elname = elname.translate(elall, elnolet)        
        charge = int(self.oxidation)
        charge = 3 if elname.decode() == 'Ce' else charge        
        charge_key = 2 if elname.decode() == 'Ce' else charge

        # Get the element from the periodic table
        element = pt.elements.isotope(elname)
        
        gL = 1./2. + (L*(L+1)-S*(S+1))/(2*J*(J+1))
        gS = 1. + (S*(S+1)-L*(L+1))/(J*(J+1))
        gJ = gL + gS # Lande splitting factor        
    
        # Return the form factor
        Q = np.sqrt(np.sum(np.square(Qm),axis=1))
        j0 = element.ion[charge].magnetic_ff[charge_key].j0_Q(Q)
        j2 = element.ion[charge].magnetic_ff[charge_key].j2_Q(Q)
        if orbital:
            fQ = (gS*j0 + gL*j0 + gL*j2)/gJ #Lovesey, Eq. 7.26 for full J
        else:
            fQ = j0 + j2*(gJ-2)/gJ #Lovesey, Eq. 7.23 for S with quenched L
        if return_Q:
            return Q, fQ, gJ
        else:
            return np.repeat(fQ.reshape((len(Qm),1)), 3, axis=1)
    #  
    @property
    def phase(self, N=None):
        """
        Should actually just be an 
        """
        return
    #    
    def addMoment(self, m, phi=0, normalize=True):
        """"""
        if self.phi is not None: phi=self.phi
        m = np.asanyarray(m)*np.exp(1j*phi)
        if normalize:
            try:
                m/= np.linalg.norm(m)
            except ValueError:
                pass
        self.moment = self.mu*m.reshape((1,3))
        return
    #  
    def getMoment(self, N=None):
        """"""
        if N is not None:
            self.moment = self.moment.reshape((1,3)) # may not need
            return np.repeat(self.moment, N, axis=0)
        else:
            return self.moment
    #
    def setMomentFromIR(self, m, phi=0):
        
        return
    #
    def setMomentSize(self, mu=None):
        if mu is None:
            pass
        else:
            self.mu = mu
            self.addMoment(self.moment)
        return
    #
    def setPhase(self,phi=None):
        """"""
        if phi is None:
            pass
        else:
            self.phi = phi
            self.addMoment(self.moment)        
        return
    #    
    def getMomentSize(self):
        """
        TODO:
        * Confirm defintion of moment size in terms of self.moment, then normalize by proper factor to put something like np.norm(self.moment) in Bohr magneton.
        """
        return np.linalg.norm(self.moment)
    #
    def setParent(self, parent):
        """"""
        if parent is None: pass
        elif not isinstance(parent, (MagneticStructure, BasisVectorCollection)): raise TypeError('MagneticAtoms expect a MagneticStructure or BasisVectorCollection type as their parent. Please provide a proper object to reference.') 
        else: setattr(self, parent.familyname, parent)        
        return
    #    
#
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
    #
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
#
class MagAtomGroup(AtomGroup):
    """
    ...
    ----------
    Additional Attributes:
    ...
    ----------
    TODO:
    """
    def __init__(self, magatoms=None):
        """
        FIX > self is now the dict (not list)
        """
        super(MagAtomGroup, self).__init__()
        if magatoms is not None: self.magatoms = list(magatoms)
        self.setqms()
        return
    #
    def checkMagAtomGroup(self):
        """
        This function will implement a check on moments of the constituent atoms to make sure they are of the right type and shape as expected.
        """
        return
    #
    def setqms(self):
        """
        TODO:
        * This should be present in the BasisVectorCollection only.
        """
        return
    #
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
    #
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
    #
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
    #
    def placeAtoms(self, struc):
        """"""
        
        # Place the atoms in their locations
        for site in struc.sites:
            atom = Atom(self.getElementName(site), self.getOxidationState(site), label=site.species_string)
            atom.setLocation(site.frac_coords, site.coords, *self.abc_angles)
            self.atoms.append(atom)        
        
        return
    #
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
    #    
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
    #
    def setNames(self):
        """
        Eventually make this point to the AtomGroup.names (or just be able to pull it as a Child)
        """
        for atom in self.atoms:
            self.names.append(atom.element)
        return
    #
    def getUniqueLabels():
        """
        TODO:
        * Move to AtomGroup
        """
        
        return
    #
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
    #
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
    #    
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
    #
    def setNuclearStructureFactor(self, Q=None, units=None):
        """
        *! This and similar operations should probably happen at the level of Crystal!!!
        """       
        if Q is not None:
            self.Q = Q
        # Construct the MagneticStructureFactorModel with 
        self.Fn = pyData.NuclearStructureFactorModel(self.Q, np.zeros(np.max(self.Q.shape), dtype=complex))#, units=units)
        return
    #    
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
    #   
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
    #   
    def setParents(self, parents, **kwargs):
        """
        This method sets the parent reference of NuclearStructures to a Cystal and returns a TypeError if the parent is of the wrong type
        """    
        errmsg = 'NuclearStructures expect a Crystal or NucRepGroup type as their parent. Please provide a Crystal type object to reference'
        errmsg = kwargs['errmsg'] if 'errmsg' in kwargs else errmsg
        
        types  = (Crystal, pyRep.NucRepGroup)
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
#
class MagneticStructure(NuclearStructure):
    """"""
    def __init__(self, magnames=None, magatoms=None, nuclear=None, qms=None,\
                 Q=None, Qmax=7, Fexp=None, parents=None, plane=None):
        """
        TODO:
        * Needs the BasisVectorGroups added to each magatom still.
        <done> No direct input of MagneticStructure without a NuclearStructure.
        """
        # Set up the MagneticStructure family
        self.familyname = 'magnetic'        
        self.setParents(parents) #<< Needs work? See Crystal.
        try:
            self.magrepgroup = self.crystal.magrepgroup
        except:
            pass
        
        assert(isinstance(nuclear, NuclearStructure))
        self.nuclear = nuclear
        
        self.magnames = magnames
        self.Q = self.makeQ(Qmax=Qmax, plane=plane) if Q is None else Q
        self.Fexp = None if Fexp is None else Fexp
        
        # Preallocate fitting fields.
        self.fitter = None
        self.res = None
        
        if magatoms is None: self.magatoms = MagAtomGroup()        
        elif not isinstance(magatoms, (MagAtomGroup)): raise TypeError('The magatoms variable expects input to a be a MagAtomGroup object.')
        else: self.magatoms = magatoms
        
        if self.nuclear is not None: self.setMagneticStructure()
  
        return
    #
    def setMagneticStructure(self):
        """
        TODO:
        <done> Modified to add the form factor via self.setFormFactor()
        <done> Implemented by matching the names in magnames to the names of atoms in self.atoms
        * Make this construct the MagAtomGroup (and change name)
        * Could also be more efficient...
        """
        if not hasattr(self, 'magatoms'): self.magatoms = MagAtomGroup() 
        count = 0
        for atom in self.nuclear.atoms:
            for magname in self.magnames:
                if atom.element.decode() == magname:
                    count += 1
                    label = str('m') + magname + str(count)
                    self.magatoms[label] = MagAtom(atom=atom, label=label)

        #print ''
        #self.setFormFactor()
        
        return 
    #    
    def setMagneticStructureFactor(self, Q=None, units=None):
        """
        TODO:
        * Combine with getStructureFactor
        *! This and similar operations should probably happen at the level of Crystal!!!
        """
        
        if Q is None:
            # make the coordinates
            coords = []
            for qm in self.qms:
                coords.append(self.Q+qm)
            coords = np.vstack(coords).reshape(len(self.qms)*len(self.Q),3)    
            sidx = np.argsort(np.linalg.norm(self.rlu2ang(coords), axis=1))
            coords = coords[sidx,:]
        else:
            coords = np.asanyarray(Q)
        
        # Construct the MagneticStructureFactorModel with 
        self.Fm = pyData.MagneticStructureFactorModel(coords, np.zeros(coords.shape, dtype=np.complex128), units=units)
        
        return
    #
    def getMagneticStructureFactor(self, gjs=None, useDebyeWaller=False,
                                   squared=True, returned=False, 
                                   scale_factor=1., Qm=None, update=True,
                                   S=1/2, L=3, plane='hhl', from_IR=True, **kwargs):
        """
        gj is the Lande g-factor
        TODO:
        * Update to include new class structure for MagneticStructureFactorModel
        * Need a way to check that the atom in each calculation loop is in the proper location for its moment and phase.
        <done> Confident that the form factor is computed with Qm rather than Q.
        """
        # working from Eq. 59 in Chapter 1 of Chatterji
        gn = -3.82608545 # neutron g-factor from: http://physics.nist.gov/cgi-bin/cuu/Value?gnn|search_for=all!
        gamma = gn/2
        r0 = np.sqrt(0.07941124) # electron 'radius' in sqrt(barn)     
        
        if Qm is None:
            self.setMagneticStructureFactor()
            Qm = 1.*self.Fm.coords
            N = len(Qm)  
            for magatom in list(self.magatoms.values()):
                d = magatom.d
                # Get the moment and form factor for the particular ion in question   
                fd   = magatom.ff(Qm,**kwargs)
                fd   = fd+0j
                md = np.repeat(magatom.moment, N, axis=0)
                Qmp = np.repeat(np.inner(Qm, d).reshape(N,1), 3, axis=1)
            
                # Determine the contribution to the structure factor. This only has contributions from magnetic ions.

                Fm0 = (gamma*r0/2)*fd*md* np.exp(2.*np.pi*1.j * Qmp)
                self.Fm.values += Fm0
                # see pg. 291 of Lovesey vol. 2 for the spin density calculation
                                 
            # Get a unit vector in direction of the magnetic peaks.
            Q = self.rlu2ang(Qm) #nodeepcopy
            Qh = np.zeros(Q.shape)
            Qnorm = np.repeat(np.linalg.norm(Q,axis=1),3)
            idx = np.where(Qnorm==0)
            Qnorm[idx] = 1
            Qnorm = Qnorm.reshape((Q.shape))
            Qh = Q / Qnorm
            
        
            # Caclulate the projection of the magnetic structure factor onto the 
            # perpendicular plane
            self.Fm.values =   np.cross(Qh, np.cross(self.Fm.values, Qh))
            
            # Constants have been checked. 
            # I feel confident they are correct so that the norm of the fourier component is the size of the moment when only one harmonic is visible.
            if squared:
                #self.Fm.values = np.abs(self.Fm.values)**2.
                self.Fm.values = np.abs(np.sum(self.Fm.values.conjugate() * self.Fm.values, axis=1))
                self.Fm.values.reshape(N,1)              
                self.Fm.values *= scale_factor
                if returned: return self.Fm # make sure numpy has implemented this correctly for complex numbers.    
            else:
                if returned: return  self.Fm 
        else:
            Qm = np.asanyarray(Qm)
            Fm = 0.+0j            
            N = len(Qm) 
            if len(Qm.shape)==1:
                N=1
                Qm=Qm.reshape(1,len(Qm))
            for magatom in list(self.magatoms.values()):
                d = magatom.d
                # Get the moment and form factor for the particular ion in question   
                fd   = magatom.ff(Qm,**kwargs)
                fd   = fd+0j
                md = np.repeat(magatom.moment, N, axis=0)
                Qmp = np.repeat(np.inner(Qm, d).reshape(N,1), 3, axis=1)
                
                # Determine the contribution to the structure factor. This only has contributions from magnetic ions. 
                Fm0 = (gamma*r0/2)*fd*md* np.exp(2.*np.pi*1.j * Qmp)
                Fm += Fm0
                # see pg. 291 of Lovesey vol. 2 for the spin density calculation

            # Get a unit vector in direction of the magnetic peaks.
            Q = self.rlu2ang(Qm) #nodeepcopy
            Qh = np.zeros(Q.shape)
            Qnorm = np.repeat(np.linalg.norm(Q,axis=1),3)
            idx = np.where(Qnorm==0)
            Qnorm[idx] = 1
            Qnorm = Qnorm.reshape((Q.shape))
            Qh = Q / Qnorm
        
            # Caclulate the projection of the magnetic structure factor onto the 
            # perpendicular plane 
            Fm =   np.cross(Qh, np.cross(Fm, Qh))
            
            if update:
                #self.setMagneticStructureFactor(Q=Qm)
                self.Fm.values = Fm
                self.Fm.coords = Qm            
            # Constants have been checked. 
            # I feel confident they are correct so that the norm of the fourier component is the size of the moment when only one harmonic is visible.
            if squared:
                #self.Fm.values = np.abs(self.Fm.values)**2.
                Fm = np.abs(np.sum(Fm.conjugate() * Fm, axis=1))
                Fm.reshape(N,1)              
                Fm *= scale_factor
                if update:
                    self.Fm.values = Fm
                    self.Fm.coords = Qm
                return Fm
            else:
                return  Fm             
    #
    def setMagneticRefinement(self, params, **kwargs):
        """"""
        self.fitter = Minimizer(self.residual, params, **kwargs)
        return
    #
    def refineMagneticStructure(self, params=None, **kwargs):
        self.res = self.fitter.minimize(params=params, **kwargs)
        return self.res
    #
    def residual(self, params, **kwargs):
        self.update(params, Qm=self.Fexp.coords, returned=False, update=True, 
                    **kwargs)
        data = self.Fexp.values
        calc = self.Fm.values
        err = self.Fexp.errors
        res = (data - calc) / err
        return res  
    #
    def update(self, params, **kwargs):
        Nrep = self.crystal.magrepgroup.IR0 
        # vector direction
        coeffs = {}
        coeffs.update(params.valuesdict())
        irrep=self.crystal.magrepgroup['G'+str(Nrep)]
        for bvg in list(irrep.values()):
            name = irrep.name+'_'+bvg.name 
            try:
                bvg.coeff = coeffs[name]
            except:
                pass
        # moment size
        _cnt = 1
        for magatom in list(self.magatoms.values()):
            magatom.setMomentSize(params['mu'+str(_cnt)])
            magatom.setPhase(params['phi'+str(_cnt)])
            magatom.addMoment(self.magrepgroup.getMagneticMoment(d=magatom.d,Nrep=Nrep)) 
            _cnt+=1
            
        self.getMagneticStructureFactor(**kwargs)
        return
    #
    def rlu2ang(self, Q):
        """
        TODO:
        * implement so that lattice constants are accessed at the crystal level.
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
    #
    def getMagneticDiffraction():
        """"""
        
        return
    #
    def claimChildren(self, family=['magatoms']):
        """
        This function sets the Children according to the NuclearStructure defintion
        """
        super(MagneticStructure, self).claimChildren(family=family)
        return
    #
    def setParents(self, parents):
        """
        This method sets the parent reference of NuclearStructures to a Cystal and returns a TypeError if the parent is of the wrong type
        """    
        errmsg = 'MagneticStructures expect a Crystal or MagRepGroup type as their parent. Please provide a Crystal type object to reference'
        types  = (Crystal, pyRep.MagRepGroup)
        
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
    #
    def setSibling(self, family=['nuclear']):
        """"""
        return
#
class ChargeDensity(NuclearStructure):
    """
    A ChargeDensity class extends the NuclearStructure to include charge density which could be fitted against charge sensitive probes (e.g., x-ray).
    """
    def __init__(self, cifname=None, structure_info=None):
        """"""
        
        # Set up the Crystal family
        self.familyname = 'charge'        


        return
    #
#
class Crystal(object):
    """
    A Crystal object is a collection of a NuclearStructure and its subclasses.
    TODO:
    * Long term, it would be nice to include exotic orders (e.g., a la Lucille Savary & Sentill Todadri)
    * Need a specified set<obj> function for each Child object that also associates with the Parent crystal. 
    * Allow input of CIF, MCIF (eventually), Sarah (eventually), etc. files in order to generate the whole Crystal from scatch
    """
    def __init__(self, nuclear=None, mag=None, charge=None, magrepgroup=None, nucrepgroup=None, spacegroup=None):
        
        # Initialize the Structures
        self.nuclear = nuclear
        self.magnetic= mag
        self.charge  = charge
        # self.exoticorders
        
        # Initialize the RepGroups
        self.magrepgroup = magrepgroup
        self.nucrepgroup = nucrepgroup
        
        # Initialize the symmetry
        self.spacegroup  = spacegroup
        
        # Initialize the data container
        self.data = {}
        self.Qm = None
        self.Fm_exp = None
        self.Qn = None
        self.Fn_exp = None
        
        # Set up the Crystal family
        self.familyname = 'crystal'
        
        self.claimChildren()
        return
    #
    def getMagneticMoments(self, Nrep=1, bvs=None, coeffs=None, mu=None):
        """
        TODO:
        * should be called setMagneticMoments?
        """
        self.magrepgroup.IR0 = Nrep
        if coeffs is None: coeffs={}
        for irrep in list(self.magrepgroup.values()):
        # reset the coefficients for each basis vector group
            for bvg in list(irrep.values()):
                name = irrep.name+'_'+bvg.name 
                try:
                    bvg.coeff = coeffs[name]
                except:
                    print('Using default coefficient of 1 for Basis Vector.')
                    bvg.coeff = 1
        for magatom in list(self.magnetic.magatoms.values()):
            magatom.setMomentSize(mu)
            magatom.addMoment(self.magrepgroup.getMagneticMoment(d=magatom.d, Nrep=Nrep))
        
        return
    #
    def setChild(self, child):
        """"""
        setattr(self, child.familyname, child)
        return
    #
    def claimChildren(self, family=['nuclear', 'magnetic', 'charge']):
        """
        This is performed in the init stage so that all consitituents of the Crystal may back-refernece it by name
        """
        children, labels, = getFamilyAttributes(self, family, return_labels=True)
        for child, label in zip(children,labels):
            if child is not None: 
                setattr(child,'crystal',self)
                setattr(self, 'label',  child)
        return
    #
    def setAliases(self, 
                        source_alias_pairs={'magrepgroup':'magnetic'}):
        """
        This is performed in the init stage so that some children can access
        certain global componenets directly.
        """
        source_attrs = source_alias_pairs.keys()
        alias_attrs  = source_alias_pairs.values()
        source_objs  = getFamilyAttributes(self, source_attrs)
        alias_objs   = getFamilyAttributes(self, alias_attrs)
        for sa, so, ao in zip(source_attrs, source_objs, alias_objs):
            setattr(ao, sa, so)
        return
    #    
    def setStructureFactor(self, Qm=None, Fm_exp=None, Fm_err=None, 
                           Qn=None, Fn_exp=None, Fn_err=None, 
                           units=None, **kwargs):
        """"""
        if (Qn is None) and (Qm is None):
            print('Need Q values and F values!')
            
        if ((not(Qn is None)) and (not(Fn_exp is None))):
            self.Qn     = Qn
            self.Fn_exp = pyData.StructureFactorModel(Qn, Fn_exp, units=None) 
            self.Fn_err = Fn_err  
            self.Fn = self.nuclear.Fn
            
        if ((not(Qm is None)) and (not(Fm_exp is None))):
            if self.Qm is Qm:
                pass
            else:
                self.Qm =Qm
                
            if self.Fm_exp is Fm_exp:
                pass
            else:
                self.magnetic.Fexp = pyData.StructureFactorModel(Qm, 
                                        Fm_exp, Fm_err, units=None)
                self.Fm_exp = self.magnetic.Fexp
            #
            Fm = np.asanyarray(self.calc_Fm(**kwargs))             
            self.magnetic.Fm.coords = self.Qm #REFERENCE        
            self.magnetic.Fm.values = Fm        
            self.Fm = self.magnetic.Fm
            self.magnetic.magrepgroup
            
        return
    #
    def rietveld_refinement(self, Nreps_fit=[], Qs_fit=None):
        """
        Driver for the refinement
        TODO:
        * Compute the scale factor from nuclear peaks separately.
        """
        params = Parameters()
        params.add('scale_factor', value=1., vary=True, min=1.e-12, max=None, expr=None)  # Later, should do this separately...          
        #out_dict = {}
        Nreps = []
        for irrep in list(self.magrepgroup.values()):
            # For each irrep...
            if Nreps_fit.count(irrep.N) > 0:            
                Nreps.append(irrep.N)
                print('Fitting Irrep #G'+str(irrep.N))
                for bvg in list(irrep.values()):
                    # Add a parmeter for each basis vector (group)...
                    name = irrep.name+'_'+bvg.name
                    params.add('rcoeff_'+name, value=1, vary=True, min=None, max=None, expr=None)
                    params.add('ccoeff_'+name, value=1., vary=True, min=None, max=None, expr=None)
            else:
                for bvg in list(irrep.values()):
                    # Add a constant zero parmeter for each basis vector (group)...
                    name = irrep.name+'_'+bvg.name
                    params.add('rcoeff_'+name, value=0, vary=False, min=None, max=None, expr=None)
                    params.add('ccoeff_'+name, value=0., vary=False, min=None, max=None, expr=None)                
                    
            
        # Set the data structure factor and show it:
        # Show the data:
        if Qs_fit is None: Qs_fit = self.F.coords
        vmax = 0.
        for data in self.data:
            self.data[str(data)].setStructureFactor(Qs_fit)  
            vmax = max(vmax,self.data[str(data)].F.values.max())
            
        for data in self.data:
            self.data[str(data)].F.plotStructureFactor(vmax=vmax)        
        
        # Now pass in the arguments and perform the fits, for each irrep, and add the result to the list
        res_args = ((self,))
        res_kws  = {'Nreps':Nreps}
        #res = minimize(self.residual, params, args=res_args, kws=res_kws, method='leastsq')
        res = minimize(self.residual, params, args=res_args, kws=res_kws, method='nelder')
        #out_dict[irrep.name] = res
        out = res
        self.F.plotStructureFactor(vmax=vmax)
        return out
    #
    def plot(self):
        vis = StructureVis(**kwargs)
        vis.set_structure(self)# FIX
        vis.show()
        return vis
    #
    def calc_Fm(self, Nrep=2, bvs=[2], mu=1.5,
                 **kwargs):        
        rep_name = 'G'+str(Nrep)
        rep_names=[]
        for bv in bvs:
            rep_names.append(rep_name+'_psi'+str(bv))

        coeffs={}
        for i_ in range(len(self.magnetic.magatoms)):
            for rn_ in rep_names:
                coeffs[rn_+'_'+str(i_+1)]=mu

        self.getMagneticMoments(Nrep=Nrep, bvs=bvs, coeffs=coeffs)
        Fm = self.magnetic.getMagneticStructureFactor(Qm=self.Qm, 
                                                      squared=True)
        return Fm
    #
    @staticmethod
    def residual(params, self, Nreps=[1]):
        """
        Must give difference between all peaks and update the crystal.
        Must also include an overall scale factor (same for both magnetic and nuclear intensities)
        """
        coeffs = {} # move to self?
        for param in params:
            coeffs[str(param)] = params[str(param)]
            
        for irrep in list(self.magrepgroup.values()): #Nrep in Nreps:
            Nrep = irrep.N
            # First update the magnetic moments according to the modified basis vector coefficients:
            self.getMagneticMoments(Nrep=Nrep, coeffs=coeffs)
            #self.getNuclearSites(params=params) # for adjusting after fitting the lattice?
        
        # Then update the structure factor calculations
        # add units to all these eventually
        self.nuclear.setNuclearStructureFactor()
        self.nuclear.getNuclearStructureFactor(scale_factor=params['scale_factor'])  
        self.magnetic.setMagneticStructureFactor()
        self.magnetic.getMagneticStructureFactor(scale_factor=params['scale_factor'])
        self.setStructureFactor() 
        
        # ...
        # Anything else?
        
        # Then, compute the difference between the new model and the data
        res = 0.
        for data in list(self.data.values()):
            res += (data.F.values - self.F.values[data.idx]) / data.F.errors
        return res
    #
#
class Powder(object):
    """
    A Powder object is a collection of NuclearStructure and its subclasses, with supplements to treat isotropically averaged experimental data.
    """
    def __init__(self):
        """"""
        return
    #
#
#
# --------------
# Define some general helper methods
# --------------

def getTrimmedAttributes(obj):
    """
    This function returns a list of attributes of the input object (class) as a list of tuples each of the form: ('field', <instance>)
    pulled from: http://stackoverflow.com/questions/9058305/getting-attributes-of-a-class
    """
    attributes = inspect.getmembers(obj, lambda a:not(inspect.isroutine(a)))
    return [a for a in attributes if not(a[0].startswith('__') and a[0].endswith('__'))]
#
def getFamilyAttributes(obj, family, return_labels=False):
    """"""
    attr = getTrimmedAttributes(obj)
    attributes = []
    if family is not None:
        for fam in family:                    
            attributes.append(attr.pop(attr.index((fam, obj.__getattribute__(fam)))))
    
    attrs = []
    lbls  = []
    for attribute in attributes:
        lbl, attr, = attribute
        lbls.append(lbl)
        attrs.append(attr)
    if return_labels:
        return attrs, lbls
    else:
        return attrs
#
def stripDigits(name):
    """"""
    pstr = string.digits 
    name = name.encode('ascii','ignore')
    allNames = string.maketrans('','')
    nodig = allNames.translate(allNames, pstr)
    name = name.translate(allNames, nodig)        
    return name
#