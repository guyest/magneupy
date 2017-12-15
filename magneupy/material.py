import typing
from cmath import polar
from lmfit import minimize, Parameters
import numpy as np
from collections import OrderedDict
from itertools import combinations_with_replacement, product
import periodictable as pt
from pymatgen.io.cif import CifFile, CifParser
import string

from .util.functions import getFamilyAttributes
from .rep.rep import BasisVectorCollection, NucRepGroup, MagRepGroup
from .data.data import StructureFactorModel, NuclearStructureFactorModel

rec2pol = np.vectorize(polar)


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
        return pt.elements.isotope(self.element.decode()).neutron.b_c / 10.

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
    """
    # Set the initial state for various fields
    familyname = 'nuclear'
    atoms = []
    names = []

    def __init__(self, cifname=None, structure_info=None, Q=None, Qmax=7, parents=None, plane=None):
        """"""
        # Set up the NuclearStructure family
        self.setParents(parents)

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
            struc     = cifparser.get_structures(False)[0]

            # Set the lattice definition parameters
            self.setLattice(struc)

            # Place the atoms in their locations
            self.placeAtoms(struc)

            # set Spacegroup in H-M string form by default.
            self.setSpaceGroup(cifblock.data)

        elif structure_info is not None:
            assert(isinstance(structure_info, dict))
            self.__dict__.update(structure_info)


        return

    def setSpaceGroup(self, cifdict):
        try:
            self.spacegroup = cifdict['_symmetry_space_group_name_H-M']
            print('Extracted nuclear spacegroup from provided CIF file...')
        except:
            self.spacegroup = None
            print('Failed to extract nuclear spacegroup from provided CIF file... Did you forget to include it? ')
            pass
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
        self._matrix = struc.lattice.matrix
        self.basis = (self._matrix[0,:], self._matrix[1,:], self._matrix[2,:])
        ar = 2.*np.pi * np.cross(self.basis[1],self.basis[2]) / self.volume
        br = 2.*np.pi * np.cross(self.basis[2], self.basis[0]) / self.volume
        cr = 2.*np.pi * np.cross(self.basis[0], self.basis[1]) / self.volume
        self.recip = (ar,br,cr)

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
                    Qiter = combinations_with_replacement(qs, 2)
            else:
                qs = -1.*np.hstack((np.arange(Qmax)+1., -np.arange(Qmax+1.)))
                if sym:
                    Qiter = product(qs, repeat=2)
                else:
                    Qiter = combinations_with_replacement(qs, 2)

            Q = []
            for q in Qiter:
                Q.append(np.asanyarray(q))
            Q = np.asanyarray(Q)
            Q = np.vstack((Q[:,0],Q[:,0],Q[:,1])).transpose()

        if plane == 'h0l':
            if firstQuad:
                qs = np.arange(Qmax+1.)
                if sym:
                    Qiter = product(qs, repeat=2)
                else:
                    Qiter = combinations_with_replacement(qs, 2)
            else:
                qs = -1.*np.hstack((np.arange(Qmax)+1., -np.arange(Qmax+1.)))
                if sym:
                    Qiter = product(qs, repeat=2)
                else:
                    Qiter = combinations_with_replacement(qs, 2)

            Q = []
            for q in Qiter:
                Q.append(np.asanyarray(q))
            Q = np.asanyarray(Q)
            Q = np.vstack((Q[:,0],0*Q[:,0],Q[:,1])).transpose()

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
        self.Fn = NuclearStructureFactorModel(self.Q, np.zeros(np.max(self.Q.shape), dtype=complex))#, units=units)
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
                return np.sqrt(scale_factor)*Fn

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


class Crystal(object):
    """
    A Crystal object is a collection of a NuclearStructure and its subclasses.
    TODO:
    * Long term, it would be nice to include exotic orders (e.g., a la Lucille Savary & Sentill Todadri)
    * Need a specified set<obj> function for each Child object that also associates with the Parent crystal. 
    * Allow input of CIF, MCIF (eventually), Sarah (eventually), etc. files in order to generate the whole Crystal from scatch
    """
    _maginit = {}  # type: typing.Mapping[str,list]

    def __init__(self, cif=None, maginfo=None, cifname=None, charge=None, magrepgroup=None, nucrepgroup=None,
                 spacegroup=None, name='', **kwargs):

        # Initialize the RepGroups
        self.magrepgroup = magrepgroup
        self.nucrepgroup = nucrepgroup

        # Initialize the data container
        self.data = {}
        self.Qm = None
        self.Fm_exp = None
        self.Qn = None
        self.Fn_exp = None

        # Set up the Crystal family
        self.familyname = 'crystal'
        self.name = name

        # Initialize the Structures
        try:
            self.nuclear = NuclearStructure(cifname=cif, parents=self, **kwargs)
        except: # make file-not-found-error handling
            self.nuclear = kwargs['nuclear'] if kwargs['nuclear'] else None
        self.spacegroup = self.nuclear.spacegroup

        if maginfo:
            from . import magnetic
            self._maginit = maginfo
            self.magrepgroup = MagRepGroup()
            try:
                self.magnetic = magnetic.MagneticStructure.from_parent(self)
                self.magnetic.prepareMagneticStructure()
            except:
                raise
        elif 'magnetic' in kwargs.keys():
            self.magnetic = kwargs['magnetic']
        else:
            self.magnetic = None

        self.claimChildren()
        return

    @property
    def maginit(self):
        return self._maginit

    def getMagneticMoments(self, bvs=None, coeffs=None, mu=None):
        """
        TODO:
        * should be called setMagneticMoments?
        """
        Nrep = int(list(self.magrepgroup.keys())[0][1]) # needs to be more robust.
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
        self.magnetic.getMagneticStructureFactor()
        return

    def setChild(self, child):
        """"""
        setattr(self, child.familyname, child)
        return

    def claimChildren(self, family=['nuclear', 'magnetic']):
        """
        This is performed in the init stage so that all consitituents of the Crystal may back-refernece it by name
        """
        children, labels, = getFamilyAttributes(self, family, return_labels=True)
        for child, label in zip(children,labels):
            if child is not None:
                setattr(child,'crystal',self)
                setattr(self, 'label',  child)
        return

    def setAliases(self, source_alias_pairs={'magrepgroup':'magnetic'}):
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

    def setStructureFactor(self, Qm=None, Fm_exp=None, Fm_err=None, Qn=None, Fn_exp=None, Fn_err=None, **kwargs):
        """"""
        if (Qn is None) and (Qm is None):
            print('Need Q values and F values!')

        if ((not(Qn is None)) and (not(Fn_exp is None))):
            self.Qn     = Qn
            self.Fn_exp = StructureFactorModel(Qn, Fn_exp, Fn_err, units=None)
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
                self.magnetic.Fexp = StructureFactorModel(Qm, Fm_exp, Fm_err, units=None)
                self.Fm_err = Fm_err
                self.Fm_exp = self.magnetic.Fexp
            #
            Fm = np.asanyarray(self.calc_Fm(**kwargs))
            self.magnetic.Fm.coords = self.Qm #REFERENCE
            self.magnetic.Fm.values = Fm
            self.Fm = self.magnetic.Fm
            self.magnetic.magrepgroup
        return

    def loadStructureFactor(self, filename: str, typstr=None):
        H,K,L,F,Ferr = np.loadtxt(filename, unpack=True)
        Q = np.hstack((H,K,L,))
        if typstr is None:
            print("Please input the type string for the structure factor: 'nuc' or 'mag'.")
        elif typstr is ('nuc' or 'nuclear'):
            self.setStructureFactor(Qn=Q, Fn_exp=F, Fn_err=Ferr)
        elif typstr is ('mag' or 'magnetic'):
            self.setStructureFactor(Qm=Q, Fm_exp=F, Fm_err=Ferr)
        else:
            print("No valid input type string. Sorry!")
        return

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

    def plot(self):
        """Placeholder"""
        return

    def calc_Fm(self, Nrep=2, bvs=[2], mu=1.5, **kwargs):
        rep_name = 'G'+str(Nrep)
        rep_names=[]
        for bv in bvs:
            rep_names.append(rep_name+'_psi'+str(bv))

        coeffs={}
        for i_ in range(len(self.magnetic.magatoms)):
            for rn_ in rep_names:
                coeffs[rn_+'_'+str(i_+1)]=mu

        self.getMagneticMoments(Nrep=Nrep, bvs=bvs, coeffs=coeffs)
        Fm = self.magnetic.getMagneticStructureFactor(Qm=self.Qm, squared=True)
        return Fm

    def gen_magrepgroup(self):
        """"""
        self.magnetic.gen_magrepgroup()
        return

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
