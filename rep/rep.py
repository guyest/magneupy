"""
Update this to not use '*'
"""
#from pyCrystal import *
#from pyData import *

import numpy, string, inspect, re, fnmatch, pandas
from collections import Iterable, namedtuple, OrderedDict, deque

# This section has built-in classes for various uses (i.e. the irrep class used for fitting)
# TODO
# * Make all classes have optional input which defaults to the default value of the corresponding object.
#
# -----
# First for the parts pertaining to the ordering
# pyRep
# -----
class Rep(OrderedDict):
    """
    This serves as a base class for the different magnetic and structure representation formalisms.
    """
    def __init__(self):
        """"""
        super(Rep,self).__init__()
        return
#
class BasisVector(numpy.ndarray):
    """
    TODO:
    <done> Extends ndarray (see: http://docs.scipy.org/doc/numpy/user/basics.subclassing.html) 
    """
    def __new__(bv, input_array, d=None, Nbv=None, Nrep=None, Natom=None, Nunique_atom=None, norm=True):
        # Input array is an already formed ndarray instance, but we want to normalize it
        if norm:
            norm_fac = numpy.linalg.norm(input_array)
            if norm_fac > 0.:
                input_array = input_array / norm_fac
            #print input_array
        # Then cast to be our class type        
        obj = numpy.asanyarray(input_array).view(bv)
        # add the new attribute to the created instance
        obj.d = d
        obj.Nbv = Nbv
        obj.Nrep = Nrep
        obj.Natom = Natom   
        obj.Nunique_atom = Nunique_atom
        obj.name = 'atom'+str(Natom)+'_'+str(Nunique_atom)
        # Finally, we must return the newly created object:
        return obj  
    
    def __array_finalize__(self,obj):
        """
        TODO:
        * Should raise a better error
        """
        if obj is None: return
        if not (obj.shape == (3,)): raise TypeError("The input array is of the wrong shape.")
        for name in ['d', 'Nbv', 'Nrep', 'Nmagatom', 'name']:
            setattr(self, name, getattr(obj, name, None))
        return
    #
#
class BasisVectorGroup(OrderedDict):
    """
    BasisVectorGroups are composed of all BasisVectors for equivalent atoms produced by a single Rep. 
    For the full set of BasisVectors for a single atom including contributions from all Reps (and qm), use a BasisVectorCollection.
    """    
    def __init__(self, basisvectors=[], Nbv=1, Nunique_atom=1, names=None, orbit=None):
        """
        Nbv should be given by the Order of the Irrep times the number of Copies.
        I decided not to have the option to input a BasisVectorGroup directly. It is simple to make a list of BasisVectors.
        TODO:
        """       
        OrderedDict.__init__(self)
        #super(BasisVectorGroup, self).__init__()
        # Set the coefficient to a default value:
        self.setCoeff()
        self.name = 'psi'+str(Nbv)+'_'+str(Nunique_atom)
        
        return    
    #
    def setBasisVectors(self, basisvectors):
        """"""
        basisvectors = list(basisvectors)
        for basisvector in basisvectors:
            self[basisvector.name] = basisvector  
        self.setCoeff()
        return
    #
    def setCoeff(self, coeff=1.+1j*0.):
        """"""
        self.coeff = coeff
        return
    #
    def addBasisVector(self, basisvector):
        """
        TODO:
        * Check for the precense of that basisvector already
        """
        self[basisvector.name] = basisvector
        # set a coeff
        return
    #
    def getMagneticMoment(self, d):
        """"""
        m = numpy.asanyarray([0.,0.,0.], dtype=numpy.complex_)
        for bv in list(self.values()):
            if numpy.isclose(d,bv.d).all():
                m += bv * self.coeff        
        return m
    #
    def checkBasisVectors(self):
        # Perform some checks to make sure the objects are what we think they are
        assert(isinstance(self.basisvectors, list))
        
        # Composed of BasisVectors
        for basisvector in self.basisvectors:
            assert(isinstance(basisvector, BasisVector))
            
        # For the same atom
        testbv = self.basisvectors[0]
        for basisvector in self.basisvectors:
            assert(testbv.d == basisvector.d)  
        return
    #
    #def __len__(self): return len(self.basisvectors)
#
class BasisVectorCollection(Rep):
    """
    A BasisVectorCollection is a collection of BasisVectorGroups from different Reps but corresponding to the same atom. 
    It is broken up into named fields corresponding to each of the ordering wavevectors.
    """
    def __init__(self, basisvectorgroups=[BasisVectorGroup()], Nbv=1, names=None):
        """
        TODO:
        * Have named fields corresponding to each of the dynamically determined ordering wavevectors (namedtuple)
        """
        __slots__ = ()
        
        _fields = ('qm1', 'qm2')
    
        def __new__(_cls, qm1, qm2):
            'Create new instance of BasisVectorCollection(qm1, qm2)'
            return _tuple.__new__(_cls, (qm1, qm2))
    
        @classmethod
        def _make(cls, iterable, new=tuple.__new__, len=len):
            'Make a new BasisVectorCollection object from a sequence or iterable'
            result = new(cls, iterable)
            if len(result) != 2:
                raise TypeError('Expected 2 arguments, got %d' % len(result))
            return result
    
        def __repr__(self):
            'Return a nicely formatted representation string'
            return 'BasisVectorCollection(qm1=%r, qm2=%r)' % self
    
        def _asdict(self):
            'Return a new OrderedDict which maps field names to their values'
            return OrderedDict(list(zip(self._fields, self)))
    
        def _replace(_self, **kwds):
            'Return a new BasisVectorCollection object replacing specified fields with new values'
            result = _self._make(list(map(kwds.pop, ('qm1', 'qm2'), _self)))
            if kwds:
                raise ValueError('Got unexpected field names: %r' % list(kwds.keys()))
            return result
    
        def __getnewargs__(self):
            'Return self as a plain tuple.  Used by copy and pickle.'
            return tuple(self)
    
        __dict__ = _property(_asdict)
    
        def __getstate__(self):
            'Exclude the OrderedDict from pickling'
            pass
    
        qm1 = _property(_itemgetter(0), doc='Alias for field number 0')
    
        qm2 = _property(_itemgetter(1), doc='Alias for field number 1')        
        return    
    #
    def append(self):
        """"""
        self.bvs
        return
    #
#   
class Irrep(OrderedDict):
    """
    An Irrep class is composed of BasisVectorGroups corresponding to each atomic site in the compound.
    It contains identifying information for the Irrep in Kovalev notation.
    ...
    ----------
    Attributes:
    ...
    ----------
    TODO:
    <done> Moved frac_coords to within each BasisVector itself
    * Access the different atomic sites by a named field (namedtuple)
    """
    def __init__(self, qm=None, sg=None, N=None, Natoms=None, copies=None, order=None, bvg=None):
        """
        This will create an object to control the irrep and basis vectors from Sarah (or elsewhere) used for fitting the magnetic structure.
        TODO:
        * Long term, it would be nice if all the input here could be pulled from Sarah output. Certainly possible, but would take some careful work.
        """
        OrderedDict.__init__(self)
        # Set irrep number, how many copies present, and its order
        self.N      = N
        self.copies = copies
        self.order  = order        
        
        # Set the total number of basis vectors groups and atoms for this irrep
        self.bvg   = bvg
        self.Natoms = Natoms
        
        # Set the name of the irrep
        self.setName()
        
        # Set a flag for tracking whether all the basis vectors have been added.
        # Eventually want a more elegant way of doing this
        self.defined= False
        
        return
    #
    def setName(self, name=None):
        """"""
        if name is not None:
            self.name = name
        else:
            self.name = 'G'+str(self.N)
        return
    #
    def addBasisVectorGroup(self, frac_coord, psi):
        """"""
        
        self.psis
        return
    #
    def getBasisVectorGroup(self, atom):
        """
        """
        return
    #
    def checkIrrep(self):
        """
        TODO: 
        * Implement all the checks required to make sure the Irrep is sound.
        """
        assert(len(bvg) == Natoms)
        # ...
        return
    #
    def __add__(self, other):
        """"""
        return Corep([self, other])
    def __eq__(self, other):
        """"""
        return self.N == other.N
    #
    #
    # Do these make sense?
    #def __gt__(self, other):
        #""""""
        #return self.order > other.order
    ##
    #def __ge__(self, other):
        #""""""
        #return self.order >= other.order
    ##
    #def __lt__(self, other):
        #""""""
        #return self.order < other.order
    ##    
    #def __le__(self, other):
        #""""""
        #return self.order <= other.order
    ##    
    def __repr__(self): return repr(self.name)
    def __str__(self): return str(self.copies)+'$\Gamma_{'+str(self.N)+'}^{'+str(self.order)+'}'
#
class Corep(Irrep):
    """
    A Corep class combines two or more Irreps into a single Corep which otherwise behaves just the same way. The net effect is to increase the size of the constituitive BasisVectorGroups.
    ...
    ----------
    Additional Attributes:
    ...
    ----------
    TODO:
    * 
    """
    def __init__(self, irreps):
        
        return
#
class MSG(Rep):
    """
    This is a Magnetic Space Group object.
    ...
    ----------
    Attributes:
    ...
    ----------
    TODO:
    * So far from being ready.
    """
    def __init__(self):    
        
        return NotImplemented
    #
#
class RepGroup(OrderedDict):
    """
    > NOTE: This is a MUTEABLE type.
    TODO:
    
    * Will be the base class for MagRepGroup (and others) and may be able to replace them altogether... (later)
    <done> Implemented as a subclass of OrderedDict.
        * May want to edit the __getitem___ method so that its returns RepGroup[key-1]. That way irreps, etc. can be referenced by number as well. 
    """
    def __init__(self, reps=None, crystal=None, repcollection=None, basisvectorgroup=None, **kwargs):
        """
        TODO:
        * Add ability to input reps as a list and pull the names for dictionary labels
        """
        self.setFamilyName()
        super(RepGroup, self).__init__()
        
        self.basisvectorgroup = basisvectorgroup
        self.bvg = self.basisvectorgroup # alias
        self.setRepCollection(repcollection, rcname=kwargs['rcname'])
            
        # Ready the input
        self.setReps(reps) # also claims reps as children
        return
    #
    def setFamilyName(self, name='repgroup'):
        self.familyname = name   
        return
    #
    def setRepCollection(self, repcollection, name='repcollection'):
        """"""
        if repcollection is not None:
            if not hasattr(repcollection, RepCollection): raise TypeError ('The repcollection field is for a RepCollection type object (or subclass).')
            setattr(self, name, repcollection)
        else:
            setattr(self, name, repcollection)
        return
    #
    def setReps(self, reps):
        """"""
        if reps is None: pass
        elif not isinstance(reps, Iterable): reps = list(reps)
        else:
            for rep in reps:
                if not isinstance(rep, Rep): raise TypeError('The reps variable should be a Reps instance or subclass.')
                self[rep.name] = rep
                rep.setParents(self, child=rep)
                
        # Also set access to dict values by attribute
        for rep in self:
            setattr(self, rep.name, rep)
        return
    #   
    def getBasisVectorCollection(self, d):
        """
        This function will return the BasisVectorCollection object corresponding to the set of BasisVectors at a particular magnetic site.
        """
        return
    #
    def setParents(self, parents, **kwargs):
        """
        This method sets the parent reference of NuclearStructures to a Cystal and returns an AssertionError if the parent is of the wrong type
        """    
        errmsg = 'RepGroups expect a Crystal or RepCollection object as parent. Plase give an appropiate object to reference.'
        errmsg = kwargs['errmsg'] if 'errmsg' in kwargs else errmsg
        
        types  = (RepCollection, Crystal)
        types  = kmwargs['types'] if 'types'  in kwargs else types
        
        if hasattr(parents, '__iter__'):
            for parent in parents: 
                if parent is None: pass
                elif not isinstance(parent, types): raise TypeError(errmsg) 
                else: setattr(self, parent.familyname, parent)
                parent.setChild(self)
        else:
            self.setParents([parents], errmsg=errmsg, types=types)        
        return
    #
    def claimChildren(self, family = ['basisvectorcollection', 'basisvectorgroup'], child=None):
        """
        This is performed in the init stage so that all consitituents of the Nuclear Structure may back-refernece it by name.
        TODO:
        * Perhaps this is modified to include AtomGroups and/or to replace atoms with an AtomGroup
        * Need to include checks here that say whether the Child is of the right type. (Somewhat redundant as it is handled by the Child as well.)
        """
        if child is None:
            for each_child in self:
                each_child.setParents(self)
        else:
            child.setParents(self)
        
        attr = getFamilyAttributes(self, family)
        for a in attr:
            label, child = a
            if child is not None: child.setParent(self)        
        return 
    # 

#
class NucRepGroup(RepGroup):
    """
    TODO:
    * For implementing structural distortions
    """
    pass
#
class MagRepGroup(OrderedDict):
    """
    A MagRep class is a collection of Reps (MSG, Irrep, Corep, etc.) for magnetic order in a given system.
    There are the same number of Reps as there are ordering wavevectors, qm. 
    The fitting can be performed for any combination of those wavevectors and their corresponding Reps.
     ...
    ----------
    Additional Attributes:
    ...
    ----------
    TODO:
    * Long term, the idea is to incorporate the option to use magnetic spacegroups (MSGs) as well
    * Long term, it would be nice if this could be read directly from Sarah. Should be quite doable.
    ** Even more ideally, porting the functionality of Sarah and/or Bilbao to Python would be FANTASTIC. (See pycrystfml: in-progress...)
    
    """
    def __init__(self, reps=None, crystal=None, repcollection=None, sarahfile=None, basisvectorgroup=None, **kwargs):
        """
        Need to fix class relations...
        """
        self.setFamilyName()
        OrderedDict.__init__(self)
        
        self.IR0 = None
        self.basisvectorgroup = basisvectorgroup
        self.bvg = self.basisvectorgroup # alias
        self.setRepCollection(repcollection, rcname='magrepcollection')
            
        # Ready the input
        self.setReps(reps) # also claims reps as children      
        #super(MagRepGroup, self).__init__(reps=reps, crystal=crystal, repcollection=repcollection, rcname='magrepcollection')
        self.mrc = self.magrepcollection
        if sarahfile is not None: 
            self.readSarahSummary(sarahfile)
            return        
        return
    #
    def setReps(self, reps):
        """"""
        if reps is None: pass
        elif not isinstance(reps, Iterable): reps = list(reps)
        else:
            for rep in reps:
                if not isinstance(rep, Rep): raise TypeError('The reps variable should be a Reps instance or subclass.')
                self[rep.name] = rep
                rep.setParents(self, child=rep)
                
        # Also set access to dict values by attribute
        for rep in self:
            setattr(self, rep.name, rep)
        return
    #      
    def setRepCollection(self, repcollection, rcname='magrepcollection'):
        """
        TODO:
        * Should implement with super()...
        """
        if repcollection is not None:
            if not hasattr(repcollection, MagRepCollection): raise TypeError ('The repcollection field in a MagRepGroup is for a MagRepCollection type object.')
            setattr(self, rcname, repcollection)
        else:
            setattr(self, rcname, repcollection)
        return
    #   
    def addBasisVector(self, bv, Nirrep=None, Nbv=None, Nunique_atom=None, Natom=None):
        """
        TODO:
        * Needs to be finished passing on and correlating the information...
        """
        #if 'G'+str(Nirrep) in self.keys():
            
        self['G'+str(Nirrep)]['psi'+str(Nbv)+'_'+str(Nunique_atom)].addBasisVector(bv)
        
        return
    #
    def sarah2pyRep(self, lines, lid=None, **kwargs):
        """"""
        #line = str(line)
        if lid is None: raise TypeError('Please provide an identifier for the line to select the proper algorithm for translation')
        
        if lid == 'ATOM-IR':
            """
            Return the value of the basis vector and the atom number
            """
            
            Natom, _0, bv = lines.partition(':')
        
            # The atom number is:
            Natom = int(re.findall(r'\d+', str(Natom))[0])
            
            bvr, _0, bvi = bv.partition('+') 
            bvr = numpy.asanyarray([float(i) for i in re.findall( r'[-+]?\d*\.\d+|\d+',bvr)])
            bvi = numpy.asanyarray([float(i) for i in re.findall( r'[-+]?\d*\.\d+|\d+',bvi)])
            
            # The basis vector is:
            bv = bvr + 1j*bvi
            
            return  bv, Natom
        #
        elif lid == 'VECTOR':
            _0, _1, qm = lines.partition('=')
            qm = numpy.asanyarray([float(i) for i in re.findall(r'[+,-]?\d+', qm)])            
            return qm
        #
        elif lid == 'N-O':            
            start = lines.index('ORDERS OF THE REPRESENTATIONS:')+1
            stop  = lines.index('APPLICATION OF ANTIUNITARY THEORY LEADS TO THE FOLLOWING COREPRESENTATIONS:') 
            deq   = deque(lines[start:stop])
            lines[:stop] = []
            
            Nirreps = []
            Oirreps = []
            while len(deq) is not 0:
                Nirrep, _1, Oirrep = deq.popleft().partition(':')
                Nirrep = int(Nirrep)
                Oirrep = int(Oirrep)
                Nirreps.append(Nirrep)
                Oirreps.append(Oirrep)
                self['G'+str(Nirrep)] = Irrep(qm=self.qm, sg=None, N=Nirrep, Natoms=None, 
                                        copies=None, 
                                        order=Oirrep, 
                                        bvg=None)            
            return Nirreps, Oirreps
        #
        elif lid =='COREP':
            self.hasCorep = False
            start = lines.index('APPLICATION OF ANTIUNITARY THEORY LEADS TO THE FOLLOWING COREPRESENTATIONS:') + 2
            stop  = lines.index('COORDINATES OF PRINCIPAL ATOMS:')
            deq   = deque(lines[start:stop])
            
            Crs = []; Irs = []; Os = []
            foundAll = False
            while not foundAll:
                l = deq.popleft()
                _0, _1, l = l.partition(')')
                
                if len(l) > 0:
                    # Get the Corep symbol
                    Cr = str(re.findall('[ABC]', l))
                    Crs.append(Cr)
                    
                    # Get the Irreps order and numbers contributing to each Corep
                    O, Ir1, Ir2 = re.findall(r'\d+', l)
                    Os.append(O)
                    Irs.append((int(Ir1),int(Ir2)))
                
                # Check if all the Irreps are represented
                found = True
                foundIrreps = list(numpy.asanyarray(Irs).flatten())
                for Nirrep in self.Nirreps:
                    found *= (Nirrep in foundIrreps)
                foundAll = found  
                
            if 'C' in Crs: self.hasCoreps = True
            return Crs, Os, Irs
        #
        elif lid == 'COORDS':
            Nunique_a = kwargs['Nunique_a']
            Natoms = 0; ds = {}
            for l in lines:
                Natom, _0, d = l.partition(':')
                
                # Grab the atom number
                Natom = int(re.findall(r'\d+', Natom)[0])
                Natoms = max(Natom, Natoms)
                #print Natom
                
                # Grab the coordinate of the atom
                dx, dy, dz = tuple(re.findall(r'[.]?\d+', d))
                ds[str(Natom)+'_'+str(Nunique_a)] = numpy.asanyarray([float(dx),float(dy),float(dz)])   
            return ds, Natoms
    #
    def readSarahSummary(self, filename):
        """
        Definitely not most efficient, but quick and dirty
        """
        # Get the data just as a list of strings for each line
        data = pandas.read_table(filename)
        lines = []
        for l in list(data.values[:][:]):
            lines.append(str(l[0])) 
        for l in lines:
            lines[lines.index(l)] = str(l)        
            
        # Get the descriptive information by reading the beginning lines
        # Read in the ordering wavevector, qm:
        self.qm = self.sarah2pyRep(fnmatch.filter(lines, 'VECTOR K*=*')[0], lid='VECTOR')        
        
        # Get a list of Irreps and their orders, while also making those Irreps in the MagRepGroup along the way.   
        Nirreps, Oirreps = self.sarah2pyRep(lines, lid='N-O')
        self.Nirreps = Nirreps
            
        # The total number of Irreps is determined as:
        self.Nreps = len(self.Nirreps)
                
        # Now get information regarding any possible Correps
        Crs, Os, Irs = self.sarah2pyRep(lines, lid='COREP')
        
        # Find out how many distinct atoms (orbits) are generated. This sets the number of BasisVectorGroups
        # can do a fnmatch.filter for 'ORBITS ARE PRESENT' if you wish
        idx_unique_atoms = []
        for l in fnmatch.filter(lines, 'ANALYSIS FOR ATOM*'):
            ## Get the atom names here too.
            idx_unique_atoms.append(lines.index(l)+2)
            
        #idx_unique_atoms[-1] = len(lines)
        #idx_unique_atoms = [m.start() for m in re.compile(re.escape(lines)).finditer('ANALYSIS FOR ATOM*')]
                        
        # Get the indicies for the coordiantes to end the substring used to extract the positions of the atoms
        idx_coords_stop = []; i=0
        for l in fnmatch.filter(lines, 'DECOMPOSITION OF THE MAGNETIC REPRESENTATION INTO IRs OF Gk:'):
            """
            *!! FIX !!*
            """        
            idx_coords_stop.append(lines.index(l, idx_unique_atoms[i]))
            i+=1            
            
        a_start = idx_unique_atoms[0]
        for a_stop in idx_unique_atoms[1:]+[-1]:
            
            a_lines = lines[a_start:a_stop]
            Nunique_a = idx_unique_atoms.index(a_start)+1
            #print Nunique_a
            
            # Get the atomic positions as a dict labeled by the atom number
            # eventually NEED to check against the CIF for atom labels or make sure that they are adjusted according to the fractional coordinates.
            ds, Natoms = self.sarah2pyRep(lines[a_start:idx_coords_stop[idx_unique_atoms.index(a_start)]], lid='COORDS', Nunique_a=Nunique_a)

            # Search for the lines beginning the Irreps so as to later get the BasisVectors from them
            subl_irrep = deque(fnmatch.filter(a_lines, 'IR #*, BASIS VECTOR: #*(ABSOLUTE NUMBER:#*)'))
            # This starts each BVG then find all the atom lines between this and the '******' line to add to a single basis vector number (Nbv)
            
            while len(subl_irrep) is not 0:
                """
                TODO:
                * Handle flags for when to increase the Atom#, etc. in this loop through the file so we know where to look for the fractional coordiantes.
                """
                # Grab the string for each Irrep-BasisVector pair
                l = subl_irrep.popleft()
                #print l
                
                # Parse the Irrep and BasisVector number from the line
                Nirrep, Nbv, Nbv_abs = [int(s) for s in re.findall(r'\d+', l)]  
                
                # Add a BasisVectorGroup to the Irrep for each set of atoms sharing a basis vector
                self['G'+str(Nirrep)]['psi'+str(Nbv)+'_'+str(Nunique_a)] = BasisVectorGroup(basisvectors=[], 
                                                        Nbv=Nbv, Nunique_atom=Nunique_a,
                                                        names=None, 
                                                        orbit=None)
                
                # Grab the index of the lines for each atom's basis vectors in the current group and combine into deque
                start = lines.index(l)+1
                stop  = start + Natoms+1 # need to handle the setting of Natoms
                deq = deque(lines[start:stop])
                
                # Go through the deque and add the basis vector values to the MagRepGroup
                while len(deq) is not 0:
                    line = deq.popleft()
                    #print line
                    if ('*' not in line) and ('#' not in line):
                        bv, Natom = self.sarah2pyRep(line, lid='ATOM-IR')
                        assert(Natom <= Natoms)
                        bv = BasisVector(bv, d=ds[str(Natom)+'_'+str(Nunique_a)], Nbv=Nbv, Nrep=Nirrep, Natom=Natom, Nunique_atom=Nunique_a)
                        self.addBasisVector(bv, Nirrep, Nbv, Nunique_a, Natom)
                    else:
                        pass
            
            a_start = a_stop
        
        if self.hasCorep:
            """
            Need to restructure the MRG to have Coreps. 
            This is actually easier than it sounds because we can just ignore the Irreps making up the Coreps or force their coefficients to vary together.
            """
            # Make correps
            #self
            pass
        return
    #    
    def setFamilyName(self, name='magrepgroup'):
        self.familyname = 'magrepgroup' 
        return
    #
    def getMagneticMoment(self, d, Nrep=None):
        """"""
        if Nrep is None: Nrep = self.IR0
        m = numpy.asanyarray([0., 0., 0.], dtype=numpy.complex_)
        for bvg in list(self['G'+str(Nrep)].values()):
            m += bvg.getMagneticMoment(d)
        return m
    #
    def setBasisVectorCollection(self, basisvectorcollection=None):
        """"""
        self.basisvectorcollection = basisvectorcollection
        self.bvc = self.basisvectorcollection # alias
        return
    #
    def setParents(self, parents):
        """"""
        errmsg = 'MagRepGroups expect a Crystal or MagRepCollection object as parent. Plase give an appropiate object to reference.'
        types = (MagRepCollection,Crystal)
        super(MagRepGroup, self).__setParents__(parents, errmsg=errmsg, types=types)
        return
    #
    def claimChildren(self, family = ['basisvectorcollection', 'basisvectorgroup'], child=None):
        """
        This is performed in the init stage so that all consitituents of the Nuclear Structure may back-refernece it by name.
        TODO:
        * Perhaps this is modified to include AtomGroups and/or to replace atoms with an AtomGroup
        * Need to include checks here that say whether the Child is of the right type. (Somewhat redundant as it is handled by the Child as well.)
        """
        if child is None:
            for each_child in self:
                each_child.setParents(self)
        else:
            child.setParents(self)
        
        attr = getFamilyAttributes(self, family)
        for a in attr:
            label, child = a
            if child is not None: child.setParent(self)        
        return 
    #     
#
class RepCollection(OrderedDict):
    """
    > NOTE: This is a MUTEABLE type.
    TODO:
    * Decide if this is neccessary...
    """
    
class MagRepCollection(RepCollection):
    """"""
    pass

def getTrimmedAttributes(obj):
    """
    This function returns a list of attributes of the input object (class) as a list of tuples each of the form: ('field', <instance>)
    pulled from: http://stackoverflow.com/questions/9058305/getting-attributes-of-a-class
    """
    attributes = inspect.getmembers(obj, lambda a:not(inspect.isroutine(a)))
    return [a for a in attributes if not(a[0].startswith('__') and a[0].endswith('__'))]

def getFamilyAttributes(obj, family):
    """"""
    attr = getTrimmedAttributes(obj)
    attributes = []
    if family is not None:
        for fam in family:                    
            attributes.append(attr.pop(attr.index((fam, obj.__getattribute__(fam)))))            
    return attributes