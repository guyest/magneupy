import numpy as np
import periodictable as pt
from lmfit import Minimizer

from magneupy.helper.functions import *
from magneupy.rep.rep import BasisVectorCollection, MagRepGroup
from magneupy.crystal.nuclear import Atom, AtomGroup, NuclearStructure
from magneupy.crystal.crystal import Crystal

class MagAtom(Atom):
    """
    ...
    ----------
    Additional Attributes:
    ...
    ----------
    TODO:
    <done> See get_form_factor to include the form factor generator for this species at a given Q.
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

    def ff(self, Qm, **kwargs):
        """
        Convenience alias for get_form_factor
        """
        return self.get_form_factor(Qm, **kwargs)

    def get_form_factor(self, Qm, rlu=True, S=1 / 2, L=3,
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

    @property
    def phase(self, N=None):
        """
        Should actually just be an 
        """
        return

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

    def getMoment(self, N=None):
        """"""
        if N is not None:
            self.moment = self.moment.reshape((1,3)) # may not need
            return np.repeat(self.moment, N, axis=0)
        else:
            return self.moment

    def setMomentFromIR(self, m, phi=0):

        return

    def setMomentSize(self, mu=None):
        if mu is None:
            pass
        else:
            self.mu = mu
            self.addMoment(self.moment)
        return

    def setPhase(self,phi=None):
        """"""
        if phi is None:
            pass
        else:
            self.phi = phi
            self.addMoment(self.moment)
        return

    def getMomentSize(self):
        """
        TODO:
        * Confirm defintion of moment size in terms of self.moment, then normalize by proper factor to put something like np.norm(self.moment) in Bohr magneton.
        """
        return np.linalg.norm(self.moment)

    def setParent(self, parent):
        """"""
        if parent is None: pass
        elif not isinstance(parent, (MagneticStructure, BasisVectorCollection)): raise TypeError('MagneticAtoms expect a MagneticStructure or BasisVectorCollection type as their parent. Please provide a proper object to reference.')
        else: setattr(self, parent.familyname, parent)
        return


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

    def checkMagAtomGroup(self):
        """
        This function will implement a check on moments of the constituent atoms to make sure they are of the right type and shape as expected.
        """
        return

    def setqms(self):
        """
        TODO:
        * This should be present in the BasisVectorCollection only.
        """
        return


class MagneticStructure(NuclearStructure):
    """"""
    def __init__(self, magnames=None, magatoms=None, nuclear=None, qms=None, \
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

    def setMagneticRefinement(self, params, **kwargs):
        """"""
        self.fitter = Minimizer(self.residual, params, **kwargs)
        return

    def refineMagneticStructure(self, params=None, **kwargs):
        self.res = self.fitter.minimize(params=params, **kwargs)
        return self.res

    def residual(self, params, **kwargs):
        self.update(params, Qm=self.Fexp.coords, returned=False, update=True,
                    **kwargs)
        data = self.Fexp.values
        calc = self.Fm.values
        err = self.Fexp.errors
        res = (data - calc) / err
        return res

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

    def getMagneticDiffraction():
        """"""

        return

    def claimChildren(self, family=['magatoms']):
        """
        This function sets the Children according to the NuclearStructure defintion
        """
        super(MagneticStructure, self).claimChildren(family=family)
        return

    def setParents(self, parents):
        """
        This method sets the parent reference of NuclearStructures to a Cystal and returns a TypeError if the parent is of the wrong type
        """
        errmsg = 'MagneticStructures expect a Crystal or MagRepGroup type as their parent. Please provide a Crystal type object to reference'
        types  = (Crystal, MagRepGroup)

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

    def setSibling(self, family=['nuclear']):
        """"""
        return


