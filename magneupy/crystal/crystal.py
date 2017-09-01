import string
from cmath import polar
from collections import OrderedDict
from itertools import combinations_with_replacement, product

import numpy as np
import periodictable as pt
from lmfit import minimize, Parameters, Minimizer
from pymatgen.io.cif import CifFile, CifParser

from magneupy.helper.functions import *
from magneupy.rep import *
from magneupy.crystal.nuclear import *
from magneupy.crystal.magnetic import *

rec2pol = np.vectorize(polar)

class Crystal(object):
    """
    A Crystal object is a collection of a NuclearStructure and its subclasses.
    TODO:
    * Long term, it would be nice to include exotic orders (e.g., a la Lucille Savary & Sentill Todadri)
    * Need a specified set<obj> function for each Child object that also associates with the Parent crystal. 
    * Allow input of CIF, MCIF (eventually), Sarah (eventually), etc. files in order to generate the whole Crystal from scatch
    """
    def __init__(self, nuclear=None, mag=None, cifname=None, charge=None, magrepgroup=None, nucrepgroup=None,
                 spacegroup=None):

        # Initialize the Structures
        if type(nuclear) is str:
            try:
                self.nuclear = NuclearStructure(cifname=nuclear)
            except:
                pass
        elif type(cifname) is str:
            try:
                self.nuclear = NuclearStructure(cifname=cifname)
            except:
                pass
        else:
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

    def setChild(self, child):
        """"""
        setattr(self, child.familyname, child)
        return

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
        vis = StructureVis(**kwargs)
        vis.set_structure(self)# FIX
        vis.show()
        return vis

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
        Fm = self.magnetic.getMagneticStructureFactor(Qm=self.Qm,
                                                      squared=True)
        return Fm

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
