"""
TODO:
v For Data, subclassing a xarray.Dataset
* Make sure to have a raw_plot option for all the DataContainers
* Note: Variable defined outside of any functions in a class are available
as attributes to ALL instances!
* Check out "sympy" in more detail -- has Clebsch-Gordon coefficients!
"""

from ..PDAt import *# functions, gunits, gureg
#from ggm.PDAt.fit import GaussianModel
#from functions import GaussianModel
from .fit import GaussianModel
#from ggm import gunits, gQ, gureg
#import pubplot


#import pyRep, pyCrystal
import warnings

# Pick one of the following?
# The winner is ... pint! (for now)
# natu has the most physics-like features but isn't sufficiently developed
# astropy.unit is an awesome option as well, but burried in a astro-centric module
# cfunits has good options and may be ideal for archival with xarray
import pint
#import astropy.unit, cfunits, natu

# Use pubplot instead (have a "sketch" submodule with no TeX)
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D

# Other namespace imports
import sys, tabulate, pandas, xarray, h5py, os, lmfit#, liborigin
"""
For liborigin use liborigin.parseOriginFile('filename').  Out put is a dict
which holds various things. The key that matters (as far as I can tell) is 
'spreads', which holds a list of various data sets (Spreadsheets). Each item 
in the list is a list of Spreadsheets again, each having a list of
SpeadColumns in the 'columns' attribute.  Each SpreadColumn has a
'data' attribute and a 'comment' attribute which seem to be used for the ...
data (duh) and the data label (comment seems weird for that though).
"""
# also: terapy for terahertz/optics (http://pythonhosted.org/terapy/)
""" 
This code is MUCH more sophistocated than the above. The hack of the beta
I provide in 'pysonal-packages' almost works. Has some bugs that were missed 
by the authors. CAN read in sheets only though, which may be enough
"""
# if the above give trouble, can use demo version of QitPlot to easily export
# (without column labels, but they can be added by hand...)
import numpy as np
import scipy.signal as sig

# Direct imports
import uncertainties as unc
from uncertainties import unumpy as unp
from numpy.matlib import repmat
from collections import namedtuple, Iterable, OrderedDict
from scipy.interpolate import griddata
from scipy import constants
from copy import deepcopy, copy
from lmfit import Model
import pickle as pkl
import time

# Deal with Mantid...
if sys.platform == 'darwin':
    mac_paths = [#'/usr/local/lib/python2.7/site-packages/', 
                 '',                 
                '/Users/Guy/Documents/Johns Hopkins/Academics/Broholm Group/Code/',
                '/Applications/MantidPlotStable.app/Contents/MacOS',
                '/Applications/MantidPlotStable.app/Contents/MacOS',
                '/Applications/MantidPlotStable.app/Contents/MacOS/../lib',
                '/Applications/MantidPlotStable.app/scripts',
                '/Applications/MantidPlotStable.app/scripts/Engineering',
                '/Applications/MantidPlotStable.app/scripts/Inelastic',
                '/Applications/MantidPlotStable.app/scripts/Reflectometry',
                '/Applications/MantidPlotStable.app/scripts/SANS',
                '/Users/Guy/Documents/Johns Hopkins/Academics/Broholm Group/Code',
                '/Applications/MantidPlotStable.app/Contents/MacOS/IPython/extensions']    
    for path in mac_paths:
        if sys.path.count(path) < 1: 
            if mac_paths.index(path) == 0:
                sys.path.insert(0, path)
            else:
                sys.path.append(path)
else:
    linux_paths = ['/opt/mantidnightly/bin/']    
    for path in linux_paths:
        if sys.path.count(path) < 1: sys.path.append(path)
#from mantid.simpleapi import mtd, LoadMD


################################# --- Data  --- ##############################
class Data(object):
    pass
#
class DataSet(xarray.Dataset): 
    """
    The most general data container I have found implemented already is an
    xarray.Dataset, which truly has a remarkable tools. It builds on the 
    excellent pandas package and provides great default plotting options with
    cues taken from Seaborne (a la prettyplotlib).
    TODO:
    v Subclassed xarray.Dataset
    """
    def archive(self, filepath, save_all=False):
        """
        Archive the in-memory data to disk at the specified filepath. Default
        is to save as a Python pickle for easy recovery in future sessions
        as well as to an HDF5/netcdf file for portability.
        """
        #for _, da in self.iteritems():
            #assert np.all(map(isinstance,da,[unc.Variable]*len(da)))
        #
        new_ds = DataSet(self.copy())#deep=True
        for key,da in list(new_ds.items()):
            if np.all(list(map(isinstance,da.values,[unc.Variable]*len(da))))\
            or np.all(list(map(isinstance,da.values,[unc.AffineScalarFunc]*len(da)))):
                da.attrs['std_devs'] = unp.std_devs(da.values)  
                da.values = unp.nominal_values(da.values)   
                print('made it')
                
            else: 
                try:
                    print('failing some')
                    
                    da.attrs['std_devs'] = da.attrs['std_devs']
                except:
                    print('failing all the way')
                    da.attrs['std_devs'] = np.nan*np.zeros_like(da.values)
        #
        new_ds.to_netcdf(filepath)
        new_ds.to_dataframe().to_csv(filepath.rstrip('.nc')+'.csv')
        new_ds.close()
        
        #for key,da in new_ds.items():
            #try:
                #da.values = da.attrs['std_devs'] 
            #except:
                #da.values = np.zeros_like(da.values)    
            #if key == 'scan':
                #print da.values
            
        
        ## Save the errors separately in text format        
        #new_ds.to_dataframe().to_csv(filepath.rstrip('.nc')+'.std_devs.csv')
        return
    #
    def load(self,filepath,**kwargs):
        """
        Uses the class method 'read_netcdf' to load in a cdf
        """
        self = self.read_netcdf(filepath,**kwargs)
        return
    #
    def get_labels(self,labels={},from_array=False):
        labels['suptitle'] = self.suptitle
        return labels
    #
    @classmethod
    def read_netcdf(cls,filepath,**kwargs):
        """
        Uses xarray.open_dataset to load a '.nc' (netcdf4) from disk and restore
        the uncertainties.Variable arrays which could not be pickled properly.
        """
        print('reading!')
        new_ds = DataSet(xarray.open_dataset(filepath,**kwargs))
        for key, da in list(new_ds.items()):
            val = da.values            
            err = da.attrs.pop('std_devs')
            if np.all(np.equal(err,0)):
                da.values = val     
            else:
                print('adding error array for '+key+'!')
                print(val.shape)
                print(err.shape)
                da.values = unp.uarray(val,err)    
                print('...finished that one.')
        #xarray.close_dataset(filepath)
        print('closed file!')
        return new_ds
    #
#
class DS(xarray.DataArray):      
    """
    This is the simplest case of a data object (when 1D). It can consist of 
    arbitrarily dimensioned data. I have chosen to stick with the xarray package
    implementation for now (this could change of course without impacting the
    higher level code, which is the beauty) and is essentially equivalent to 
    a pandas.DataFrame (to which it can also be cast.)
    
    These DataSlices are the components of the DataSet.
    
    TODO:
    * Implement DataUnits as a known attribute on each DataSlice. They could 
    be indexed for each coord/value by name (string) of that variable.
    """
    #def __init__(*args,**kwargs):
        #super(DS,self).__init__(*args,**kwargs)
        #return
    ##
    def get_labels(self,labels={},dim=None):
        dim = self.coords.dims[0] if dim is None else dim
        labels['xlabels'] = self.coords.get(dim).axislabel
        try:
            labels['ylabels'] = self.attrs['axislabels']
        except KeyError:
            labels['ylabels'] = self.attrs['ylabels']
        try:
            labels['dlabels'] = self.attrs['dlabels']  
        except KeyError:
            try:
                labels['dlabels'] = self.attrs['vlabels']
            except KeyError:
                labels['dlabels'] = self.attrs['linelabel']
        return labels
    #
    def plot(self,ps='Presentation',**kwargs):
        if len(self.dims) == 1:
            labels={}
            labels['xlabels'] = self.coords.get(self.coords.dims[0]).axislabel
            x = unp.nominal_values(list(self.coords.values())[0])
            xerr = unp.std_devs(list(self.coords.values())[0])
            
            labels['ylabels'] = self.attrs['axislabel']
            y = unp.nominal_values(self.values)
            yerr = unp.std_devs(self.values)
            
            labels['title'] = self.attrs['title']
            axes,fig = pubplot.plot(x, y, yerr=yerr, xerr=xerr, labels=labels,ps=ps,
                              **kwargs)
        else:
            raise NotImplemented('# of dimensions too great for simple plot.')
        return axes,fig
    #
    def archive(self,filename=None, save_all=False):
        """
        """
        tstamp = '_'+''.join(str(time.time()).split('.'))
        if self.name is None:
            self.name = 'unnamed'
        if filename is None:
            filename = self.name
        ds = DataSet(self.to_dataset())
        ds.archive(filename+tstamp+'.nc')
        
        if save_all:
            with open(filename+tstamp+'.pickle','w') as f:
                pkl.dump(self,f)
            df = self.to_dataframe()
            df.to_csv(filename+tstamp+'.csv')
            df.to_pickle(filename+tstamp+'.pandas.pickle')
        return
    #

#
class DataSlice(xarray.DataArray):      
    """
    This is the simplest case of a data object (when 1D). It can consist of 
    arbitrarily dimensioned data. I have chosen to stick with the xarray package
    implementation for now (this could change of course without impacting the
    higher level code, which is the beauty) and is essentially equivalent to 
    a pandas.DataFrame (to which it can also be cast.)
    
    These DataSlices are the components of the DataSet.
    
    TODO:
    * Implement DataUnits as a known attribute on each DataSlice. They could 
    be indexed for each coord/value by name (string) of that variable.
    """
    def __init__(self, value, value_error=None, value_label=None,
                 coords=None, cerrors=None, clabels=[],
                 encoding=None, axislabel=None, ticks=None, dims=None,
                 **kwargs):
        """
        Creates a DataSlice object with defaults initialized to None.
        TODO:
        * Implement ticks keyword overriding the DataLabel.ticks
        """
        if isinstance(coords,DataLabel):
            coords = coords.toCoordinates()
        elif isinstance(coords, (Coordinates,xarray.Coordinate,OrderedDict)):
            if dims is None:
                try:
                    dims = value.dims
                except:
                    raise ValueError(""" It is great that you're using Coordinates!
                    Way to go! ... but you then need to specify a list of 
                    dimensions ('dims=') so I know which are corresponding to the
                    dependence of the data given its shape.""")
        else:
            #coords = Coordinates() 
            warnings.warn('The cunits and cnames fields must be ALIGNED at this\
                      stage in development. You may not like the output if\
                      they are not... i.e. it would likely be wrong. Also,\
                      make sure the pairing is (cname,cunit). If the setup\
                      seems contrived, I agree. Just use the DataLabelGroup!\
                      Using a dict would also be fine.')
            
            isIndependent = True
            for cname, cunit in clabels:
                dl = DataLabel(cname, ticks, unit, isIndependent,
                               axislabel=axislabel)
                coords[cname] = dl
                            
        if value_error is not None:
            value = unp.uarray(value,value_error)
         
        # Pull the name and units from the value_label Coordinates object 
        # Needs to be generalize for multiple values...
        try:
            value_name = value_label.name
        except:
            try:
                value_name = value.name
            except:
                value_name = 'unknown'
        try:
            value_unit = value_label.unit
        except:
            value_unit = value.attrs['unit']            
        value_attrs= OrderedDict()
        value_attrs['unit'] = value_unit
        
        # Any remaining kwargs are passed as attributes to the data value
        for key, atr in kwargs.items():
            value_attrs[key] = atr
        if not 'value_label' in kwargs:
            value_attrs['axislabel'] = str(value_unit)
        else:
            value_attrs['axislabel'] = str(value_unit)
        value_attrs.update(kwargs)
        
        ars = (value,)
        kws = {'coords':coords, 'dims': dims, 'name':value_name, 
               'attrs':value_attrs, 'encoding':encoding}
        super(DataSlice,self).__init__(*ars, **kws)
        # Squeeze out the 1D dimensions that had to be input for compatibility
        #self = self.squeeze()
        return
    #
    def get_labels(self,labels={},dim=None):
        dim = self.coords.dims[0] if dim is None else dim
        labels['xlabels'] = self.coords.get(dim).axislabel
        labels['ylabels'] = self.attrs['axislabel']
        try:
            labels['dlabels'] = self.attrs['dlabels']  
        except KeyError:
            try:
                labels['dlabels'] = self.attrs['vlabels']
            except KeyError:
                labels['dlabels'] = self.attrs['axislabel']
        return labels
    #
    def plot(self,ps='Presentation',**kwargs):
        if len(self.dims) == 1:
            labels={}
            labels['xlabels'] = self.coords.get(self.coords.dims[0]).axislabel
            x = unp.nominal_values(list(self.coords.values())[0])
            xerr = unp.std_devs(list(self.coords.values())[0])
            
            labels['ylabels'] = self.attrs['axislabel']
            y = unp.nominal_values(self.values)
            yerr = unp.std_devs(self.values)
            
            labels['title'] = self.attrs['title']
            axes,fig = pubplot.plot(x, y, yerr=yerr, xerr=xerr, labels=labels,ps=ps,
                              **kwargs)
        else:
            raise NotImplemented('# of dimensions too great for simple plot.')
        return axes,fig
    #
    def check_units(self, every=True, **kwargs):
        """
        This method checks the units for the DataSlice. It is just for
        consistency, e.g. the 'values' should all be in the same units.
        """
        return
    #
    @classmethod
    def _test(cls, *args, **kwargs):
        """
        This function perfroms a test of the features of a DataSlice. It is 
        also useful to get acquanted with the object.
        """
        if len(args)+len(kwargs) < 1:
            
            y, coords, value_label, title, labels = cls._get_test_data()
            ds = cls(y, value_label=value_label, coords=coords)
        
        else:
            ds = cls(*args, **kwargs)
        
        ax = ds.plot(labels=labels, figsize=(8,6))
        
        return ds, ax
    #
    @classmethod
    def _get_test_data(cls):
        """
        Produce some test data, labels, and title.
        """
        x = unp.uarray(np.linspace(0,2.*np.pi,100),
                       np.abs(0.05*np.random.randn(100)))
        coords = DataLabel('x', x, 'meters', True, 
                                 axislabel=r'Distance ($m$)')
        coords = coords.toCoordinates()
        
        y = unp.sin(x)
        value_label = DataLabel('y', None, 'meters/second', False, 
                              axislabel=r'Velocity ($m/s$)')
        value_label = value_label.toCoordinates()   
        
        # Need a self.get_labels() instance method
        title = r'Progress in the average PhD'
        labels = {'xlabels': r'Distance ($m$)', 'ylabels': r'Velocity ($m/s$)',
                  'title': title}        
        
        return y, coords, value_label, title, labels
    #
    def archive(self,filename=None):
        """
        """
        tstamp = '_'+''.join(str(time.time()).split('.'))
        if filename is None:
            filename = self.name
        pkl.dump(self,open(filename+tstamp+'.pickle','w'))
        ds = self.to_dataset()
        #err = unp.std_devs(ds)# find a way to save these
        #ds.value = unp.nominal_values(ds)
        #ds.to_netcdf(filename+tstamp+'.cdf')
        return
    #
#
class DataLabel(tuple):
    """
    ** Good learning experience... BUT NEEDS TO BE REPLACED BY xarray.Coordinate
    subclass **
    This class provides a general object for labeling a data dimension. It
    consists of a plain text name for the dimension (readable on stdout), a
    Unit object for the dimension, and an optional axislabel for plotting 
    (otherwise the name will be used).
    
    Self is (name, ticks, attrs) input to dims field of DataSlice. The ticks
    input is optional to DataLabel but required in DataSlice. If it is not
    provided here, then its value will be 'None' and the data values will be 
    provided as tick labels by the DataSlice.
    
    If the coordinates are 'independent' then they must have ticks.
    
    Base code is output from namedtuple.
    """
    __slots__ = ()

    _fields = ('name', 'ticks', 'attrs')
    
    def __new__(_cls, name, ticks, unit, isIndependent,
                attrs=None, axislabel=None):
        'Create new instance of DataLabel(name, ticks, attrs)'
        
        if isIndependent and ticks is None:
            raise ValueError('Ticks must be provided for indep. variable!')
        
        try:
            unit = Unit(unit)
            #
        except:
            raise TypeError('The unit must be castable as a Unit object.')        
    
        try:
            name = str(name)
        except ValueError:
            raise TypeError('The name must be castable as a string')        
    
        if attrs is None:
            attrs = OrderedDict()
            attrs['unit'] = unit            
            attrs['axislabel'] = name if axislabel is None else axislabel
            attrs['isIndependent'] = isIndependent
        else:
            try:
                attrs = OrderedDict(attrs)
            except:
                raise TypeError('The attrs must be dict-like.')
            
        tup = (name,ticks,attrs)
        return tuple.__new__(_cls, tup)
    #
    @property
    def __dict__(self):
        return self._asdict(self)
    @property
    def unit(self):
        return self[2]['unit']    
    @property
    def is_independent(self):
        return self[2]['isIndependent']  
    @property
    def hasTicks(self):
        return self.ticks is not None
    @property
    def attrs(self):
        return self[2]     
    @property
    def ticks(self):
        return self[1]    
    @property
    def name(self):
        return self[0]     
    def addTicks(self, ticks):
        """
        Convenience method for adding ticks to a DataLabel after it is made.
        """
        self.ticks = ticks
        return
    # 
    @classmethod
    def _make(cls, iterable, new=tuple.__new__, len=len):
        'Make a new DataLabel object from a sequence or iterable'
        result = new(cls, iterable)
        if len(result) != 3:
            raise TypeError('Expected 3 arguments, got %d' % len(result))
        return result
    #
    def toCoordinates(self):
        """
        This instance method takes a DataLabel returns a Coordinates object. 
        """
        return Coordinates([self])
    #
    def __repr__(self):
        'Return a nicely formatted representation string'
        return 'DataLabel(name=%r, ticks=%r, attrs=%r)' % self
    #
    def _asdict(self):
        'Return a new OrderedDict which maps field names to their values'
        return OrderedDict(list(zip(self._fields, self)))
    #
    def _replace(_self, **kwds):
        'Return a new DataLabel object replacing specified fields with new values'
        result = _self._make(list(map(kwds.pop, ('name', 'ticks', 'attrs'), _self)))
        if kwds:
            raise ValueError('Got unexpected field names: %r' % list(kwds.keys()))
        return result
    #
    def __getnewargs__(self):
        'Return self as a plain tuple.  Used by copy and pickle.'
        return tuple(self)
    #
    def __getstate__(self):
        'Exclude the OrderedDict from pickling'
        pass
    #
#
    
#  
class Coordinates(OrderedDict):
    """
    This class extends OrderedDict to provide specific functionality for 
    handling DataLabels.
    TODO:
    * Seems that one should simply extend xarray.Coordinate...
    """
    
    def __init__(self,datalabels=[]):
        """
        Constructor
        """
        super(Coordinates,self).__init__()
        
        for dl in list(datalabels):
            # Check for ticks                
            if (not dl.hasTicks) and (dl.is_independent):
                raise ValueError('The DataLabel needs labels of some kind!')            
            
            name = dl.name
            ticks= dl.ticks
            attrs= dl.attrs
             
            self[name] = xarray.Coordinate(name, ticks, attrs=attrs)
            #self.units[name] = dl.unit        

        
        
        
        
        return
    #
    @property
    def names(self):
        """
        Returns the names of the consitituent DataLabels.
        """
        return list(self.keys())
    #
    @property
    def list(self):
        """
        Return list of DataLabel tuples for input to DataSlice.
        """
        return list(self.values())
    #
    @property
    def tick_dict(self):
        """
        Return input sensible when extra coordinates are provided.
        """
        res={}
        for key,value in self.items():
            res[key] = value.ticks
        return res
    #
    @property
    def units(self):
        """
        
        """
        res={}
        for key,value in self.items():
            res[key] = value.attrs['unit']        
        return
    #
#    
class Unit_new(gunits):
    pass
#
class Unit(object):
    """
    Thi class provides a way to assign a unit attribute to (mainly) 
    Data.values or Data.coords
    TODO:
    * Need a lookup for units by string or similar human readable input (SEE
    pints OR natu FOR THIS!)
    * Consider scimath.units from Enthought as well!
    """

    def __init__(self, unitname):
        """
        Constructor
        """
        self.name = unitname
        
        return
    #
    def __repr__(self):
        """
        String representation
        """
        return self.name
    #
# 
class ArbUnits(Unit):
    """
    The ArbUnits class is a subclass to Unit to handle cases when arbitrary 
    units are required because no true units can be assigned. In those cases,
    we must ensure that any quantities being combined have the SAME 
    (as evaluated by 'is' comparison) even though they are arbitrary.
    """
    pass
#
class ExperimentalData(Data):
    """
    ExperimentalData is a form of Data which issues a Warning upon any use
    if there are not valid values for any or all of the error or units.
    
    
    """
    def __init__(self, values, verrors, coords, cerrors, units):
        """"""
        super(ExperimentalData, self).__init__(coords=coords, values=values,
                                               verrors=verrors,
                                               cerrors=cerrors, units=units)
        return
    #
#
class ExperimentalDataSet(DataSet):
    """
    ExperimentalData is a form of Data which issues a Warning upon any use
    if there are not valid values for any or all of the error or units.
    
    
    """
    def __init__(self, values, verrors, coords, cerrors, units):
        """"""
        super(ExperimentalData, self).__init__(coords=coords, values=values,
                                               verrors=verrors,
                                               cerrors=cerrors, units=units)
        return
    #
#
class ExperimentalDataSlice(DataSlice):
    """
    ExperimentalDataSlice is a subclass of DataSlice which issues a Warning 
    upon any use if there are not valid values for any or all of the error 
    or units. It will raise an Error if there is no error provided for the 
    values and it does not know how to compute them from the data.
    
    (self, value, value_error=None, value_label=None,
                 coords=None, cerrors=None, clabels=None,
                 encoding=None, axislabel=None, ticks=None, **kwargs):
    """

    def __init__(self, value, coords, value_error=None, meta=None, **kwargs):
        """
        Constructor
        """
        #if value_error is None and np.all(unp.std_devs(np.asanyarray(value) == 0):
                #raise ValueError(""" Must provide error on the data value! 
                #This can be done either via the keyword 'value_error' or by
                #providing an uncertainties.uarray of uncertainties.ufloats 
                #with proper errors as the 'value' (first argument). """)
        kws = {}
        if meta is None:
            warnings.warn("""\n No metadata provided! This is strongly 
            discouraged! Wouldn't you like to know where your data came from
            later on? \n""")
        else:
            kws['meta'] = meta
            
        # !!!
        # Still need to CHECK FOR UNITS and convert to same ArbUnit if
        # possible and units are explicitly specified as 'None'.
        # This should be done according to pint, actually. A self.attrs['unit']
        # should be required for all ExperimentalDataSlices.
        # !!!
        
        args = (value,)
        kws['coords'] = coords
        kws.update(kwargs)
        #kws.update({'value_error':value_error, 'value_label':value_label, 
               #'cerrors':cerrors, 'clabels':clabels})
        super(ExperimentalDataSlice,self).__init__(*args, **kws)
        return
    #
    @classmethod
    def _test(cls):
        """
        This classmethod is a test of the ExperimentalDataSlice type. It will
        create arbitrary data and return a new object.
        """
        y, coords, value_label, title, labels = DataSlice._get_test_data()
        ds = ExperimentalDataSlice(y, value_label, coords,)
        ax = ds.plot(labels=labels, figsize=(8,6))
        
        return ds, ax,
    #
#   
class Calculation(DataSet):
    """
    A Calculation is a form of Data which is allowed to not have errors, but values and units are still required.
    """
    pass
    #def __init__(self, coords, values, units, errors=None):
        #""""""
        #super(Calculation, self).__init__(coords=coords, values=values, units=units, errors=errors) 
        #return
    ##
#
class Magnetization(ExperimentalDataSlice):
    """
    TODO:
    * Overhaul likely required. Temporary functionality.
    """
    #def __init__(self, value, coords=None, **kwargs):
        #super(Magnetization,self).__init__(value, coords, **kwargs)
        #return 
    def fit(self,J0=1.5,gj0=6./7):
        B = unp.nominal_values(self.magnetic_field.values.squeeze())
        T = unp.nominal_values(self.temperature.values.squeeze())

        
        mod = Model(functions.BJ_with_tanh)
        pars  = mod.make_params(independent_var=['B'], T=T, J=J0, gj=gj0, 
            B0=0., Bc1=2.71,Bc2=0, chi1=.12, chi2=.17, bgr=0., dm1=.25, dm2=0.,
            w2=0.1, w1=0.1, w=0.1, Bc=0,b=5)
        # set gj not to change        
        pars['J'].set(vary=False, min=0.1, max=7)
        pars['Bc1'].set(vary=True, min=0.5, max=3.5)
        pars['Bc2'].set(vary=True, min=3.5, max=10)
        pars['Bc'].set(vary=True, min=5.4, max=6)
        pars['chi1'].set(vary=True,min=0.0, max=2.)
        pars['chi2'].set(vary=True,min=0.0, max=2.)
        pars['dm1'].set(vary=True,min=0.0, max=2.)
        pars['dm2'].set(vary=True,min=0.0, max=2.)        
        #pars['ampT'].set(vary=True,min=0.1, max=2)
        #pars['ampl'].set(vary=False, min=0., max=1.)
        pars['w2'].set(vary=False, min=0,max=10)
        pars['w1'].set(vary=False, min=0,max=10)        
        pars['w'].set(vary=False, min=0,max=10)        
        pars['gj'].set(vary=False)
        pars['T'].set(vary=False)        
        pars['bgr'].set(vary=False)
        pars['B0'].set(vary=False)
        pars['b'].set(vary=True)        
        
        M = unp.nominal_values(self.values.squeeze())
        result = mod.fit(M, pars, B=M)  #B=B?
        self.attrs['fit_result'] = result
        lmfit.fit_report(result)
        
        M = mod.func(B, **result.values)
        
        return M, result, mod
    #

#
class Suceptibility(ExperimentalDataSlice):
    """"""
    pass
#
class Resistivity(ExperimentalDataSlice):
    pass
#
class StructureFactorModel(object):
    """
    TODO:
    <done> The standard is for the squared structure factor to be input at this point.
    """
    def __init__(self, coords=None, values=None, errors=None, units=None):
        """
        Creates an empty data object with attributes initialized to None.
        """
        self.coords = np.asanyarray(coords)
        self.values = np.asanyarray(values)
        self.errors = np.asanyarray(errors)
        return
    #
    def __getitem__(self, key):
        """
        TODO:
        Would need to be generalized to work everywhere, or be sure to overwrite for more complex (mulitdimensional) data
        """
        idx = np.equal(key, self.coords)        
        if self.errors.dtype is np.dtype(object):
            if self.values.shape == self.coords.shape:
                return self.values[idx].reshape(self.coords.shape)
            else:
                idx = np.asanyarray(idx.prod(1).reshape(self.values.shape)) # Needs to be generalized!
                #print idx 
                return self.values[idx]
        else:
            return self.values[idx], self.errors[idx]
    #
    def __setitem__(self, key, value):
        """
        FIX!!!
        """
        idx = np.equal(key, self.coords)        
        if self.errors.dtype is np.dtype(object):
            if self.values.shape == self.coords.shape:
                idx = np.asanyarray(idx.prod(1).reshape(self.values.shape[0])) # Needs to be generalized!
                self.values[idx,:] = value
            
            else:
                idx = np.asanyarray(idx.prod(1).reshape(self.values.shape)) # Needs to be generalized!
                #print idx 
                self.values[idx] = value
                
        else:
            self.values[idx] = value[0] 
            self.errors[idx] = value[1]
        return 
    #
    #
    #def __add__(self, other):
        #"""
        
        #"""
        #return Data()
    #
    #    
    #def __init__(self, coords, values, units, errors=None):
        #""""""
        #super(StructureFactorModel, self).__init__(coords, values, units, errors=None)
        #return
    #
    def gaussian(self, qx, qy, F, Qx, Qy, sigmax=0.05, sigmay=0.05):
        """"""
        return F * 1./(sigmax*np.sqrt(2.*np.pi)) * np.exp(-(qx-Qx)**2./(2.*sigmax**2.)) * 1./(sigmay*np.sqrt(2.*np.pi)) * np.exp(-(qy-Qy)**2./(2.*sigmay**2.))
    #
    def plotStructureFactor(self, step=0.01, vmin=1e-12, vmax=None, return_only=False,\
            planning=False, nm=False, xmin=None,xmax=None,ymin=None,ymax=None, plane='hhl', **kwargs):
        """
        Below is for dim0 vs dim2 only!
        TODO:
        * Generalize to any slice
        * Return an axis handle that can be displayed or saved at a later time. 
        * Improve the smoothing step so that it takes into account experimemantntal resolution properly (Q-dependent?)
        * Need to fix LogNorm (for colorbar it seems)
        * Add line for the zero axes.
        """
        
        #idx = np.where(self.coords[:,2] <= 0)
        #self.coords[idx,2] = np.abs(self.coords[idx,2])
        #idx = np.where(self.coords[:,0] <= 0)
        #self.coords[idx,0] = np.abs(self.coords[idx,0])  
        #self.coords[idx,1] = np.abs(self.coords[idx,1])        
        
        # Interpolate to get Fn larger
        qx = np.arange(self.coords[:,0].min(), self.coords[:,0].max()+step, step)
        
        if plane=='hk0':
            qy = np.arange(self.coords[:,1].min(), self.coords[:,1].max()+step, step)
        else:
            qy = np.arange(self.coords[:,2].min(), self.coords[:,2].max()+step, step)
            
        qx, qy = np.meshgrid(qx, qy)
        if planning:
            mask = NeutronSpectroscopy.TAS_mask((qx, qx, qy), **kwargs)        
        Fc = np.zeros(qx.shape)
        for i in range(len(self.values)):
            #
            if plane=='hk0':
                Fc += self.gaussian(qx, qy, self.values[i], self.coords[i,0], self.coords[i,1])#HK0
            else:
                Fc += self.gaussian(qx, qy, self.values[i], self.coords[i,0], self.coords[i,2])#HHL
        if planning: Fc = np.ma.masked_where(mask, Fc, copy=True)
        

        
        #Fc = griddata((self.coords[:,0], self.coords[:,2]), np.abs(self.values)**2., (qx,qy))
        
        # Get the fractional and itegral parts of our arrays
        #qxf, qxi = np.modf(qx)
        #qyf, qyi = np.modf(qy)
        
        """
        Need to fix for the AF and incomensurate cases! --> Use a centered gaussian, which also does away with interpolation.
        """
        
        # Get the logical index for the true Bragg peaks and set Fc to zero elsewhere
        #lidx = np.less_equal(qxf, 0.1)*np.less_equal(qyf, 0.1)
        #Fc[np.logical_not(lidx)] = 0
        
        # Now smooth the structure factor so that it looks a bit more realistic
        #Fc = self.smooth_slice(Fc, Nsmooth)
        
        # Finally plot and show the structure factor
        # Add lines for the zero axes!
        if not return_only:
            if vmax is None: vmax=0.8*Fc.max()
            cmap = plt.get_cmap('RdYlBu_r') 
            plt.figure()
            if nm: plt.pcolormesh(qx, qy, Fc, cmap=cmap, vmin=vmin, vmax=vmax, norm=LogNorm(vmin=vmin, vmax=vmax)) 
            else: plt.pcolormesh(qx, qy, Fc, cmap=cmap, vmin=vmin, vmax=vmax)
            if plane=='hk0':
                plt.xlabel('H00')#  
                plt.ylabel('0K0')#
            elif plane=='hhl':
                plt.xlabel('HH0')
                plt.ylabel('00L')
            if planning: plt.title('k_i: '+str(kwargs['ki'])+' inv. Ang. \n E_i: '+str(2.072*kwargs['ki']**2.)+' meV')
            plt.colorbar()        

        if planning: 
            # Should still be improved with object oriented graphing.
            Fc.mask = np.logical_not(mask)
            if nm: plt.pcolormesh(qx, qy, np.abs(Fc)+.01, cmap=cmap, alpha=0.04, vmin=vmin, vmax=vmax, norm=LogNorm(vmin=vmin, vmax=vmax)) 
            else:  plt.pcolormesh(qx, qy, Fc, cmap=cmap, alpha=0.04, vmin=vmin, vmax=vmax)
        

        #h.set_alpha(0.01)
        #fig = plt.figure()
        #ax = fig.add_subplot(111, projection='3d')        
        #ax.plot_surface(qx, qy, Fc, cmap='RdYlBu_r', vmin=0, vmax=0.1*Fc.max())
        if return_only:
            return Fc, qx, qy
        else:
            plt.xlim(xmin,xmax)
            plt.ylim(ymin,ymax)
            plt.grid()
            plt.show()            
            return
    #    
    @staticmethod
    def smooth_slice(S, Nsmooth, *kw):
        """
        Taken from 'pyslice' only so far.
        ...
        This function takes a (preferably masked) slice image 'Sm' (a np.ma.MaskedArray object) and returns a smoothed version.
        The convolution is performed 'N_smooth' times with a predefined kernel. If a specific kernel is required, it can be
        passed in as a keyword argument 'A=...' and should be a np.ndarray object.
        ------------
        TODO:
        * Update to be more general and work within this class structure
        * Need to add error support (see pySlice)
        """
        # Use default kernel unless there is one given as a keyword argument
        A = kw['A'] if 'A' in kw else np.array([[0.1, 0.2, 0.1], [0.2, 0.8, 0.2], [0.1, 0.2, 0.1]])
    
        for i in range(Nsmooth):
            # Perform the convolution
            S = sig.convolve(S, A, mode='same')
        return S   
    #   
    def getTable(self, crystal, gfxdir='./', fname='structure_factors.tex'):
        """
        This function will return a table (preferably latex-able) of the structure factor values (squared) and the corresponding coordinates in r.l.u, d, and lamdda. 
        TODO:
        * For magnetic, add a column that normalizes intensity to the closest Bragg peak.
        """
        QA = crystal.nuclear.rlu2ang(deepcopy(self.coords))
        headers = [r'Q (r.l.u.)', r'Q (\AA$^{-1}$)', r'd (\AA)', r'|F(Q)|$^2$']
        coords = []
        for coord in self.coords:
            coords.append('('+str(coord[0])+', '+str(coord[1])+', '+str(coord[2])+')')
        coords = np.array(coords)
        
        dat = [coords, QA, 2.*np.pi / np.linalg.norm(QA,axis=-1), self.values]
        headers = ['Q (r.l.u.)', 'd (\AA)', '|F(Q)|$^{2}$']
        dat = [coords, 2.*np.pi / np.linalg.norm(QA,axis=-1), self.values]        
        
        # Sort in order of descending peak intensity.
        idx = np.argsort(-dat[2])
        table = OrderedDict((headers[i], dat[i][idx]) for i in range(len(dat)))
        texf = open(gfxdir+fname,'w')
        print(tabulate.tabulate(table, floatfmt='.4e', headers='keys', tablefmt='latex'), file=texf)
        texf.close()   
        return
    #    
#
class StructureFactor(Data):
    """
    TODO:
    <done> The standard is for the squared structure factor to be input at this point.
    """
    def gaussian(self, qx, qy, F, Qx, Qy, sigmax=0.02, sigmay=0.02):
        """"""
        return F * 1./(sigmax*np.sqrt(2.*np.pi)) * np.exp(-(qx-Qx)**2./(2.*sigmax**2.)) * 1./(sigmay*np.sqrt(2.*np.pi)) * np.exp(-(qy-Qy)**2./(2.*sigmay**2.))
    #
    def plotStructureFactor(self, step=0.01, vmin=0, vmax=None, returned=False, save=False, show=True, gfxdir='./', gfxname='structure_factor', ext='.png'):
        """
        Below is for dim0 vs dim2 only!
        TODO:
        * Modify to make sureit works well for DATA as opposed to Calcuation.
        * Generalize to any slice
        * Return an axis handle that can be displayed or saved at a later time. 
        * Improve the smoothing step so that it takes into account experimental resolution properly (Q-dependent?)
        """
        
        #idx = np.where(self.coords[:,2] <= 0)
        #self.coords[idx,2] = np.abs(self.coords[idx,2])
        #idx = np.where(self.coords[:,0] <= 0)
        #self.coords[idx,0] = np.abs(self.coords[idx,0])  
        #self.coords[idx,1] = np.abs(self.coords[idx,1])        
        
        # Interpolate to get Fn larger
        qx = np.arange(self.coords[:,0].min(), self.coords[:,0].max()+step, step)
        qy = np.arange(self.coords[:,2].min(), self.coords[:,2].max()+step, step)
        qx, qy = np.meshgrid(qx, qy)
        Fc = np.zeros(qx.shape)
        for i in range(len(self.values)):
            Fc += self.gaussian(qx, qy, self.values[i], self.coords[i,0], self.coords[i,2])
        
        
        #Fc = griddata((self.coords[:,0], self.coords[:,2]), np.abs(self.values)**2., (qx,qy))
        
        # Get the fractional and itegral parts of our arrays
        #qxf, qxi = np.modf(qx)
        #qyf, qyi = np.modf(qy)
        
        """
        Need to fix for the AF and incomensurate cases! --> Use a centered gaussian, which also does away with interpolation.
        """
        
        # Get the logical index for the true Bragg peaks and set Fc to zero elsewhere
        #lidx = np.less_equal(qxf, 0.1)*np.less_equal(qyf, 0.1)
        #Fc[np.logical_not(lidx)] = 0
        
        # Now smooth the structure factor so that it looks a bit more realistic
        #Fc = self.smooth_slice(Fc, Nsmooth)
        
        # Finally plot and show the structure factor
        if vmax is None: vmax=0.8*Fc.max()
        plt.pcolormesh(qx, qy, Fc, cmap='RdYlBu_r', vmin=vmin, vmax=vmax)
        plt.xlabel('HH0')
        plt.ylabel('00L')
        plt.colorbar()
        #fig = plt.figure()
        #ax = fig.add_subplot(111, projection='3d')        
        #ax.plot_surface(qx, qy, Fc, cmap='RdYlBu_r', vmin=0, vmax=0.1*Fc.max())
        if save:
            plt.savefig(gfxdir+gfxname+ext)
        elif show:
            plt.show()
        
        if returned: return Fc, qx, qy
        else:        return
    #       
    def getTable(self, crystal, gfxdir='./', fname='structure_factors.tex'):
        """
        This function will return a table (preferably latex-able) of the structure factor values (squared) and the corresponding coordinates in r.l.u, d, and lamdda. 
        TODO:
        * For magnetic, add a column that normalizes intensity to the closest Bragg peak.
        """
        QA = crystal.nuclear.rlu2ang(deepcopy(self.coords))
        headers = [r'Q (r.l.u.)', r'Q (\AA$^{-1}$)', r'd (\AA)', r'|F(Q)|$^2$']
        coords = []
        for coord in self.coords:
            coords.append('('+str(coord[0])+', '+str(coord[1])+', '+str(coord[2])+')')
        coords = np.array(coords)
        
        dat = [coords, QA, 2.*np.pi / np.linalg.norm(QA,axis=-1), self.values]
        headers = ['Q (r.l.u.)', 'd (\AA)', '|F(Q)|$^{2}$']
        dat = [coords, 2.*np.pi / np.linalg.norm(QA,axis=-1), self.values]        
        
        # Sort in order of descending peak intensity.
        idx = np.argsort(-dat[2])
        table = OrderedDict((headers[i], dat[i][idx]) for i in range(len(dat)))
        texf = open(gfxdir+fname,'w')
        print(tabulate.tabulate(table, floatfmt='.4e', headers='keys', tablefmt='latex'), file=texf)
        texf.close()       
        return
    #
#
class MagneticStructureFactorModel(StructureFactorModel):
    """"""
    #def __init__(self, coords, values, units, errors=None):
        #""""""
        ##assert(values.dtype is np.dtype(complex))        
        #super(MagneticStructureFactorModel, self).__init__(coords=coords, values=values, units=units, errors=errors)       
        #return
    ##
    def zeros(self, units=None, Q=None):
        """"""
        self.coords = Q
        self.values = np.zeros(Q.shape, dtype=complex)
        self.units  = units
        return
    #
    def __add__(self, other):
        """"""
        return self.values
    #
#
class NuclearStructureFactorModel(StructureFactorModel):
    """"""
    #def __init__(self, coords, values, units, errors=None):
        #""""""
        ##assert(values.dtype is np.dtype(complex)) 
        #print type(self)
        #super(NuclearStructureFactorModel, self).__init__(coords=coords, values=values, units=units, errors=errors)       
        #return
    #
    def zeros(self, units=None, Q=None):
        """"""
        self.coords = Q
        self.values = np.zeros(Q.shape, dtype=complex)
        self.units  = units
        return
    #
    def __add__(self, other):
        """"""
        return self.values
    #
#
class Spectroscopy(ExperimentalData):
    """"""
    pass
    #
#
class SQw(ExperimentalDataSlice):
    """"""
    def bin_slice(i, e, Qx, Qy, dQx, dQy, **kw):
        """"""
        # Will need to be working from very sparse data (raw) for this to work.
        # This is not a limitation of the code, but rather a result of the simple fact
        # that one cannot increase the bin resolution without additional information. 
        
        # Find the bounding values of Q in the plane or take them from input. 
        Qxmin = kw['Qxmin'] if 'Qxmin' in kw else np.min(Qx)
        Qxmax = kw['Qxmax'] if 'Qxmax' in kw else np.max(Qx)
        Qymin = kw['Qymin'] if 'Qymin' in kw else np.min(Qy)
        Qymax = kw['Qymax'] if 'Qymax' in kw else np.max(Qy)
        
        # Set up a couple variables for the binning procedure
        bins = (np.arange(Qxmin, Qxmax, dQx), np.arange(Qymin, Qymax, dQy))
        var = np.square(np.concatenate(e))
        
        # Perform the calculation of the histogram. 
        # The choice of weights and additional division is made such that the value in the resultant array is the weighted average of the intensity values.
        I, [Qx,Qy] = np.histogramdd(np.array([Qx,Qy]).transpose(), bins=bins, weights=np.concatenate(i)/var, normed=False, returned=True)
        w1 = np.histogramdd(np.array([Qx,Qy]).transpose(), bins=bins, weights=1./var, normed=False, returned=False)
        I = I / w1
        w2 = np.histogramdd(np.array([Qx,Qy]).transpose(), bins=bins, weights=np.square(1./var), normed=False, returned=False)
        w1 = np.square(w1)
        # difficult to make error work...
        # E = (w1 / (w1 - w2)) * np.histogramdd(np.array([Qx,Qy]).transpose(), bins=bins, weights=np.square(np.concatenate(i) - ), normed=False, returned=False)
        
        
        #w2 = 
        
        
        return I, Qx, Qy
    #
    def avg_slices(i, e, rebin=False, *kw):
        """ 
        This function performs a simple weighted average of multiple slices of data. 
        The 'i' variable is a list of length N with (M x P) numpy masked arrays of slice data.
        The unbiased estimate of the population variance is taken from:
        https://en.wikipedia.org/wiki/Weighted_variance#Weighted_sample_variance
        which in turn comes from the GSL library. 
        """ 
        
        # Handle the case of a single slice, where these is nothing to average.
        if ((isinstance(i, (tuple, list)) and len(i) == 1) or isinstance(i, np.ndarray)):
            print('No averaging to be done. Returning original data as numpy array.')
            if not isinstance(i, np.ndarray):
                i = np.asarray(i[0])
                
            if not isinstance(e, np.ndarray):
                e = np.asarray(e[0])        
            return i, e
       
        else:
            # Run preliminary checks on the data types
            # Needs to be expanded. 
            # ALLOW MASKED ARRAY
            if isinstance(i, tuple): 
                i = np.asarray(i)
        
            if isinstance(e, tuple):
                e = np.asarray(e)
                
            # Need to renormalize intensities to the same counting time.
            # ... forthcoming
                
            weights = 1./np.square(e)
            
            # Need robust way to determine axis. Will also allow general treatment of all tuple sizes.
            axis = 0
            
            # Get the weighted mean in each bin
            # Need more robust determination of the axis to average over
            iw, wsum = np.ma.average(i, weights=weights, axis=axis, returned=True)
        
            # Get the weighted variance to return new error estimates
            # This is done more elegantly in some ways by pygsl, but only flat arrays are allowed it seems
            w2sum = np.sum(np.square(weights), axis=axis)
            wsum2 = np.square(wsum)
            varw, wsum = np.ma.average(np.square(i - iw), weights=weights, axis=axis, returned=True)
            varw = varw * (wsum2 / (wsum2 - w2sum))
            
            return iw, np.sqrt(varw)
    #
    def smooth_slice(Sm, E, N_smooth, *kw):
        """
        This function takes a (preferably masked) slice image 'Sm' (a numpy.ma.MaskedArray object) and returns a smoothed version.
        The convolution is performed 'N_smooth' times with a predefined kernel. If a specific kernel is required, it can be
        passed in as a keyword argument 'A=...' and should be a numpy.ndarray object.
        """
        # Use default kernel unless there is one given as a keyword argument
        A = kw['A'] if 'A' in kw else np.array([[0.1, 0.2, 0.1], [0.2, 0.8, 0.2], [0.1, 0.2, 0.1]])
        
        # Create a standary numpy.ndarry with zeros instead of nan's and/or a mask.
        if isinstance(Sm, np.ma.MaskedArray):
            S = Sm.data
            idxb = np.where(Sm.mask)
            idxg = np.where(~Sm.mask)
            S[idxb] = 0
            E[idxb] = 0
        elif isinstance(Sm, np.ndarray):
            idxb = np.where(S == (np.nan or -1.e20))
            idxg = np.where(S != (np.nan or -1.e20))
            S[idxb] = 0 
            E[idxb] = 0
        else:
            assert isinstance(Sm, (np.ma.MaskedArray, np.ndarray))
    
        # Get the weight for each bin to use in determining normalization. 
        weight = sig.convolve(~Sm.mask.astype(int), A, mode='same')
        weight[idxb] = np.nan
        
        for i in range(N_smooth):
            # Must ensure in every iteration that bins that began without data never contribute to the convolution.
            S[idxb] = 0
            E[idxb] = 0
            
            # Perform the convolution
            S = sig.convolve(S, A, mode='same')
            E = np.sqrt(sig.convolve(np.square(E), np.square(A), mode='same'))
            
            # Ensure normalization
            S[idxg] = S[idxg]/weight[idxg]
            E[idxg] = E[idxg]/weight[idxg]
        
        # Return masked arrays for easy plotting.
        S[idxb] = np.nan
        E[idxb] = np.nan
        S = np.ma.masked_invalid(S)
        E = np.ma.masked_invalid(E)
        # May be preferable to just edit the data passed in.
        #Sm.data[idxg] = S[idxg]
        #E.data[idxg] = E[idxg]    
        
        return S, E
    #
    def tabintpol(x, y, e=None, k=1, s=0):
        """
        This will have the same function as cobro.tabintpol but be more useable
        """
        
        if e is not None:
            w = 1./e**2.
        else:
            w = None
        
        return spi.UnivariateSpline(x, y, w=w, k=k, s=s)
    #    
#
class NeutronScanSet(ExperimentalDataSet):
    """
    This class consists of set of different 
    """
    pass
#
class NeutronSpectroscopy(Spectroscopy):
    """
    TODO:
    * This should eventually have all the ingredients to request an mdslice that is dynamically created and throw an error if not present.
    """
    @staticmethod
    def TAS_mask(Q, ki=4.04, E0=0., dE=0.5, dark_twotheta_min=None, dark_twotheta_max=None, twotheta_max=None, elastic=True, crystal=None, step = 1.):
        """
        For now, the default units are degrees for the theta_max and 
        TODO:
        * Deal with units more generally. Q, ks, should be input as Wavevector objects with units and theta_max should be an Angle and converted to radians. 
        * Generalize the crystal rotation to Euler angles or the like.
        * Change name 'dark_...' to 'allowed_...' and allow computation of 'allowed_...' from input of 'dark_...'
        * See the code for the planning routines in the Mantid implementation and try to adapt to make more general including sample environments. 
        """
        if not isinstance(crystal, pyCrystal.Crystal): raise TypeError('NeutronSpectroscopy requires a Crystal, pal. This might change in the future but unlikely.')   
        if isinstance(Q, tuple):
            Qt = np.zeros((Q[0].shape[0], Q[0].shape[1], 3))
            for i in range(3): 
                Qt[...,i] = Q[i]
            
        a=crystal.nuclear.a; b=crystal.nuclear.b; c=crystal.nuclear.c
        Qt[...,0] *= (2.*np.pi)/a
        Qt[...,1] *= (2.*np.pi)/b
        Qt[...,2] *= (2.*np.pi)/c
        
        # Set the magnitude of the incoming wavevector to be used irresepective of the direction of the vector in the crystal frame.
        ki_mag = ki
            
        # Convert to radians. This ia good reason to have an "Angle" object with units just like "Wavevector"
        dark_theta_bounds = np.asanyarray((dark_twotheta_min, dark_twotheta_max)).transpose() / 2.
        dark_theta_bounds = np.radians(dark_theta_bounds)
        theta_max = np.radians(twotheta_max/2.)
        
        wlist = []
        for w in np.arange(-360, 360, step):
            # This step range is definitely overdoing it -- need better way of finding all the allowed w and theta_Q without missing due to the restriction to [-pi,pi]
            w = np.radians(w)
            for i in range(len(dark_twotheta_min)):
                if np.logical_and(np.greater(w, dark_twotheta_min[i]), np.less(w, dark_twotheta_max[i])):
                    wlist.append(w)
        wlist = np.asanyarray(wlist)        
                
        mask_total = np.zeros(Qt[...,0].shape, dtype=bool)
        for w in wlist:
            # Need to compute the vector ki for each angular setting along with all the other values determined by it 
            ki = np.asanyarray([0, 0, ki_mag]) # (get the vector first; this is a good reason among others to make a wavevector object, with mag., units, etc.!)
            R = np.array([[np.cos(w), 0, np.sin(w)], [0, 1, 0], [-np.sin(w), 0, np.cos(w)]]) # rotation around the y-axis for now. To be generalized to properly represent all crystal frames. (?need it be here?)
            ki = np.dot(ki,R) # opposite rotation for ki than for w (same as doing the transpose of R)
            ki = repmat(ki.transpose(), Qt.shape[0], Qt.shape[1]).reshape(*Qt.shape)
            kf = Qt + ki
            Qt_abs = np.linalg.norm(Qt, axis=-1)
            ki_abs =  np.linalg.norm(ki, axis=-1)
            kf_abs =  np.linalg.norm(kf, axis=-1) 
            E = 2.072*(kf_abs**2. - ki_abs**2.) # energy transfer, in meV.
            
            # Need to get the sign of the part of kf perpendicular to ki in order to assign the angles correctly.
            #kf_lab = np.dot(R,kf) # need kf in the lab frame to get the angle right 
            kidkf = np.einsum('...i,...i', ki, kf)
            kidkf = np.repeat(kidkf.reshape(kidkf.shape[0], kidkf.shape[1],1), 3, axis=-1)
            kih = ki/np.repeat(ki_abs.reshape(ki_abs.shape[0], ki_abs.shape[1], 1), 3, axis=-1) # unit vector along ki
            kf_perp = kf - kidkf * kih
            pos = np.dot(np.asanyarray([0,1,0]),R) # to be generalized (it is the initial perp direction in the lab frame rotated to current angular setting) (?does it actually need to be generalized any more?)
            pos = np.sign(np.einsum('ijk,k', kf_perp, pos))
            
            # Compute the values of theta corresponding to each vector. 
            # From Eqs.1.2-3 in Shirane NS-TAS (which is wrong, so see the correction!) 
            s2theta_Q = (Qt_abs**2. - (kf_abs-ki_abs)**2.)/(4.*ki_abs*kf_abs)            
            theta_Q = np.arcsin(pos*np.sqrt(np.abs(s2theta_Q))) #
            theta_Q = np.ma.masked_invalid(theta_Q)
            
            # These are folded into the [-pi/2, pi/2] interval by the arcsin function, but we need to unfold unfold them based on the sign of the inner product between ki and kf. 
            # If the angle is greater than pi/2, then the inner product will be negative. The 1st quad then maps the 2nd, and the 4th quad maps to the 3rd.
            # The final interval of angles that we remain in are [-pi,pi]
            xneg = np.less(kidkf[...,0],0)
            idxQ2 = np.where(np.logical_and(xneg, np.greater(pos, 0))) 
            idxQ3 = np.where(np.logical_and(xneg, np.less(pos, 0))) 
            theta_Q[idxQ2] = np.pi - theta_Q[idxQ2]
            theta_Q[idxQ3] =-np.pi - theta_Q[idxQ3] # these thetas are negative so it should be adding.
            # Here we could add 2pi to all the negative values, right?
            # also deal with E for mask here in the future.
                        
            mask_ang = np.zeros(Qt[...,0].shape, dtype=bool)
            for i in range(len(dark_theta_bounds)):  
                # Here we implement all of the conditions that must be true for the sample environment and other experimental constraints on the given setup.
                mask_constraint = np.ones(Qt[...,0].shape, dtype=bool)
                
                # Get the values of the boundaries of the allowed angles
                dtmax = dark_theta_bounds[i,1]+w/2 #np.mod(dark_theta_bounds[i,1]+w/2+np.pi, np.pi) - np.pi
                dtmin = dark_theta_bounds[i,0]+w/2 #np.mod(dark_theta_bounds[i,0]+w/2+np.pi, np.pi) - np.pi
                
                # Get the constraint masks
                mask_constraint = np.logical_and(np.less(theta_Q, dtmax), mask_constraint) # below max allowed angle
                mask_constraint = np.logical_and(np.greater(theta_Q, dtmin), mask_constraint) # above min allowed angle
                mask_constraint = np.logical_and(np.less(np.abs(theta_Q), theta_max), mask_constraint)#-w/2? # total 2theta less than max
                mask_constraint = np.logical_and(np.less(np.abs(E-E0), dE), mask_constraint)
            
                # Compute the mask for the given angular setting over all the experimental constraints
                mask_ang = np.logical_or(mask_constraint, mask_ang)
            
            #idx_ang = np.where(mask_ang)
            #if len(idx_ang[0])>0 or len(idx_ang[1])>0:
                #print 'min: ', 2.*np.degrees(theta_Q[idx_ang].min()), '    max: ', 2.*np.degrees(theta_Q[idx_ang].max()) 
                
            # The mask for all angular settings is determined by whether a particular Q-point is viewed by any single one of the angular settings.
            mask_total = np.logical_or(mask_total, mask_ang)              
                 
        # The mask is for the acessible portions, so we return the logical_not in order to have "True" where the Q value is not accessible (so it is masked in the plot)
        return np.logical_not(mask_total)
    #
#
class NeutronScan(ExperimentalDataSlice):
    """
    This object is a subclass of ExperimentalDataSlice holding neutron scan 
    data as from a TAS, for example.
    TODO:
    V Made T, and H included as Coordinates!
    * ALSO need to have k and l as separate coordinates though (along with h)
    * Possible to expose all attributes directly... if I wanted.
    """
    results = {}
    def plot(self, dims=('h',), with_fit=(True,), gfxpath=None, show=False,
             save=True, **kwargs):
        """
        Default plot of a neutron scan.
        """
        if len(dims) != len(with_fit):
            raise ValueError('Dimensions of input must match.')
        N = len(dims)
        
        
        for i in range(N):
            if i>0:
                raise NotImplementedError('Only one plot at a time for now.')
            
            dim = dims[i]
            
            ax=None;fig=None
            if with_fit[i]:
                gfit = self.gfit
                ax,fig = gfit.plot(ax,fig)            
            
            coord = getattr(self,str(dim))
            var   = self.squeeze()
            
            labels={}
            labels['xlabels'] = coord.axislabel
            x = unp.nominal_values(coord.values)
            xerr = unp.std_devs(coord.values)
            
            labels['ylabels'] = var.axislabel
            y = unp.nominal_values(var.values)
            yerr = unp.std_devs(var.values)
            
            labels['title'] = self.title
            ax,fig,pbl = pubplot.plot(x, y, yerr=yerr, xerr=xerr, labels=labels, 
                                  ax=ax, fig=fig, save=save, show=show,
                                  **kwargs)        
            
            #if self.meta is not None:
                #meta = self.meta.split('\n')
                #pubplot.add_meta(ax, fig, self.meta)
            
            if gfxpath is not None:
                fig.savefig(gfxpath)
            elif save:
                raise NotImplementedError('Must supply as save path yourself!')
            
        return ax, fig, pbl
    #
    def fit(self, show=True, save=False, append=True, dims=None, ion=False,
            gfxpath=None, **kwargs):
        """
        This function will perform a fit of the scan.
        # TODO:
        * Change to use pltkwargs={}
        """        
        if show and ion:
            plt.ion()
            
        data = self.squeeze()
        dims = data.dims if dims is None else dims
        
        y = data.values
        ye= unp.std_devs(y)
        y = unp.nominal_values(y)
        
        x = OrderedDict()
        xe= OrderedDict()
        for dim in dims:
            if (dims is None) or (dim in dims):
                x[dim] = unp.nominal_values(data.coords.get(dim))
                xe[dim] = unp.std_devs(self.coords.get(dim))
        
        for key, coords in x.items():
            params = GaussianModel.guess(coords, y) 
            gfit = GaussianModel(coords, y, params, yerrs=ye) 
            gfit.auto_min()#mini_kwargs={'params':params}
            
            model_id = 'GaussianModel_'+str(key)+'_'
            self.attrs[model_id+'_fit'] = OrderedDict()
            gmf = self.attrs[model_id+'_fit']
            gmf['object'] = gfit
            self.attrs['gfit'] = gmf['object']
            
            redchi = gfit.mini_result.redchi
            if redchi < 5.:
                self.attrs['isGoodFit'] = True
            else:
                self.attrs['isGoodFit'] = False
                warnings.warn('Fit is above redchi = 5.!')
                print("redchi = "+str(redchi))
            
            if show or save:
                ax, fig, pbl = self.plot(save=save, show=show, gfxpath=gfxpath,
                                    **kwargs)
                #instead of gfit.plot()
                gmf['figure'] = fig
                gmf['axes'] = ax            
            
            ci_report = lmfit.ci_report(gfit.ci_dict)
            fit_report = lmfit.fit_report(gfit.mini_result)\
                +'\n[[Confidence Intervals]]\n'+ci_report
            print(fit_report)
            gmf['fit_report'] = fit_report
            gmf['ci_dict'] = gfit.ci_dict
            
            # Pull from the gmf['ci_dict'] the best fit parameters with error
            ci_dict = gmf['ci_dict']
            for key in list(ci_dict.keys()):
                nom = ci_dict[key][3][1]
                u   = ci_dict[key][4][1]
                l   = ci_dict[key][2][1]
                mau = np.max([u,nom])
                miu = np.min([u,nom])
                mal = np.max([l,nom])
                mil = np.min([l,nom])                
                std = np.mean([np.abs(mau-miu), np.abs(mal-mil)])
                gmf[key] = unp.uarray(nom,std)  
                            
            
            #if show:
                #plt.show()
            #if save:
                #ax.save#?  
        
        
        if show and ion:
            plt.ioff()
        return
    #
    @property
    def fig(self,):
        """
        Return the figure handle if available. Plots the image in qtconsole.
        """   
        d={}
        for key, value in self.attrs.items():
            if isinstance(value, dict):
                d[key] = value['figure']
        
        return d
    #
    @property
    def redchi(self):
        """
        String for fit report
        """
        return self.gfit.mini_result.redchi
    #    
    @property
    def fit_report(self):
        """
        String for fit report
        """
        d={}
        for key, value in self.attrs.items():
            if isinstance(value, dict):
                d[key] = value['fit_report']
        return d
    #
    @property
    def area(self):
        """
        Area with error.   
        """
        d={}
        for key, value in self.attrs.items():
            if isinstance(value, dict):
                d[key] = value['area']        
        return d
#   
    @property
    def center(self):
        """
        Center with error.   
        """
        d={}
        for key, value in self.attrs.items():
            if isinstance(value, dict):
                d[key] = value['center']        
        return d
    @property
    def sigma(self):
        """
        Center with error.   
        """
        d={}
        for key, value in self.attrs.items():
            if isinstance(value, dict):
                d[key] = value['sigma']        
        return d    
#    

    #
#   
class NeutronTOF(NeutronSpectroscopy):
    """"""
    def __init__(self, fname=None):
        """"""    
        if sys.platform == 'darwin':
            # Eventually fix so the data can be read in from nxs on mac. Works fine for linux
            if mtd.doesExist('mdhisto'):
                mdhisto = mtd['mdhisto']
            else: raise OSError('Mantid import does not work directly from python on mac (so far). Perform in Mantid shell.')
        else:
            if mtd.doesExist('mdhisto'):
                mdhisto = mtd['mdhisto']
            else:
                mdhisto = LoadMD(fname)
        
        # Set up the data arrays
        self.getHisto(mdhisto)   

        
        return
    #
    def getHisto(self, mdhisto, fname='/Users/Guy/mdhisto_00l_hh0_mhh0_DE _79412_79606.nxs'):
        """
        TODO: 
        * Later, make this act on a mdslice object from 'new_sym.py' (included in this will be reorganizing the python code structure)
        """
        
        # Get the objects defining the bin edges for each dimension in Q.
        xd = mdhisto.getDimension(0)
        yd = mdhisto.getDimension(1)
        zd = mdhisto.getDimension(2)
        Ed = mdhisto.getDimension(3)
        
        # Get the bin edges and compute the bin centers in Q.
        qx = np.linspace(xd.getMinimum(), xd.getMaximum(), xd.getNBins()+1)
        qx = (qx+np.roll(qx,1))/2.
        qx = qx[1:]
        qy = np.linspace(yd.getMinimum(), yd.getMaximum(), yd.getNBins()+1)
        qy = (qy+np.roll(qy,1))/2.
        qy = qy[1:]
        qz = np.linspace(zd.getMinimum(), zd.getMaximum(), zd.getNBins()+1)
        qz = (qz+np.roll(qz,1))/2.
        qz = qz[1:]
        E  = np.linspace(Ed.getMinimum(), Ed.getMaximum(), Ed.getNBins()+1)
        E  = (E+np.roll(E,1))/2.
        E  = E[1:]   
        
        # Get the signal array. Done as a deepcopy so that the data is writeable (C++/Python wierd thing with Mantid)
        histo = deepcopy(mdhisto.getSignalArray()).transpose().reshape(len(qx), len(qy), len(qz),len(E))
        herrsq = deepcopy(mdhisto.getErrorSquaredArray()).transpose().reshape(len(qx), len(qy), len(qz),len(E))
        
        # Apply to self:
        self.histo = histo
        self.herrsq = herrsq
        self.qx = qx
        self.qy = qy
        self.qz = qz
        self.E = E        
    
        return
    #
    def plotSlice(self, qz0=0., dqz=.05, E0=0, dE=1., herrsq=None, vmin=1.5e-3, vmax=5.e-3, cmap='RdYlBu_r'):
        """
        TODO:
        * Should still be more general so that the qz axis can be chosen to be plotted and another axis averaged over.
        * Make plot more general (obj. oriented adn with labels)
        """
        Nx = len(self.qx); Ny = len(self.qy)
        Qx, Qy = np.meshgrid(self.qy, self.qx)
        dat, errsq = self.getSlice(qz0=qz0, dqz=dqz, E0=E0, dE=dE)
        mask = np.logical_and(np.greater_equal(Qx,0), np.greater_equal(Qy,0))
        #idx = np.where(mask)
        #Qx = Qx[idx[0],:]; Qx = Qx[:,idx[1]]
        #Qy = Qy[idx[0],:]; Qy = Qy[:,idx[1]]
        #dat = dat[idx[0],:]; dat = dat[:,idx[1]]
        Qx = np.ma.masked_where(mask, Qx, copy=True)
        Qy = np.ma.masked_where(mask, Qy, copy=True)
        dat = np.ma.masked_where(mask, dat, copy=True)
        plt.pcolormesh(Qx, Qy, dat, vmin=vmin, vmax=vmax, cmap=cmap)
        return #ax #object handle for plot
    #
    def getSlice(self, qz0=0., dqz=.05, E0=0, dE=1.):
        """
        This function returns the 2D data slice for the (qx,qy) plane as defined for the histo provided. 
        The qz and E dimensions are integrated over the provided range, with the data being averaged according to a weighted mean.
        The weights are determined from the inverse variance.
        
        TODO:
        * Make sure that this works still for the case of no herrsq or (better yet?) raise an error if no herrsq is included
        """
        idxE = np.where(np.abs(self.E-E0)<=dE) 
        idxqz = np.where(np.abs(self.qz-qz0)<=dqz)
        histo = np.ma.masked_invalid(self.histo)
        if self.herrsq is not None:
            herrsq  = np.ma.masked_invalid(self.herrsq)
            w = 1./herrsq
            dat, w = np.ma.average(histo[:,:,idxqz,idxE], axis=-1, weights=w[:,:,idxqz,idxE], returned=True)
            dat, w = np.ma.average(dat[:,:,idxqz], axis=-1, weights=w[:,:,idxqz], returned=True)  
            dat = dat.squeeze()
            errsq = 1./w.squeeze()      
            # may need to better account for case of no herrsq
        else: 
            dat, w = np.ma.average(histo[:,:,idxqz,idxE], axis=-1, returned=True)
            dat, w = np.ma.average(dat[:,:,idxqz], axis=-1, returned=True)   
            dat = dat.squeeze()
            errsq = None   
    
        return dat, errsq
    #
    def makeQEdict(self, qx0=0., dqx=.1, qy0=0., dqy=.1, qz0=0., dqz=.1, E0=0, dE=1.):
        """
        TODO:
        *? Might be nicer to have a (qx, qy, qz, E) tuple and the analogs for the other values    
        """
        self.qedict = {'qx':(self.qx,qx0,dqx), 'qy':(self.qy,qy0,dqy), 'qz':(self.qz,qz0,dqz), 'E':(self.E,E0,dE)}  
        return
    #
    def setIndex(self):
        """"""
        self.idxdict = {}
        dims = list(self.qedict.keys())
        for dim in dims:
            qetuple = self.qedict[dim]
            assert(isinstance(qetuple, tuple))
            qe, qe0, dqe = qetuple
            #idxdict[dim] = np.less_equal(np.abs(qe-qe0), dqe)
            self.idxdict[dim]  = np.concatenate(np.where(np.abs(qe-qe0)<=dqe))
    
        
        ## Make repeating arrays for the center values
        #qx0 = np.repeat(qx0, len(qx)).reshape(qx.shape)
        #qy0 = np.repeat(qy0, len(qy)).reshape(qy.shape)
        #qz0 = np.repeat(qz0, len(qz)).reshape(qz.shape)
        #E0  = np.repeat(E0,  len(E)).reshape(E.shape)
        
        ## Do the same for the difference values
        #dqx = np.repeat(dqx, len(qx)).reshape(qx.shape)
        #dqy = np.repeat(dqy, len(qy)).reshape(qy.shape)
        #dqz = np.repeat(dqz, len(qz)).reshape(qz.shape)
        #dE  = np.repeat(dE,  len(E)).reshape(E.shape)  
        
        ## Make a meshgrid for all of the coordinates
        #qx, qy, qz, E = np.meshgrid(qx, qy, qz, E)
        #qx0, qy0, qz0, E0 = np.meshgrid(qx0, qy0, qz0, E0)
        #dqx, dqy, dqz, dE = np.meshgrid(dqx, dqy, dqz, dE)
        
        #dims = qedict.keys()
        #boolar = np.ones(histo.shape)
        #for dim in dims:
            #qetuple = qedict[dim]
            #assert(isinstance(qetuple, tuple))
            #qe, qe0, dqe = qetuple
            
            ## Make repeating arrays for the center values        
            #qe0 = np.repeat(qe0, len(qe)).reshape(qe.shape)
            
            ## Do the same for the difference values
            #dqe = np.repeat(dqe, len(qe)).reshape(qe.shape) 
            
            ## Make a meshgrid for all of the coordinates
            #qx, qy, qz, E = np.meshgrid(qx, qy, qz, E)
            #qx0, qy0, qz0, E0 = np.meshgrid(qx0, qy0, qz0, E0)
            #dqx, dqy, dqz, dE = np.meshgrid(dqx, dqy, dqz, dE)
            
            #boolar *= np.less_equal(qe-qd0, dqe)
            
            #idxdict[dim]  = np.where()
            
        return
    #        
    def getPeak(self, qx0=1.5, dqx=.25, qy0=1., dqy=.25, qz0=0., dqz=.1, E0=0, dE=1., fit=False):
        """
        This function returns the intensity and error for a given integration width around a peak location
        TODO:
        * Determine peak width according to a fit or implement using the EventData method for SCD in Mantid
        * Consider doing this with a masked array...
        """
        if fit:
            print('Fit is not implemented yet.')
            pass
        else:
            self.makeQEdict(qx0=qx0, dqx=dqx, qy0=qy0, dqy=dqy, qz0=qz0, dqz=dqz, E0=E0, dE=dE) # this will be incorporated into the SCdata object
            self.setIndex()
            #histo = np.ma.masked_invalid(histo[idxdict['qx'], idxdict['qy'], idxdict['qz'], idxdict['E']]) 
            histosl = self.histo[self.idxdict['qx'],...]
            histosl = histosl[:,self.idxdict['qy'],...]
            histosl = histosl[...,self.idxdict['qz'],:]
            histosl = np.ma.masked_invalid(histosl[...,self.idxdict['E']])
            #histo = np.ma.masked_invalid(histo[idxdict['qx'].min():idxdict['qx'].max()+1, idxdict['qy'].min():idxdict['qy'].max()+1, 0, idxdict['E'].min():idxdict['E'].max()+1])        
            #histo = np.ma.masked_invalid(histo[idxdict['qx'][0].min():idxdict['qx'][0].max()+1, idxdict['qy'][0].min():idxdict['qy'][0].max()+1, idxdict['qz'][0].min():idxdict['qz'][0].max()+1, idxdict['E'][0].min():idxdict['E'][0].max()+1])        
            if self.herrsq is not None: 
                herrsqsl = self.herrsq[self.idxdict['qx'],...]
                herrsqsl = herrsqsl[:,self.idxdict['qy'],...]
                herrsqsl = herrsqsl[...,self.idxdict['qz'],:]
                herrsqsl = np.ma.masked_invalid(herrsqsl[...,self.idxdict['E']])            
                w = 1./herrsqsl
                if np.all(np.greater(w.shape, 0)):
                    peaki, w = np.ma.average(histosl, axis=None, weights=w, returned=True)
                    peaki = peaki.squeeze()
                    peake = 1./w.squeeze()
                    badflag = False
                else:
                    peaki = None
                    peake = None
                    badflag = True
            else:
                peaki  = np.ma.average(histosl, axis=None, returned=True)
                peaki = peaki.squeeze()
                peake = None        
                badflag = False
        return peaki, peake, badflag
    #
    def setStructureFactor(self, Qs, qz0=0.):
        """
        Extract the peak intensities at the Q values in the provided list and set up the data StructureFactor
        """
        # Get some lists ready for storage
        dQxs = []
        dQys = []
        peakis=[]
        peakes2=[]
        idx   =[]
        #print 'setting structure factor...'
        # Loop through the Qs and integrate the intensity around that location in the data
        for i in range(0,len(Qs)):
            if np.any(np.greater(Qs[i,:], 0.)) and np.all(np.less(Qs[i,0], 5)) and np.all(np.less(Qs[i,2], 5)):
                # inefficient way of not double counting...
                use = True
                #eQ = np.vstack((dQxs,dQxs,dQys)).transpose()
                #for i in xrange(len(eQ)):
                    #use *= np.array_equal(Qs[i,:].reshape(1,3), eQ[i,:].reshape(1,3))
                if use:
                    # ^ Restricted range of applicability since no background correction, etc. ^ #
                    # Can be generalized per experiment.
                    peaki, peake2, badflag = self.getPeak(qx0=Qs[i,0], qy0=Qs[i,2], qz0=qz0)
                    if not badflag:
                        #print Qs[i,:]                        
                        # Check these coordinates!!!
                        dQxs.append(Qs[i,0])
                        dQys.append(Qs[i,2])                
                        peakis.append(peaki)
                        peakes2.append(peake2)
                        idx.append(i)

        # Check coordinates....
        dQxs = np.asanyarray(dQxs)
        dQys = np.asanyarray(dQys)
        dQs = np.vstack((dQxs,dQxs,dQys)).transpose()
        
        self.F = StructureFactor(dQs, np.ma.masked_invalid(peakis), np.sqrt(np.ma.masked_invalid(peakes2)), units=None)
        self.idx = idx
        return
    #
#
class NeutronReactor(NeutronSpectroscopy):
    """"""
    pass
    #
#
class xraySpectroscopy(Spectroscopy):
    """"""
    def __init__(self):
        """"""
        return
    #
#
class RIXS(xraySpectroscopy):
    """"""
    def __init__(self):
        """"""
        return
    #
#
class REXS(xraySpectroscopy):
    """"""
    def __init__(self):
        """"""
        return
    #
#
class ARPES(Spectroscopy):
    """"""
    def __init__(self):
        """"""
        return
    #
#
class TRARPES(ARPES):
    
    """"""
    def __init__(self):
        """"""
        return
    #
#
#----------------------------------------------------------------------------#

########################### --- Physical Entities --- ########################
## LOOK AT USING pint.contexts HERE!!!
##############################################################################
class Wavevector(gQ):
    pass
#
class MomentumTransfer(gQ):
    """"""
    pass
    #
#
#----------------------------------------------------------------------------#

############################### --- Data Readers --- #########################
class DataContainer(OrderedDict):
    """
    The DataContainer object is 
    This would be the equivalent of a collection of various Data objects, eg:
      - Different runs from a magnetic field sweep of nuclear peaks
      - sucsceptibility at different freq., H, along with heat capacity
    """
    def __init__(self):
        """
        Constructor for a DataContainer instance.
        """
        
        return
    #
    def place(self, data):
        """
        Place a 
        """
        
        return
    #
#
class DataReader(object):
    """
    A DataReader object is a custom reader for a specific output data file. 
    TODO:
    v Generalized for many datasources.
    """
    datasources = OrderedDict()  
    known_exts = OrderedDict()    
    readers = OrderedDict()
    reader_inferred = OrderedDict()    
    data = OrderedDict()
    
    def __init__(self, datasources, keys, **kwargs):
        """
        Set the source to be read. 
        """
        if isinstance(keys, str):
            keys = [str(keys)]
        elif not isinstance(datasources, Iterable):
            datasources = [datasources]
        elif isinstance(datasources, Iterable):
            datasources = list(datasources)
        else:
            datasources = datasources
            
        self.datasources.update(list(zip(keys,datasources)))
        for key,value in self.datasources.items():
            key = str(key)
            try:
                self.datasources[key] = DataSource(value)            
            except:
                raise
                
            self.reader_inferred[key] = False
        return
    #
    def populate(self, datacontainer):
        """
        Establish the data references.
        """
        self.data
        return
    #
#
class DataSource(object):
    """"""
    def __init__(self, origin, kind=None):
        """
        TODO:
        Need handler functions to deal with the form of the data source 
        (i.e., string for filename, pointer to file, etc.)
        """
        if isinstance(origin, type(DataSource)):
            self = origin
        else:
            self.origin = origin
        self.infer_kind(kind=kind)            
        self.set_ext()
        return 
    #   
    def set_ext(self):
        """
        Set the extension of origin if it is a file.
        """
        self.ext = self.get_ext()
        return
    #
    def get_ext(self):
        """
        Get the extension of origin if it is a file.
        """
        if self.isFileName:
            ext = '.'+self.filename.split('.')[-1]
        else:
            ext = ''
        return ext
    #
    def infer_kind(self, kind=None):
        """
        """
        self.kind=kind
        self.isFileName   = True and os.path.isfile(str(self.origin))
        if self.isFileName:
            self.kind = str
            self.filename = str(self.origin)
        else:
            self.isObj  = isinstance(self.kind, object) and not self.isFileName
            
        # check various extentions to set self.target_fmt
            
            
        return
    #
    def _same(self, obj):
        """
        Check that the type of input is same.
        """
        if isinstance(type(obj), type(DataSource)):
            pass
        else:
            raise TypeError
        return
    #
    def close(self):
        """
        Close the datasource if needed.
        """
        try:
            self.close()
        except:
            # if there is no close(), nothing to worry about right?
            pass
        return
    #
    def __repr__(self):
        if self.isFileName:
            res = self.filename
        else:
            res = ''
        return res
#
class QdPPMSReader(DataReader):
    """
    So far this is just pulled from 'acmc_reader.py'. Much work to be done to make it more general!
    TODO:
    * Use header to determine the type of option used for the experiment (e.g. VSM or ACMS)
    """
    addenda = OrderedDict()
    def __init__(self, filename, key='qdppms',addenda_filename=None, usecols=None, skiprows=None):
        with open(filename, 'r') as f:
            lines = f.readlines()
            idx = lines.index('[Data]\n') 
        skiprows = idx+1
        try:
            self.data[key] = pandas.read_csv(filename, usecols=usecols, skiprows=skiprows)
        except:
            skiprows = 30
            self.data[key] = pandas.read_csv(filename, usecols=usecols, skiprows=skiprows)
        
        with open(addenda_filename, 'r') as f:
            lines = f.readlines()
            idx = lines.index('[Data]\n') 
        skiprows = idx+1
        try:
            self.addenda[key] = pandas.read_csv(addenda_filename, usecols=usecols, skiprows=skiprows)
        except:
            skiprows = 30
            self.addenda[key] = pandas.read_csv(addenda_filename, usecols=usecols, skiprows=skiprows)        
        
        return
#
    def chivT_plot(data, save=False, view=True, svname='mvt',label=None, method='VSM', histogram=True, all_data=False, h0=1000, dT=0.5, mass=0.54e-3, fum=578, fundamental_units=False):
        """"""
        # Get the data
        if method == 'VSM':
            h = data['Magnetic Field (Oe)']
            # For units of (mu_o) ([mu_B per f.u. per]) [10^4 Oe] (Tesla), use the fundamental_units factor. (SI), [Gaussian]
            chi = data['Moment (emu)'] / ( h * mass / fum)
            if fundamental_units: chi /= (9.274e-24 * 6.022e23 * (1.e6/(4.*np.pi)) * 4.*np.pi*1.e-7) 
            T   = data['Temperature (K)']
            err = data['M. Std. Err. (emu)'] / ( h * mass / fum) 
            if fundamental_units: err /= (9.274e-24 * 6.022e23 * (1.e6/(4.*np.pi)) * 4.*np.pi*1.e-7)
            
        
        h = np.asanyarray(h)
        if all_data:
            idx = np.where(np.equal(h,h))[0]
        else:
            idx = np.where(np.abs(h-h0)<10)[0]
        chi = np.asanyarray(chi)[idx]
        T = np.asanyarray(T)[idx]
        err = np.asanyarray(err)[idx]   
        
        if all_data:
            # include more data
            fname = '/Users/Guy/Documents/Johns Hopkins/Academics/Broholm Group/Experiments/CeAuSb2/Zero Field/data/Fisk/susceptibility_with_error.txt'
            T2,chic2,errc2,chiab2,errab2 = np.loadtxt(fname,unpack=True)    
            T=np.concatenate((T,T2))
            #chi=np.concatenate((chi,chic2*0.86))
            #err=np.concatenate((err,errc2*0.86))
            #chi=np.concatenate((chi,chic2))
            #err=np.concatenate((err,errc2))  
            chi=np.concatenate((chi,chiab2))
            err=np.concatenate((err,errab2))         
        
        
        if histogram:
            Ti = np.arange(1.8,300,dT)
            Tdigit = np.digitize(T, Ti)
            Tcounts, Tbins = np.histogram(T, bins=Ti)
            Tcenters = 0.5*(Tbins[1:]+Tbins[:-1])
            chihist = np.zeros(len(Tcounts))
            chierr = np.zeros(len(Tcounts))
            for i in range(len(Tcounts)):
                idx = np.where((i+1) == Tdigit)
                chihist[i] = np.ma.average(chi[idx], weights=1./err[idx]**2.)#chi[idx].sum()/Tcounts[i] #
                chierr[i]  = np.sqrt(1./np.sum(1./err[idx]**2.))
            #ax.errorbar(1./Tcenters, chihist, yerr=chierr, marker='.', mfc='r')
            T = Tcenters
            chi = chihist
            err = chierr
            idx = np.where(np.logical_not(np.isnan(chi)))
            chi = chi[idx]
            T=T[idx]
            err=err[idx]  
            idx = np.where(np.logical_not(np.isnan(err)))
            chi = chi[idx]
            T=T[idx]
            err=err[idx]         
    
        if save:  
            savefig(svname+'.pdf',bbox_inches='tight')
        elif view:
            fig = plt.figure()
            ax = fig.add_subplot(111)        
            ax.errorbar(T, chi, yerr=err, marker='.', mfc='b', label=label)
            #ax.set_ylim(0,3.e-6)
            ax.set_xlabel('Temperature (K)')
            ax.set_ylabel(r'Susceptibility (emu / Oe $\cdot$ mg)')  
            ax.legend()        
            plt.show()
            
        return chi, T, err
    #      
    def mvtf_scan_plot(data, h=170, dh=10, molar_mass=270.05, sample_mass=32.3, Nm=2, save=False, svname='mvtf', show=True, rasterized=True, 
                       component='imag', oplot=False, norm=None, meas_dir='susceptibility/mvtf/',
                       gfxdir = '/Users/Guy/Documents/Johns Hopkins/Academics/Broholm Group/Experiments/Cu2OSeO3/PPMS/gfx/'):
        """"""
        idx = np.where(np.abs(data['Magnetic Field (Oe)']-h)<dh)[0]
        
        if component == 'imag': 
            #data = data[(data['Magnetic Field (Oe)']-h < dh) & (data["M'' (emu)"] > 0) ]#& ((np.logical_not(np.isnan(data["M'' (emu)"]))).bool())]            
            m = np.asanyarray(data["M'' (emu)"]*molar_mass/(constants.N_A*Nm*constants.codata.value('Bohr magneton')*sample_mass))[idx]
            m/= np.asanyarray(data["Moment (Oe)"])
            if norm is not None: 
                norm = LogNorm(vmin=np.nanmin(m[np.where(m>0)]), vmax=np.nanmax(m))
                vmin=1.e-6#np.nanmin(m)#1.e-8
                vmax=3.e-4#np.nanmax(m)#1.e-6  
            else: 
                vmin=1.e-6#np.nanmin(m)#1.e-8
                vmax=1.1e-4#np.nanmax(m)#1.e-6               
        elif component == 'real': 
            #data = data[(data['Magnetic Field (Oe)']-h < dh) & (data["M'' (emu)"] > 0) ]#& ((np.logical_not(np.isnan(data["M'' (emu)"]))).bool())]            
            m = np.asanyarray(data["M' (emu)"]*molar_mass/(constants.N_A*Nm*constants.codata.value('Bohr magneton')*sample_mass))[idx]
            if norm is not None: 
                norm = LogNorm(vmin=np.nanmin(m[np.where(m>0)]), vmax=np.nanmax(m))
                vmin=np.nanmin(m[np.where(m>0)])#1.e-8
                vmax=np.nanmax(m[np.where(m>0)])#1.e-6  
            else: 
                vmin=np.nanmin(m[np.where(m>0)])#1.e-8
                vmax=np.nanmax(m[np.where(m>0)])#1.e-6            
        else: return
        f = np.asanyarray(data['Frequency (Hz)'])[idx]
        #f = data['Frequency (Hz)']
        t = np.asanyarray(data['Temperature (K)'])[idx]
        #t = data['Temperature (K)']    
        idx = np.where(np.logical_and(m>0, np.logical_not(np.isnan(m)))) 
        #m=m.iloc[idx]; t=t.iloc[idx]; f=f.iloc[idx]
        m=m[idx]; t=t[idx]; f=f[idx]
        
        # Make the figure using gridspec. Should make this part more modular so it can be used elsewhere.
        if save:
            pubfig.pub_ready()
            figsize = pubfig.fig_size()
        else:
            rc('text', usetex=True)
            figsize = (8,6)
        figsl = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(1,2, width_ratios=[8,1], wspace=0)
        pax = plt.subplot(gs[0])
        pax.set_xlabel('Temperature (K)')
        pax.set_ylabel(r'Frequency (Hz)')
        pax.set_title(r'Susceptibility (emu / Oe $\cdot$ mol)'+' at H='+str(round(h))+' Oe')
        cax = plt.subplot(gs[1])
        
        # Interpolate the data
        fi1d = np.logspace(1, 4, 1000)#get with log10
        ti = np.linspace(56, 59, 1000)#get min/max from data
        ti, fi = np.meshgrid(ti, fi1d)        
        mi = griddata((t,f), m, (ti, fi), method='linear')  
        
        if oplot:
            # find max trend line
            fm=[]; tm=[]; mm=[]
            for fp in fi1d:
                idx = np.where(fp==fi)
                max_idx = np.where(np.nanmax(mi[idx])==mi)
                #max_idx = np.intersect1d(idx, max_idx)
                fm.append(fi[max_idx])
                tm.append(ti[max_idx])
                mm.append(mi[max_idx])
        
            fm = np.asarray(fm).squeeze()  
            tm = np.asarray(tm).squeeze()
            mm = np.asarray(mm).squeeze()
        
        cmap = plt.get_cmap('RdYlBu_r')
        cmap.set_bad(alpha=0.)    
         
        pc    = pax.pcolormesh(ti, fi, mi, rasterized=rasterized, vmin=vmin, vmax=vmax, cmap=cmap, norm=norm)
        pscat = pax.scatter(t, f, c=m, marker='o', linewidths=0.25, s=2, alpha=0.5,  vmin=vmin, vmax=vmax, cmap=cmap, norm=norm)
        if oplot: pmax  = pax.plot(tm, fm, c='k', alpha=0.5, marker='.', ms=3., linestyle='', markevery=30)
        pax.set_yscale('log')
        pax.set_xlim(np.nanmin(ti),np.nanmax(ti))
        pax.set_ylim(np.nanmin(fi), np.nanmax(fi))
        cb = plt.colorbar(pc, cax=cax)
        cb.solids.set_rasterized(rasterized)    
        if save:
            if norm is not None: svname=svname+'_logscale'
            if not oplot: plt.savefig(gfxdir+meas_dir+svname+'.pdf',bbox_inches='tight')
            else: plt.savefig(gfxdir+meas_dir+svname+'_with_max_trace.pdf',bbox_inches='tight')
        elif show:
            plt.show()
        return
    #
    def read_MvH(self, key, limd=None, lima=None, t=2, dt=1, dh=.05, molar_mass=580.6, sample_mass=4.9e-3, Nm=1, title=None,el='Ce'):
        """"""
        if limd is None:
            data = self.data[key]  
        else:
            l=[]
            for lim in limd:
                lo = lim[0]
                hi = lim[1]
                l.append(self.data[key][lo:hi])
            data = pandas.concat(l,copy=False)
        addenda = self.addenda[key] if lima is None else self.data[key][lima[0]:lima[1]]
        
        M_emu_to_SI = 1.e-3
        M_to_molar = molar_mass /  (sample_mass * constants.N_A)
        mfac = M_emu_to_SI *  M_to_molar / (Nm*constants.codata.value('Bohr magneton'))
        # 
            
        idx = np.where(np.abs(data['Temperature (K)']-t)<dt)[0]
        Hm0 = np.asanyarray(data['Magnetic Field (Oe)'])[idx]/1.e4 #to Tesla 
        # add Herr
        Herr = 0.
        M = np.asanyarray(data['M-DC (emu)'])[idx] * mfac
        Merr = np.asanyarray(data['M-Std.Dev. (emu)'])[idx] * mfac
        idx = np.where(np.logical_not(np.isnan(M)))[0]
        Hm0 = Hm0[idx]
        #Hm0 = unp.uarray(Hm0,Herr)
        M   = M[idx]
        Merr= Merr[idx]
        M = unp.uarray(M,Merr)
        
        if addenda is not None:
            idx = np.where(np.abs(addenda['Temperature (K)']-t)<dt)[0]
            aHm0 = np.asanyarray(addenda['Magnetic Field (Oe)'])[idx]/1.e4 #to Tesla
            # add aHerr
            aHerr = 0.            
            aM = np.asanyarray(addenda['M-DC (emu)'])[idx] * mfac       
            aMerr = np.asanyarray(addenda['M-Std.Dev. (emu)'])[idx] * mfac
            idx = np.where(np.logical_not(np.isnan(aM)))[0]
            aHm0 = aHm0[idx] 
            aHm0 = unp.uarray(aHm0,aHerr)            
            aM = aM[idx]
            aMerr= aMerr[idx]
            aM = unp.uarray(aM,aMerr)
            
            # Need many more checks on whether the data is proper to subtract from
            # each other
            #...

            
            Hcenters = Hm0.view()
            aHcenters=aHm0.view()
            
            try:
                Hi = np.arange(unp.nominal_values(Hcenters.min()),unp.nominal_values(Hcenters.max()+dh),dh)
            except:
                Hi = np.arange(unp.nominal_values(Hcenters.min()),unp.nominal_values(Hcenters.max()+dh),-dh)
            aHi = np.arange(unp.nominal_values(aHcenters.min()),unp.nominal_values(aHcenters.max()+dh),dh)      
            
            Hdigit = np.digitize(unp.nominal_values(Hm0), unp.nominal_values(Hi))            
            aHdigit = np.digitize(unp.nominal_values(aHm0), unp.nominal_values(aHi))
            
            Hcounts, Hbins = np.histogram(unp.nominal_values(M), bins=unp.nominal_values(Hi))
            aHcounts, aHbins = np.histogram(unp.nominal_values(aM), bins=unp.nominal_values(aHi))
            
            Hcenters = 0.5*(Hbins[1:]+Hbins[:-1])
            aHcenters = 0.5*(aHbins[1:]+aHbins[:-1])
            
            #assert(len(Hcounts)==len(aHcounts))
            Mhistn = np.zeros(len(Hcounts))
            Mhistn = unp.uarray(Mhistn,Mhistn)
            #Merrn = np.zeros(len(Hcounts))
            aMhistn = np.zeros(len(aHcounts))
            aMhistn = unp.uarray(aMhistn,aMhistn)            
            #aMerrn = np.zeros(len(aHcounts))            
            for i in range(len(Hcounts)):
                idx = np.where((i+1) == Hdigit)
                Mhistn[i] = M[idx].mean()

            for i in range(len(aHcounts)):
                idx = np.where((i+1) == aHdigit)
                aMhistn[i] = aM[idx].mean()
         
            Hm0 = Hcenters
            aHm0=aHcenters
            # Find bad bins:
            Mhistn = unp.uarray(unp.nominal_values(Mhistn),unp.std_devs(Mhistn))
            aMhistn = unp.uarray(unp.nominal_values(aMhistn),unp.std_devs(aMhistn))          
            idx = np.where(np.logical_not(np.equal(unp.std_devs(aMhistn),0.)))
            #Hm0 = Hm0[idx]
            #Mhistn = Mhistn[idx]
            aHm0 = aHm0[idx]
            aMhistn=aMhistn[idx]
            
            idx = np.where(np.logical_not(np.equal(unp.std_devs(Mhistn),0.)))
            Hm0 = Hm0[idx]
            Mhistn = Mhistn[idx]
            #aMhistn=aMhistn[idx]              
            # Get subtracted data
            
            # Fit a linear and paramagnetic term to the addenda and subtract!                         
            
            mod = Model(self.para)
            pars  = mod.make_params(slope=-1, amplitude=0.1, width=0.1)
            
            result = mod.fit(unp.nominal_values(aMhistn), pars, x=aHm0)  
            self.result = result
            y = self.para(Hm0, result.params['slope'].value, result.params['amplitude'].value, result.params['width'].value)
            # get errors on y
            #...
            
            #M = Mhistn - aMhistn
            M = Mhistn - y
            idx = np.where(np.logical_not(unp.isnan(M)))
            Hm0 = Hm0[idx]         
            M   = M[idx]
            
        print('mass:', sample_mass, '/n', 'Saturation Magnetization at '+str(t)+' K:', M.max(), '/n', end=' ') 
        #later use a specific data set for magnetization data versus 
        #temp, field, and freq
        F = np.zeros(len(M))        
        F = np.resize(F,1)#reshape(1)    
        F = unp.uarray(F,F*0.001)        
        F = DataLabel('frequency', F, 'rlu', True, \
                    axislabel=r'$(r.l.u.)$')
        T = t*np.ones(len(M))
        T = np.resize(T,1)#reshape(1)
        T = unp.uarray(T,T*0.001)
        T = DataLabel('temperature', T, 'Kelvin', True, 
                      axislabel=r'T ($K$)')
        #Hm0 = np.resize(Hm0,) #reshape(1)
        Hm0 = DataLabel('magnetic_field', Hm0, 'Tesla', True, 
                      axislabel=r'$\mu_\circ$H ($T$)')
        
        coords = Coordinates([Hm0,T,F])#
    
        # * Signal MUST BE RESHAPED TO MATCH THE COORDINATE SHAPE *
        M = M.reshape(len(Hm0.ticks),len(T.ticks),len(F.ticks))
        M_label = DataLabel('Magnetization', M, 'Bohr Magneton', False, 
                            axislabel=r'$\bm{\mu}\cdot\bm{H} (\mu_B$/)'+el)
        if title is None: title = 'Magnetization vs T, H, F'
        
    
        mzn = Magnetization(M, coords, value_error=None, value_label=M_label, clabels=(Hm0,T,F,),
            title=title, meta='magnetization', dims=('magnetic_field','temperature','frequency',))
        return mzn
    # 
    @staticmethod
    def para(x, slope, amplitude, width):
        "line+tanh"
        return slope * x + amplitude*np.tanh(x/width) 
    #
    @staticmethod    
    def rho_fermi_liquid(T, pfl, A2):
        """
        Canonical quadratic in temperature dependence of the resisitivity
        associated with Fermi Liquid like behavior. Other origins are also
        possible, e.g. spin disorder in gapless ferromagnets. 
        """
        return pfl + np.abs(A2)*T**2.  
    #
    @staticmethod
    def rho_residual(T, p0):
        """
        Residual temperature independent contribution to the resistivity 
        arising from impurities, etc.
        """
        return p0
    #
    @staticmethod
    def rho_magnon_gap(T, pmg, D, Delta):
        """
        Contribution to resisitivity coming from the opening of a magnon gap.
        """
        return pmg + (np.abs(D)*T/np.abs(Delta))*(1.+2.*T/np.abs(Delta))*np.exp(-np.abs(Delta)/T)
    #
    @staticmethod
    def rho_magnon_gap_vH(H, H0, pmg, T0, D, gap0):
        """
        Contribution to resisitivity coming from the opening of a magnon gap.
        """
        gap = gap0*(1.-(H/H0)**2.)
        return pmg + (D*T0/gap)*(1.+2.*T0/gap)*np.exp(-gap/T0)
    #    
    @staticmethod
    def rho_kondo(T, pk, A5, Jk):
        return
    #
    def mvth_scan_plot(chi,t,h,save=False,svname='mvth_scan',rasterized=True, molar_mass=270.05, sample_mass=32.3, Nm=2,
                       gfxdir = '/Users/Guy/Documents/Johns Hopkins/Academics/Broholm Group/Experiments/Cu2OSeO3/PPMS/gfx/',
                       meas_dir='susceptibility/', vmin=None, vmax=None, norm=None):
        """"""
        # Compute the scaling factor for Gaussian/cgs emu to SI units of molar susceptibility per magnetic atom
        x_emu_to_SI = 4.*np.pi * 1.e-6
        M_emu_to_SI = 1.e-3
        nodim_to_molar = (constants.N_A * unit_cell_volume*1.e-24) # cm^3 / mol; from molar_mass * (constants.N_A * unit_cell_volume*1.e-24) / molar_mass)
        M_to_molar = mag_molar_mass /  (sample_mass * constants.N_A)
        fu_to_mag = Nm * mag_molar_mass / molar_mass 
        #     
        
        idx = np.where(np.logical_and(chi>0, np.logical_not(np.isnan(x))))
        chi =  data["M'' (emu)"] * emu_to_SI * nodim_to_molar * fu_to_mag; t=t[idx]; h=h[idx]
        chi/= data['Moment (emu)']
        
        # Make the figure using gridspec. Should make this part more modular so it can be used elsewhere.
        if save:
            pubfig.pub_ready()
            figsize = pubfig.fig_size()
        else:
            figsize = (8,6)
        figsl = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(1,2, width_ratios=[8,1], wspace=0)
        pax = plt.subplot(gs[0])
        pax.set_xlabel('Temperature (K)')
        pax.set_ylabel('Magnetic Field (Oe)')   
        pax.set_title(r'Susceptibility (emu / Oe $\cdot$ mg)')
        cax = plt.subplot(gs[1])
        
        ti = np.linspace(np.nanmin(t), np.nanmax(t), 1000)
        #else: ti = np.logspace(np.log10(np.nanmin(t)), np.log10(np.nanmax(t)), 1000)
        hi = np.linspace(np.nanmin(h), np.nanmax(h), 1000)
        ti, hi = np.meshgrid(ti, hi)        
        chi_interp = griddata((t,h), chi, (ti, hi), method='linear') 
        
        cmap = plt.get_cmap('RdYlBu_r')
        cmap.set_bad(alpha=0.) 
        
        if vmin is None: vmin=np.nanmin(chi)#1.e-8
        if vmax is None: vmax=np.nanmax(chi)#1.e-6
        if norm is not None: norm = LogNorm(vmin=np.min(chi[np.where(chi>0)]), vmax=np.nanmax(chi))
        pc    = pax.pcolormesh(ti, hi, chi_interp, rasterized=rasterized, vmin=vmin, vmax=vmax, cmap=cmap, norm=norm)
        pscat = pax.scatter(t, h, c=chi, marker='o', linewidths=0.25, s=2, alpha=0.5,  vmin=vmin, vmax=vmax, cmap=cmap, norm=norm)    
        cb = plt.colorbar(pc, cax=cax)
        cb.solids.set_rasterized(rasterized)    
        pax.axis([56,59,0,300])
            
        if save:
            if norm is None: plt.savefig(gfxdir+meas_dir+svname+'.pdf',bbox_inches='tight')
            else: plt.savefig(gfxdir+meas_dir+svname+'_logscale.pdf',bbox_inches='tight')
        else:
            plt.show()
        return
    #
    def cvht_scan_plot(filename, datdir='/Users/Guy/Documents/Johns Hopkins/Academics/Broholm Group/Experiments/Cu2OSeO3/PPMS/First Order/',
                       gfxdir='/Users/Guy/Documents/Johns Hopkins/Academics/Broholm Group/Experiments/Cu2OSeO3/PPMS/gfx/',
                       usecols=['Temperature (K)', ' Field (Oe)',' HCSample (mJ/mole-K)'], ext='.dat', meas_dir='heat capacity/',
                       save=False, show=True, svname='cvth', from_single_slope_save=True, oplot=False, rasterized=True, skiprows=13,
                       histogram=False):
        """"""
        if from_single_slope_save: 
            data = acms_data(datdir+filename+ext, usecols=usecols, from_single_slope_save=from_single_slope_save)
            c = np.asanyarray(data[usecols[2]])
            h = np.asanyarray(data[usecols[1]])
            t = np.asanyarray(data[usecols[0]])
            idx=np.where(np.logical_not(np.isnan(t)))
            t=t[idx]; h=h[idx]; c=c[idx]        
        else: 
            data = acms_data(datdir+filename+ext, usecols=usecols, from_single_slope_save=from_single_slope_save,skiprows=skiprows)
        
            c = np.asanyarray(data[usecols[1]])
            h = np.asanyarray(data[usecols[3]])
            t = np.asanyarray(data[usecols[0]])
            idx=np.where(np.logical_not(np.isnan(t)))
            t=t[idx]; h=h[idx]; c=c[idx]
        
       
        if histogram:
            temps = data['Temperature (K)']
            temps=np.asanyarray(temps)
            idx=np.where(np.abs(h)<=5.)
            h=data[' Field (Oe)']
            idx=np.where(np.abs(h)<=5.)
            temps=temps[idx]
            c = data[' HCSample (\xb5J/K)']
            c=np.asanyarray(c)
            c=c[idx]
            tbins = np.linspace(np.nanmin(temps), np.nanmax(temps), 160)
            tbins = np.arange(np.nanmin(temps), np.nanmax(temps), 0.05)
            tbins
            tbins_lin = np.linspace(np.nanmin(temps), np.nanmax(temps), 160)
            tbins_lin
            tdigit = np.digitize(temps, tbins)
            tcounts, tbins_out = np.histogram(temps, bins=tbins)
            tcenters = 0.5*(tbins[1:]+tbins[:-1])
            chist = np.zeros(len(tcounts))
            cerr = np.zeros(len(tcounts))
            for i in range(len(tcounts)):
                idx = np.where((i+1) == tdigit)
                chist[i] = c[idx].sum()/tcounts[i] #np.average(c[idx], weights=None)
                cerr[i]  = c[idx].std()
            idx_plt = list(range(40,len(tcenters)-60))
            #plt.plot(tcenters[:-2], chist[:-2], '.')    
            plt.errorbar(tcenters[idx_plt], chist[idx_plt], yerr=cerr[idx_plt], marker='.')
        
        
        # Make the figure using gridspec. Should make this part more modular so it can be used elsewhere.
        if save:
            pubfig.pub_ready()
            figsize = pubfig.fig_size()
        else:
            rc('text', usetex=True)
            figsize = (8,6)
        figsl = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(1,2, width_ratios=[8,1], wspace=0)
        pax = plt.subplot(gs[0])
        pax.set_xlabel('Temperature (K)')
        pax.set_ylabel(r'Magnetic field (Oe)')
        pax.set_title(r'Heat Capacity (mJ / mol $\cdot$ K)')
        cax = plt.subplot(gs[1])
    
        hi1d = np.linspace(np.nanmin(h), np.nanmax(h), 1000)
        ti = np.linspace(np.nanmin(t), 59, 1000)
        ti = np.linspace(57.6, 58.6, 1000)
        ti, hi = np.meshgrid(ti, hi1d)        
        ci = griddata((t,h), c, (ti, hi), method='linear')  
    
        if oplot:
            # find max trend line
            hm=[]; tm=[]; cm=[]
            for hp in hi1d:
                idx = np.where(hp==hi)
                max_idx = np.where(np.nanmax(ci[idx])==ci)
                #max_idx = np.intersect1d(idx, max_idx)
                hm.append(hi[max_idx])
                tm.append(ti[max_idx])
                cm.append(ci[max_idx])
        
            hm = np.asarray(hm).squeeze()  
            tm = np.asarray(tm).squeeze()
            cm = np.asarray(cm).squeeze()
    
    
        cmap = plt.get_cmap('RdYlBu_r')
        cmap.set_bad(alpha=0.)    
        
        c0 = np.nanmin(ci)
        c1 = np.nanmax(ci)  
        cd = c1-c0
        
        vmin = c1-cd
        vmax = c1+.03*cd
        norm = None#LogNorm(vmin=np.nanmin(c[np.where(c>0)]), vmax=np.nanmax(c))
    
        pc    = pax.pcolormesh(ti, hi, ci, rasterized=rasterized, vmin=vmin, vmax=vmax, cmap=cmap, norm=norm)
        pscat = pax.scatter(t, h, c=c, alpha=0.7, marker='o', s=5, vmin=vmin, vmax=vmax, cmap=cmap, norm=norm)
        if oplot: pmax  = pax.plot(tm, hm, c=1., alpha=1., marker='.', ls='--', markevery=50)
        pax.set_xlim(np.nanmin(ti),np.nanmax(ti))
        pax.set_ylim(np.nanmin(hi), np.nanmax(hi))
        cb = plt.colorbar(pc, cax=cax)
        cb.solids.set_rasterized(rasterized)    
        if save:
            if not oplot: plt.savefig(gfxdir+meas_dir+svname+'.pdf',bbox_inches='tight')
            else: plt.savefig(gfxdir+meas_dir+svname+'_with_max_trace.pdf',bbox_inches='tight')
        elif show:
            plt.show()
        return    
    #
    def heat_capacity_plot(data, pax=None, save=False,svname='heat_capacity',rasterized=True, Cerr=None, single_slope=False, show=True, Hlist=[135], dH=5, marker=''):
        """"""
        colors = itertools.cycle(['.b', '.g', '.r', '.c', '.m', '.y'])
        for H in Hlist:
            color = next(colors)
            print(color)
            if not single_slope:
                #idx = np.where(np.abs(data['Temperature (K)']-t)<dt)[0]
                T = data['Temperature (K)']#[idx]
                C = data[' HCSample (mJ/mole-K)']#[idx]
                #print 'mass:', sample_mass, '/n', 'Saturation Magnetization at 5 K:', M[90], '/n',     
            else:
                idx = np.where(np.abs(data[' Field (Oe)']-H)<dH)[0]
                T = data['Temperature (K)'][idx]
                C = data[' HCSample (mJ/mole-K)'][idx]
                #print 'mass:', sample_mass, '/n', 'Saturation Magnetization at 5 K:', M[90], '/n',             
            
            
            ## Get the entropy
            #sh  = h.shape
            #T   = np.zeros(sh)
            #S   = np.zeros(sh)
            #cT  = np.zeros(sh) 
            #cTm= np.zeros(sh)
            #Sm = np.zeros(sh)
            #tD = 1.
            #dh=5
            ##hlist = [0,10,135,260,315,475]
            #hlist = [0]
            #for hp in hlist:
            #idx = np.where(np.abs(hp-h)<=dh)
            #tint=np.linspace(np.min(t), np.max(t),1000)
            #cint = np.linspace(np.nanmin(c), np.nanmax(c),1000)
            #tmp = physfit.entropy(T=t[idx], c=c[idx])
            #T[idx] = tmp[0]
            #S[idx] = tmp[1]
            #cT[idx]= tmp[2]
            #tmp = physfit.entropy(T=t[idx], c=c[idx]/1.e3 - ((t[idx]/tD)**3.)*((2.*np.pi**2.)/5)*(constants.k**4.)/((constants.hbar*constants.c)**3.))
            #Sm[idx] = tmp[1]
            #cTm[idx] = tmp[2]
            #idx = np.where(np.abs(h-0)<dh)
            #plt.plot(T[idx]**2., cT[idx], 'ob', T[idx]**2., cT[idx]-cTm[idx], 'oy')
            ##plt.plot(T[idx], Sm[idx]/((1.e3/2) * constants.N_A*constants.k * np.log(2.)), '^g', T[idx], cTm[idx], '^k')
            #plt.show()           
            
            
            
            
            if pax is None and Hlist.index(H) is 0:
                # Make the figure using gridspec. Should make this part more modular so it can be used elsewhere.
                if save:
                    pubfig.pub_ready()
                    figsize = pubfig.fig_size()
                else:
                    rc('text', usetex=True)
                    figsize = (8,6)
                figsl = plt.figure(figsize=figsize)
                gs = gridspec.GridSpec(1,1, width_ratios=[1], wspace=0)
                pax = plt.subplot(gs[0])
                pax.set_xlabel('Temperature (K)')
                pax.set_ylabel(r'Heat Capacity (mJ/mole-K)')
            if Cerr is not None: pc = pax.errorbar(T, C, yerr=Cerr)
            else: pc = pax.plot(T, C, marker+color)    
        if save:
            savefig(svname+'.pdf',bbox_inches='tight')
        elif show:
            plt.show()
                
        if Cerr is not None: return T, C, Cerr, pax
        else: return T, C, pax
    #
    def chi_v_th_with_cvht_vol_plot():
        """"""
        #x, y, z = np.ogrid[-10:10:20j, -10:10:20j, -10:10:20j]
        #s = np.sin(x*y*z)/(x*y*z)    
        # 
        ti = np.linspace(np.nanmin(56), np.nanmax(62), 100)
        hi = np.linspace(np.nanmin(0), np.nanmax(300), 100)
        fi = np.logspace(1,4,100)
        #ti_d, hi_d, fi_d = np.meshgrid(ti, hi, fi, sparse=False)
        ti, hi, fi = np.meshgrid(ti, hi, fi, sparse=True)
        chi_i = griddata((t, h, f), chi, (ti, hi, fi), method='nearest')
        #
        src = mlab.pipeline.scalar_field(chi_i)
        mlab.pipeline.iso_surface(src, contours=[chi_i.min()+0.1*chi_i.ptp(), ], opacity=0.1)
        mlab.pipeline.iso_surface(src, contours=[chi_i.max()-0.1*chi_i.ptp(), ],)
        mlab.pipeline.image_plane_widget(src,
                                         plane_orientation='z_axes',
                                         slice_index=33,
                                         )    
        
        return
    #
#
class PSIReader(DataReader):
    """
    TODO:
    * Possible to automatically close the file if the DataReader object is 
    deleted?
    """
    def __init__(self, datasources, scan_keys, **kwargs):
        """
        TODO:
        * Generalize to work with many scans!
        """
        # Only allow data files through if they are valid
        if isinstance(datasources,Iterable):
            for ds in datasources:
                if isinstance(ds, str) and not os.path.isfile(ds):
                    datasources.remove(ds)
                    warnings.warn('Found a bad path specifier for datafile \
                                  - removing from list!')
                        
            
        super(PSIReader,self).__init__(datasources, scan_keys, **kwargs)
        self.set_known_ext_dict()
        self.infer_readers()
        self._read_all()
        #self._set_root()
        return
    #
    def place_data(self, data, data_container=None):
        """
        Plcae the input data in the proper container if associated and
        associate any (optional) input DataContainer first.
        """
        try:
            self.datacontainer.place(data)
        except AttributeError:
            if data_container is None:
                raise AttributeError('No DataContainer associated or '
                                     'provided. Please supply and retry.')
            elif not isinstance(data_container, DataContainer):
                raise TypeError('Input must be of instance of DataContainer.')
            else:
                self.datacontainer = data_container
                self.datacontainer.place(data)
        return
    #
    def _read_all(self):
        """
        Instantiate readers for all the associated datasources
        """
        for key in self.datasources.keys():
            self.read(key)
        return
    #
    def read(self,key):
        """
        Read in the datasource and assign to an associated DataContainer
        """
        key=str(key)
        
        source = self.datasources[key]
        ext = self.datasources[key].ext        

        if self.reader_inferred: 
            pass
        # case of filename
        elif self.datasource.isFileName:
            if ext in list(self.known_exts.keys()):
                self.infer_reader(ext=ext)
                
        # case of h5py.File object
        # case of pytables ojbect
        # case of pandas.HDFStore 
        else:
            raise TypeError('Cant read that!')
        
        
        # case of filename        
        if source.isFileName:
            self.data[key] = self.readers[key](source.filename)
            

        return self.data[key]
    #
    def infer_readers(self, ext=None, obj=None):
        """
        Infer the read function to use for the given extension or object.
        TODO:
        * Finish the other cases
        """
        for key,source in self.datasources.items():
                
            if source.isFileName:
                ext = source.ext if ext is None else ext
                if ext in self.known_ext_groups['HDF']:
                    self.readers[key] = h5py.File   
            else:
                raise TypeError('Cannot find a reader for that type of source!')
            # Other cases...
        
        return
    #
    def set_known_ext_dict(self):
        """
        Set up the OrderedDict of file extensions known to this reader.
        """
        self.known_exts = OrderedDict()
        self.known_ext_groups = OrderedDict()
        
        #
        group_key = 'HDF'
        hdf_exts = ['.hd5', '.hdf', '.hdf5', '.h5'] # '.nxs'?, netcdf?        
        self.known_ext_groups[group_key] = hdf_exts
        
        
        for group_key,value_list in self.known_ext_groups.items():
            for value_key in value_list:
                self.known_exts[value_key] = group_key
            
        return
    #    
    def add_ext(self, read_fcn):
        """
        User may provide a function that will read the given DataSource and 
        input to the associated DataContainer
        """
        
        return
    #
#    
class RITA2Reader(PSIReader):
    """
    Read in HDF5 files and parse into different useful slices of data to be 
    stored as Data tables for current and later use. 
    """
    #
    #def get_magnetic_field_trace?
    #
    
    def create_NeutronScanSet(self):
        """
        This method will construct NeutronScanSet (subclass of DataSet) from 
        the datasources stored in the reader.
        """
        nset = NeutronScanSet()
        for key in self.data.keys():
            ns = self.get_single_scan(key)
            nset.add_scan(ns)
        self.close_all()
        return nset
    #
    def close_all(self):
        """
        Close all the datasources.
        """
        for value in self.datasources.values():
            value.close()
        return
    #
    def _get_sweep_dir(self,nom,pnom,std,pstd,name=None):
        """
        Get sweep direction for input variable
        """
        sweep_dir='unknown'
        if (nom-pnom) > (std+pstd):
            sweep_dir = 'up sweep' 
        elif (nom-pnom) < (std+pstd):
            sweep_dir = 'down sweep'
        elif np.abs(nom-pnom) < (std+pstd): 
            sweep_dir = 'no change'
        else: 
            warnings.warn(str(name)+'sweep direction unknown!')   
            
        return sweep_dir
    #
    def get_single_scan(self, key, dim='h', verbose=False):
        """
        This instance method will access the data field and pull from it the
        information required for a neutron scan.
        """
        
        if verbose:
            _loc = "in"+__name__+"> "
        
        key = str(key)   
        scn = int(key)
        datum = self.data[key]
        reader=self
            
        if verbose:
            print(_loc+"Scan #: " + str(scn))        
        
        #Get previous data
        try:
            idx = list(self.data.keys()).index(key)-1
            pkey= list(self.data.keys())[idx]
            pdatum = self.data[pkey]
            pmagnetic_fields = pdatum.get_magnetic_fields(pkey)
            ptemperatures = pdatum.get_temperatures(pkey)
        
        except:
            pmagnetic_fields = np.array([0])
            ptemperatures    = np.array([400])
            
        pmf_nom = pmagnetic_fields.mean()
        pmf_std = pmagnetic_fields.std()
        ptemp_nom = ptemperatures.mean()
        ptemp_std = ptemperatures.std()
        
        neutron_scan = None
        try:
            if str(datum['entry1/user/name'].value[0]) == 'Guy' and\
               str(datum['entry1/sample/name'].value[0]) == 'CeAuSb2':
                
                # These should be created as Coordinates right here!
                mon = reader.get_monitor(key)
                I, qmap = reader.get_counts(key,mon=mon)
                Q, qmap_coords = reader.get_hkl(key)
                


                magnetic_fields = reader.get_magnetic_fields(key)
                mf_nom = magnetic_fields.mean()
                mf_std = magnetic_fields.std()
                mf_str = str(mf_nom)+' T' if mf_nom >= 1. \
                    else str(1.e3*mf_nom)+' mT'  
                
                temperatures = reader.get_temperatures(key)
                temp_nom = temperatures.mean()
                temp_std = temperatures.std()
                temp_str = str(temp_nom)+' K' if temp_nom >= 1. \
                    else str(1.e3*temp_nom)+' mK' 
                
                scan_command = reader.get_scan_command(key)
                
                if verbose:
                    print(scan_command)
                    print(Q['h'].mean(), I.max())

                # Implement this for temp as well! (make it a function.)
                mf_sweep_dir = self._get_sweep_dir(mf_nom,pmf_nom,
                    mf_std,pmf_std, name='magnetic field ')
                temp_sweep_dir = self._get_sweep_dir(temp_nom,ptemp_nom,
                    temp_std,ptemp_std, name='temperature ')
                
                #if (mf_nom-pmf_nom) > (mf_std+pmf_std):
                    #mf_sweep_dir = 'up sweep' 
                #elif (mf_nom-pmf_nom) < (mf_std+pmf_std):
                    #mf_sweep_dir = 'down sweep'
                #elif np.abs(mf_nom-pmf_nom) < (mf_std+pmf_std): 
                    #mf_sweep_dir = 'no change'
                #else: 
                    #warnings.warn('magnetic field sweep direction unknown!')
                    
                    
                    # get title and info about plot
                title='Scan of ('+str(Q['h'].mean())+','+str(Q['k'].mean())+\
                    ','+str(Q['l'].mean())+')\n in '+mf_str+' along c at '+temp_str

                meta = ''
                #meta+='\n\n'
                meta+= r'$\mathbf{Start Time}$: '\
                    +datum['entry1/start_time'].value[0]+'\t\t'
                meta+= r'$\mathbf{End Time}$: '\
                    +datum['entry1/end_time'].value[0]+'\n'
                meta+= r'$\mathbf{Scan Command}$: '+scan_command+'\n'
                meta+= r'$\mathbf{Temperature}$: '+temp_str+\
                    ' ('+temp_sweep_dir+')'+'\t\t'
                meta+= r'$\mathbf{Magnetic Field}$: '+mf_str+\
                    ' ('+mf_sweep_dir+')'+'\n'
                meta+=''
                #more to add...                                       
                    
    
                Q[dim] = DataLabel(dim, Q[dim], 'rlu', True, 
                                   axislabel=r'$(r.l.u.)$')
                # adding H and T to coordinates... 
                T = unp.uarray(temp_nom, temp_std)
                T = np.resize(T,1)#reshape(1)
                T = DataLabel('temperature', T, 'Kelvin', True, 
                              axislabel=r'T ($K$)')
                B = unp.uarray(mf_nom, mf_std)
                B = np.resize(B,1) #reshape(1)
                B = DataLabel('magnetic_field', B, 'Tesla', True, 
                              axislabel=r'B ($T$)')
                
                
                coords = Coordinates([Q[dim],T,B])#
                #if len(B.ticks) !=1 or len(T.ticks) !=1:
                    #print 'stop'
                    #raise Exception
            
                # * Intensity MUST BE RESHAPED TO MATCH THE COORDINATE SHAPE *
                dimtup = (len(I),1,1)                
                I = I.reshape(dimtup)
                I_label = DataLabel('I', I, 'Intensity', False, 
                                    axislabel=r'cts/mon')
                
            
                neutron_scan = NeutronScan(I, coords, value_label=I_label,
                                           meta=meta, title=title,
                                           dims=[dim,'temperature','magnetic_field'])       
        except:
            raise
        
        reader.datasources[key].close()
        reader.datasources[pkey].close()
        return neutron_scan
    #
    @staticmethod
    def _finder(name):
        clb = lambda path: path if name in path else None
        return clb
    #
    def get_magnetic_fields(self, key, name='magnetic_field' ):
        """
        Get the magnetic fields array
        """
        data = self[key]
        
        clb = self._finder(name)
        hkey = data.visit(clb)
        mf = data[hkey]
        return mf.value
    #
    def get_temperatures(self, key, name='entry1/sample/temperature'):
        """
        Get the temps.
        """
        data = self[key]
        
        clb = self._finder(name)
        hkey = data.visit(clb)
        temp = data[hkey]
        return temp.value
    #
    def get_monitor(self, key, name='entry1/control/data'):
        
        data = self[key]
        
        clb = self._finder(name)
        hkey = data.visit(clb)
        monitor = data[hkey]
        return monitor.value
    #
    def get_counts(self, key, mon=1., name='RITA-2/detectorwindows/counts'):
        """
        Get the map of detector counts.
        """
        data = self[key]
        
        clb = self._finder(name)
        hkey = data.visit(clb)
        qmap = data[hkey].value[:,:9]
        
        y = qmap[:,4]/mon 
        ye = np.sqrt(qmap[:,4])/mon
        idx= np.where(y==0)
        ye[idx] = 1./mon[idx] # Fix the error on zero counts    
        I = unp.uarray(y, ye)
        
        if len(mon) > 1.:
            if not np.all(np.equal(mon,mon[0])):
                raise NotImplemented(""" Monitor counts differing between 
                points not yet implemented! """)
            
            unit_str = 'cts/'+str(mon)
        else:
            unit_str = 'cts'
            
            # Make into a Coordinate here?

        return I, qmap
    #
    def get_hkl(self, key, name='entry1/data/Q'):
        """
        Get the h,k, and l values and return as a tuple
        """
        data = self[key]
        
        qmap_coords=[]
        for dim in list('hkl'):
            clb = self._finder(name+dim)
            hkey = data.visit(clb)
            qmap_coords.append(data[hkey].value)
        
        h,k,l = tuple(qmap_coords)
        q=OrderedDict()
        qe=OrderedDict()
        q['h']= h[:,4]
        qe['h'] = 0. # can find some way to put an error on this
        q['k'] = k[:,4]
        qe['k'] = 0.   
        q['l'] = l[:,4]
        qe['l'] = 0.
        
        Q = OrderedDict()
        for dim in list('hkl'):
            Q[dim] = unp.uarray(q[dim],qe[dim])
            
        unit_string = r'(r.$\ell$.u.)'
              
        return Q, (h,k,l,)
    #
    def get_scan_command(self, key, name='entry1/scancommand'):
        """
        Gets the scan command used for the current scan being read.
        """
        data = self[key]
        
        clb = self._finder(name)
        hkey = data.visit(clb)
        sc  = data[hkey]
        return sc.value[0]        
    #
    def _set_root(self):
        """
        Set the RITA2 as root
        """
        
        return
    #
    def __getitem__(self,key):
        """
        Return the data at str(key) 
        """
        return self.data[str(key)]
    #

    @classmethod
    def test(cls,first=6758,last=6758):
        """
        Test.
        """
        workspace_dir_path = '/Users/Guy/Documents/Johns Hopkins/Academics/Broholm Group/Experiments/CeAuSb2/PSI-Field/'
        workspace_dir_name = 'workspace/'
        data_proc_dir = 'data_processed/'
        data_dir = 'data/'
        gfx_dir  = 'gfx/'
        fn_root  = 'rita22015n00'
        fn_ext   = '.hdf'# generalize with a function to check for other variants (like the default '.h5')        
        
        reader = None        
        scans = list(range(first, last+1)) 
        for scn in scans:
            key = str(scn)
            print("Scan #: " + key)
    
            # Get the file
            fn = workspace_dir_path+data_dir+fn_root+str(scn)+fn_ext
            ds  = DataSource(fn)
            
            if scn == scans[0]:
                pds = DataSource(data_dir+fn_root+str(scn-1)+fn_ext)
                if pds.isFileName:
                    prev_reader = cls(pds,scn-1)# Needs to be previous key...
                    pmagnetic_fields = prev_reader.get_magnetic_fields()
                    pmf_nom = pmagnetic_fields.mean()  
                    pmf_std = pmagnetic_fields.std()
                else:
                    # Check if 'test'+fn_ext exists and make it if not.
                    pds = DataSource(data_dir+fn_root+'test'+fn_ext)                  
                    pmagnetic_fields = np.array(0)
                    pmf_nom = pmagnetic_fields.mean()
                    pmf_std = pmagnetic_fields.std()  
            
            if ds.isFileName: 
                # if exists
                reader = cls(ds,[scn]) 
                dathdf = reader.data[key]
    
                try:
                    if str(dathdf['entry1/user/name'].value[0]) == 'Guy' and\
                       str(dathdf['entry1/sample/name'].value[0]) == 'CeAuSb2':
                        # Extract information from the file:
                        # Eventually could be file from any neutron source. 
                        # Ideally, should be uniformized to hdf data before 
                        # processing.
                        
                        mon = reader.get_monitor(key)
                        I, qmap = reader.get_counts(key, mon=mon)
                        Q, qmap_coords = reader.get_hkl(key)

                        magnetic_fields = reader.get_magnetic_fields(key)
                        mf_nom = magnetic_fields.mean()
                        mf_std = magnetic_fields.std()
                        mf_str = str(mf_nom)+' T' if mf_nom >= 1. else str(1.e3*mf_nom)+' mT'  
                        
                        temperatures = reader.get_temperatures(key)
                        temp_nom = temperatures.mean()
                        temp_std = temperatures.std()
                        temp_str = str(temp_nom)+' K' if temp_nom >= 1. else str(1.e3*temp_nom)+' mK' 
                        
                        scan_command = reader.get_scan_command(key)
                        
                        print(scan_command)
                        print(Q['h'].mean(), I.max())
    
                        # Implement this:
                        sweep_dir='unknown'
                        if (mf_nom-pmf_nom) > (mf_std+pmf_std):
                            sweep_dir = 'up sweep' 
                        elif (mf_nom-pmf_nom) < (mf_std+pmf_std):
                            sweep_dir = 'down sweep'
                        elif np.abs(mf_nom-pmf_nom) < (mf_std+pmf_std): 
                            sweep_dir = 'no change'
                        else:
                            warnings.warn('Do not know if it is up or down sweep!')
                            
                        ## The following would be done in the DataSlice ##

                        replot=True
                        # Make these dicts!
                        coord_labels=[r'$(hh0)$',r'$(kk0)$',r'$()00\ell)$']
                        count_label='cts/mon'
                        for dim,x in list(Q.items()):
                            
                            if dim == 'h':
                                gfxfn = gfx_dir+fn_root+str(scn)+'_scan_'+dim+'.pdf'
                                if (not os.path.isfile(gfxfn)) or replot:   # can adjust 'replot' so it works for individual scans to be singled out
                                    x=unp.nominal_values(Q[dim])
                                    xe=unp.std_devs(Q[dim])
                                    y=unp.nominal_values(I)
                                    ye=unp.std_devs(I)
                                    
                                    xu = x.max(); xl = x.min()
                                    yu = y.max(); yl = y.min()
                                    params=lmfit.Parameters()
                                    params.add(name='area', value=3.e-4, min=0., vary=True)
                                    params.add(name='sigma', value=3.e-3, min=0., max=xu,
                                               vary=True)
                                    params.add(name='center', value=0.864, vary=True,
                                               min=xl, max=xu)
                                    params.add(name='background', value=0., vary=True,
                                               min=0., max=yu)
                                    params.add(name='slope', value=0., vary=True)                        
                                    # Fit teh data
                                    # std err issue !!!!!!!!!!!!!!
                                    gfit = GaussianModel(unp.nominal_values(x), unp.nominal_values(y), params, yerrs=ye)
                                    gfit.auto_min(verbose=True)
        
                                    # get title and info about plot
                                    title='Scan of ('+str(Q['h'].mean())+','+str(Q['k'].mean())+','+str(Q['l'].mean())+')\n in '+mf_str+' along c at '+temp_str
        
                                    meta = ''
                                    #meta+='\n\n'
                                    meta+= r'$\mathbf{Start Time}$: '+dathdf['entry1/start_time'].value[0]+'\t\t'
                                    meta+= r'$\mathbf{End Time}$: '+dathdf['entry1/end_time'].value[0]+'\n'
                                    meta+= r'$\mathbf{Scan Command}$: '+scan_command+'\n'
                                    meta+= r'$\mathbf{Temperature}$: '+temp_str+'\t\t'
                                    meta+= r'$\mathbf{Magnetic Field}$: '+mf_str+' ('+sweep_dir+')'+'\n'
                                    meta+=''
                                    #more to add...                        
        
                                    gfit.plot(show=False, save=False,
                                              title=title, meta=meta,
                                              xlabel=coord_labels[0], ylabel=count_label,
                                              gfxpath=gfxfn)
                except:
                    raise               
        return reader
    #
#
class IEXYReader(DataReader):
    """"""
    def create_sqw(self,xa=[1,1,0],ya=[0,0,1],w=0.):
        I,E,x,y = self.read_iexy()
        h = xa[0]*x + ya[0]*y
        k = xa[1]*x + ya[1]*y
        l = xa[2]*x + ya[2]*y
        Q = np.a=sanyarray(a)
        
        # * Intensity MUST BE RESHAPED TO MATCH THE COORDINATE SHAPE *
        #dimtup = (len(I))                
        I_label = DataLabel('I', I, 'Intensity', False, axislabel=r'cts/mon')
        
        sqw = SQw(I, Q, value_error=E, meta=self.meta)
        return sqw
    #
    def read_iexy_as_powder():
        return
    #
    def read_iexy(self, key=0, raw=False, hires=True, folded=True, iflag=-1.e20):
        """"""
        # Load in the data
        i, e, x, y = np.loadtxt(self.datasources[key], unpack=True)

        # Mark the invalid data points in the intensity array as 'NaN' using the default value from Mslice
        # and create a masked array for the data.
        idx = np.where(i == iflag)[0]
        i[idx] = np.nan # where outputs a tuple. seems that taking the zeroth element is good.
        i = np.ma.masked_invalid(i)
        e[idx] = np.nan # 'uncertainties' module needs positive std_devs or 'nan'. Regardless, 'nan' is a better label than '-1'
        e = np.ma.masked_invalid(e)               

        # Get the dimensions of the matrix array
        Nx = np.where(x == x[0])[0].size
        Ny = np.where(y == y[0])[0].size
        
        # We are assuming that the last (x, y) arrays are the same as for all the other sets. We should have a check on that. 
        return i, e, x, y
    
    #
    def make_slice(i, x, y, e=None, Nx=None, Ny=None, iflag=-1.e20):
    
        if Nx is None:
            Nx = np.where(x == x[0])[0].size
        if Ny is None:
            Ny = np.where(y == y[0])[0].size      
    
        if not isinstance(i, np.ma.MaskedArray):
            idx = np.where(i == iflag)[0]
            i[idx] = np.nan # where outputs a tuple. seems that taking the zeroth element is good.
            i = np.ma.masked_invalid(i)
    
        # Reshape the data for use with pcolormesh.
        X = x.reshape((Ny,Nx))
        Y = y.reshape((Ny,Nx))
        I = i.reshape((Ny,Nx))
        if e is not None:
            E = e.reshape((Ny,Nx))
    
        # Generate a masked array so that 'no data' is distinguisheable from 'low counts'
        if not isinstance(I, np.ma.MaskedArray):
            I = np.ma.masked_invalid(I) 
    
        if e is None:
            return I, X, Y
        else:
            return I, E, X, Y
    #    
############################### --- Data Types --- ###########################
# ... needed?
#
#----------------------------------------------------------------------------#

############################### --- Instruments --- ##########################
class Instrument(object):
    pass
#
class FacilityInstrument(Instrument):
    pass
#
class LabInstrument(Instrument):
    pass
#
class RITA2(FacilityInstrument):
    """"""
    pass
#
class MACS(FacilityInstrument):
    """"""
    pass
#
class PPMS(LabInstrument):
    pass
#
#----------------------------------------------------------------------------#
if __name__ == '__main__':
    #r = RITA2Reader.test()
    #nsc = r.get_single_scan(6758)
    #nsc.fit()
    #print nsc
    
    datroot = '/Users/Guy/data/processed/'
    datdir  = 'CeAuSb2/2015-10-1_PSI_RITA2/magnetic_field_sweep/'
    fn_root = 'gaussian_params_'
    ext = '.nc'
    # For the following load in the Dataset with all of the different kinds of 
    # traces. This is very convenient for plotting!
    gp_vHup_100mK = DataSet.read_netcdf(datroot+datdir+fn_root+'vHup_100mK'+ext)
    gp_vHup_100mK.archive('/Users/Guy/Downloads/tmp/test.nc')