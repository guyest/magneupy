import warnings
#import pint
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import sys, tabulate, pandas, xarray, h5py, os, lmfit
import numpy as np
import scipy.signal as sig
import uncertainties as unc
from uncertainties import unumpy as unp
from numpy.matlib import repmat
from collections import namedtuple, Iterable, OrderedDict
from copy import deepcopy, copy
from lmfit import Model
import pickle as pkl
import time


class Data(object):
    pass


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

    def load(self,filepath,**kwargs):
        """
        Uses the class method 'read_netcdf' to load in a cdf
        """
        self = self.read_netcdf(filepath,**kwargs)
        return

    def get_labels(self,labels={},from_array=False):
        labels['suptitle'] = self.suptitle
        return labels

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
    def __init__(self, value, value_error=None, value_label=None, coords=None, cerrors=None, clabels=[],
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

    def check_units(self, every=True, **kwargs):
        """
        This method checks the units for the DataSlice. It is just for
        consistency, e.g. the 'values' should all be in the same units.
        """
        return

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
    
    def __new__(_cls, name, ticks, unit, isIndependent, attrs=None, axislabel=None):
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

    @classmethod
    def _make(cls, iterable, new=tuple.__new__, len=len):
        'Make a new DataLabel object from a sequence or iterable'
        result = new(cls, iterable)
        if len(result) != 3:
            raise TypeError('Expected 3 arguments, got %d' % len(result))
        return result

    def toCoordinates(self):
        """
        This instance method takes a DataLabel returns a Coordinates object. 
        """
        return Coordinates([self])

    def __repr__(self):
        'Return a nicely formatted representation string'
        return 'DataLabel(name=%r, ticks=%r, attrs=%r)' % self

    def _asdict(self):
        'Return a new OrderedDict which maps field names to their values'
        return OrderedDict(list(zip(self._fields, self)))

    def _replace(_self, **kwds):
        'Return a new DataLabel object replacing specified fields with new values'
        result = _self._make(list(map(kwds.pop, ('name', 'ticks', 'attrs'), _self)))
        if kwds:
            raise ValueError('Got unexpected field names: %r' % list(kwds.keys()))
        return result

    def __getnewargs__(self):
        'Return self as a plain tuple.  Used by copy and pickle.'
        return tuple(self)

    def __getstate__(self):
        'Exclude the OrderedDict from pickling'
        pass


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

    @property
    def names(self):
        """
        Returns the names of the consitituent DataLabels.
        """
        return list(self.keys())

    @property
    def list(self):
        """
        Return list of DataLabel tuples for input to DataSlice.
        """
        return list(self.values())

    @property
    def tick_dict(self):
        """
        Return input sensible when extra coordinates are provided.
        """
        res={}
        for key,value in self.items():
            res[key] = value.ticks
        return res

    @property
    def units(self):
        """
        
        """
        res={}
        for key,value in self.items():
            res[key] = value.attrs['unit']        
        return


class Unit(object):
    """
    This class provides a way to assign a unit attribute to (mainly) 
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

    def __repr__(self):
        """
        String representation
        """
        return self.name


class ArbUnits(Unit):
    """
    The ArbUnits class is a subclass to Unit to handle cases when arbitrary 
    units are required because no true units can be assigned. In those cases,
    we must ensure that any quantities being combined have the SAME 
    (as evaluated by 'is' comparison) even though they are arbitrary.
    """
    pass


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


class Suceptibility(ExperimentalDataSlice):
    """"""
    pass


class Resistivity(ExperimentalDataSlice):
    pass


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

    def gaussian(self, qx, qy, F, Qx, Qy, sigmax=0.05, sigmay=0.05):
        """"""
        return F * 1./(sigmax*np.sqrt(2.*np.pi)) * np.exp(-(qx-Qx)**2./(2.*sigmax**2.)) * 1./(sigmay*np.sqrt(2.*np.pi)) * np.exp(-(qy-Qy)**2./(2.*sigmay**2.))

    def plotStructureFactor(self, step=0.01, vmin=1e-12, vmax=None, return_only=False,
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


class StructureFactor(Data):
    """
    TODO:
    <done> The standard is for the squared structure factor to be input at this point.
    """
    def gaussian(self, qx, qy, F, Qx, Qy, sigmax=0.02, sigmay=0.02):
        """"""
        return F * 1./(sigmax*np.sqrt(2.*np.pi)) * np.exp(-(qx-Qx)**2./(2.*sigmax**2.)) * 1./(sigmay*np.sqrt(2.*np.pi)) * np.exp(-(qy-Qy)**2./(2.*sigmay**2.))

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

    def __add__(self, other):
        """"""
        return self.values


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


class Spectroscopy(ExperimentalData):
    """"""
    pass


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


class NeutronScanSet(ExperimentalDataSet):
    """
    This class consists of set of different 
    """
    pass


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
    def plot(self, dims=('h',), with_fit=(True,), gfxpath=None, show=False, save=True, **kwargs):
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

    def fit(self, show=True, save=False, append=True, dims=None, ion=False, gfxpath=None, **kwargs):
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

    @property
    def redchi(self):
        """
        String for fit report
        """
        return self.gfit.mini_result.redchi

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


class NeutronReactor(NeutronSpectroscopy):
    """"""
    pass


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