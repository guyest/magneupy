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
from scipy.interpolate import griddata
from scipy import constants
from copy import deepcopy, copy
from lmfit import Model
import pickle as pkl
import time

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
            if str(datum['entry1/user/name'].value[0]) == 'Guy' and \
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
                title='Scan of ('+str(Q['h'].mean())+','+str(Q['k'].mean())+ \
                      ','+str(Q['l'].mean())+')\n in '+mf_str+' along c at '+temp_str

                meta = ''
                #meta+='\n\n'
                meta+= r'$\mathbf{Start Time}$: ' \
                       +datum['entry1/start_time'].value[0]+'\t\t'
                meta+= r'$\mathbf{End Time}$: ' \
                       +datum['entry1/end_time'].value[0]+'\n'
                meta+= r'$\mathbf{Scan Command}$: '+scan_command+'\n'
                meta+= r'$\mathbf{Temperature}$: '+temp_str+ \
                       ' ('+temp_sweep_dir+')'+'\t\t'
                meta+= r'$\mathbf{Magnetic Field}$: '+mf_str+ \
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
                    if str(dathdf['entry1/user/name'].value[0]) == 'Guy' and \
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
