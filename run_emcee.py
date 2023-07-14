import os
os.environ['OMP_NUM_THREADS'] = '1'
import stsynphot as stsyn
import matplotlib.pyplot as plt
import numpy as np
from astropy.io.fits import getheader
from astropy.io import fits
import numpy as np
import astropy.units as u
from matplotlib import pyplot as plt
from astropy.visualization import quantity_support
from astropy.io.fits import getheader
from specutils import Spectrum1D
from glob import glob 
#from astropy.visualization import quantity_support
#quantity_support()  # for getting units on the axes below  
from specutils.manipulation import box_smooth
from specutils.manipulation import LinearInterpolatedResampler
import emcee
import corner
from multiprocessing import Pool

#grab anything that contains '.fits'
files = glob('*.fits')
print(files)
# hdr = getheader(files[1])
# print(hdr)
print(files[1])

def lowres(t_eff, log_g, rv, ebv, radius, wavelength, i):
    """
    Takes effective temperature, surface gravity, radial velocity, reddening, and radius (in solar radii)
    and an array of wavelength points and 
    outputs corresponding model flux evaluated on the wavelength array
    """

    model = stsyn.grid_to_spec('ck04models', t_eff, -0.25, log_g) #model from Castelli & Kurucz (2004)
    model.z = rv/(299792458/1000) #speed of light converted from m/s to km/s
    
    reddening = stsyn.spectrum.ebmvx('lmcavg', ebv) #average of total/selective dimming
    
    
    reddenedmodel = reddening*model
    
    reddenedmodel_flux = reddenedmodel(wavelength, flux_unit='flam')
    
    reddenedmodel_flux*=(radius/(2.15134e12))**2

    
    
    
    hdulist = fits.open(str(files[i]))
    hdr = getheader(files[i])
    spec1d = Spectrum1D.read(files[i])
    filename = files[i]
    # The spectrum is in the first HDU of this file.
    with fits.open(filename) as f:  
        specdata = f[0].data
    text = colored('----------------\nThis graph is: ' + files[i] + ' and array ' + str(i), 'red', attrs=['bold'])  
    print(text)  
    print(specdata)
    
    print(spec1d)
    
    #lowres
    spec_smooth = box_smooth(spec1d, width=1500)
    wavelength_grid = np.arange(spec1d.spectral_axis.value.min(), spec1d.spectral_axis.value.max(), 20) * u.AA
    linear = LinearInterpolatedResampler()
    spec_interp = linear(spec_smooth, wavelength_grid)
    
    
    ax = plt.subplots()[1] 
    ax.plot(spec_interp.spectral_axis.value, spec_interp.flux.value)
    ax.set_xlim(3000,9700)
    #ax.set_ylim(0,2e-13)
    ax.set_ylabel('Flux (erg/cm^2/s/Å)')
    ax.set_xlabel('Wavelength (Å)')
    plt.plot(wavelength, reddenedmodel_flux)
    plt.show
    diff=np.diff(spec1d.spectral_axis.value) #find resolution diff
    print(diff)
    
    
def logprior(theta):
    t_eff, log_g, rv, ebv, radius = theta #theta is an array containing parameter values, which we unpack into individual variables
    if (4000 <= t_eff <= 12500) & (-.5<=log_g<=3) & (-50<=rv<=50) & (0<=ebv<=2.5) & (1<=radius<=1000): 
        try: #this step makes sure the parameter combination loads the correct model
            model = stsyn.grid_to_spec('ck04models', t_eff, -0.25, log_g)
            return model
        except: #if this breaks stsynphot...
            return -np.inf #if there's no model here, this is a bad parameter set! so return negative infinity!!
    else: #if we're also outside of the parameter bounds...
        return -np.inf #it's bad! So we return negative infinity!!
    
    
def logprob(theta, wavelength, flux):
    lprior=logprior(theta)
    if lprior==-np.inf:
        return -np.inf
    t_eff, log_g, rv, ebv, radius = theta
    model=lprior

    model.z = rv/(299792458/1000) #speed of light converted from m/s to km/s

    reddening = stsyn.spectrum.ebmvx('lmcavg', ebv) #average of total/selective dimming

    
    reddenedmodel = reddening*model

    reddenedmodel_flux = reddenedmodel(wavelength, flux_unit='flam')
    
    reddenedmodel_flux*=(radius/(2.15134e12))**2


    
    
    error=flux/200

    
    chi=((flux-reddenedmodel_flux.value)**2)/(error**2)
    return chi.sum() * -0.5

def choosefile(i):
    hdulist = fits.open(str(files[i]))
    hdr = getheader(files[i])
    spec1d = Spectrum1D.read(files[i])
    filename = files[i]
# The spectrum is in the first HDU of this file.
    with fits.open(filename) as f:  
        specdata = f[0].data
    
    #lowres
    spec_smooth = box_smooth(spec1d, width=1500)
    wavelength_grid = np.arange(spec1d.spectral_axis.value.min(), spec1d.spectral_axis.value.max(), 20) * u.AA
    linear = LinearInterpolatedResampler()
    spec_interp = linear(spec_smooth, wavelength_grid)
    
    wavelength = spec_interp.spectral_axis.value
    flux = spec_interp.flux.value
    fitsname=files[i]
    
    
    return wavelength, flux, fitsname


def mcmc(theta)
    for i in range(39):
        wavelength, flux, fitsname = choosefile(i)
        pos = theta + 1e-4 * np.random.randn(32, 5) #this initializes our walkers in a small little random ball around your starting guess
 
        nwalkers, ndim = pos.shape
 
        filename = fitsname + '_emcee.h5' 
        backend = emcee.backends.HDFBackend(filename)
 
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(
                nwalkers, ndim, logprob, args=(wavelength, flux), backend=backend, pool=pool
            )
        sampler.run_mcmc(pos, 5000, progress=True);


if __name__ == '__main__':
    theta=[6549.725, 1.619, 20, 5e-4, 264]
    mcmc(theta)
    
    
    