from timeseries_routines import *
from astropy.io import fits
import numpy as np
import glob, os, sys, time
import matplotlib.pyplot as plt
from scipy import stats
from scipy.ndimage.measurements import center_of_mass
from scipy.interpolate import LinearNDInterpolator
from scipy.optimize import minimize_scalar
from scipy.optimize import leastsq
from scipy.optimize import least_squares
from scipy.interpolate import UnivariateSpline
from photutils import CircularAperture
from photutils import aperture_photometry
from photutils import CircularAnnulus
import batman
from tabulate import tabulate
from IPython.display import HTML
import emcee
import corner
import collections

# Miscellaneous functions
def getldcoeffs(Teff, logg, z, Tefferr, loggerr, zerr, law, channel, quiet = False):
    """Get the limb darkening coefficients and their estimated errors from the
    tables downloaded from Sing et al. by doing a 3 dimensional
    linear interpolation using scipy."""

    if not quiet: print "\nInterpolating {} limb darkening coefficients for channel {}...".format(law,channel)
    # Paths where the tables are stored
    ch1path = "/Users/cbaxter/PhD/code/LD3.6Spitzer.txt"
    ch2path = "/Users/cbaxter/PhD/code/LD4.5Spitzer.txt"

    # Read in the required table
    if channel == 'ch1':
        table = np.genfromtxt(ch1path, skip_header=13, dtype=float)
    elif channel == 'ch2':
        table = np.genfromtxt(ch2path, skip_header=13, dtype=float)

    # 3D array of discrete values of teff, logg and z
    points = np.array([table.T[0], table.T[1], table.T[2]]).T

    if law == "linear": index = [0]
    elif law == "quadratic": index = [1,2]
    elif law == "nonlinear": index = [6,7,8,9]
    else: pass

    coeffs = np.zeros(len(index))
    coeffs_err = np.zeros(len(index))

    for i in range(len(index)):
        # All possible values of desired limb darkening coefficient (indexed)
        values = table.T[3+index[i]]
        # 3D Interpolates
        interp = LinearNDInterpolator(points,values)
        coeffs[i] = interp.__call__([Teff,logg,z])

        # Estimate the error on the interpolated result based on errors Teff,logg,z
        coeffsTU = interp.__call__(np.array([Teff+Tefferr,logg,z]))
        coeffsTL = interp.__call__(np.array([Teff-Tefferr,logg,z]))
        coeffsgU = interp.__call__(np.array([Teff,logg+loggerr,z]))
        coeffsgL = interp.__call__(np.array([Teff,logg-loggerr,z]))
        coeffszU = interp.__call__(np.array([Teff,logg,z+zerr]))
        coeffszL = interp.__call__(np.array([Teff,logg,z-zerr]))

        coeffs_err[i] = np.sqrt( ((coeffsTU - coeffsTL)/2.)**2 + ((coeffsgU - coeffsgL)/2.)**2 + ((coeffszU - coeffszL)/2.)**2 )

    if not quiet: print "\tCoeff(s): {}".format(coeffs)
    if not quiet: print "\tCoeff Err(s): {}".format(coeffs_err)

    return coeffs.tolist(), coeffs_err.tolist()

# Functions for initial least squares fitting of the data
def flatten(x):
    "Fucntion to flatten a list of lists of floats with undefined shape."
    if isinstance(x, collections.Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]

def update_params(coeffs_dict, params):
    """Function to update the parameters for both batman and a normal dictionary."""
    for key in coeffs_dict.keys():
        try:
            # Batman
            params.__dict__[key] = coeffs_dict[key]
        except:
            # Normal dictionary
            params[key] = coeffs_dict[key]
    return params

def model_poly(coeffs, t, x, y, coeffs_dict, coeffs_tuple, fix_coeffs, batman_params, poly_params, components = False):
    """Make a quadratic function of order 2 with no cross terms."""
    x0, y0 = np.floor(np.mean(x))+0.5, np.floor(np.mean(y))+0.5

    # List of batman parameter names for fitting
    batman_coeffs_tuple = [key for key in coeffs_tuple[0:9] if key not in fix_coeffs]

    # dictionary of batman parameters for fitting
    batman_coeffs_dict = dict(item for item in coeffs_dict.items() if item[0] in coeffs_tuple[0:9] and item[0] not in fix_coeffs)

    if 'u' in batman_coeffs_tuple:
        # Index of the limb darkening coeff in names for fitting
        index = batman_coeffs_tuple.index("u")
        # Deal with all kinds of limb darkening
        if coeffs_dict['limb_dark'] == "uniform":
            batman_coeffs = coeffs[:len(batman_coeffs_tuple) - 1]
            batman_coeffs_dict['u'] = []
        elif coeffs_dict['limb_dark'] == "linear":
            batman_coeffs = coeffs[:len(batman_coeffs_tuple)]
            batman_coeffs_dict['u'] = [batman_coeffs[index]]
        elif coeffs_dict['limb_dark'] == "nonlinear":
            batman_coeffs = coeffs[:len(batman_coeffs_tuple) + 3]
            batman_coeffs_dict['u'] = [batman_coeffs[index],batman_coeffs[index+1],batman_coeffs[index+2],batman_coeffs[index+3]]
        elif coeffs_dict['limb_dark'] == "quadratic" or ld_opt == "squareroot" or ld_opt == "logarithmic" or ld_opt =="exponential":
            batman_coeffs = coeffs[:len(batman_coeffs_tuple) + 1]
            batman_coeffs_dict['u'] = [batman_coeffs[index],batman_coeffs[index+1]]
    else:
        batman_coeffs = coeffs[:len(batman_coeffs_tuple)]

    # Update the local dictionary
    for key in batman_coeffs_tuple:
        if key != 'u':
            batman_coeffs_dict[key] = batman_coeffs[batman_coeffs_tuple.index(key)]

    # Update the global batman dictionary
    batman_params = update_params(batman_coeffs_dict, batman_params)

    # Repeat above for all the normal polynomial parameters
    poly_coeffs_tuple = [key for key in coeffs_tuple[9:] if key not in fix_coeffs]
    poly_coeffs = coeffs[len(batman_coeffs):]
    poly_coeffs_dict = dict(item for item in coeffs_dict.items() if item[0] in coeffs_tuple[9:] and item[0] not in fix_coeffs)

    for key in poly_coeffs_tuple:
        poly_coeffs_dict[key] = poly_coeffs[poly_coeffs_tuple.index(key)]

    poly_params = update_params(poly_coeffs_dict, poly_params)

    # Create the transit model
    m = batman.TransitModel(batman_params, t)
    transit = m.light_curve(batman_params)

    # Create the polynomial model including a temporal ramp
    K1, K2, K3, K4, K5 = poly_params['K1'], poly_params['K2'], poly_params['K3'], poly_params['K4'], poly_params['K5']
    f, g, h = poly_params['f'], poly_params['g'], poly_params['h']

    F = 1. + K1*(x-x0) + K2*((x-x0)**2) + K3*(y-y0) + K4*((y-y0)**2) + K5*(x-x0)*(y-y0)

    ramp = f + g*t + h*t**2

    # Full model
    new_flux = transit * F * ramp

    if components:
        return transit, F, ramp
    else:
        return new_flux

def function_poly(coeffs, t, x, y, lc, coeffs_dict, coeffs_tuple, fix_coeffs, batman_params, poly_params):
    """Find the difference between a quadratic function and the lightcurve."""
    new_flux = model_poly(coeffs, t, x, y, coeffs_dict, coeffs_tuple, fix_coeffs, batman_params, poly_params)
    return lc - new_flux

def fit_function_poly(coeffs_dict, coeffs_tuple, fix_coeffs, t, x, y, lc):

    # Initialise ALL of the batman parameters
    batman_params = batman.TransitParams()
    batman_params.t0 = coeffs_dict['t0']                      #time of inferior conjunction
    batman_params.per = coeffs_dict['per']                #orbital period
    batman_params.rp = coeffs_dict['rp']                      #planet radius (in units of stellar radii)
    batman_params.a = coeffs_dict['a']                        #semi-major axis (in units of stellar radii)
    batman_params.inc = coeffs_dict['inc']                    #orbital inclination (in degrees)
    batman_params.ecc = coeffs_dict['ecc']                    #eccentricity
    batman_params.w = coeffs_dict['w']                        #longitude of periastron (in degrees)
    batman_params.u = coeffs_dict['u']
    batman_params.limb_dark = coeffs_dict['limb_dark']

    # Initialise the systematic polynomial model parameters
    poly_params = dict(item for item in coeffs_dict.items() if item[0] in coeffs_tuple[9:])

    # Create a list of coefficients for feaeding into scipy's least_squares
    fittable_coeffs = [ coeffs_dict[key] for key in coeffs_tuple if key not in fix_coeffs ]
    fittable_coeffs = flatten(fittable_coeffs)

    # Run the least squares
    optimum_result = scipy.optimize.least_squares(function_poly,
                                                    fittable_coeffs,
                                                    bounds = [[0]+[-np.inf]*(len(fittable_coeffs)-1), [np.inf]*(len(fittable_coeffs))],
                                                    args=(t, x, y, lc, coeffs_dict, coeffs_tuple, fix_coeffs, batman_params, poly_params))

    return optimum_result, batman_params, poly_params

def model_PLD(coeffs, t, Pns, coeffs_dict, coeffs_tuple, fix_coeffs, batman_params, PLD_params, components = False):
    """Create the PLD model for fitting based on Deming 2016."""

    # List of batman parameter names for fitting
    batman_coeffs_tuple = [key for key in coeffs_tuple[0:9] if key not in fix_coeffs]

    # dictionary of batman parameters for fitting
    batman_coeffs_dict = dict(item for item in coeffs_dict.items() if item[0] in coeffs_tuple[0:9] and item[0] not in fix_coeffs)

    if 'u' in batman_coeffs_tuple:
        # Index of the limb darkening coeff in names for fitting
        index = batman_coeffs_tuple.index("u")
        # Deal with all kinds of limb darkening
        if coeffs_dict['limb_dark'] == "uniform":
            batman_coeffs = coeffs[:len(batman_coeffs_tuple) - 1]
            batman_coeffs_dict['u'] = []
        elif coeffs_dict['limb_dark'] == "linear":
            batman_coeffs = coeffs[:len(batman_coeffs_tuple)]
            batman_coeffs_dict['u'] = [batman_coeffs[index]]
        elif coeffs_dict['limb_dark'] == "nonlinear":
            batman_coeffs = coeffs[:len(batman_coeffs_tuple) + 3]
            batman_coeffs_dict['u'] = [batman_coeffs[index],batman_coeffs[index+1],batman_coeffs[index+2],batman_coeffs[index+3]]
        elif coeffs_dict['limb_dark'] == "quadratic" or ld_opt == "squareroot" or ld_opt == "logarithmic" or ld_opt =="exponential":
            batman_coeffs = coeffs[:len(batman_coeffs_tuple) + 1]
            batman_coeffs_dict['u'] = [batman_coeffs[index],batman_coeffs[index+1]]
    else:
        batman_coeffs = coeffs[:len(batman_coeffs_tuple)]

    # Update the local dictionary
    for key in batman_coeffs_tuple:
        if key != 'u':
            batman_coeffs_dict[key] = batman_coeffs[batman_coeffs_tuple.index(key)]

    # Update the global batman dictionary
    batman_params = update_params(batman_coeffs_dict, batman_params)

    # Repeat above for all the normal PLD parameters
    PLD_coeffs_tuple = [key for key in coeffs_tuple[9:] if key not in fix_coeffs]
    PLD_coeffs = coeffs[len(batman_coeffs):]
    PLD_coeffs_dict = dict(item for item in coeffs_dict.items() if item[0] in coeffs_tuple[9:] and item[0] not in fix_coeffs)

    for key in PLD_coeffs_tuple:
        PLD_coeffs_dict[key] = PLD_coeffs[PLD_coeffs_tuple.index(key)]

    PLD_params = update_params(PLD_coeffs_dict, PLD_params)

    # Create the transit model
    m = batman.TransitModel(batman_params, t)
    DE = m.light_curve(batman_params)

    # Create the PLD model including a temporal ramp
    Cs = [PLD_params['c1'], PLD_params['c2'], PLD_params['c3'], PLD_params['c4'], PLD_params['c5'], PLD_params['c6'], PLD_params['c7'], PLD_params['c8'], PLD_params['c9']]
    g, h = PLD_params['g'], PLD_params['h']
    pixels = np.zeros( Pns.shape )
    for i in range(len(Cs)):
        pixels[i] = Cs[i]*Pns[i]
    pixels = np.sum(pixels, axis=0)

    ramp = g*t + h*t**2

    # Full model
    new_dSt = (DE + pixels + ramp)

    if components:
        return DE, pixels, ramp
    else:
        return new_dSt

def function_PLD(coeffs, t, Pns, lc, coeffs_dict, coeffs_tuple, fix_coeffs, batman_params, PLD_params):
    """Find the difference between a PLD model and the lightcurve."""
    new_flux = model_PLD(coeffs, t, Pns, coeffs_dict, coeffs_tuple, fix_coeffs, batman_params, PLD_params)
    return lc - new_flux

def fit_function_PLD(coeffs_dict, coeffs_tuple, fix_coeffs, t, timeseries, centroids, lc, boxsize=(3,3)):

    # Create the Pns = the inidividual pixel timeseries
    el1, el2 = int(np.floor(np.mean(centroids[:,1]))), int(np.floor(np.mean(centroids[:,0])))
    Parray = np.zeros((boxsize[0]*boxsize[1], len(timeseries)))
    for i in range(boxsize[0]):
        for j in range(boxsize[1]):
            Parray[boxsize[0]*i+j] = pix_timeseries(timeseries, el1+i-(boxsize[0]/2), el2+j-boxsize[1]/2)
    sumP = np.sum(Parray, axis = 0)
    Pns = Parray/sumP

    # Initialise ALL of the batman parameters
    batman_params = batman.TransitParams()
    batman_params.t0 = coeffs_dict['t0']                      #time of inferior conjunction
    batman_params.per = coeffs_dict['per']                #orbital period
    batman_params.rp = coeffs_dict['rp']                      #planet radius (in units of stellar radii)
    batman_params.a = coeffs_dict['a']                        #semi-major axis (in units of stellar radii)
    batman_params.inc = coeffs_dict['inc']                    #orbital inclination (in degrees)
    batman_params.ecc = coeffs_dict['ecc']                    #eccentricity
    batman_params.w = coeffs_dict['w']                        #longitude of periastron (in degrees)
    batman_params.u = coeffs_dict['u']
    batman_params.limb_dark = coeffs_dict['limb_dark']

    # Initialise the systematic model parameters
    PLD_params = dict(item for item in coeffs_dict.items() if item[0] in coeffs_tuple[9:])

    # Create a list of coefficients for feaeding into scipy's least_squares
    fittable_coeffs = [ coeffs_dict[key] for key in coeffs_tuple if key not in fix_coeffs ]
    fittable_coeffs = flatten(fittable_coeffs)

    # Run the least squares
    optimum_result = scipy.optimize.least_squares(function_PLD,
                                                    fittable_coeffs,
                                                    bounds = [[-np.inf]*len(fittable_coeffs), [np.inf]*len(fittable_coeffs)],
                                                    args=(t, Pns, lc, coeffs_dict, coeffs_tuple, fix_coeffs, batman_params, PLD_params))

    return optimum_result, batman_params, PLD_params, Pns

# Fit checking functions
def chi(popt, t, lc, lcerr, coeffs_dict, coeffs_tuple, fix_coeffs, batman_params, params,
        x=None, y=None, Pns=None, method = None):
    if method == 'PLD':
        mod = model_PLD(popt, t, Pns, coeffs_dict, coeffs_tuple, fix_coeffs, batman_params, params)
        chi2 = ((lc - mod)/ lcerr)**2
        return np.sum(chi2)
    elif method == 'poly':
        mod = model_poly(popt, t, x, y, coeffs_dict, coeffs_tuple, fix_coeffs, batman_params, params)
        chi2 = ((lc - mod)/ lcerr)**2
        return np.sum(chi2)

def BIC(popt, t, lc, lcerr, coeffs_dict, coeffs_tuple, fix_coeffs, batman_params, params,
        x=None, y=None, Pns=None, method = None):
    if method == 'PLD':
        chi2 = chi(popt, t, lc,lcerr, coeffs_dict, coeffs_tuple, fix_coeffs, batman_params, params,Pns=Pns, method =method)
        return chi2 - len(popt)*np.log(len(lc))
    elif method == 'poly':
        chi2 = chi(popt, t, lc,lcerr, coeffs_dict, coeffs_tuple, fix_coeffs, batman_params, params, x=x,y=y,method = method)
        return np.sum(chi2) - len(popt)*np.log(len(lc))

def gelman_rubin(chain):
    ssq = np.var(chain, axis=1, ddof=1) # var for each walker
    W = np.mean(ssq, axis=0) # average var over walkers
    thb = np.mean(chain, axis=1) # average value for each walker

    thbb = np.mean(thb, axis=0) # average value over all walkers
    m = chain.shape[0] # number of walkers
    n = chain.shape[1] # number of steps
    B = (n / (m - 1.)) * np.sum((thbb - thb)**2, axis=0) # between walker variance
    var_th = (n - 1.) / n * W + 1. / n * B  # total parameter variance?
    Rhat = np.sqrt(var_th / W)
    return Rhat

# Pipeline functions
def runFullPipeline(path, sigma_badpix, nframes_badpix,
               method_bkg, method_cent,
               ramp_time,
               sigma_clip_cent, iters_cent,
               radius_photom,
               sigma_clip_phot, iters_photom,
               size_bkg_box = None, radius_bkg_ann = None, size_bkg_ann = None, size_cent_bary = None, quiet = False,  passenger57 = False,
               plot = False, AOR = None, planet = None, channel = None, sysmethod=None):
               # Must be sigma_clip_phot because function is called sigma_clip_photom!!!
    """
    timeseries = original timeseries from the first part of the pipeline
    sigma_badpix = number of sigma away from median to clip the bad pixels
    nframes_badpix = number of frames to use as the sliding median filter for badpixel masking
    method_bkg = method for which to subtract the background
    method_cent = method for which to do the centroiding
    ramp_time = time [sec] for which to discard the first frames
    frametime = time between frames from the first part of the pipeline
    sigma_clip_cent = number of sigma away from median to clip the centroid positions
    iters_cent = number of iterations to perform the centroid sigma clipping
    radius_photom = radius over which to perform aperture photometry
    sigma_clip_photom = number of sigma away from median to clip the photometry
    iters_photom = numer of iterations to perform the photometry sigma clipping
    size_bkg_box = (if method_bkg == 'Box') size of the box to background subtract
    radius_bkg_ann = (if method_bkg == 'Annulus') radius of annulus to background subtract
    size_bkg_ann = (if method_bkg == 'Annulus') size of annulus to background subtract
    size_cent_bary = (if method_cent == 'Barycenter') size of box to use for barycenter centroiding
    """

    # Read in the data
    data_info = read_files(path)
    exptime = data_info.exptime()
    readnoise = data_info.readnoise()
    gain = data_info.gain()
    fluxconv = data_info.fluxconv()
    framtime = data_info.framtime()
    MJysr2lelectrons = exptime*gain/fluxconv

    #Create timeseries, midtimes and maskseries
    timeseries = data_info.create_timeseries()
    midtimes = data_info.midtimes()

    print "\t Exptime = {}, Readnoise = {}, Gain = {}, Fluxconv = {}, Framtime = {}".format(exptime, readnoise, gain, fluxconv, framtime)
    print "\t MJy/sr to electrons conversion factor = {}".format(MJysr2lelectrons)

    #Fix bad pixles
    timeseries = fast_bad_pix_mask(timeseries, sigma_badpix, nframes_badpix, quiet = quiet)

    #Subtract background
    timeseries, background = bck_subtract(timeseries, method = method_bkg,
                                           boxsize = size_bkg_box,
                                           radius = radius_bkg_ann,
                                           size = size_bkg_ann,
                                           quiet = quiet, plot = plot, AOR = AOR, planet = planet, channel = channel,sysmethod=sysmethod)

    #Get rid of first half hour of observations before any clipping
    timeseries, midtimes, background  = discard_ramp(timeseries, midtimes, background, ramp_time, framtime, quiet = quiet, passenger57 = passenger57)

    #Centroid barycenter
    centroids = centroid(timeseries, method=method_cent, boxsize = size_cent_bary, quiet = quiet, plot = plot, AOR = AOR, planet = planet, channel = channel,sysmethod=sysmethod)

    #Clip barycenter centroids twice
    timeseries, centroids, midtimes, background = sigma_clip_centroid(timeseries, centroids, midtimes, background, sigma_clip_cent, iters_cent, quiet = quiet, plot = plot, AOR = AOR, planet = planet, channel = channel,sysmethod=sysmethod)

    #Aperture photometry to create lightcurve
    lightcurve = aperture_photom(timeseries, centroids, radius_photom, quiet = quiet)

    #Clip photometry twice
    lightcurve, timeseries, centroids, midtimes, background = sigma_clip_photom(lightcurve, timeseries, centroids, midtimes, background, sigma_clip_phot, iters_photom, quiet = quiet, plot = plot, AOR = AOR, planet = planet, channel = channel,sysmethod=sysmethod)

    return lightcurve*MJysr2lelectrons, timeseries, centroids, midtimes, background

def runPipeline(timeseries_badpixmask, midtimes,
               method_bkg = None, method_cent = None,
               ramp_time = None, frametime = None,
               sigma_clip_cent = None, iters_cent = None,
               radius_photom = None,
               sigma_clip_phot = None, iters_photom = None,
               size_bkg_box = None, radius_bkg_ann = None, size_bkg_ann = None, size_cent_bary = None, passenger57 = False,
               quiet = False, plot = False, AOR = None, planet = None, channel = None, sysmethod = None):
               # Must be sigma_clip_phot because function is called sigma_clip_photom!!!
    """
    timeseries_badpixmask = original timeseries from the first part of the pipeline but have already
                            performed the bad pixel masking
    method_bkg = method for which to subtract the background
    method_cent = method for which to do the centroiding
    ramp_time = time [sec] for which to discard the first frames
    frametime = time between frames from the first part of the pipeline
    sigma_clip_cent = number of sigma away from median to clip the centroid positions
    iters_cent = number of iterations to perform the centroid sigma clipping
    radius_photom = radius over which to perform aperture photometry
    sigma_clip_photom = number of sigma away from median to clip the photometry
    iters_photom = numer of iterations to perform the photometry sigma clipping
    size_bkg_box = (if method_bkg == 'Box') size of the box to background subtract
    radius_bkg_ann = (if method_bkg == 'Annulus') radius of annulus to background subtract
    size_bkg_ann = (if method_bkg == 'Annulus') size of annulus to background subtract
    size_cent_bary = (if method_cent == 'Barycenter') size of box to use for barycenter centroiding
    """

    #Subtract background
    timeseries, background = bck_subtract(timeseries_badpixmask, method=method_bkg,
                                           boxsize = size_bkg_box,
                                           radius = radius_bkg_ann,
                                           size = size_bkg_ann, quiet = quiet,
                                           plot = plot, AOR = AOR, planet = planet, channel = channel,sysmethod=sysmethod)

    #Get rid of first half hour of observations before any clipping
    timeseries, midtimes, background = discard_ramp(timeseries, midtimes, background, ramp_time, frametime, quiet = quiet, passenger57 = passenger57)

    #Centroid barycenter
    centroids = centroid(timeseries, method=method_cent, boxsize = size_cent_bary, quiet = quiet,
                         plot = plot, AOR = AOR, planet = planet, channel = channel,sysmethod=sysmethod)

    #Clip barycenter centroids twice
    timeseries, centroids, midtimes, background = sigma_clip_centroid(timeseries, centroids, midtimes, background, sigma_clip_cent, iters_cent, quiet = quiet,
                                                          plot = plot, AOR = AOR, planet = planet, channel = channel,sysmethod=sysmethod)

    #Aperture photometry to create lightcurve
    lightcurve = aperture_photom(timeseries, centroids, radius_photom, quiet = quiet)

    #Clip photometry twice
    lightcurve, timeseries, centroids, midtimes, background = sigma_clip_photom(lightcurve, timeseries, centroids, midtimes, background, sigma_clip_phot, iters_photom, quiet = quiet,
                                                                    plot = plot, AOR = AOR, planet = planet, channel = channel,sysmethod=sysmethod)

    return lightcurve, timeseries, centroids, midtimes, background

def plot_lightcurve(t,  lc, lcerr, popt, coeffs_dict, coeffs_tuple, fix_coeffs, batman_params, sys_params,
                    x = None, y = None, Pns = None, errors = False, binsize = 50,
                    name = None, channel = None, orbit=None, savefile = False, TT_hjd = None,
                    method = 'PLD', color = 'r', scale = None, filext = None):
    """Function for plotting the lightcurve
    errors = Bool do or do not plot the errorbars.
    binsize = number of points to include in a bin
    """

    binsize = int(binsize)

    if name == None:
        warnings.warn( "What planetary system are we looking at!? -- setting to 'unknown'" )
        name = 'unknown'
    if channel == None:
        warnings.warn( "What channel are we looking at!? -- setting to 'unknown'" )
        channel = 'unknown'

    if method == 'poly':

        transit, F, ramp = model_poly(popt, t, x, y, coeffs_dict, coeffs_tuple, fix_coeffs, batman_params, sys_params, components = True)
        optflux = transit*F*ramp

        # Correct the lightcurve and bin the data and the optimum values
        corrected_data = lc / (F*ramp)
        start, end = 0, binsize
        binned_data, binned_opt, binned_times = [], [], []
        while end < len(corrected_data):
            binned_data.append( np.mean(corrected_data[start:end]) )
            binned_opt.append( np.mean(transit[start:end]) )
            binned_times.append((t)[start])
            start += binsize
            end += binsize

        # Calculate the residuals
        residuals = lc - optflux
        binned_residuals = []
        for i in range(len(binned_data)):
            binned_residuals.append(binned_data[i] - binned_opt[i])

        # Make the plot, 3 plots in one.
        fig = plt.figure(figsize=(15, 6))
        frame1=fig.add_axes((.1,.6,.8,.4))
        frame1.axes.get_xaxis().set_visible(False)
        frame2=fig.add_axes((.1,.2,.8,.4))
        frame2.axes.get_xaxis().set_visible(False)
        frame3=fig.add_axes((.1,.0,.8,.2))

        frame1.plot(t, lc*scale, 'ko', markersize=2, label='Raw data')
        frame1.plot(t, optflux*scale, color, label='Best fit full model', linewidth =2)
        frame1.set_title("{0} - {1} lightcurve".format(name, channel))
        frame1.set_ylabel("Raw [e-]")
        frame1.legend(loc = 'best')

        frame2.plot(binned_times, binned_data, 'ko', markersize = 4, label='Binned data (x{})'.format(binsize))
        frame2.plot(binned_times, binned_opt, color = color, label='Best fit transit model')
        frame2.set_ylabel("Corrected & Normalised")
        frame2.legend(loc = 'best')

        frame3.plot(binned_times,binned_residuals, 'ko', markersize = 4)
        frame3.axhline(0, color = color)
        frame3.set_xlabel("Time [bjd]")
        frame3.set_ylabel("Residuals")

        if savefile:
            plt.savefig("/Users/cbaxter/PhD/SpitzerTransits/{0}/{0}_{1}_{2}_lcPoly_{3}.png".format(name, orbit, channel, filext))
            plt.close()

    elif method == 'PLD':

        DE, pixels, ramp = model_PLD(popt, t, Pns, coeffs_dict, coeffs_tuple, fix_coeffs, batman_params, sys_params, components = True)

        optflux = DE + pixels + ramp

        # Correct the lightcurve and bin the data and the optimum values
        corrected_data = lc - pixels - ramp
        start, end = 0, binsize
        binned_data, binned_opt, binned_times = [], [], []
        while end < len(corrected_data):
            binned_data.append( np.mean(corrected_data[start:end]) )
            binned_opt.append( np.mean(DE[start:end]) )
            binned_times.append((t)[start])
            start += binsize
            end += binsize

        # Calculate the residuals
        residuals = lc - optflux
        binned_residuals = []
        for i in range(len(binned_data)):
            binned_residuals.append(binned_data[i] - binned_opt[i])

        # Make the plot, 3 plots in one.
        fig = plt.figure(figsize=(15, 6))
        frame1=fig.add_axes((.1,.6,.8,.4))
        frame1.axes.get_xaxis().set_visible(False)
        frame2=fig.add_axes((.1,.2,.8,.4))
        frame2.axes.get_xaxis().set_visible(False)
        frame3=fig.add_axes((.1,.0,.8,.2))

        frame1.plot(t, lc*scale, 'ko', markersize=2, label='Raw data')
        frame1.plot(t, optflux*scale, color, label='Best fit full model', linewidth =2)
        frame1.set_title("{0} - {1} lightcurve".format(name, channel))
        frame1.set_ylabel("Raw [e-]")
        frame1.legend(loc = 'best')

        frame2.plot(binned_times, binned_data, 'ko', markersize = 4, label='Binned data (x{})'.format(binsize))
        frame2.plot(binned_times, binned_opt, color = color, label='Best fit transit model')
        frame2.set_ylabel("Corrected & Normalised")
        frame2.legend(loc = 'best')

        frame3.plot(binned_times,binned_residuals, 'ko', markersize = 4)
        frame3.axhline(0, color = color)
        frame3.set_xlabel("Time [bjd]")
        frame3.set_ylabel("Residuals")

        if savefile:
            plt.savefig("/Users/cbaxter/PhD/SpitzerTransits/{0}/{0}_{1}_{2}_lcPLD_{3}.png".format(name, orbit, channel,filext))
            plt.close()

    else:
        warnings.warn( "What model do you want to plot?!" )

def inflate_errs(popt, t, lc, lcerr, coeffs_dict, coeffs_tuple,
                 fix_coeffs, batman_params, params,
                 x=None, y=None, Pns=None, method = None):
    """Function to inflate the errors so we have a reduced chi2 of 1."""

    print "\nInflating {} errors...".format(method)

    if method == 'poly':

        redChi2 = chi(popt, t, lc, lcerr, coeffs_dict, coeffs_tuple,
                      fix_coeffs, batman_params, params,
                      x=x,y=y, method = method)/(len(lc)-len(popt))

        print "\t Original Reduced Chi2: {:.2f}".format(redChi2)

        newlcerr = np.sqrt(redChi2)*lcerr
        new_redChi2 = chi(popt, t, lc, newlcerr, coeffs_dict, coeffs_tuple,
                          fix_coeffs, batman_params, params,
                          x=x,y=y, method = method)/(len(lc)-len(popt))

        print "\t New Reduced Chi2: {:.2f}".format(new_redChi2)

    elif method == 'PLD':

        redChi2 = chi(popt, t, lc, lcerr, coeffs_dict, coeffs_tuple,
                      fix_coeffs, batman_params, params,
                      Pns=Pns, method = method)/(len(lc)-len(popt))

        print "\t Original Reduced Chi2: {:.2f}".format(redChi2)

        newlcerr = np.sqrt(redChi2)*lcerr
        new_redChi2 = chi(popt, t, lc, newlcerr, coeffs_dict, coeffs_tuple,
                          fix_coeffs, batman_params, params,
                          Pns=Pns, method = method)/(len(lc)-len(popt))

        print "\t New Reduced Chi2: {:.2f}".format(new_redChi2)

    return newlcerr

def fold_inc(sampler, chaintype, labels):
    "Function to fold over the inclination."

    print "\t Folding over the inclination."

    index = labels.index("inc")

    if chaintype == 'chain':
        samples = sampler.chain

        for j in range(samples.shape[0]):
            for i in range(samples.shape[1]):
                if samples.T[index][i][j] > 90.:
                    samples.T[index][i][j] = 180. - samples.T[index][i][j]

    elif chaintype == 'flatchain':
        samples = sampler.flatchain

        for i in range(samples.shape[0]):
            if samples.T[index][i] > 90.:
                samples.T[index][i] = 180. - samples.T[3][i]

    return samples

def mcmc_results(sampler, popt, t, lc, lcerr, x, y, Pns, background, coeffs_dict, coeffs_tuple, fix_coeffs, batman_params, params,scale, labels, planet, AOR, channel, method, saveplots = True):

    print "\nMCMC {} results...".format(method)
    samples = fold_inc(sampler, "chain", labels)
    samples2 = fold_inc(sampler, "flatchain", labels)

    if saveplots:

        # Blue for polynomial model, red for PLD
        if method == 'poly': c = 'b'
        elif method == 'PLD': c = 'r'

        ################################# Pns Plot #################################################

        if method == 'PLD':
            fig, ax = plt.subplots(figsize=(15,5))
            for i in range(len(Pns)):
                ax.plot(midtimes,Pns[i], alpha=0.7, label = 'P{}'.format(i+1))

            ax.set_xlabel('Time')
            ax.set_ylabel('Normalised Pixel Flux')
            ax.legend()
            plt.savefig("/Users/cbaxter/PhD/SpitzerTransits/{0}/{0}_{1}_{2}_{3}_Pns.png".format(planet, AOR, channel, method))
            plt.close()
        else:
            pass

        ################################# Gelmin Rubin Diagnostics #################################

        print "\t Saving {} results and Gelman-Rubin to file...".format(method)

        f = open("/Users/cbaxter/PhD/SpitzerTransits/{0}/{0}_{1}_{2}_{3}_GelminRubin.txt".format(planet, AOR, channel,method), 'w')

        print >> f, ("\tParam\tlsq_mu\t\tmu\t\tsigma\tGelman-Rubin")
        print >> f, ("\t=============================================================")
        for var, lsq, chain in list(zip(labels, popt, ([samples[:,:,i:i+1] for i in range(samples.shape[2])]))):
            print >> f, ("\t{0:}\t{1: 1.3e}\t{2: 1.3e}\t{3:1.3e}\t{4:1.4g}".format(
                var,
                lsq,
                chain.reshape(-1, chain.shape[-1]).mean(axis=0)[0],
                chain.reshape(-1, chain.shape[-1]).std(axis=0)[0],
                gelman_rubin(chain)[0]))

        f.close()

        ################################# Walkers Plots ############################################

        print "\t Plotting walkers and saving to file..."
        fig, ax = plt.subplots(len(labels), 1, figsize=(10, 5*len(labels)))
        for d in range(len(labels)):
            for w in range(100):
                ax[d].plot(samples[w,:,d], c=c, alpha=0.1)
            ax[d].set_xlabel(r'${\rm Steps}$')
            ax[d].set_ylabel(labels[d])

        plt.savefig("/Users/cbaxter/PhD/SpitzerTransits/{0}/{0}_{1}_{2}_{3}_walkers.png".format(planet, AOR, channel,method))
        plt.close()

        ################################# Corner Plot ##############################################

        print "\t Plotting corner plot and saving to file..."
        fig = corner.corner(samples2,
                               quantiles=[0.16, 0.5, 0.84],
                               labels = labels,
                               show_titles=True, title_kwargs={"fontsize": 12}, color =c)

        plt.savefig("/Users/cbaxter/PhD/SpitzerTransits/{0}/{0}_{1}_{2}_{3}_corner.png".format(planet, AOR, channel, method))
        plt.close()

        ################################# Lightcurve Plot ###########################################
        print "\t Plotting lightcurve and saving to file..."
        if method == 'poly': c = 'b'
        elif method == 'PLD': c = 'r'
        plot_lightcurve(t,  lc, lcerr, popt, coeffs_dict, coeffs_tuple, fix_coeffs, batman_params, params,
                    x=x,y=y,Pns = Pns, errors = False, binsize = 50,
                    name = planet, channel = channel, orbit=AOR, savefile = True, TT_hjd = None,
                    method = method, color = c, scale = scale, filext = "mcmc")

        ################################# RMS vs Binsize Plot #######################################

        print "\t Plotting rms vs binsize and saving to file..."
        std_devs, binsizes = [],[]

        for i in range(1,1000):
            binsize = i
            start, end = 0, binsize
            binned_lc, binned_lcerr, binned_t, binned_transit = [], [], [], []
            while end < len(lc):
                binned_lc.append( np.mean(lc[start:end]) )
                binned_t.append(t[start + binsize/2])
                binned_lcerr.append(np.sqrt(np.sum(np.sqrt(lc)[start:end]**2))/binsize)
                if method == 'poly':
                    binned_transit.append( np.mean(model_poly(popt, t, x, y, coeffs_dict, coeffs_tuple, fix_coeffs, batman_params, params)[start:end]) )
                elif method == 'PLD':
                    binned_transit.append( np.mean(model_PLD(popt, t, Pns, coeffs_dict, coeffs_tuple, fix_coeffs, batman_params, params)[start:end]) )
                start += binsize
                end += binsize

            std_devs.append(np.std(np.array(binned_lc) - np.array(binned_transit)))
            binsizes.append(binsize)

        fig = plt.figure(figsize=(10, 5))
        frame1=fig.add_axes((.1,.3,.8,.6))
        frame1.set_title('Real data - frame SNR: {:.0f}'.format(np.sqrt(scale)))
        frame2=fig.add_axes((.1,.1,.8,.2))

        frame1.plot(np.array(binsizes),np.array(std_devs),'k',label="RMS scatter" )
        frame1.plot(np.array(binsizes),std_devs[0]/(np.sqrt(np.array(binsizes))), c , label="Theoretical N^-0.5")
        #frame1.xlabel("binsize")
        frame1.set_ylabel("stddev")
        frame1.set_yscale('log')
        frame1.set_xscale('log')
        frame1.xaxis.set_visible(False)
        frame1.legend(loc='best')

        frame2.plot(np.array(binsizes), np.array(std_devs) -  std_devs[0]/(np.sqrt(np.array(binsizes))) , 'ko', markersize=1)
        frame2.axhline(0., color=c, linewidth=2)
        frame2.set_xscale('log')
        plt.savefig("/Users/cbaxter/PhD/SpitzerTransits/{0}/{0}_{1}_{2}_{3}_rmsVSbinsize.png".format(planet, AOR, channel, method))
        plt.close()

        ################################# Background vs residuals Plot #############################

        print "\t Plotting background vs residuals and saving to file..."
        if method == 'poly':
            residuals = function_poly(popt, t, x, y, lc, coeffs_dict, coeffs_tuple, fix_coeffs, batman_params, params)
        elif method == 'PLD':
            residuals = function_PLD(popt, t, Pns, lc, coeffs_dict, coeffs_tuple, fix_coeffs, batman_params, params)

        plt.plot(residuals, background, 'd', markersize = 4, color = c)
        plt.xlabel("Residuals")
        plt.ylabel("Background Flux (image units)")
        plt.savefig("/Users/cbaxter/PhD/SpitzerTransits/{0}/{0}_{1}_{2}_{3}_bkgVSresiduals.png".format(planet, AOR, channel, method))
        plt.close()

        ##################################### Flux Vs x&y plot #####################################

        print "\t Plotting Flux vs xy and saving to file..."
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        ax[0].plot(x, lc, 'o', color = c, markersize = 2, alpha = 0.5)
        ax[0].set_title('x-pos vs Flux')
        ax[1].plot(y, lc, 'o', color = c , markersize = 2, alpha = 0.5)
        ax[1].set_title('y-pos vs Flux')
        plt.savefig("/Users/cbaxter/PhD/SpitzerTransits/{0}/{0}_{1}_{2}_{3}_fluxVSxy.png".format(planet, AOR, channel, method))
        plt.close()


    # Average parameter values
    avgs = np.mean(samples2,axis=0)
    stds = np.std(samples2,axis=0)

    return avgs, stds

# Functions for MCMC parameter exploration of PLD

def lnprob_PLD(theta, t, Pns, lc, lcerrs, bounds,
               coeffs_dict, coeffs_tuple, fix_coeffs, batman_params, PLD_params):

    lp = lnprior_PLD(theta, bounds, batman_params)

    if not np.isfinite(lp): return -np.inf

    return lp + lnlike_PLD(theta, t, Pns, lc, lcerrs, coeffs_dict, coeffs_tuple, fix_coeffs, batman_params, PLD_params)

def lnlike_PLD(theta, t, Pns, lc, lcerrs,
               coeffs_dict, coeffs_tuple, fix_coeffs, batman_params, PLD_params):

    mod = model_PLD(theta,  t, Pns, coeffs_dict, coeffs_tuple, fix_coeffs, batman_params, PLD_params)

    return -0.5*np.sum((((lc-mod)/lcerrs)**2) + np.log(2.*np.pi*(lcerrs**2)))

def lnprior_PLD(theta, bounds, batman_params):

    for i in range(len(theta)):
        if not bounds[0][i] < theta[i] < bounds[1][i]:
            return -np.inf

    e = batman_params.__dict__['ecc']
    a = batman_params.__dict__['a']
    inc = batman_params.__dict__['inc']
    if not inc < 180.: return -np.inf
    omega = batman_params.__dict__['w']

    b = ((1 - e**2)/(1 + e * np.sin(omega*np.pi/180.))) * a * np.cos(inc*np.pi/180.)
    if not b <= 1.:
        return -np.inf

    return 0.0

def mcmc_PLD(initial, data, nwalkers = 100, burnin_steps = 1000, production_steps = 2000, plot = False):

    print "\nStarting MCMC..."

    ndim = len(initial)
    p0 = [np.array(initial) + 1e-8 * np.random.randn(ndim)
            for i in xrange(nwalkers)]

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_PLD, args=(data))

    print("\t Running burn-in: {} steps".format(burnin_steps))
    p0, lnp, _ = sampler.run_mcmc(p0, burnin_steps)
    sampler.reset()

    print("\t Running production: {} steps".format(production_steps))
    p0, _, _ = sampler.run_mcmc(p0, production_steps)

    print("\t Mean acceptance fraction: {0:.3f}"
           .format(np.mean(sampler.acceptance_fraction)))

    return sampler

# Functions for MCMC parameter exploration of polynomial
def lnprob_poly(theta, t, x,y, lc, lcerrs, bounds,
               coeffs_dict, coeffs_tuple, fix_coeffs, batman_params, poly_params):

    lp = lnprior_poly(theta, bounds, batman_params)

    if not np.isfinite(lp): return -np.inf

    return lp + lnlike_poly(theta, t, x,y, lc, lcerrs, coeffs_dict, coeffs_tuple, fix_coeffs, batman_params, poly_params)

def lnlike_poly(theta, t, x,y, lc, lcerrs,
               coeffs_dict, coeffs_tuple, fix_coeffs, batman_params, poly_params):

    mod = model_poly(theta,  t, x,y, coeffs_dict, coeffs_tuple, fix_coeffs, batman_params, poly_params)

    return -0.5*np.sum((((lc-mod)/lcerrs)**2) + np.log(2.*np.pi*(lcerrs**2)))

def lnprior_poly(theta, bounds, batman_params):

    for i in range(len(theta)):
        if not bounds[0][i] < theta[i] < bounds[1][i]:
            return -np.inf

    e = batman_params.__dict__['ecc']
    a = batman_params.__dict__['a']
    inc = batman_params.__dict__['inc']
    if not inc < 180.: return -np.inf
    omega = batman_params.__dict__['w']

    b = ((1 - e**2)/(1 + e * np.sin(omega*np.pi/180.))) * a * np.cos(inc*np.pi/180.)
    if not b <= 1.:
        return -np.inf

    return 0.0

def mcmc_poly(initial, data, nwalkers = 100, burnin_steps = 1000, production_steps = 2000, plot = False):

    print "\nStarting MCMC..."

    ndim = len(initial)
    p0 = [np.array(initial) + 1e-8 * np.random.randn(ndim)
            for i in xrange(nwalkers)]

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_poly, args=(data))

    print("\t Running burn-in: {} steps".format(burnin_steps))
    p0, lnp, _ = sampler.run_mcmc(p0, burnin_steps)
    sampler.reset()

    print("\t Running production: {} steps".format(production_steps))
    p0, _, _ = sampler.run_mcmc(p0, production_steps)

    print("\t Mean acceptance fraction: {0:.3f}"
           .format(np.mean(sampler.acceptance_fraction)))

    return sampler
