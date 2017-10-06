from timeseries_routines import *
from astropy.io import fits
import numpy as np
import glob, os, sys, time
import matplotlib.pyplot as plt
from scipy import stats
from scipy.ndimage.measurements import center_of_mass
from scipy.interpolate import LinearNDInterpolator
from scipy.optimize import minimize_scalar, leastsq, least_squares
from scipy.interpolate import UnivariateSpline
from photutils import CircularAperture, aperture_photometry, CircularAperture
import batman
from tabulate import tabulate
from IPython.display import HTML
import emcee, corner, collections, warnings
from matplotlib import gridspec

# Miscellaneous functions
def getldcoeffs(Teff, logg, z, Tefferr, loggerr, zerr, law, channel, quiet = False):
    """Get the limb darkening coefficients and their estimated errors from the
    tables downloaded from Sing et al. by doing a 3 dimensional
    linear interpolation using scipy."""

    if not quiet: print "\nInterpolating {} limb darkening coefficients for {}...".format(law,channel)
    # Paths where the tables are stored
    ch1path = "{}/PhD/code/LD3.6Spitzer.txt".format(os.getenv('HOME'))
    ch2path = "{}/PhD/code/LD4.5Spitzer.txt".format(os.getenv('HOME'))

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

    if not quiet: print "\t Coeff(s): {}".format(coeffs)
    if not quiet: print "\t Coeff Err(s): {}".format(coeffs_err)

    return coeffs.tolist(), coeffs_err.tolist()

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

def flatten(x):
    "Fucntion to flatten a list of lists of floats with undefined shape."
    if isinstance(x, collections.Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]

# Functions for initial least squares fitting of the data
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

def make_bounds(coeffs_tuple, fix_coeffs, t=None, fix_coeffs_channels = None, normal = True,
            gaussian_priors = False, prior_params = None, coeffs_dict = None):
    """Function to make a bounds list for the LSQ and the mcmc"""

    if normal:
        if fix_coeffs_channels != None:
            fittable_coeffs_labels = [ key for key in coeffs_tuple if key not in fix_coeffs and key not in fix_coeffs_channels]
        else:
            fittable_coeffs_labels = [ key for key in coeffs_tuple if key not in fix_coeffs]

        bounds = [[-np.inf]*(len(fittable_coeffs_labels)), [np.inf]*(len(fittable_coeffs_labels))]

        if 't0' in fittable_coeffs_labels:
            ind = np.where(np.array(fittable_coeffs_labels) == 't0')
            ind = ind[0][0]
            bounds[0][ind], bounds[1][ind] = t[0], t[-1]
        if 'rp' in fittable_coeffs_labels:
            ind = np.where(np.array(fittable_coeffs_labels) == 'rp')
            ind = ind[0][0]
            bounds[0][ind], bounds[1][ind] = 0., 1.
        if 'a' in fittable_coeffs_labels:
            ind = np.where(np.array(fittable_coeffs_labels) == 'a')
            ind = ind[0][0]
            bounds[0][ind]= 0.
        if 'inc' in fittable_coeffs_labels:
            ind = np.where(np.array(fittable_coeffs_labels) == 'inc')
            ind = ind[0][0]
            bounds[0][ind], bounds[1][ind] = 0., 180.
        if 'ecc' in fittable_coeffs_labels:
            ind = np.where(np.array(fittable_coeffs_labels) == 'ecc')
            ind = ind[0][0]
            bounds[0][ind], bounds[1][ind] = 0., 1.
        if 'w' in fittable_coeffs_labels:
            ind = np.where(np.array(fittable_coeffs_labels) == 'w')
            ind = ind[0][0]
            bounds[0][ind], bounds[1][ind] = 0., 360.

        if gaussian_priors:
            for param in prior_params:
                ind = np.where(np.array(fittable_coeffs_labels) == param)[0][0]
                bounds[0][ind] = coeffs_dict[param] - coeffs_dict['{}_err'.format(param)]
                bounds[1][ind] = coeffs_dict[param] + coeffs_dict['{}_err'.format(param)]

        return bounds

    else:
        fittable_coeffs_labels = [ key for key in coeffs_tuple if key in fix_coeffs_channels]

        bounds = [[-np.inf]*(len(fittable_coeffs_labels)), [np.inf]*(len(fittable_coeffs_labels))]

        if 't0' in fittable_coeffs_labels:
            ind = np.where(np.array(fittable_coeffs_labels) == 't0')
            ind = ind[0][0]
            bounds[0][ind], bounds[1][ind] = t[0], t[-1]
        if 'rp' in fittable_coeffs_labels:
            ind = np.where(np.array(fittable_coeffs_labels) == 'rp')
            ind = ind[0][0]
            bounds[0][ind], bounds[1][ind] = 0., 1.
        if 'a' in fittable_coeffs_labels:
            ind = np.where(np.array(fittable_coeffs_labels) == 'a')
            ind = ind[0][0]
            bounds[0][ind]= 0.
        if 'inc' in fittable_coeffs_labels:
            ind = np.where(np.array(fittable_coeffs_labels) == 'inc')
            ind = ind[0][0]
            bounds[0][ind], bounds[1][ind] = 0., 180.
        if 'ecc' in fittable_coeffs_labels:
            ind = np.where(np.array(fittable_coeffs_labels) == 'ecc')
            ind = ind[0][0]
            bounds[0][ind], bounds[1][ind] = 0., 1.
        if 'w' in fittable_coeffs_labels:
            ind = np.where(np.array(fittable_coeffs_labels) == 'w')
            ind = ind[0][0]
            bounds[0][ind], bounds[1][ind] = 0., 180.

        if gaussian_priors:
            for param in prior_params:
                ind = np.where(np.array(fittable_coeffs_labels) == param)[0][0]
                bounds[0][ind] = coeffs_dict[param] - coeffs_dict['{}_err'.format(param)]
                bounds[1][ind] = coeffs_dict[param] + coeffs_dict['{}_err'.format(param)]

        return bounds

# Polynomial fitting functions
def model_poly(coeffs, t, x, y, coeffs_dict, coeffs_tuple, fix_coeffs, batman_params, poly_params, components = False):
    """Make a quadratic function of order 2 with cross terms."""
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

def fit_function_poly(coeffs_dict, coeffs_tuple, fix_coeffs, t, x, y, lc, gaussian_priors = False, prior_params = None):

    # Initialise ALL of the batman parameters
    batman_params = batman.TransitParams()
    batman_params.t0 = coeffs_dict['t0']                      #time of inferior conjunction
    batman_params.per = coeffs_dict['per']                    #orbital period
    batman_params.rp = coeffs_dict['rp']                      #planet radius (in units of stellar radii)
    batman_params.a = coeffs_dict['a']                        #semi-major axis (in units of stellar radii)
    batman_params.inc = coeffs_dict['inc']                    #orbital inclination (in degrees)
    batman_params.ecc = coeffs_dict['ecc']                    #eccentricity
    batman_params.w = coeffs_dict['w']                        #longitude of periastron (in degrees)
    batman_params.u = coeffs_dict['u']
    batman_params.limb_dark = coeffs_dict['limb_dark']

    # Initialise the systematic polynomial model parameters
    poly_params = dict(item for item in coeffs_dict.items() if item[0] in coeffs_tuple[9:])

    # Create a list of coefficients for feeding into scipy's least_squares
    fittable_coeffs = [ float(coeffs_dict[key]) for key in coeffs_tuple if key not in fix_coeffs]
    fittable_coeffs_labels = [ key for key in coeffs_tuple if key not in fix_coeffs]
    fittable_coeffs = flatten(fittable_coeffs)

    bounds = make_bounds(coeffs_tuple, fix_coeffs, t, gaussian_priors = gaussian_priors, prior_params = prior_params, coeffs_dict = coeffs_dict)

    # Run the least squares
    optimum_result = scipy.optimize.least_squares(function_poly,
                                                    fittable_coeffs,
                                                    bounds = bounds,
                                                    args=(t, x, y, lc, coeffs_dict, coeffs_tuple, fix_coeffs, batman_params, poly_params))

    return optimum_result, batman_params, poly_params

def prayer_bead_poly(coeffs, t, x, y, lc, coeffs_dict, coeffs_tuple, fix_coeffs, batman_params, poly_params,
                        nskip, planet, AOR, channel, method, foldext='', plot = True):

    # Create list of parameters that are being fit
    labels = [ key for key in coeffs_tuple if key not in fix_coeffs ]

    # Create a master model based on the least squares
    master_model = model_poly(coeffs, t, x, y, coeffs_dict, coeffs_tuple, fix_coeffs, batman_params, poly_params)

    # And a master residuals which will be permuated
    master_residuals = function_poly(coeffs, t, x, y, lc, coeffs_dict, coeffs_tuple, fix_coeffs, batman_params, poly_params)

    prayer_bead = np.ones((len(master_residuals)/nskip,len(coeffs)))

    # Loop over the number of iterations required
    for j in range(len(master_residuals)/nskip):

        residuals = np.zeros(len(master_residuals))

        # Loop over all the datapoints in the residuals and replace by the shifted point
        for i in range(len(master_residuals)):

            # If we have reached the end of the array loop back around to the beginning
            if (i+j*nskip+1) > len(master_residuals):
                residuals[i] = master_residuals[j*nskip + i - len(master_residuals)]
            else:
                residuals[i] = master_residuals[j*nskip + i]

        # Create new simulated light curve - model + residuals
        sim_lc = residuals + master_model

        # Refit the simualted light curve
        optimum_result_sim, batman_params, poly_params = fit_function_poly(coeffs_dict, coeffs_tuple, fix_coeffs, t, x, y, sim_lc)

        popt_sim = optimum_result_sim.x

        # Save the optimum parameters to an array
        prayer_bead[j] = popt_sim


    # If desired plot the corner plot
    if plot:
        fig = corner.corner(prayer_bead,
                               quantiles=[0.16, 0.5, 0.84],
                               labels = labels,
                               show_titles=True, title_kwargs={"fontsize": 12}, color = 'b')

        plt.savefig("{5}/PhD/SpitzerTransits/{0}{4}/{0}_{1}_{2}_{3}_prayerBead_corner.png".format(planet, AOR, channel, method,foldext,os.getenv('HOME')))
        plt.close()

    return prayer_bead

# PLD fitting functions
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

def fit_function_PLD(coeffs_dict, coeffs_tuple, fix_coeffs, t, timeseries, centroids, lc, boxsize=(3,3), gaussian_priors = False, prior_params = None):

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
    fittable_coeffs = [ float(coeffs_dict[key]) for key in coeffs_tuple if key not in fix_coeffs ]
    fittable_coeffs = flatten(fittable_coeffs)

    bounds = make_bounds(coeffs_tuple, fix_coeffs, t, gaussian_priors = gaussian_priors, prior_params = prior_params, coeffs_dict = coeffs_dict)

    # Run the least squares
    optimum_result = scipy.optimize.least_squares(function_PLD,
                                                    fittable_coeffs,
                                                    bounds = bounds,
                                                    args=(t, Pns, lc, coeffs_dict, coeffs_tuple, fix_coeffs, batman_params, PLD_params))

    return optimum_result, batman_params, PLD_params, Pns

def prayer_bead_PLD(coeffs, t, Pns, lc, timeseries, centroids, coeffs_dict, coeffs_tuple, fix_coeffs, batman_params, PLD_params,
                    nskip, planet, AOR, channel, method, foldext='', plot = True):

    # Create list of parameters that are being fit
    labels = [ key for key in coeffs_tuple if key not in fix_coeffs ]

    # Create a master model based on the least squares
    master_model = model_PLD(coeffs, t, Pns, coeffs_dict, coeffs_tuple, fix_coeffs, batman_params, PLD_params)

    # And a master residuals which will be permuated
    master_residuals = function_PLD(coeffs, t, Pns, lc, coeffs_dict, coeffs_tuple, fix_coeffs, batman_params, PLD_params)

    prayer_bead = np.ones((len(master_residuals)/nskip,len(coeffs)))

    # Loop over the number of iterations required
    for j in range(len(master_residuals)/nskip):

        residuals = np.zeros(len(master_residuals))

        # Loop over all the datapoints in the residuals and replace by the shifted point
        for i in range(len(master_residuals)):

            # If we have reached the end of the array loop back around to the beginning
            if (i+j*nskip+1) > len(master_residuals):

                residuals[i] = master_residuals[j*nskip + i - len(master_residuals)]
            else:
                residuals[i] = master_residuals[j*nskip + i]

        # Create new simulated light curve - model + residuals
        sim_lc = residuals + master_model

        # Refit the simualted light curve
        optimum_result_sim, batman_params, PLD_params, Pns = fit_function_PLD(coeffs_dict, coeffs_tuple, fix_coeffs, t, timeseries, centroids, sim_lc)

        popt_sim = optimum_result_sim.x

        # Save the optimum parameters to an array
        prayer_bead[j] = popt_sim

    # If desired plot the corner plot
    if plot:
        fig = corner.corner(prayer_bead,
                               quantiles=[0.16, 0.5, 0.84],
                               labels = labels,
                               show_titles=True, title_kwargs={"fontsize": 12}, color = 'r')

        plt.savefig("{5}/PhD/SpitzerTransits/{0}{4}/{0}_{1}_{2}_{3}_prayerBead_corner.png".format(planet, AOR, channel, method,foldext, os.getenv('HOME')))
        plt.close()

    return prayer_bead

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
        chi2 = chi(popt, t, lc, lcerr, coeffs_dict, coeffs_tuple, fix_coeffs, batman_params, params,Pns=Pns, method =method)
        return np.sum(chi2) - len(popt)*np.log(len(lc))
    elif method == 'poly':
        chi2 = chi(popt, t, lc, lcerr, coeffs_dict, coeffs_tuple, fix_coeffs, batman_params, params, x=x,y=y,method = method)
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
               method_bkg, method_cent, plotting_binsize,
               ramp_time, end_time,
               sigma_clip_cent, iters_cent, nframes_cent,
               radius_photom,
               sigma_clip_phot, iters_photom, nframes_photom,
               x0guess = None, y0guess = None,
               size_bkg_box = None, radius_bkg_ann = None, size_bkg_ann = None, size_cent_bary = None, quiet = False,  passenger57 = False,
               plot = False, AOR = None, planet = None, channel = None, sysmethod=None, foldext = ''):
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
    timeseries = fast_bad_pix_mask(timeseries, sigma_badpix, nframes_badpix, quiet = quiet, foldext=foldext)

    #Subtract background
    timeseries, background = bck_subtract(timeseries, method = method_bkg,
                                           boxsize = size_bkg_box,
                                           radius = radius_bkg_ann,
                                           size = size_bkg_ann,
                                           quiet = quiet, plot = plot, plotting_binsize = plotting_binsize, AOR = AOR, planet = planet, channel = channel,sysmethod=sysmethod, foldext=foldext)

    #Get rid of first half hour of observations before any clipping
    timeseries, midtimes, background  = discard_ramp(timeseries, midtimes, background, ramp_time, end_time, framtime, quiet = quiet, passenger57 = passenger57, foldext=foldext)

    #Centroid barycenter
    centroids = centroid(timeseries, method=method_cent, boxsize = size_cent_bary, quiet = quiet, plot = plot, AOR = AOR, planet = planet, channel = channel,sysmethod=sysmethod, foldext=foldext, x0guess=x0guess, y0guess=y0guess)

    #Clip barycenter centroids twice
    timeseries, centroids, midtimes, background = sigma_clip_centroid(timeseries, centroids, midtimes, background, sigma_clip_cent, iters_cent, nframes_cent, quiet = quiet, plot = plot, AOR = AOR, planet = planet, channel = channel,sysmethod=sysmethod, foldext=foldext)

    #Aperture photometry to create lightcurve
    lightcurve = aperture_photom(timeseries, centroids, radius_photom, quiet = quiet, foldext=foldext)

    #Clip photometry twice
    # Doing this isn't really fair because the points could be valid! Need to do it after the initial fit.
    #lightcurve, timeseries, centroids, midtimes, background = sigma_clip_photom(lightcurve, timeseries, centroids, midtimes, background, sigma_clip_phot, iters_photom, nframes_photom, quiet = quiet, plot = plot, AOR = AOR, planet = planet, channel = channel,sysmethod=sysmethod, foldext=foldext)

    lightcurve, timeseries, centroids, midtimes, background = sigma_clip_photom(lightcurve, timeseries, centroids, midtimes, background,
                                                                5, 1, len(lightcurve),
                                                                quiet = quiet, plot = plot, AOR = AOR, planet = planet, channel = channel,sysmethod=sysmethod, foldext=foldext)

    return lightcurve*MJysr2lelectrons, timeseries, centroids, midtimes, background

def runPipeline(timeseries_badpixmask, midtimes,
               method_bkg = None, method_cent = None, plotting_binsize = None,
               ramp_time = None, end_time = None, frametime = None,
               sigma_clip_cent = None, iters_cent = None, nframes_cent = None, x0guess=None, y0guess=None,
               radius_photom = None,
               sigma_clip_phot = None, iters_photom = None, nframes_photom = None,
               size_bkg_box = None, radius_bkg_ann = None, size_bkg_ann = None, size_cent_bary = None, passenger57 = False,
               quiet = False, plot = False, AOR = None, planet = None, channel = None, sysmethod = None, foldext = ''):
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
                                           plot = plot, plotting_binsize = plotting_binsize, AOR = AOR, planet = planet, channel = channel,sysmethod=sysmethod, foldext=foldext)

    #Get rid of first half hour of observations before any clipping
    timeseries, midtimes, background = discard_ramp(timeseries, midtimes, background, ramp_time, end_time, frametime, quiet = quiet, passenger57 = passenger57, foldext=foldext)

    #Centroid barycenter
    centroids = centroid(timeseries, method=method_cent, boxsize = size_cent_bary, quiet = quiet,
                         plot = plot, AOR = AOR, planet = planet, channel = channel,sysmethod=sysmethod, foldext=foldext,
                         x0guess=x0guess, y0guess=y0guess)

    #Clip barycenter centroids twice
    timeseries, centroids, midtimes, background = sigma_clip_centroid(timeseries, centroids, midtimes, background, sigma_clip_cent, iters_cent, nframes_cent, quiet = quiet,
                                                          plot = plot, AOR = AOR, planet = planet, channel = channel,sysmethod=sysmethod, foldext=foldext)

    #Aperture photometry to create lightcurve
    lightcurve = aperture_photom(timeseries, centroids, radius_photom, quiet = quiet, foldext=foldext)

    #Clip photometry twice
    # Doing this isn't really fair because the points could be valid! Need to do it after the initial fit.
    #lightcurve, timeseries, centroids, midtimes, background = sigma_clip_photom(lightcurve, timeseries, centroids, midtimes, background, sigma_clip_phot, iters_photom, nframes_photom, quiet = quiet,
        #                                                            plot = plot, AOR = AOR, planet = planet, channel = channel,sysmethod=sysmethod, foldext=foldext)

    lightcurve, timeseries, centroids, midtimes, background = sigma_clip_photom(lightcurve, timeseries, centroids, midtimes, background, 5, 1, len(lightcurve), quiet = quiet,
                                                                    plot = plot, AOR = AOR, planet = planet, channel = channel,sysmethod=sysmethod, foldext=foldext)

    return lightcurve, timeseries, centroids, midtimes, background

def plot_lightcurve(t,  lc, lcerr, popt, coeffs_dict, coeffs_tuple, fix_coeffs, batman_params, sys_params,
                    x = None, y = None, Pns = None, errors = False, binsize = 50,
                    name = None, channel = None, orbit = None, savefile = False, TT_hjd = None,
                    method = 'PLD', color = 'r', scale = None, filext = None, foldext = '',
                    showCuts = False, ncutstarts = None, cutstartTime = None,
                    cutends = False):

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
        rms = np.sqrt(np.sum(residuals**2)/len(residuals))
        chi2 = chi(popt, t, lc, lcerr, coeffs_dict, coeffs_tuple, fix_coeffs, batman_params, sys_params, x=x, y=y, method = 'poly')/(len(lc)-len(popt))
        bic = BIC(popt, t, lc, lcerr, coeffs_dict, coeffs_tuple, fix_coeffs, batman_params, sys_params, x=x, y=y, method = 'poly')

        binned_residuals = []
        for i in range(len(binned_data)):
            binned_residuals.append(binned_data[i] - binned_opt[i])

        # Make the plot, 3 plots in one.
        fig = plt.figure(figsize=(15, 6))
        #plt.title("{0}: rms={1}, red_chi2={2}, BIC={3}".format(name, rms, chi2, bic))
        frame1=fig.add_axes((.1,.6,.8,.4))
        frame1.axes.get_xaxis().set_visible(False)
        frame2=fig.add_axes((.1,.2,.8,.4))
        frame2.axes.get_xaxis().set_visible(False)
        frame3=fig.add_axes((.1,.0,.8,.2))

        frame1.plot(t, lc*scale, 'ko', markersize=2, label='Raw data')
        frame1.plot(t, optflux*scale, color, label='Best fit poly model', linewidth =2)
        frame1.set_title("{0} - {1} - {2} lightcurve".format(name, channel, orbit))
        frame1.set_ylabel("Raw [e-]")
        frame1.legend(loc = 'best')

        frame2.plot(binned_times, binned_data, 'ko', markersize = 4, label='Binned data (x{})'.format(binsize))
        frame2.plot(binned_times, binned_opt, color = color, label='Best fit transit model')
        frame2.set_ylabel("Corrected & Normalised")
        frame2.legend(loc = 'lower left')
        frame2.annotate('RMS={0:.3e}\n'.format(rms) + r'$\chi_{red}^2$' + '={0:.3e} \nBIC={1:.3e}'.format(chi2, bic),
                xy=(0.85, 0.2), xycoords='axes fraction',bbox={'facecolor':color, 'alpha':0.5, 'pad':10})
        frame3.plot(binned_times,binned_residuals, 'ko', markersize = 4)
        frame3.axhline(0, color = color)
        frame3.set_ylabel("Residuals")

        if showCuts:
            for j in range(ncutstarts):
                if cutends:
                    frame1.axvline(t[-1] - j*(float(cutstartTime)/(60.*24.)), color = 'k', ls='dashed')
                    frame2.axvline(t[-1] - j*(float(cutstartTime)/(60.*24.)), color = 'k', ls='dashed')
                    frame3.axvline(t[-1] - j*(float(cutstartTime)/(60.*24.)), color = 'k', ls='dashed')
                else:
                    frame1.axvline(j*(float(cutstartTime)/(60.*24.)), color = 'k', ls='dashed')
                    frame2.axvline(j*(float(cutstartTime)/(60.*24.)), color = 'k', ls='dashed')
                    frame3.axvline(j*(float(cutstartTime)/(60.*24.)), color = 'k', ls='dashed')

        plt.xlabel("Time [bjd]")

        if savefile:
            plt.savefig("{5}/PhD/SpitzerTransits/{0}{4}/{0}_{1}_{2}_Poly_lc_{3}.png".format(name, orbit, channel, filext, foldext, os.getenv('HOME')),bbox_inches='tight')
            plt.close()
        else:
            plt.show()
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
        rms = np.sqrt(np.sum(residuals**2)/len(residuals))
        chi2 = chi(popt, t, lc, lcerr, coeffs_dict, coeffs_tuple, fix_coeffs, batman_params, sys_params, Pns=Pns, method = 'PLD')/(len(lc)-len(popt))
        bic = BIC(popt, t, lc, lcerr, coeffs_dict, coeffs_tuple, fix_coeffs, batman_params, sys_params, Pns=Pns, method = 'PLD')

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
        frame1.plot(t, optflux*scale, color, label='Best fit PLD model', linewidth =2)
        frame1.set_title("{0} - {1} - {2} lightcurve".format(name, channel, orbit))
        frame1.set_ylabel("Raw [e-]")
        frame1.legend(loc = 'best')

        frame2.plot(binned_times, binned_data, 'ko', markersize = 4, label='Binned data (x{})'.format(binsize))
        frame2.plot(binned_times, binned_opt, color = color, label='Best fit transit model')
        frame2.set_ylabel("Corrected & Normalised")
        frame2.legend(loc = 'lower left')
        frame2.annotate('RMS={0:.3e}\n'.format(rms) + r'$\chi_{red}^2$' + '={0:.3e} \nBIC={1:.3e}'.format(chi2, bic),
                xy=(0.85, 0.2), xycoords='axes fraction',bbox={'facecolor':color, 'alpha':0.5, 'pad':10})

        frame3.plot(binned_times,binned_residuals, 'ko', markersize = 4)
        frame3.axhline(0, color = color)
        frame3.set_ylabel("Residuals")

        plt.xlabel("Time [bjd]")

        if showCuts:
            for j in range(ncutstarts):
                if cutends:
                    frame1.axvline(t[-1] - j*(float(cutstartTime)/(60.*24.)), color = 'k', ls='dashed')
                    frame2.axvline(t[-1] - j*(float(cutstartTime)/(60.*24.)), color = 'k', ls='dashed')
                    frame3.axvline(t[-1] - j*(float(cutstartTime)/(60.*24.)), color = 'k', ls='dashed')
                else:
                    frame1.axvline(j*(float(cutstartTime)/(60.*24.)), color = 'k', ls='dashed')
                    frame2.axvline(j*(float(cutstartTime)/(60.*24.)), color = 'k', ls='dashed')
                    frame3.axvline(j*(float(cutstartTime)/(60.*24.)), color = 'k', ls='dashed')

        if savefile:
            plt.savefig("{5}/PhD/SpitzerTransits/{0}{4}/{0}_{1}_{2}_PLD_lc_{3}.png".format(name, orbit, channel,filext,foldext, os.getenv('HOME')),bbox_inches='tight')
            plt.close()
        else:
            plt.show()
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

    """Function to fold over the inclination from the mcmc samples so that we do
    not have an average inclination of 90 degrees from the degeneracy."""

    indices = [i for i, s in enumerate(labels) if 'inc' in s]

    if chaintype == 'chain':

        print "\t Folding over the inclination for chain."

        samples = sampler.chain

        for k in range(len(indices)): #loop over multiple inclinations

            index = indices[k]

            for j in range(samples.shape[0]):
                for i in range(samples.shape[1]):
                    if samples.T[index][i][j] > 90.:
                        samples.T[index][i][j] = 180. - samples.T[index][i][j]

    elif chaintype == 'flatchain':

        print "\t Folding over the inclination for flatchain."

        samples = sampler.flatchain

        for k in range(len(indices)): #loop over multiple inclinations

            index = indices[k]

            for i in range(samples.shape[0]):
                if samples.T[index][i] > 90.:
                    samples.T[index][i] = 180. - samples.T[index][i]

    return samples

def t0_BJD(samples, chaintype, labels, Tinitial):
    """Function to turn the change the value of t0 in smaples into real BJD
    based on the start time of the observations (Tinitial)."""

    index = labels.index("t0")

    if chaintype == 'chain':
        samp = np.zeros(samples.shape)
        print "\t Changing t0 to BJD for chain..."

        for j in range(samples.shape[0]):
            for i in range(samples.shape[1]):
                samp[j][i] = samples[j][i]
                samp.T[index][i][j] = samples.T[index][i][j] + Tinitial

    elif chaintype == 'flatchain':
        samp = np.zeros(samples.shape)
        print "\t Changing t0 to BJD for flatchain..."

        for i in range(samples.shape[0]):
            samp[i] = samples[i]
            samp.T[index][i] = samples.T[index][i] + Tinitial

    return samp

def t0_Tinitial(samples, chaintype, labels, cuttime):
    """Function to turn the change the value of t0 in smaples into t0
    relative to original start time of the observations (Tinitial).
    This is for the cutstart script...."""

    index = labels.index("t0")

    if chaintype == 'chain':
        samp = np.zeros(samples.shape)
        print "\t Changing t0 relative to start of observations for chain..."

        for j in range(samples.shape[0]):
            for i in range(samples.shape[1]):
                samp[j][i] = samples[j][i]
                samp.T[index][i][j] = samples.T[index][i][j] - cuttime

    elif chaintype == 'flatchain':
        samp = np.zeros(samples.shape)
        print "\t Changing t0 relative to start of observations for flatchain..."

        for i in range(samples.shape[0]):
            samp[i] = samples[i]
            samp.T[index][i] = samples.T[index][i] - cuttime

    return samp

def sigma_clip_residuals(popt, t, background, lc, lcerr, coeffs_dict, coeffs_tuple, fix_coeffs, batman_params, params, sigmaclip, nframes,
        x=None, y=None, Pns=None, quiet = False, planet=None, AOR=None, channel=None, method=None,foldext='',plot = False, timeseries=None, centroids=None):

    if method == 'poly': c, c2 ='b', '#ff7f0e'
    elif method == 'PLD': c, c2 = 'r', 'c'

    if not quiet: print "\nClipping frames >{0} sigma away from residuals...".format(sigmaclip)

    originalt = t

    if method == 'poly':
        optflux = model_poly(popt, t, x, y, coeffs_dict, coeffs_tuple, fix_coeffs, batman_params, params, components=False)
        residuals = lc - optflux

        flagged_frames = np.full(residuals.shape, True, dtype=bool)

        for i in range(len(residuals)):
            # idx_start = i-(nframes/2)
            # idx_end = i+1+(nframes/2)
            # if idx_start < 0:
            #     idx_start=0
            fullmedian = np.median(residuals)
            fullstd = np.std(residuals)
            # median = np.median(np.append(residuals[idx_start:i],residuals[i+1:idx_end]))
            # std = np.std(np.append(residuals[idx_start:i],residuals[i+1:idx_end]))
            # if (abs(residuals[i] - median) > sigmaclip*std) or (abs(residuals[i] - fullmedian) > sigmaclip*fullstd):
            #     flagged_frames[i] = False
            if (abs(residuals[i] - fullmedian) > sigmaclip*fullstd):
                flagged_frames[i] = False
            else:
                pass

        lc = lc[flagged_frames,]
        lcerr = lcerr[flagged_frames,]
        t = t[flagged_frames,]
        x = x[flagged_frames,]
        y = y[flagged_frames,]
        background = background[flagged_frames,]

        nflagged = len(flagged_frames) - np.sum(flagged_frames)

        if not quiet: print "\t Clipped {0} frames.".format(nflagged)

        if plot:
            fig, ax = plt.subplots(figsize = (20,5))
            ax.plot(originalt[~flagged_frames], residuals[~flagged_frames], 'o', c=c2, markersize = 8)
            ax.plot(originalt, residuals, 'd', markersize = 4, c=c)
            ax.set_xlabel("Time (BJD)")
            ax.set_ylabel("Residuals")
            plt.savefig("{5}/PhD/SpitzerTransits/{0}{4}/{0}_{1}_{2}_{3}_residualClip.png".format(planet, AOR, channel, method,foldext, os.getenv('HOME')))
            plt.close()

        return t, x, y, lc, lcerr, background

    if method == 'PLD':
        optflux = model_PLD(popt, t, Pns, coeffs_dict, coeffs_tuple, fix_coeffs, batman_params, params, components = False)
        residuals = lc - optflux

        flagged_frames = np.full(residuals.shape, True, dtype=bool)

        for i in range(len(residuals)):
            idx_start = i-(nframes/2)
            idx_end = i+1+(nframes/2)
            if idx_start < 0:
                idx_start=0
            fullmedian = np.median(residuals)
            fullstd = np.std(residuals)
            median = np.median(np.append(residuals[idx_start:i],residuals[i+1:idx_end]))
            std = np.std(np.append(residuals[idx_start:i],residuals[i+1:idx_end]))
            if (abs(residuals[i] - median) > sigmaclip*std) or (abs(residuals[i] - fullmedian) > sigmaclip*fullstd):
                flagged_frames[i] = False
            else:
                pass

        lc = lc[flagged_frames,]
        lcerr = lcerr[flagged_frames,]
        t = t[flagged_frames,]
        x = x[flagged_frames,]
        y = y[flagged_frames,]
        Pns = Pns[:,flagged_frames]
        background = background[flagged_frames,]
        timeseries = timeseries[flagged_frames,::]
        centroids = centroids[flagged_frames,:]

        nflagged = len(flagged_frames) - np.sum(flagged_frames)

        if not quiet: print "\t Clipped {0} frames.".format(nflagged)

        if plot:
            sig = np.std(residuals)
            fig, ax = plt.subplots(1,2, figsize = (20,5))
            gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1])
            ax0 = plt.subplot(gs[0])
            ax0.plot(originalt[~flagged_frames], residuals[~flagged_frames], 'o', c=c2, markersize = 8)
            ax0.plot(originalt, residuals, 'd', markersize = 4, color=c)
            ax0.set_xlabel("Time (BJD)")
            ax0.set_ylabel("Residuals")
            ax1 = plt.subplot(gs[1])
            ax1.hist(residuals,bins=50,orientation='horizontal',color=c)
            ax1.axhline(sig*sigmaclip, color=c2, linestyle = 'dashed')
            ax1.axhline(-sig*sigmaclip, color=c2, linestyle = 'dashed')
            plt.savefig("{5}/PhD/SpitzerTransits/{0}{4}/{0}_{1}_{2}_{3}_residualClip.png".format(planet, AOR, channel, method,foldext, os.getenv('HOME')))
            plt.close()

        return t, x, y, Pns, lc, lcerr, background, timeseries, centroids

# Functions for MCMC parameter exploration of PLD
def lnprob_PLD(theta, t, Pns, lc, lcerrs, bounds,
               coeffs_dict, coeffs_tuple, fix_coeffs, batman_params, PLD_params,
               gaussian_proirs = False, prior_coeffs = None):

    fitted_coeffs = [key for key in coeffs_tuple if key not in fix_coeffs]
    lp = lnprior_PLD(theta, bounds, batman_params, fitted_coeffs, gaussian_proirs, prior_coeffs, coeffs_dict)

    if not np.isfinite(lp): return -np.inf
    if np.isnan(lp): return -np.inf

    return lp + lnlike_PLD(theta, t, Pns, lc, lcerrs, coeffs_dict, coeffs_tuple, fix_coeffs, batman_params, PLD_params)

def lnlike_PLD(theta, t, Pns, lc, lcerrs,
               coeffs_dict, coeffs_tuple, fix_coeffs, batman_params, PLD_params):

    mod = model_PLD(theta,  t, Pns, coeffs_dict, coeffs_tuple, fix_coeffs, batman_params, PLD_params)

    return -0.5*np.sum((((lc-mod)/lcerrs)**2) + np.log(2.*np.pi*(lcerrs**2)))

def lnprior_PLD(theta, bounds, batman_params, fitted_coeffs, gaussian_priors, prior_params, coeffs_dict):

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

    prior = 0.0
    if gaussian_priors:
        for param in prior_params:
            # create normal distribution
            mu = coeffs_dict['{}'.format(param)]
            sigma = coeffs_dict['{}_err'.format(param)]
            nd = scipy.stats.norm(mu,sigma)

            #Find index of param in theta
            index = fitted_coeffs.index(param)

            # Calculate pdf and add the log to the prior
            prior += np.log(nd.pdf(theta[index]))

    return prior


def mcmc_PLD(initial, data, nwalkers = 200, burnin_steps = 1000, production_steps = 2000, plot = False):

    print "\nStarting MCMC..."

    ndim = len(initial)
    p0 = [np.array(initial) + 1e-11 * np.random.randn(ndim)
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
               coeffs_dict, coeffs_tuple, fix_coeffs, batman_params, poly_params,
               gaussian_proirs = False, prior_coeffs = None):

    fitted_coeffs = [key for key in coeffs_tuple if key not in fix_coeffs]

    lp = lnprior_poly(theta, bounds, batman_params, fitted_coeffs, gaussian_proirs, prior_coeffs, coeffs_dict)

    if not np.isfinite(lp): return -np.inf
    if np.isnan(lp): return -np.inf

    return lp + lnlike_poly(theta, t, x,y, lc, lcerrs, coeffs_dict, coeffs_tuple, fix_coeffs, batman_params, poly_params)

def lnlike_poly(theta, t, x,y, lc, lcerrs,
               coeffs_dict, coeffs_tuple, fix_coeffs, batman_params, poly_params):

    mod = model_poly(theta,  t, x,y, coeffs_dict, coeffs_tuple, fix_coeffs, batman_params, poly_params)

    return -0.5*np.sum((((lc-mod)/lcerrs)**2) + np.log(2.*np.pi*(lcerrs**2)))

def lnprior_poly(theta, bounds, batman_params, fitted_coeffs, gaussian_priors, prior_params, coeffs_dict):

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

    prior = 0.0
    if gaussian_priors:
        for param in prior_params:
            # create normal distribution
            mu = coeffs_dict['{}'.format(param)]
            sigma = coeffs_dict['{}_err'.format(param)]
            nd = scipy.stats.norm(mu,sigma)

            #Find index of param in theta
            index = fitted_coeffs.index(param)

            # Calculate pdf and add the log to the prior
            prior += np.log(nd.pdf(theta[index]))

    return prior


def mcmc_poly(initial, data, nwalkers = 200, burnin_steps = 1000, production_steps = 2000, plot = False):

    print "\nStarting MCMC..."

    ndim = len(initial)
    p0 = [np.array(initial) + 1e-11 * np.random.randn(ndim)
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

# Plotting and convieniently saving the MCMC results...
def mcmc_results(sampler, popt, t, lc, lcerr, x, y, Pns, background,
                coeffs_dict, coeffs_tuple, fix_coeffs, batman_params, params,scale,
                labels, planet, AOR, channel, method, Tinitial, saveplots = True, foldext = '',
                fix_coeffs_channels = None, AOR_No = None):

    print "\nMCMC {} results...".format(method)

    # Fold over the inclination
    if 'inc' not in fix_coeffs:
        # We need this try/except for the cases when we have fixed the inclination...
        samples = fold_inc(sampler, "chain", labels)
        samples_fc = fold_inc(sampler, "flatchain", labels)
    else:
        samples = sampler.chain
        samples_fc = sampler.flatchain

    if AOR_No != None:

        print "\t Creating samples array for AOR number {} with name {}...".format(AOR_No,AOR)

        x1 = len([key for key in coeffs_tuple[0:9] if key not in fix_coeffs
                      and key not in fix_coeffs_channels])
        x2 = len([key for key in coeffs_tuple[0:9] if key in fix_coeffs_channels])
        x3 = len([key for key in coeffs_tuple if key not in fix_coeffs
                      and key not in fix_coeffs_channels])

        popt = np.concatenate((popt[AOR_No*x3:AOR_No*x3 + x1],
                               popt[-x2:],
                               popt[AOR_No*x3 + x1 :(AOR_No+1)*x3 ]))

        samples = np.concatenate((samples[:,:,AOR_No*x3:AOR_No*x3 + x1],
                               samples[:,:,-x2:],
                               samples[:,:,AOR_No*x3 + x1 :(AOR_No+1)*x3 ]), axis = 2)

        samples_fc = np.concatenate((samples_fc[:,AOR_No*x3:AOR_No*x3 + x1],
                               samples_fc[:,-x2:],
                               samples_fc[:,AOR_No*x3 + x1 :(AOR_No+1)*x3 ]), axis = 1)

    labels = [ key for key in coeffs_tuple if key not in fix_coeffs ]

    if saveplots:

        # Blue for polynomial model, red for PLD
        if method == 'poly': c, cmap = 'b', 'Blues'
        elif method == 'PLD': c, cmap = 'r', 'Reds'

        ################################# Pns Plot #################################################

        if method == 'PLD':
            fig, ax = plt.subplots(figsize=(15,5))
            for i in range(len(Pns)):
                ax.plot(t,Pns[i], alpha=0.7, label = 'P{}'.format(i+1))

            ax.set_xlabel('Time')
            ax.set_ylabel('Normalised Pixel Flux')
            ax.legend()
            plt.savefig("{5}/PhD/SpitzerTransits/{0}{4}/{0}_{1}_{2}_{3}_Pns.png".format(planet, AOR, channel, method, foldext, os.getenv('HOME')))
            plt.close()
        else:
            pass

        ################################# Gelman Rubin Diagnostics #################################

        print "\t Saving {} results and Gelman-Rubin to file...".format(method)

        f = open("{5}/PhD/SpitzerTransits/{0}{4}/{0}_{1}_{2}_{3}_GelmanRubin.txt".format(planet, AOR, channel, method, foldext, os.getenv('HOME')), 'w')

        print >> f, ("Param\tlsq_mu\tmu\tsigma\tGelman-Rubin")
        print >> f, ("=============================================================")
        for var, lsq, chain in list(zip(labels, popt, ([samples[:,:,i:i+1] for i in range(samples.shape[2])]))):
            print >> f, ("{0:}\t{1: 1.3e}\t{2: 1.3e}\t{3: 1.3e}\t{4:1.4g}".format(
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

        plt.savefig("{5}/PhD/SpitzerTransits/{0}{4}/{0}_{1}_{2}_{3}_walkers.png".format(planet, AOR, channel,method,foldext, os.getenv('HOME')))
        plt.close()

        ################################# Corner Plot ##############################################

        print "\t Plotting corner plot and saving to file..."
        fig = corner.corner(samples_fc,
                               quantiles=[0.16, 0.5, 0.84],
                               labels = labels,
                               show_titles=True, title_kwargs={"fontsize": 12}, color =c)

        plt.savefig("{5}/PhD/SpitzerTransits/{0}{4}/{0}_{1}_{2}_{3}_corner.png".format(planet, AOR, channel, method,foldext, os.getenv('HOME')))
        plt.close()

        ################################# Lightcurve Plot ###########################################

        print "\t Plotting lightcurve and saving to file..."

        avgs = np.mean(samples_fc,axis=0)

        plot_lightcurve(t,  lc, lcerr, avgs, coeffs_dict, coeffs_tuple, fix_coeffs, batman_params, params,
                    x=x, y=y, Pns = Pns, errors = False, binsize = 50,
                    name = planet, channel = channel, orbit=AOR, savefile = True, TT_hjd = None,
                    method = method, color = c, scale = scale, filext = "mcmc",foldext=foldext)

        # Make t0 in BJD instead of just start of observation time
        if 't0' not in fix_coeffs:
            # We need this try/except for the cases when we have fixed the t0...
            #print samples[0][0][0], samples_fc[0][0]
            samples = t0_BJD(samples, "chain", labels, Tinitial)
            #print samples[0][0][0], samples_fc[0][0]
            samples_fc = t0_BJD(samples_fc, "flatchain", labels, Tinitial)
            #print samples[0][0][0], samples_fc[0][0]
        else:
            pass

        ################################# RMS vs Binsize Plot #######################################

        print "\t Plotting rms vs binsize and saving to file..."
        std_devs, binsizes = [],[]

        for i in range(1,100): #TODO change this for real run
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
        plt.savefig("{5}/PhD/SpitzerTransits/{0}{4}/{0}_{1}_{2}_{3}_rmsVSbinsize.png".format(planet, AOR, channel, method,foldext, os.getenv('HOME')))
        plt.close()

        ################################# Background Vs residuals Plot #############################

        print "\t Plotting background vs residuals and saving to file..."
        if method == 'poly':
            residuals = function_poly(popt, t, x, y, lc, coeffs_dict, coeffs_tuple, fix_coeffs, batman_params, params)
        elif method == 'PLD':
            residuals = function_PLD(popt, t, Pns, lc, coeffs_dict, coeffs_tuple, fix_coeffs, batman_params, params)

        plt.plot(residuals, background, 'd', markersize = 4, color = c)
        plt.xlabel("Residuals")
        plt.ylabel("Background Flux (image units)")
        plt.savefig("{5}/PhD/SpitzerTransits/{0}{4}/{0}_{1}_{2}_{3}_bkgVSresiduals.png".format(planet, AOR, channel, method,foldext, os.getenv('HOME')))
        plt.close()

        ##################################### Flux Vs x&y plot #####################################

        print "\t Plotting Flux vs xy and saving to file..."
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        ax[0].scatter(x, lc, c = t, s = 2, alpha = 0.5, cmap = cmap)
        ax[0].set_xlabel('x-position')
        ax[0].set_ylabel('Normalised Flux')
        ax[1].scatter(y, lc, c = t , s = 2, alpha = 0.5, cmap = cmap)
        ax[1].set_xlabel('y-position')
        ax[1].set_ylabel('Normalised Flux')
        plt.savefig("{5}/PhD/SpitzerTransits/{0}{4}/{0}_{1}_{2}_{3}_fluxVSxy.png".format(planet, AOR, channel, method,foldext, os.getenv('HOME')))
        plt.close()

        ##################################### Residuals Vs x&y plot #####################################

        print "\t Plotting Flux vs xy and saving to file..."
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        ax[0].scatter(residuals, x, c = t, s = 2, alpha = 0.5, cmap = cmap)
        ax[0].set_ylabel('x-position')
        ax[0].set_xlabel('Residuals')
        ax[1].scatter(residuals, y, c = t , s = 2, alpha = 0.5, cmap = cmap)
        ax[1].set_ylabel('y-position')
        ax[1].set_xlabel('Residuals')
        plt.savefig("{5}/PhD/SpitzerTransits/{0}{4}/{0}_{1}_{2}_{3}_xyVsResiduals.png".format(planet, AOR, channel, method,foldext, os.getenv('HOME')))
        plt.close()

        ##################################### Total Results #####################################

        # Average parameter values
        avgs = np.mean(samples_fc,axis=0)
        stds = np.std(samples_fc,axis=0)
        percentiles = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples_fc, [16, 50, 84],
                                              axis=0)))
        percentiles = np.array(percentiles)
        meds = percentiles[:,0]
        pos = percentiles[:,1]
        neg = percentiles[:,2]

        # Calculate the rms, bic and chi2
        # TODO check which chi2 this is, the one before or after the errors have been inflated?

        if method == 'poly':
            optflux = model_poly(popt, t, x, y, coeffs_dict, coeffs_tuple, fix_coeffs, batman_params, params, components = False)
            residuals = lc - optflux
            rms = np.sqrt(np.sum(residuals**2)/len(residuals))
            chi2 = chi(popt, t, lc, lcerr, coeffs_dict, coeffs_tuple, fix_coeffs, batman_params, params, x=x,y=y, method = 'poly')
            bic = BIC(popt, t, lc, lcerr, coeffs_dict, coeffs_tuple, fix_coeffs, batman_params, params, x=x,y=y, method = 'poly')

        if method == 'PLD':
            optflux = model_PLD(popt, t, Pns, coeffs_dict, coeffs_tuple, fix_coeffs, batman_params, params, components = False)
            residuals = lc - optflux
            rms = np.sqrt(np.sum(residuals**2)/len(residuals))
            chi2 = chi(popt, t, lc, lcerr, coeffs_dict, coeffs_tuple, fix_coeffs, batman_params, params, Pns=Pns, method = 'PLD')
            bic = BIC(popt, t, lc, lcerr, coeffs_dict, coeffs_tuple, fix_coeffs, batman_params, params, Pns=Pns, method = 'PLD')

        ##################################### Final Results File #####################################
        # been inferred and parameters that have been fixed.
        print "\t Creating final {} results file...".format(method)

        f = open("{5}/PhD/SpitzerTransits/{0}{4}/{0}_{1}_{2}_{3}_FinalParams.txt".format(planet, AOR, channel, method, foldext, os.getenv('HOME')), 'w')

        #Calculate the transit depth
        fittable_coeffs_labels = [ key for key in coeffs_tuple if key not in fix_coeffs ]
        if 'rp' in fittable_coeffs_labels:
            rpindex = np.where(np.array(fittable_coeffs_labels) == 'rp')
            depth = avgs[rpindex[0][0]]**2
            depth_err = np.sqrt(2 * depth * stds[rpindex[0][0]])
        else:
            depth = coeffs_dict['rp']**2
            depth_err = 0.

        # Calulate the impact paramter
        if 'a' in fittable_coeffs_labels:
            aindex = np.where(np.array(fittable_coeffs_labels) == 'a')
            a, a_err = avgs[aindex[0][0]], stds[aindex[0][0]]
        else:
            a, a_err = coeffs_dict['a'], coeffs_dict['a_err']
        if 'inc' in fittable_coeffs_labels:
            incindex = np.where(np.array(fittable_coeffs_labels) == 'inc')
            inc, inc_err = avgs[incindex[0][0]], stds[incindex[0][0]]
        else:
            inc, inc_err = coeffs_dict['inc'], coeffs_dict['inc_err']

        b = a*np.cos(inc*np.pi/180.)
        b_err = np.sqrt( (np.cos( inc * np.pi/180. ) * a_err)**2 + \
                ( a * np.sin( inc * np.pi/180. ) * inc_err)**2 )


        print >> f, ("Fitted Parameters")
        print >> f, ("Param\tLSQ_mu  \tMCMC_mu  \tMCMC_err\tMCMC_med\tMCMC_+err\tMCMC_-err")
        print >> f, ("-------------------------------------------------------------")
        for var, lsq, mcmc, err, med, po, ne in list(zip(labels,popt,avgs,stds,meds,pos,neg)):
            print >> f, ("{0:}\t{1: 1.4e}\t{2: 1.4e}\t{3: 1.4e}\t{4: 1.4e}\t{5: 1.4e}\t{6: 1.4e}".format(
                var,
                lsq,
                mcmc,
                err,
                med,
                po,
                ne))

        print >> f, ("\nFixed Parameters")
        print >> f, ("Param\tValue")
        print >> f, ("-------------------------------------------------------------")
        for var, val in list(zip(fix_coeffs,[coeffs_dict[key] for key in fix_coeffs])):
            print >> f, ("{0:}\t{1:}".format(
                var,
                val))

        print >> f, ("\nInferred Parameters")
        print >> f, ("Param\tMCMC_mu\tMCMC_err")
        print >> f, ("-------------------------------------------------------------")
        print >> f, ("Depth\t{0: 1.4e}\t{1: 1.4e}".format(float(depth), float(depth_err)))
        print >> f, ("b\t{0: 1.4e}\t{1: 1.4e}".format(float(b), float(b_err)))

        print >> f, ("\nStatistical Tests")
        print >> f, ("Param\tValue")
        print >> f, ("-------------------------------------------------------------")
        print >> f, ("Chi2\t{}".format(chi2))
        print >> f, ("rms\t{}".format(rms))
        print >> f, ("BIC\t{}".format(bic))
        print >> f, ("Npoints\t{}".format(len(lc)))
        print >> f, ("Nparams\t{}".format(len(popt)))
        print >> f, ("Chi2red\t{}".format(chi2/( len(lc)- len(popt) )))

        f.close()

        f = open("{5}/PhD/SpitzerTransits/{0}{4}/{0}_{1}_{2}_{3}_FinalParamsLatex.txt".format(planet, AOR, channel,method,foldext, os.getenv('HOME')), 'w')

        ## Create text file with the same as above but latex...
        print >> f, ("Fitted Parameters \n \hline")
        print >> f, ("Param & LSQ_mu & MCMC_mu & MCMC_err")
        print >> f, ("\hline")
        for var, lsq, mcmc, err in list(zip(labels,popt,avgs,stds)):
            print >> f, ("{0:} & {1: 1.4e} & {2: 1.4e} & {3: 1.4e}".format(
                var,
                lsq,
                mcmc,
                err))

        print >> f, ("\nFixed Parameters \n \hline")
        print >> f, ("Param & Value")
        print >> f, ("\hline")
        for var, val in list(zip(fix_coeffs,[coeffs_dict[key] for key in fix_coeffs])):
            print >> f, ("{0:} & {1:}".format(
                var,
                val))

        print >> f, ("\nInferred Parameters \n \hline")
        print >> f, ("Param & MCMC_mu & MCMC_err")
        print >> f, ("\hline")
        print >> f, ("Depth & {0: 1.4e} & {1: 1.4e}".format(float(depth), float(depth_err)))
        print >> f, ("b & {0: 1.4e} & {1: 1.4e}".format(float(b), float(b_err)))

        print >> f, ("\n\tStatistical Tests \n \hline")
        print >> f, ("Param & Value")
        print >> f, ("\hline")
        print >> f, ("Chi2 & {}".format(chi2))
        print >> f, ("rms & {}".format(rms))
        print >> f, ("BIC & {}".format(bic))
        print >> f, ("Npoints & {}".format(len(lc)))
        print >> f, ("Nparams & {}".format(len(popt)))
        print >> f, ("Chi2red & {}".format(chi2/( len(lc)- len(popt) )))

        f.close()

    return avgs, stds, meds, pos, neg, rms, chi2, bic

def weightedMean(averages, stddevs):

    """Function to calculate the weighted mean and standard deviation of parameters,
    Works for array of parameters of individual."""

    ndatapoints = averages.shape[0]

    try:
        # There might be some problems with this part of the code
        nparams = averages.shape[1]
        weighted_means = np.zeros(nparams)
        total_stddevs = np.zeros(nparams)
        for i in range(nparams):
            stddevs2 = np.zeros(stddevs[i].shape[1])
                for j in range(len(stddevs[i].T)):
                stddevs2[j] = stddevs[i].T[j].max()
            weighted_mean = np.sum(averages[i]/stddevs2**2, axis = 0)/ np.sum(1./stddevs2**2, axis = 0)
            weighted_means[i] = weighted_mean
            fdis2 =  np.sum( ((averages[i] - weighted_mean)**2) / ((stddevs2**2) * (ndatapoints - 1))  , axis =0)
            total_variance = fdis2 * (1 / np.sum(1/(stddevs2**2), axis =0) )
        total_stddevs[i] = np.sqrt(total_variance)
        return weighted_means, total_stddevs

    except:
        stddevs2 = np.zeros(stddevs.shape[1])
        for j in range(len(stddevs.T)):
            stddevs2[j] = stddevs.T[j].max()
        weighted_mean = np.sum(averages/stddevs2**2, axis = 0)/ np.sum(1./stddevs2**2, axis = 0)
        fdis2 =  np.sum( ((averages - weighted_mean)**2) / ((stddevs2**2) * (ndatapoints - 1))  , axis =0)
        total_variance = fdis2 * (1 / np.sum(1/(stddevs2**2), axis =0) )
        total_stddev = np.sqrt(total_variance)
        return weighted_mean, total_stddev

def nsigma(val1, err1, val2, err2):

    """Function to return the number of sigma between 2 values with errors."""

    diff = abs(val1-val2)
    differr = np.sqrt(err1**2 + err2**2)
    nsigma = diff/differr
    return differr, nsigma

def t0check(resultFile, resultType, method, period, perioderr):

    """Function to return the number of sigma between the timing of two transits,
    accounts also for the error in the period. """

    inputData = np.genfromtxt(resultFile, dtype=None, delimiter=', ', comments='#')

    if resultType == 'LSQ':
        k = np.where(inputData[0] == 't0_lsq')[0][0]
    elif resultType == 'Mean':
        k = np.where(inputData[0] == 't0_mu')[0][0]
    elif resultType == 'Median':
        k = np.where(inputData[0] == 't0_med')[0][0]
    else:
        raise ValueError("Result type not recognised, please chose Mean, Median or LSQ.")

    l = np.where(inputData[0] == 't0_std')[0][0]

    t0s = [float(line[k]) for line in inputData if method in line]
    t0errs = [float(line[l]) for line in inputData if method in line]

    if len(t0s) == 2:
        nperiods = round((t0s[1] - t0s[0])/period)
        t0_diff = (t0s[1] - t0s[0]) - nperiods*period
        t0_diff_err = np.sqrt( ( t0errs[1] )**2 + ( t0errs[0] )**2  + (nperiods*perioderr)**2 )
        sigma_sec = t0_diff_err*60*60*24
        nsigma = t0_diff/t0_diff_err
    else:
        raise ValueError("Not implemented for more than 2 transits yet.")

    return sigma_sec, nsigma

def pipelineOptPlots(planet, channel, method, AOR, sampleLabels, saveplots = True, foldext =''):

    """Function to plot grids from the pipeline optimisation part of the pipeline
    loads in the information from numpy files and creates a colour map of the chi2
    so we can visually see which is the best parameters to use """

    if method == 'PLD': c, cmap = 'r', 'Reds'
    elif method == 'poly': c, cmap = 'b', 'Blues'

    files = glob.glob("{2}/PhD/SpitzerTransits/{0}{1}/*.npy".format(planet,foldext, os.getenv('HOME')))

    bkg_methods_params = np.load("{5}/PhD/SpitzerTransits/{0}{3}/{0}_{1}_{2}_bkgMethodsParams.npy".format(planet, AOR, channel,foldext, os.getenv('HOME')))
    cent_methods_params = np.load("{5}/PhD/SpitzerTransits/{0}{3}/{0}_{1}_{2}_centMethodsParams.npy".format(planet, AOR, channel,foldext, os.getenv('HOME')))
    photom_methods_params = np.load("{5}/PhD/SpitzerTransits/{0}{3}/{0}_{1}_{2}_photomMethodsParams.npy".format(planet, AOR, channel,foldext, os.getenv('HOME')))

    Samples = np.load("{5}/PhD/SpitzerTransits/{0}{4}/{0}_{1}_{2}_{3}_pipelineOptSamples.npy".format(planet, AOR, channel,method,foldext, os.getenv('HOME')))
    chi2Grid = np.load("{5}/PhD/SpitzerTransits/{0}{4}/{0}_{1}_{2}_{3}_Chi2grid.npy".format(planet, AOR, channel,method,foldext, os.getenv('HOME')))

    # Plot the Aperture radius against the background method
    ind = np.where(chi2Grid == chi2Grid.min())
    yticks = bkg_methods_params
    xticks = photom_methods_params
    plt.imshow(chi2Grid[:,ind[1][0],:,], cmap = cmap)
    plt.xlabel("Aperture radius")
    plt.xticks( np.arange(len(xticks)), xticks)
    plt.ylabel("Background Method")
    plt.yticks( np.arange(len(yticks)), yticks)
    plt.colorbar(label = r'$\chi^2_{red}$')
    if saveplots:
        plt.savefig("{5}/PhD/SpitzerTransits/{0}{4}/{0}_{1}_{2}_{3}_GRID_PhotomRadiusVSBkgMethod.png".format(planet, AOR, channel,method,foldext, os.getenv('HOME')),bbox_inches='tight')
    plt.close()

    # Plot the Aperture radius against the centroiding method
    xticks = photom_methods_params
    yticks = cent_methods_params
    plt.imshow(chi2Grid[ind[0][0],:,:,], cmap = cmap)
    plt.xlabel("Aperture Radius")
    plt.xticks( np.arange(len(xticks)), xticks)
    plt.ylabel("Centroiding Method")
    plt.yticks( np.arange(len(yticks)), yticks)
    plt.colorbar(label = r'$\chi^2_{red}$')
    if saveplots:
        plt.savefig("{5}/PhD/SpitzerTransits/{0}{4}/{0}_{1}_{2}_{3}_GRID_PhotomRadiusVSCentMethod.png".format(planet, AOR, channel,method,foldext, os.getenv('HOME')),bbox_inches='tight')
    plt.close()

    # Plot the Centroiding method against the background method
    xticks = cent_methods_params
    yticks = bkg_methods_params
    plt.imshow(chi2Grid[:,:,ind[2][0],], cmap = cmap)
    plt.xlabel("Centroiding Method")
    plt.xticks( np.arange(len(xticks)), xticks)
    plt.ylabel("Background Method")
    plt.yticks( np.arange(len(yticks)), yticks)
    plt.colorbar(label = r'$\chi^2_{red}$')
    if saveplots:
        plt.savefig("{5}/PhD/SpitzerTransits/{0}{4}/{0}_{1}_{2}_{3}_GRID_CentMethodVSBkgMethod.png".format(planet, AOR, channel,method,foldext, os.getenv('HOME')),bbox_inches='tight')
    plt.close()

    # Plot the corner plot of the samples, focussing on the transit parameters
    fig = corner.corner(Samples,
                               quantiles=[0.16, 0.5, 0.84], title_fmt='.5f',
                               labels = sampleLabels,
                               show_titles=True, title_kwargs={"fontsize": 12}, color = c)
    if saveplots:
        plt.savefig("{5}/PhD/SpitzerTransits/{0}{4}/{0}_{1}_{2}_{3}_GRID_Corner.png".format(planet, AOR, channel,method,foldext, os.getenv('HOME')),bbox_inches='tight')
    plt.close()

def parameter_plots(result_file, fitted_params, resultType, planet, plotPublished = False, publishedDataFile = None, saveplot=True, foldext=''):

    """
    Function to plot the parameters in channel 1 and channel 2 for all AORs and both the polynomial
    and the PLD models. Also takes an input file of published values for plotting and calculating the
    number of sigma for each channel.

    result_file = result file from the
    fitted_params = list from the original pipeline
    resultType = whether we are plotting the mean median or lsq """

    inputData = np.genfromtxt(result_file, dtype=None, delimiter=', ', comments='#')

    # Find out the indices of the type of average and the error
    if 't0' in str(inputData[0]):
        values = [s for i, s in enumerate(inputData[0]) if 't0' in s]
        indices = [i for i, s in enumerate(inputData[0]) if 't0' in s]
        Nvalues = len(indices)
        Nbefore = indices[0]

        if resultType == 'Mean':
            k = values.index('t0_mu') + Nbefore
            e = values.index('t0_std') + Nbefore

        elif resultType == 'Median':
            k = values.index('t0_med') + Nbefore
            e = values.index('t0_poserr') + Nbefore

        elif resultType == 'LSQ':
            k = values.index('t0_lqs') + Nbefore
            e = values.index('t0_std') + Nbefore

        else:
            raise ValueError("Result type not recognised, please chose Mean, Median or LSQ.")
    else:
        raise ValueError("Not implemented indices selection if we have not fit for t0")

    # If we have some published values, create a table of the Nsigma difference
    if plotPublished:
        print "\nOpening file for writing nsigma from published..."
        f = open("{2}/PhD/SpitzerTransits/{0}{1}/{0}_NsigmaFromPublished.txt".format(planet, foldext, os.getenv('HOME')), 'w')
        print >> f, ("Param\tAOR\tChannel\tMethod\tPub_val\tPub_err\tMy_val\tMy_err\tSigma\tNsigdiff")
        print >> f, ("==============================================================================================")

    f2 = open("{2}/PhD/SpitzerTransits/{0}{1}/{0}_NsigmaBetweenChannels.txt".format(planet, foldext, os.getenv('HOME')), 'w')
    print >> f2, ("Param\tMethod\tch1_val\tch1_err\tch2_val\tch2_err\tSigma\tNsigdiff")
    print >> f2, ("==============================================================================================")

    for p in range(len(fitted_params)):

        # Figure out which parameter we are looking at
        param = fitted_params[p]

        # Get the AOR labels for poly and PLD and check that they are the same for channel 1
        ch1polyAORs = [line[0] for line in inputData if 'ch1' in line and 'poly' in line]
        ch1PLDAORs = [line[0] for line in inputData if 'ch1' in line and 'PLD' in line]

        if ch1polyAORs != ch1PLDAORs:
            warnings.warn("The PLD and poly AORs for channel 1 are not the same, plotted labels might be wrong!")

        # Get the Mean and stddev of poly and PLD for the parameter that we are looking at for channel 1
        ch1polyMean = [float(line[k+p*Nvalues]) for line in inputData if 'ch1' in line and 'poly' in line]
        ch1PLDMean = [float(line[k+p*Nvalues]) for line in inputData if 'ch1' in line and 'PLD' in line]
        if resultType == 'Mean' or resultType == 'LSQ':
            ch1polyStds = np.array([float(line[e+p*Nvalues]) for line in inputData if 'ch1' in line and 'poly' in line])
            ch1PLDStds = np.array([float(line[e+p*Nvalues]) for line in inputData if 'ch1' in line and 'PLD' in line])
        elif resultType == 'Median':
            # asymetric errors
            ch1polyStds = np.array([[float(line[e+1+p*Nvalues]),float(line[e+p*Nvalues])]  for line in inputData if 'ch1' in line and 'poly' in line]).T
            ch1PLDStds = np.array([[float(line[e+1+p*Nvalues]),float(line[e+p*Nvalues])] for line in inputData if 'ch1' in line and 'PLD' in line]).T

        # Get the AOR labels for poly and PLD and check that they are the same for channel 2
        ch2polyAORs = [line[0] for line in inputData if 'ch2' in line and 'poly' in line]
        ch2PLDAORs = [line[0] for line in inputData if 'ch2' in line and 'PLD' in line]

        if ch2polyAORs != ch2PLDAORs:
            warnings.warn("The PLD and poly AORs for channel 2 are not the same, plotted labels might be wrong!")

        # Get the Mean and stddev of poly and PLD for the parameter that we are looking at for channel 2
        ch2polyMean = [float(line[k+p*Nvalues]) for line in inputData if 'ch2' in line and 'poly' in line]
        ch2PLDMean = [float(line[k+p*Nvalues]) for line in inputData if 'ch2' in line and 'PLD' in line]
        if resultType == 'Mean' or resultType == 'LSQ':
            ch2polyStds = np.array([float(line[e+p*Nvalues]) for line in inputData if 'ch2' in line and 'poly' in line])
            ch2PLDStds = np.array([float(line[e+p*Nvalues]) for line in inputData if 'ch2' in line and 'PLD' in line])
        elif resultType == 'Median':
            # asymmetric errors
            ch2polyStds = np.array([[float(line[e+1+p*Nvalues]),float(line[e+p*Nvalues])]  for line in inputData if 'ch2' in line and 'poly' in line]).T
            ch2PLDStds = np.array([[float(line[e+1+p*Nvalues]),float(line[e+p*Nvalues])] for line in inputData if 'ch2' in line and 'PLD' in line]).T

        # Calculate the weighted means...
        if len(ch1polyMean) > 1. and param != 't0':
            WMch1poly, TSch1poly = weightedMean(np.array(ch1polyMean), np.array(ch1polyStds))
            WMch1PLD, TSch1PLD = weightedMean(np.array(ch1PLDMean), np.array(ch1PLDStds))
            WMch2poly, TSch2poly = weightedMean(np.array(ch2polyMean), np.array(ch2polyStds))
            WMch2PLD, TSch2PLD = weightedMean(np.array(ch2PLDMean), np.array(ch2PLDStds))

        # Initialise plotting the parameters
        fig = plt.figure(figsize=(8,5))
        ax1 = fig.add_axes([0.1, 0.1, 0.4, 0.8])
        ax2 = fig.add_axes([0.5, 0.1, 0.4, 0.8])
        ax1.set_title(r'Ch1 ($3.6\mu m$)')
        ax2.set_title(r'Ch2 ($4.5\mu m$)')
        ax1.set_ylabel(param)

        # Channel 1
        # polynomial
        ax1.errorbar(np.array(range(len(ch1polyMean)))-0.05, ch1polyMean, yerr = ch1polyStds, marker = 'o', linestyle='', color ='b', label = 'Poly')
        # PLD
        ax1.errorbar(np.array(range(len(ch1PLDMean)))+0.05, ch1PLDMean, yerr = ch1PLDStds, marker = 'o', linestyle='', color = 'r', label = 'PLD')
        # labels
        ax1.set_xticks(range(len(ch1PLDMean)))
        ax1.set_xticklabels(ch1polyAORs, rotation='vertical')


        # Channel2
        # polynomial
        ax2.errorbar(np.array(range(len(ch2polyMean )))-0.05, ch2polyMean, yerr = ch2polyStds, marker = 'o', linestyle='', color ='b')
        # PLD
        ax2.errorbar(np.array(range(len(ch2PLDMean )))+0.05, ch2PLDMean, yerr = ch2PLDStds, marker = 'o', linestyle='', color = 'r')
        # labels
        ax2.set_xticks(range(len(ch2PLDMean)))
        ax2.set_xticklabels(ch2polyAORs, rotation='vertical')

        # Plot the weighted means
        if len(ch1polyMean) > 1. and param != 't0':
            ax1.axhline(WMch1poly, color = 'b')
            ax1.axhspan(WMch1poly - TSch1poly,
                      WMch1poly + TSch1poly,
                      alpha = 0.3, color='b', label='Weighted Mean')
            ax1.axhline(WMch1PLD, color = 'r')
            ax1.axhspan(WMch1PLD - TSch1PLD,
                      WMch1PLD + TSch1PLD,
                      alpha = 0.3, color='r',label='Weighted Mean')

            ax2.axhline(WMch2poly, color = 'b')
            ax2.axhspan(WMch2poly - TSch2poly,
                      WMch2poly + TSch2poly,
                      alpha = 0.3, color='b')
            ax2.axhline(WMch2PLD, color = 'r')
            ax2.axhspan(WMch2PLD - TSch2PLD,
                      WMch2PLD + TSch2PLD,
                      alpha = 0.3, color='r')

        if plotPublished:
            # Plot the published values if they they are given
            # and calculate Nsigma from published and save to a file
            data = np.genfromtxt(publishedDataFile, dtype=None, delimiter=', ')

            if param in data.T[0].tolist():

                pIdx = data.T[0].tolist().index(param)
                ax1.axhline(float(data[pIdx][1]), color = 'k', linestyle = 'dashed')
                ax1.axhspan(float(data[pIdx][1]) - float(data[pIdx][2]),
                            float(data[pIdx][1]) + float(data[pIdx][2]),
                            alpha = 0.3, color='k',label='Literature')

                ax2.axhline(float(data[pIdx][3]),color = 'k', linestyle = 'dashed')
                ax2.axhspan(float(data[pIdx][3]) - float(data[pIdx][4]),
                            float(data[pIdx][3]) + float(data[pIdx][4]),
                            alpha = 0.3, color='k',label='Literature')

                # Save the number of sigma to a file
                # Deal with polynomial first
                for i in range(len(ch1polyAORs)):
                    print >> f, ('{0:}\t{1:}\t{2:}\t{3:}\t{4:.4f}\t{5:.4f}\t{6:.4f}\t{7:.4f}\t{8:.4f}\t{9:.4f}'.format(
                        param, ch1polyAORs[i], 'ch1', 'poly',
                        float(data[pIdx][1]), float(data[pIdx][2]), #channel 1
                        float(ch1polyMean[i]), float(ch1polyStds.T[i].max()), # choose the biggest error
                        *nsigma(float(data[pIdx][1]), float(data[pIdx][2]),float(ch1polyMean[i]), float(ch1polyStds.T[i].max()))))

                for i in range(len(ch2polyAORs)):
                    print >> f, ('{0:}\t{1:}\t{2:}\t{3:}\t{4:.4f}\t{5:.4f}\t{6:.4f}\t{7:.4f}\t{8:.4f}\t{9:.4f}'.format(
                        param, ch2polyAORs[i], 'ch2', 'poly',
                        float(data[pIdx][3]), float(data[pIdx][4]), #channel 2
                        float(ch2polyMean[i]), float(ch2polyStds.T[i].max()), # choose the biggest error
                        *nsigma(float(data[pIdx][3]), float(data[pIdx][4]),float(ch2polyMean[i]), float(ch2polyStds.T[i].max()))))

                # PLD
                for i in range(len(ch1PLDAORs)):
                    print >> f, ('{0:}\t{1:}\t{2:}\t{3:}\t{4:.4f}\t{5:.4f}\t{6:.4f}\t{7:.4f}\t{8:.4f}\t{9:.4f}'.format(
                        param, ch1PLDAORs[i], 'ch1', 'PLD',
                        float(data[pIdx][1]), float(data[pIdx][2]), #channel 1
                        float(ch1PLDMean[i]), float(ch1PLDStds.T[i].max()), # choose the biggest error
                        *nsigma(float(data[pIdx][1]), float(data[pIdx][2]),float(ch1PLDMean[i]), float(ch1PLDStds.T[i].max()))))

                for i in range(len(ch2PLDAORs)):
                    print >> f, ('{0:}\t{1:}\t{2:}\t{3:}\t{4:.4f}\t{5:.4f}\t{6:.4f}\t{7:.4f}\t{8:.4f}\t{9:.4f}'.format(
                        param, ch2PLDAORs[i], 'ch2', 'PLD',
                        float(data[pIdx][3]), float(data[pIdx][4]), #channel 2
                        float(ch2PLDMean[i]), float(ch2PLDStds.T[i].max()), # choose the biggest error
                        *nsigma(float(data[pIdx][3]), float(data[pIdx][4]),float(ch2PLDMean[i]), float(ch2PLDStds.T[i].max()))))
            else:
                pass

        # Write the difference between the two channels to a text file
        if param != 't0' and param != 'rp':
            if len(ch1polyMean) > 1.:
                # polynomial
                for i in range(len(ch1polyAORs)):
                    print >> f2, ('{0:}\t{1:}\t{2:.4f}\t{3:.4f}\t{4:.4f}\t{5:.4f}\t{6:.4f}\t{7:.4f}'.format(
                        param, 'poly',
                        float(WMch1poly[i]), float(TSch1poly.T[i].max()), #channel 1
                        float(WMch2poly[i]), float(TSch2poly.T[i].max()), # choose the biggest error
                        *nsigma(float(WMch1poly[i]), float(TSch1poly.T[i].max()),
                                float(WMch2poly[i]), float(TSch2poly.T[i].max()))))

                # PLD
                for i in range(len(ch1polyAORs)):
                    print >> f2, ('{0:}\t{1:}\t{2:.4f}\t{3:.4f}\t{4:.4f}\t{5:.4f}\t{6:.4f}\t{7:.4f}'.format(
                        param, 'PLD',
                        float(WMch1PLD[i]), float(TSch1PLD.T[i].max()), #channel 1
                        float(WMch2PLD[i]), float(TSch2PLD.T[i].max()), # choose the biggest error
                        *nsigma(float(WMch1PLD[i]), float(TSch1PLD.T[i].max()),
                                float(WMch2PLD[i]), float(TSch2PLD.T[i].max()))))
            else:
                # polynomial
                for i in range(len(ch1polyAORs)):
                    print >> f2, ('{0:}\t{1:}\t{2:.4f}\t{3:.4f}\t{4:.4f}\t{5:.4f}\t{6:.4f}\t{7:.4f}'.format(
                        param, 'poly',
                        float(ch1polyMean[i]), float(ch1polyStds.T[i].max()), #channel 1
                        float(ch2polyMean[i]), float(ch2polyStds.T[i].max()), # choose the biggest error
                        *nsigma(float(ch1polyMean[i]), float(ch1polyStds.T[i].max()),
                                float(ch2polyMean[i]), float(ch2polyStds.T[i].max()))))

                # PLD
                for i in range(len(ch1polyAORs)):
                    print >> f2, ('{0:}\t{1:}\t{2:.4f}\t{3:.4f}\t{4:.4f}\t{5:.4f}\t{6:.4f}\t{7:.4f}'.format(
                        param, 'PLD',
                        float(ch1PLDMean[i]), float(ch1PLDStds.T[i].max()), #channel 1
                        float(ch2PLDMean[i]), float(ch2PLDStds.T[i].max()), # choose the biggest error
                        *nsigma(float(ch1PLDMean[i]), float(ch1PLDStds.T[i].max()),
                                float(ch2PLDMean[i]), float(ch2PLDStds.T[i].max()))))
        else:
            pass


        # Set the y limits of both plots to be the same sacle and to include both ch1 and ch2 points
        axis1ylim = ax1.get_ylim()
        axis2ylim = ax2.get_ylim()
        if ax1.get_ylim()[0] < ax2.get_ylim()[0]:
            ymin = ax1.get_ylim()[0]
        else:
            ymin = ax2.get_ylim()[0]
        if ax1.get_ylim()[1] < ax2.get_ylim()[1]:
            ymax = ax2.get_ylim()[1]
        else:
            ymax = ax1.get_ylim()[1]
        ax1.set_ylim(ymin,ymax)
        ax2.set_ylim(ax1.get_ylim())
        ax2.get_yaxis().set_visible(False)

        ax1.legend(loc = 'best')


        # Save if required
        if saveplot:
            plt.savefig("{3}/PhD/SpitzerTransits/{0}{1}/{0}_{2}.png".format(planet,foldext,param,os.getenv('HOME')),bbox_inches='tight')
        plt.close()
    if plotPublished:
        f.close()
    else:
        pass
    f2.close()

def compare_parameter_plots(result_file, result_file_eccentric, fitted_params, resultType, planet,
                    plotPublished = False, publishedDataFile = None, saveplot=True, foldext=''):

    """
    Function to plot the parameters in channel 1 and channel 2 for all AORs and both the polynomial
    and the PLD models. Also takes an input file of published values for plotting and calculating the
    number of sigma for each channel.

    result_file = result file from the
    fitted_params = list from the original pipeline
    resultType = whether we are plotting the mean median or lsq """

    # Check if we are wanting to compare the eccentric and circular resutls.

    inputData_circular = np.genfromtxt(result_file, dtype=None, delimiter=', ', comments='#')
    inputData_eccentric = np.genfromtxt(result_file_eccentric, dtype=None, delimiter=', ', comments='#')

    inputData = [inputData_circular, inputData_eccentric]

    # Find out the indices of the type of average and the error
    if 't0' in str(inputData[0][0]):
        values = [s for i, s in enumerate(inputData[0][0]) if 't0' in s]
        indices = [i for i, s in enumerate(inputData[0][0]) if 't0' in s]
        Nvalues = len(indices)
        Nbefore = indices[0]

        if resultType == 'Mean':
            k = values.index('t0_mu') + Nbefore
            e = values.index('t0_std') + Nbefore

        elif resultType == 'Median':
            k = values.index('t0_med') + Nbefore
            e = values.index('t0_poserr') + Nbefore

        elif resultType == 'LSQ':
            k = values.index('t0_lqs') + Nbefore
            e = values.index('t0_std') + Nbefore

        else:
            raise ValueError("Result type not recognised, please chose Mean, Median or LSQ.")
    else:
        raise ValueError("Not implemented indices selection if we have not fit for t0")

    # If we have some published values, create a table of the Nsigma difference
    if plotPublished:
        print "\nOpening file for writing nsigma from published..."
        f = open("{3}/PhD/SpitzerTransits/{0}{1}/{0}_NsigmaFEccentricCircular.txt".format(planet, foldext), 'w')
        print >> f, ("Param\tAOR\tChannel\tMethod\tCirc_val\tCirc_err\tEcc_val\tEcc_err\tSigma\tNsigdiff")
        print >> f, ("==============================================================================================")

    for p in range(len(fitted_params)):

        # Figure out which parameter we are looking at
        param = fitted_params[p]

        # Initialise plotting the parameters
        fig = plt.figure(figsize=(8,5))
        ax1 = fig.add_axes([0.1, 0.1, 0.4, 0.8])
        ax2 = fig.add_axes([0.5, 0.1, 0.4, 0.8])
        ax1.set_title(r'Ch1 ($3.6\mu m$)')
        ax2.set_title(r'Ch2 ($4.5\mu m$)')
        ax1.set_ylabel(param)

        markers = ['o', '*']
        offset = [0,0.01]
        lab = [' cir',' ecc']

        # create lists to save the parameters to so that we can compare the circular and elliptical results
        ch1polyMean_list, ch1PLDMean_list, ch1polyStds_list, ch1PLDStds_list = [],[],[],[]
        ch2polyMean_list, ch2PLDMean_list, ch2polyStds_list, ch2PLDStds_list = [],[],[],[]

        for c in range(len(inputData)):

            # Get the AOR labels for poly and PLD and check that they are the same for channel 1
            ch1polyAORs = [line[0] for line in inputData[c] if 'ch1' in line and 'poly' in line]
            ch1PLDAORs = [line[0] for line in inputData[c] if 'ch1' in line and 'PLD' in line]

            if ch1polyAORs != ch1PLDAORs:
                warnings.warn("The PLD and poly AORs for channel 1 are not the same, plotted labels might be wrong!")

            # Get the Mean and stddev of poly and PLD for the parameter that we are looking at for channel 1
            ch1polyMean = [float(line[k+p*Nvalues]) for line in inputData[c] if 'ch1' in line and 'poly' in line]
            ch1PLDMean = [float(line[k+p*Nvalues]) for line in inputData[c] if 'ch1' in line and 'PLD' in line]
            if resultType == 'Mean' or resultType == 'LSQ':
                ch1polyStds = np.array([float(line[e+p*Nvalues]) for line in inputData[c] if 'ch1' in line and 'poly' in line])
                ch1PLDStds = np.array([float(line[e+p*Nvalues]) for line in inputData[c] if 'ch1' in line and 'PLD' in line])
            elif resultType == 'Median':
                # asymetric errors
                ch1polyStds = np.array([[float(line[e+1+p*Nvalues]),float(line[e+p*Nvalues])]  for line in inputData[c] if 'ch1' in line and 'poly' in line]).T
                ch1PLDStds = np.array([[float(line[e+1+p*Nvalues]),float(line[e+p*Nvalues])] for line in inputData[c] if 'ch1' in line and 'PLD' in line]).T

            ch1polyMean_list.append(ch1polyMean)
            ch1PLDMean_list.append(ch1PLDMean)
            ch1polyStds_list.append(ch1polyStds)
            ch1PLDStds_list.append(ch1PLDStds)


            # Get the AOR labels for poly and PLD and check that they are the same for channel 2
            ch2polyAORs = [line[0] for line in inputData[c] if 'ch2' in line and 'poly' in line]
            ch2PLDAORs = [line[0] for line in inputData[c] if 'ch2' in line and 'PLD' in line]

            if ch2polyAORs != ch2PLDAORs:
                warnings.warn("The PLD and poly AORs for channel 2 are not the same, plotted labels might be wrong!")

            # Get the Mean and stddev of poly and PLD for the parameter that we are looking at for channel 2
            ch2polyMean = [float(line[k+p*Nvalues]) for line in inputData[c] if 'ch2' in line and 'poly' in line]
            ch2PLDMean = [float(line[k+p*Nvalues]) for line in inputData[c] if 'ch2' in line and 'PLD' in line]
            if resultType == 'Mean' or resultType == 'LSQ':
                ch2polyStds = np.array([float(line[e+p*Nvalues]) for line in inputData[c] if 'ch2' in line and 'poly' in line])
                ch2PLDStds = np.array([float(line[e+p*Nvalues]) for line in inputData[c] if 'ch2' in line and 'PLD' in line])
            elif resultType == 'Median':
                # asymmetric errors
                ch2polyStds = np.array([[float(line[e+1+p*Nvalues]),float(line[e+p*Nvalues])]  for line in inputData[c] if 'ch2' in line and 'poly' in line]).T
                ch2PLDStds = np.array([[float(line[e+1+p*Nvalues]),float(line[e+p*Nvalues])] for line in inputData[c] if 'ch2' in line and 'PLD' in line]).T

            ch2polyMean_list.append(ch2polyMean)
            ch2PLDMean_list.append(ch2PLDMean)
            ch2polyStds_list.append(ch2polyStds)
            ch2PLDStds_list.append(ch2PLDStds)

            # Calculate the weighted means...
            WMch1poly, TSch1poly = weightedMean(np.array(ch1polyMean), np.array(ch1polyStds))
            WMch1PLD, TSch1PLD = weightedMean(np.array(ch1PLDMean), np.array(ch1PLDStds))
            WMch2poly, TSch2poly = weightedMean(np.array(ch2polyMean), np.array(ch2polyStds))
            WMch2PLD, TSch2PLD = weightedMean(np.array(ch2PLDMean), np.array(ch2PLDStds))

            # Channel 1
            # polynomial
            ax1.errorbar(np.array(range(len(ch1polyMean)))-0.05+offset[c], ch1polyMean, yerr = ch1polyStds, marker = markers[c], linestyle='', color ='b', label = 'Poly' + lab[c])
            # PLD
            ax1.errorbar(np.array(range(len(ch1PLDMean)))+0.05+offset[c], ch1PLDMean, yerr = ch1PLDStds, marker = markers[c], linestyle='', color = 'r', label = 'PLD' + lab[c])
            # labels
            ax1.set_xticks(range(len(ch1PLDMean)))
            ax1.set_xticklabels(ch1polyAORs, rotation='vertical')


            # Channel2
            # polynomial
            ax2.errorbar(np.array(range(len(ch2polyMean )))-0.05+offset[c], ch2polyMean, yerr = ch2polyStds, marker = markers[c], linestyle='', color ='b')
            # PLD
            ax2.errorbar(np.array(range(len(ch2PLDMean )))+0.05+offset[c], ch2PLDMean, yerr = ch2PLDStds, marker = markers[c], linestyle='', color = 'r')
            # labmarkers[c]
            ax2.set_xticks(range(len(ch2PLDMean)))
            ax2.set_xticklabels(ch2polyAORs, rotation='vertical')

            # Plot the weighted means
            if len(ch1polyMean) > 1.:
                ax1.axhline(WMch1poly, color = 'b')
                ax1.axhspan(WMch1poly - TSch1poly,
                          WMch1poly + TSch1poly,
                          alpha = 0.3, color='b',label='Weighted Mean')
                ax1.axhline(WMch1PLD, color = 'r')
                ax1.axhspan(WMch1PLD - TSch1PLD,
                          WMch1PLD + TSch1PLD,
                          alpha = 0.3, color='r',label='Weighted Mean')

                ax2.axhline(WMch2poly, color = 'b')
                ax2.axhspan(WMch2poly - TSch2poly,
                          WMch2poly + TSch2poly,
                          alpha = 0.3, color='b')
                ax2.axhline(WMch2PLD, color = 'r')
                ax2.axhspan(WMch2PLD - TSch2PLD,
                          WMch2PLD + TSch2PLD,
                           alpha = 0.3, color='r')


        if plotPublished:
            # Plot the published values if they they are given
            # and calculate Nsigma from published and save to a file
            data = np.genfromtxt(publishedDataFile, dtype=None, delimiter=', ')

            if param in data.T[0].tolist():

                pIdx = data.T[0].tolist().index(param)
                ax1.axhline(float(data[pIdx][1]), color = 'k', linestyle = 'dashed')
                ax1.axhspan(float(data[pIdx][1]) - float(data[pIdx][2]),
                            float(data[pIdx][1]) + float(data[pIdx][2]),
                            alpha = 0.3, color='k',label='Literature')

                ax2.axhline(float(data[pIdx][3]),color = 'k', linestyle = 'dashed')
                ax2.axhspan(float(data[pIdx][3]) - float(data[pIdx][4]),
                            float(data[pIdx][3]) + float(data[pIdx][4]),
                            alpha = 0.3, color='k',label='Literature')


            # Save the number of sigma to a file
            # This doesn't need to be in param loop because we want to do it for all the parameters
            # regardless if we have published values.
            # polynomial
            for i in range(len(ch1polyAORs)):
                print >> f, ('{0:}\t{1:}\t{2:}\t{3:}\t{4:.4f}\t{5:.4f}\t{6:.4f}\t{7:.4f}\t{8:.4f}\t{9:.4f}'.format(
                    param, ch1polyAORs[i], 'ch1', 'poly',
                    float(ch1polyMean_list[0][i]), float(ch1polyStds_list[0].T[i].max()), #channel 1
                    float(ch1polyMean_list[1][i]), float(ch1polyStds_list[1].T[i].max()), # choose the biggest error
                    *nsigma(float(ch1polyMean_list[0][i]), float(ch1polyStds_list[0].T[i].max()),
                            float(ch1polyMean_list[1][i]), float(ch1polyStds_list[0].T[i].max()))))

            for i in range(len(ch2polyAORs)):
                print >> f, ('{0:}\t{1:}\t{2:}\t{3:}\t{4:.4f}\t{5:.4f}\t{6:.4f}\t{7:.4f}\t{8:.4f}\t{9:.4f}'.format(
                    param, ch2polyAORs[i], 'ch2', 'poly',
                    float(ch2polyMean_list[0][i]), float(ch2polyStds_list[0].T[i].max()), #channel 1
                    float(ch2polyMean_list[1][i]), float(ch2polyStds_list[1].T[i].max()), # choose the biggest error
                    *nsigma(float(ch2polyMean_list[0][i]), float(ch2polyStds_list[0].T[i].max()),
                            float(ch2polyMean_list[1][i]), float(ch2polyStds_list[0].T[i].max()))))
            #PLD
            for i in range(len(ch1PLDAORs)):
                print >> f, ('{0:}\t{1:}\t{2:}\t{3:}\t{4:.4f}\t{5:.4f}\t{6:.4f}\t{7:.4f}\t{8:.4f}\t{9:.4f}'.format(
                    param, ch1PLDAORs[i], 'ch1', 'PLD',
                    float(ch1PLDMean_list[0][i]), float(ch1PLDStds_list[0].T[i].max()), #channel 1
                    float(ch1PLDMean_list[1][i]), float(ch1PLDStds_list[1].T[i].max()), # choose the biggest error
                    *nsigma(float(ch1PLDMean_list[0][i]), float(ch1PLDStds_list[0].T[i].max()),
                            float(ch1PLDMean_list[1][i]), float(ch1PLDStds_list[0].T[i].max()))))

            for i in range(len(ch2PLDAORs)):
                print >> f, ('{0:}\t{1:}\t{2:}\t{3:}\t{4:.4f}\t{5:.4f}\t{6:.4f}\t{7:.4f}\t{8:.4f}\t{9:.4f}'.format(
                    param, ch2PLDAORs[i], 'ch2', 'PLD',
                    float(ch2PLDMean_list[0][i]), float(ch2PLDStds_list[0].T[i].max()), #channel 1
                    float(ch2PLDMean_list[1][i]), float(ch2PLDStds_list[1].T[i].max()), # choose the biggest error
                    *nsigma(float(ch2PLDMean_list[0][i]), float(ch2PLDStds_list[0].T[i].max()),
                            float(ch2PLDMean_list[1][i]), float(ch2PLDStds_list[0].T[i].max()))))
            pass

        # Set the y limits of both plots to be the same sacle and to include both ch1 and ch2 points
        axis1ylim = ax1.get_ylim()
        axis2ylim = ax2.get_ylim()
        if ax1.get_ylim()[0] < ax2.get_ylim()[0]:
            ymin = ax1.get_ylim()[0]
        else:
            ymin = ax2.get_ylim()[0]
        if ax1.get_ylim()[1] < ax2.get_ylim()[1]:
            ymax = ax2.get_ylim()[1]
        else:
            ymax = ax1.get_ylim()[1]
        ax1.set_ylim(ymin,ymax)
        ax2.set_ylim(ax1.get_ylim())
        ax2.get_yaxis().set_visible(False)

        ax1.legend(loc = 'best')


        # Save if required
        if saveplot:
            plt.savefig("{3}/PhD/SpitzerTransits/{0}{1}/{0}_{2}_eccentricCircular.png".format(planet,foldext,param, os.getenv('HOME')),bbox_inches='tight')
        #plt.close()
    if plotPublished:
        f.close()
    else:
        pass

# Pipeling fitting function for fitting N transits and allowing us to fix parameters
# between the transits, such as a or inc

def function_poly_Ntransits(coeffs, t_list, x_list, y_list, lc_list,
                            coeffs_dict_list,
                            coeffs_tuple, fix_coeffs, fix_coeffs_channels,
                            batman_params_list, poly_params_list):

    """Find the difference between a quadratic function and the lightcurve."""

    residuals = []

    x1 = len([key for key in coeffs_tuple[0:9] if key not in fix_coeffs
               and key not in fix_coeffs_channels])
    x2 = len([key for key in coeffs_tuple[0:9] if key in fix_coeffs_channels])
    x3 = len([key for key in coeffs_tuple if key not in fix_coeffs
                          and key not in fix_coeffs_channels])

    for i in range(len(coeffs_dict_list)):

        coeffs_fit = np.concatenate((coeffs[i*x3:i*x3 + x1],
                                       coeffs[-x2:],
                                       coeffs[i*x3 + x1 :(i+1)*x3 ]))

        new_flux = model_poly(coeffs_fit, t_list[i], x_list[i], y_list[i], coeffs_dict_list[i],
                              coeffs_tuple, fix_coeffs, batman_params_list[i], poly_params_list[i])

        residuals.append(lc_list[i]-new_flux)

    return np.array(np.concatenate(residuals))

def fit_function_poly_Ntransits(coeffs_dict_list, coeffs_tuple,
                                fix_coeffs, fix_coeffs_channels,
                                t_list, x_list, y_list, lc_list):

    batman_params_list, poly_params_list = [],[]
    fittable_coeffs, bounds_list = [],[]

    for i in range(len(coeffs_dict_list)):

        batman_params = batman.TransitParams()
        batman_params.t0 = coeffs_dict_list[i]['t0']                      #time of inferior conjunction
        batman_params.per = coeffs_dict_list[i]['per']                #orbital period
        batman_params.rp = coeffs_dict_list[i]['rp']                      #planet radius (in units of stellar radii)
        batman_params.a = coeffs_dict_list[i]['a']                        #semi-major axis (in units of stellar radii)
        batman_params.inc = coeffs_dict_list[i]['inc']                    #orbital inclination (in degrees)
        batman_params.ecc = coeffs_dict_list[i]['ecc']                    #eccentricity
        batman_params.w = coeffs_dict_list[i]['w']                        #longitude of periastron (in degrees)
        batman_params.u = coeffs_dict_list[i]['u']
        batman_params.limb_dark = coeffs_dict_list[i]['limb_dark']

        batman_params_list.append(batman_params)

        poly_params = dict(item for item in coeffs_dict_list[i].items() if item[0] in coeffs_tuple[9:])

        poly_params_list.append(poly_params)

        fittable_coeffs_i = [ float(coeffs_dict_list[i][key]) for key in coeffs_tuple if key not in fix_coeffs
                          and key not in fix_coeffs_channels]

        bounds_list.append(make_bounds(coeffs_tuple, fix_coeffs, t_list[i], fix_coeffs_channels))

        fittable_coeffs.append(fittable_coeffs_i)

    # COnstruct an array of parameters for fitting Ntransits*N+n where:
    # Ntransits = no of transits that we are fitting together
    # N = no of free parameters in each individual transit fit
    # n = np of parameters fixed between the seperate transits
    bounds_list.append(make_bounds(coeffs_tuple, fix_coeffs, t_list[i], fix_coeffs_channels, normal=False))

    bounds = np.concatenate([np.array(o) for o in bounds_list],axis=1).tolist()
    fittable_coeffs.append([ float(coeffs_dict_list[i][key]) for key in coeffs_tuple if key in fix_coeffs_channels]) # this might go wrong when we try to flatten it

    fittable_coeffs = flatten(fittable_coeffs)

    optimum_result = scipy.optimize.least_squares(function_poly_Ntransits,
                                                fittable_coeffs,
                                                bounds = bounds,
                                                args=(t_list, x_list, y_list, lc_list, coeffs_dict_list, coeffs_tuple,
                                                      fix_coeffs, fix_coeffs_channels, batman_params_list, poly_params_list))

    return optimum_result.x, batman_params_list, poly_params_list

def lnprob_poly_Ntransits(theta, t_list, x_list, y_list, lc_list, lcerrs_list, bounds,
                coeffs_dict_list, coeffs_tuple, fix_coeffs, fix_coeffs_channels,
                batman_params_list, poly_params_list):

    x1 = len([key for key in coeffs_tuple[0:9] if key not in fix_coeffs
               and key not in fix_coeffs_channels])
    x2 = len([key for key in coeffs_tuple[0:9] if key in fix_coeffs_channels])
    x3 = len([key for key in coeffs_tuple if key not in fix_coeffs
                          and key not in fix_coeffs_channels])

    lnlikelihoods = []
    lnpriors = []

    for i in range(len(coeffs_dict_list)):

        coeffs_fit = np.concatenate((theta[i*x3:i*x3 + x1],
                                       theta[-x2:],
                                       theta[i*x3 + x1 :(i+1)*x3 ]))

        bounds_fit = np.concatenate((np.array(bounds)[:,i*x3:i*x3 + x1],
                                       np.array(bounds)[:,-x2:],
                                       np.array(bounds)[:,i*x3 + x1 :(i+1)*x3 ]),axis=1)

        lnlike = lnlike_poly(coeffs_fit, t_list[i], x_list[i], y_list[i], lc_list[i], lcerrs_list[i],
                             coeffs_dict_list[i], coeffs_tuple, fix_coeffs,
                             batman_params_list[i], poly_params_list[i])

        lnlikelihoods.append(lnlike)

        # This could be an issue with multiplying by "a" and "inc" twice if the
        # priors are not uniform, but for now it works. TODO
        lp = lnprior_poly(coeffs_fit, bounds_fit, batman_params_list[i])

        if not np.isfinite(lp): return -np.inf
        if np.isnan(lp): return -np.inf

        lnpriors.append(lp)

    # Sum the log probabilities
    return np.sum(lnpriors) + np.sum(lnlikelihoods)

def mcmc_poly_Ntransits(initial, data, nwalkers = 100, burnin_steps = 1000, production_steps = 2000, plot = False):

    print "\nStarting MCMC..."

    ndim = len(initial)
    p0 = [np.array(initial) + 1e-9 * np.random.randn(ndim)
            for i in xrange(nwalkers)]

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_poly_Ntransits, args=(data))

    print("\t Running burn-in: {} steps".format(burnin_steps))
    p0, lnp, _ = sampler.run_mcmc(p0, burnin_steps)
    sampler.reset()

    print("\t Running production: {} steps".format(production_steps))
    p0, _, _ = sampler.run_mcmc(p0, production_steps)

    print("\t Mean acceptance fraction: {0:.3f}"
           .format(np.mean(sampler.acceptance_fraction)))

    return sampler

def function_PLD_Ntransits(coeffs, t_list, Pns_list, lc_list,
                            coeffs_dict_list,
                            coeffs_tuple, fix_coeffs, fix_coeffs_channels,
                            batman_params_list, PLD_params_list):

    """Find the difference between a quadratic function and the lightcurve."""

    residuals = []

    x1 = len([key for key in coeffs_tuple[0:9] if key not in fix_coeffs
               and key not in fix_coeffs_channels])
    x2 = len([key for key in coeffs_tuple[0:9] if key in fix_coeffs_channels])
    x3 = len([key for key in coeffs_tuple if key not in fix_coeffs
                          and key not in fix_coeffs_channels])

    for i in range(len(coeffs_dict_list)):

        coeffs_fit = np.concatenate((coeffs[i*x3:i*x3 + x1],
                                       coeffs[-x2:],
                                       coeffs[i*x3 + x1 :(i+1)*x3 ]))

        new_flux = model_PLD(coeffs_fit, t_list[i], Pns_list[i], coeffs_dict_list[i],
                              coeffs_tuple, fix_coeffs, batman_params_list[i], PLD_params_list[i])

        residuals.append(lc_list[i]-new_flux)

    return np.array(np.concatenate(residuals))

def fit_function_PLD_Ntransits(coeffs_dict_list, coeffs_tuple,
                                fix_coeffs, fix_coeffs_channels,
                                t_list, timeseries_list, centroids_list, lc_list,
                                boxsize=(3,3)):


    batman_params_list, PLD_params_list = [],[]
    fittable_coeffs, bounds_list = [],[]
    Pns_list = []

    for i in range(len(coeffs_dict_list)):

        el1, el2 = int(np.floor(np.mean(centroids_list[i][:,1]))), int(np.floor(np.mean(centroids_list[i][:,0])))
        Parray = np.zeros((boxsize[0]*boxsize[1], len(timeseries_list[i])))
        for k in range(boxsize[0]):
            for j in range(boxsize[1]):
                Parray[boxsize[0]*k+j] = pix_timeseries(timeseries_list[i], el1+k-(boxsize[0]/2), el2+j-boxsize[1]/2)
        sumP = np.sum(Parray, axis = 0)
        Pns = Parray/sumP
        Pns_list.append(Pns)

        batman_params = batman.TransitParams()
        batman_params.t0 = coeffs_dict_list[i]['t0']                      #time of inferior conjunction
        batman_params.per = coeffs_dict_list[i]['per']                #orbital period
        batman_params.rp = coeffs_dict_list[i]['rp']                      #planet radius (in units of stellar radii)
        batman_params.a = coeffs_dict_list[i]['a']                        #semi-major axis (in units of stellar radii)
        batman_params.inc = coeffs_dict_list[i]['inc']                    #orbital inclination (in degrees)
        batman_params.ecc = coeffs_dict_list[i]['ecc']                    #eccentricity
        batman_params.w = coeffs_dict_list[i]['w']                        #longitude of periastron (in degrees)
        batman_params.u = coeffs_dict_list[i]['u']
        batman_params.limb_dark = coeffs_dict_list[i]['limb_dark']

        batman_params_list.append(batman_params)

        PLD_params = dict(item for item in coeffs_dict_list[i].items() if item[0] in coeffs_tuple[9:])

        PLD_params_list.append(PLD_params)

        fittable_coeffs_i = [ float(coeffs_dict_list[i][key]) for key in coeffs_tuple if key not in fix_coeffs
                          and key not in fix_coeffs_channels]

        bounds_list.append(make_bounds(coeffs_tuple, fix_coeffs, t_list[i], fix_coeffs_channels))

        fittable_coeffs.append(fittable_coeffs_i)

    # COnstruct an array of parameters for fitting Ntransits*N+n where:
    # Ntransits = no of transits that we are fitting together
    # N = no of free parameters in each individual transit fit
    # n = np of parameters fixed between the seperate transits
    bounds_list.append(make_bounds(coeffs_tuple, fix_coeffs, fix_coeffs_channels = fix_coeffs_channels, normal=False))

    bounds = np.concatenate([np.array(o) for o in bounds_list],axis=1).tolist()
    fittable_coeffs.append([ float(coeffs_dict_list[i][key]) for key in coeffs_tuple if key in fix_coeffs_channels]) # this might go wrong when we try to flatten it

    fittable_coeffs = flatten(fittable_coeffs)

    optimum_result = scipy.optimize.least_squares(function_PLD_Ntransits,
                                                fittable_coeffs,
                                                bounds = bounds,
                                                args=(t_list, Pns_list, lc_list, coeffs_dict_list, coeffs_tuple,
                                                      fix_coeffs, fix_coeffs_channels, batman_params_list, PLD_params_list))

    return optimum_result.x, batman_params_list, PLD_params_list, Pns_list

def lnprob_PLD_Ntransits(theta, t_list, Pns_list, lc_list, lcerrs_list, bounds,
                coeffs_dict_list, coeffs_tuple, fix_coeffs, fix_coeffs_channels,
                batman_params_list, PLD_params_list):

    """Function to calculate the log probability of the transit model,
    plus the systematic model plus the prior probabilities of the parameters """

    x1 = len([key for key in coeffs_tuple[0:9] if key not in fix_coeffs
               and key not in fix_coeffs_channels])
    x2 = len([key for key in coeffs_tuple[0:9] if key in fix_coeffs_channels])
    x3 = len([key for key in coeffs_tuple if key not in fix_coeffs
                          and key not in fix_coeffs_channels])

    lnlikelihoods = []
    lnpriors = []

    for i in range(len(coeffs_dict_list)):

        # Create an array from the full joint array of parameters for the joint
        # fit
        coeffs_fit = np.concatenate((theta[i*x3:i*x3 + x1],
                                       theta[-x2:],
                                       theta[i*x3 + x1 :(i+1)*x3 ]))

        bounds_fit = np.concatenate((np.array(bounds)[:,i*x3:i*x3 + x1],
                                       np.array(bounds)[:,-x2:],
                                       np.array(bounds)[:,i*x3 + x1 :(i+1)*x3 ]),axis=1)

        lnlike = lnlike_PLD(coeffs_fit, t_list[i], Pns_list[i], lc_list[i], lcerrs_list[i],
                             coeffs_dict_list[i], coeffs_tuple, fix_coeffs,
                             batman_params_list[i], PLD_params_list[i])

        lnlikelihoods.append(lnlike)

        # This could be an issue with multiplying by "a" and "inc" twice if the
        # priors are not uniform, but for now it works. TODO
        lp = lnprior_PLD(coeffs_fit, bounds_fit, batman_params_list[i])

        if not np.isfinite(lp): return -np.inf
        if np.isnan(lp): return -np.inf

        lnpriors.append(lp)

    return np.sum(lnpriors) + np.sum(lnlikelihoods)

def mcmc_PLD_Ntransits(initial, data, nwalkers = 100, burnin_steps = 1000, production_steps = 2000, plot = False):

    print "\nStarting MCMC..."

    ndim = len(initial)
    p0 = [np.array(initial) + 1e-9 * np.random.randn(ndim)
            for i in xrange(nwalkers)]

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_PLD_Ntransits, args=(data))

    print("\t Running burn-in: {} steps".format(burnin_steps))
    p0, lnp, _ = sampler.run_mcmc(p0, burnin_steps)
    sampler.reset()

    print("\t Running production: {} steps".format(production_steps))
    p0, _, _ = sampler.run_mcmc(p0, production_steps)

    print("\t Mean acceptance fraction: {0:.3f}"
           .format(np.mean(sampler.acceptance_fraction)))

    return sampler
