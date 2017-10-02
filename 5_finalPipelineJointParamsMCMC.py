# imports
from astropy.io import fits
import numpy as np
import glob, os, sys, time
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy import stats
from scipy.ndimage.measurements import center_of_mass
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
import os

# Custom imports
sys.path.insert(0, '{}/PhD/code/mySpitzerTransit/'.format(os.getenv('HOME')))
from timeseries_routines import *
from transitFitting_routines import *
sys.path.insert(1, '{}/PhD/code/'.format(os.getenv('HOME')))
from ProgressBar import *

try:
    planetinFile = sys.argv[1]
    pipelineinFile = sys.argv[2]
    burninsteps = int(sys.argv[3])
    runsteps = int(sys.argv[4])
except:
    raise ValueError('You need to provide input planet file and pipeline file as arguements and burnin and runsteps.')

try:
    publishedDataFile = sys.argv[5]
    plotPublished = True
    print "Using literature planet data from {}".format(publishedDataFile)
except:
    publishedDataFile = None
    plotPublished = False

print "Using planet data from {}".format(planetinFile)
print "Using pipeline information from {}".format(pipelineinFile)

# Read in the planet information from a text file
inputData = np.genfromtxt(planetinFile, dtype=None, delimiter=': ', comments='#')
planet = inputData[0][1]
AORs = inputData[1][1].split(', ')
channels = inputData[2][1].split(', ')
t0s = inputData[3][1].split(', ')
cutstarts = [float(x) for x in inputData[4][1].split(', ')]
cutends = [float(x) for x in inputData[5][1].split(', ')]
posGuess = [float(x) for x in inputData[6][1].split(', ')]
ldlaw = inputData[int(np.where(inputData.T[0]=='limb_dark')[0])][1]
foldext = raw_input("Provide folder extension (hit ENTER for None): ")

PP = np.genfromtxt(pipelineinFile, dtype = None, skip_header = 1, delimiter = ', ', comments = '#')

leastsquares, averages, medians, stdevs, rmss, chi2s, BICs, poserrs, negerrs = [],[],[],[],[],[],[],[],[]

#Create list for dictionaries and for the results from running the pipeine
coeffs_dict_poly_list, coeffs_dict_PLD_list = [],[]
lc_list_poly, lcerr_list_poly, t_list_poly, x_list_poly, y_list_poly, bkg_list_poly, scales_list_poly, T0_list_poly = [],[],[],[],[],[],[],[]
lc_list_PLD, lcerr_list_PLD, t_list_PLD, centroids_list_PLD, timeseries_list_PLD, bkg_list_PLD, scales_list_PLD, T0_list_PLD = [],[],[],[],[],[],[],[]

for i in range(len(PP)): # loop over the AORs

    method = PP[i][0]
    channel = PP[i][2]

    AOR = PP[i][1]
    # Figure out which cutstarts and cutends to use
    cutstart = cutstarts[np.where(np.array(AORs) == AOR)[0][0]]
    cutend = cutends[np.where(np.array(AORs) == AOR)[0][0]]
    t0guess = t0s[np.where(np.array(AORs) == AOR)[0][0]]

    #AOR = 'r47032832'
    bkg_method = PP[i][4]
    #bkg_method = "Box"
    bkg_boxsize = None if PP[i][5] == 'None' else int(PP[i][5])
    #bkg_boxsize = 4
    bkg_annradius = None if PP[i][6] == 'None' else int(PP[i][6])
    bkg_annsize = None if PP[i][7] == 'None' else int(PP[i][7])
    cent_method = PP[i][8]
    cent_sizebary = None if PP[i][9] == 'None' else int(PP[i][9])
    photom_radius = float(PP[i][10])

    datapath = "{3}/PhD/SpitzerData/{0}/{1}/{2}/bcd/".format(planet,AOR,channel, os.getenv('HOME'))

    # Read in the planet information from a text file
    # Create dictionary and fill of the stellar parameters (for limb darkening)
    star_params = {'Teff':0,'logg':0,'z':0,'Tefferr':0,'loggerr':0,'zerr':0}
    for key in star_params:
        star_params[key] = float(inputData[int(np.where(inputData.T[0]==key)[0])][1])

    # Get limb darkening coefficients
    ldcoeffs, ldcoeffs_err = getldcoeffs(star_params['Teff'],star_params['logg'],star_params['z'],
                                         star_params['Tefferr'],star_params['loggerr'],star_params['zerr'],
                                         law = ldlaw, channel = channel, quiet = False)

    if method == 'poly':

        # Create a dictionary of the polynomial parameters...
        coeffs_tuple_poly = ('t0', 'per', 'rp', 'a', 'inc', 'ecc', 'w', 'u', 'limb_dark',
                       'K1', 'K2', 'K3', 'K4', 'K5',
                       'f', 'g', 'h')

        coeffs_dict_poly = dict()
        for label in coeffs_tuple_poly:
            if label != 'u':
                try:
                    coeffs_dict_poly[label] = float(inputData[int(np.where(inputData.T[0]==label)[0])][1].split(', ')[i/2])
                except:
                    try: # If there is nothing to split
                        coeffs_dict_poly[label] = float(inputData[int(np.where(inputData.T[0]==label)[0])][1])
                    except: # the limb darkening law
                        coeffs_dict_poly[label] = inputData[int(np.where(inputData.T[0]==label)[0])][1]

        # and a list of polynomial parameters to fix
        # this doesn't need to be in the loop
        fix_coeffs_poly = inputData[int(np.where(inputData.T[0]=='fixcoeffs_poly')[0])][1].split(', ')
        fix_coeffs_channels_poly = inputData[int(np.where(inputData.T[0]=='fixcoeffs_channels_poly')[0])][1].split(', ')
        if fix_coeffs_channels_poly == ',':
            fix_coeffs_channels_poly = []
        else:
            pass

        # Get the errors of the parameters that are used for calculating the depth and b
        try:
            if 'a' in fix_coeffs_poly:
                coeffs_dict_poly['a_err'] = float(inputData[int(np.where(inputData.T[0]=='a_err')[0])][1])
        except:
            raise ValueError("Trying to fix a without giving an error!")
        try:
            if 'inc' in fix_coeffs_poly:
                coeffs_dict_poly['inc_err'] = float(inputData[int(np.where(inputData.T[0]=='inc_err')[0])][1])
        except:
            raise ValueError("Trying to fix inclination without giving an error!")
        try:
            if 'per' in fix_coeffs_poly:
                coeffs_dict_poly['per_err'] = float(inputData[int(np.where(inputData.T[0]=='per_err')[0])][1])
        except:
            raise ValueError("Trying to fix period without giving an error!")

        coeffs_dict_poly['t0'] = t0guess
        coeffs_dict_poly['u'] = ldcoeffs # Must be list!

        # If the AOR is the same as the one before do not run the pipeline again (deleted)
        lightcurve, timeseries, centroids, midtimes, background = runFullPipeline(datapath, 4, 30,
                       method_bkg = bkg_method, method_cent = cent_method, plotting_binsize = 64,
                       ramp_time = cutstart*60, end_time = cutend*60,
                       sigma_clip_cent = 4, iters_cent = 2, nframes_cent=30,
                       radius_photom = photom_radius,
                       x0guess = posGuess[0], y0guess = posGuess[1],
                       sigma_clip_phot = 4, iters_photom = 2, nframes_photom=30,
                       size_bkg_box = bkg_boxsize, radius_bkg_ann = bkg_annradius, size_bkg_ann = bkg_annsize,
                       size_cent_bary = cent_sizebary, passenger57 = True,
                       quiet = False, plot = True, AOR = AOR, planet = planet, channel = channel, sysmethod = method, foldext='')

        lc = lightcurve
        lcerr = np.sqrt(lc)
        scale = np.median(lc[:100])
        print "\nGuess scale: {}".format(scale)
        lc, lcerr = lc/scale, lcerr/scale
        t = (midtimes - midtimes[0])
        x, y = centroids.T[1], centroids.T[0]

        # Initial fit to get a new scale and increase the errors
        result, batman_params, poly_params = fit_function_poly(coeffs_dict_poly, coeffs_tuple_poly,
                                                               fix_coeffs_poly, t, x, y, lc)

        # Calculate a new scale from after the initial first fit of the lightcurve
        Tdur = (batman_params.__dict__['per']/np.pi) * np.arcsin( np.sqrt( (1 + batman_params.__dict__['rp'])**2 - (batman_params.__dict__['a']*np.cos(batman_params.__dict__['inc']*np.pi/180.))**2 ) / batman_params.__dict__['a'] )
        t0 = batman_params.__dict__['t0']

        exclude0 = find_nearest(t,t0-Tdur/2.)
        exclude1 = find_nearest(t,t0+Tdur/2.)

        newlc = np.delete(lightcurve, xrange(exclude0,exclude1))
        scale = np.median(newlc)
        lc, lcerr = lightcurve/scale, np.sqrt(lightcurve)/scale

        print "\nNew scale poly: {}".format(scale)



        # Least squares again
        result, batman_params, poly_params = fit_function_poly(coeffs_dict_poly, coeffs_tuple_poly, fix_coeffs_poly, t, x, y, lc)
        popt = result.x

        # Plot the least squares result
        plot_lightcurve(t,  lc, lcerr, popt, coeffs_dict_poly, coeffs_tuple_poly, fix_coeffs_poly, batman_params, poly_params,
                            x=x, y=y, errors = False, binsize = 50,
                            name = planet, channel = channel, orbit = AOR, savefile = True, TT_hjd = None,
                            method = method, color = 'b', scale = scale, filext = "lsq")

        # Perform a sigma clip of the photometry
        t, x, y, lc, lcerr, bkg = sigma_clip_residuals(popt, t, background, lc, lcerr, coeffs_dict_poly,
                                                          coeffs_tuple_poly, fix_coeffs_poly, batman_params,
                                                          poly_params, 4, 30, x=x, y=y, quiet = False,
                                                          planet=planet, AOR=AOR, channel=channel, method=method,
                                                          foldext=foldext, plot=True)

        # Least squares again
        result, batman_params, poly_params = fit_function_poly(coeffs_dict_poly, coeffs_tuple_poly, fix_coeffs_poly, t, x, y, lc)
        popt = result.x

        # Inflate the errors
        newlcerr = inflate_errs(popt, t, lc, lcerr, coeffs_dict_poly, coeffs_tuple_poly, fix_coeffs_poly, batman_params, poly_params,x=x,y=y, method = method)

        # Write stuff to lists for looping
        coeffs_dict_poly_list.append(coeffs_dict_poly)
        lc_list_poly.append(lc)
        lcerr_list_poly.append(newlcerr)
        t_list_poly.append(t)
        x_list_poly.append(x)
        y_list_poly.append(y)
        bkg_list_poly.append(bkg)
        T0_list_poly.append(midtimes[0])
        scales_list_poly.append(scale)

    if method == 'PLD':

        # Create a dictionary of the PLD paramters...
        coeffs_tuple_PLD = ('t0', 'per', 'rp', 'a', 'inc', 'ecc', 'w', 'u', 'limb_dark',
                       'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9',
                       'g', 'h')
        coeffs_dict_PLD = dict()
        for label in coeffs_tuple_PLD:
            if label != 'u':
                try:
                    coeffs_dict_PLD[label] = float(inputData[int(np.where(inputData.T[0]==label)[0])][1].split(', ')[i/2])
                except:
                    try: # If there is nothing to split
                        coeffs_dict_PLD[label] = float(inputData[int(np.where(inputData.T[0]==label)[0])][1])
                    except: # the limb darkening law
                        coeffs_dict_PLD[label] = inputData[int(np.where(inputData.T[0]==label)[0])][1]

        # and a list of PLD parameters to fix
        # this doesn't need to be in the loop
        fix_coeffs_PLD = inputData[int(np.where(inputData.T[0]=='fixcoeffs_PLD')[0])][1].split(', ')
        fix_coeffs_channels_PLD = inputData[int(np.where(inputData.T[0]=='fixcoeffs_channels_PLD')[0])][1].split(', ')

        if fix_coeffs_channels_PLD == ',':
            fix_coeffs_channels_PLD = []
        else:
            pass

        # Get the errors of the parameters that are used for calculating the depth and b
        try:
            if 'a' in fix_coeffs_PLD:
                coeffs_dict_PLD['a_err'] = float(inputData[int(np.where(inputData.T[0]=='a_err')[0])][1])
        except:
            raise ValueError("Trying to fix a without giving an error!")
        try:
            if 'inc' in fix_coeffs_PLD:
                coeffs_dict_PLD['inc_err'] = float(inputData[int(np.where(inputData.T[0]=='inc_err')[0])][1])
        except:
            raise ValueError("Trying to fix inclination without giving an error!")
        try:
            if 'per' in fix_coeffs_PLD:
                coeffs_dict_PLD['per_err'] = float(inputData[int(np.where(inputData.T[0]=='per_err')[0])][1])
        except:
            raise ValueError("Trying to fix period without giving an error!")

        coeffs_dict_PLD['t0'] = t0guess
        coeffs_dict_PLD['u'] = ldcoeffs # Must be list!

        # If the AOR is the same as the one before do not run the pipeline again (deleted)
        lightcurve, timeseries, centroids, midtimes, background = runFullPipeline(datapath, 4, 30,
                       method_bkg = bkg_method, method_cent = cent_method, plotting_binsize = 64,
                       ramp_time = cutstart*60, end_time = cutend*60,
                       sigma_clip_cent = 4, iters_cent = 2, nframes_cent=30,
                       radius_photom = photom_radius,
                       x0guess = posGuess[0], y0guess = posGuess[1],
                       sigma_clip_phot = 4, iters_photom = 2, nframes_photom=30,
                       size_bkg_box = bkg_boxsize, radius_bkg_ann = bkg_annradius, size_bkg_ann = bkg_annsize,
                       size_cent_bary = cent_sizebary, passenger57 = True,
                       quiet = False, plot = True, AOR = AOR, planet = planet, channel = channel, sysmethod = method, foldext='')

        lc = lightcurve
        lcerr = np.sqrt(lc)
        scale = np.median(lc[:100])
        print "\nGuess scale: {}".format(scale)
        lc, lcerr = lc/scale, lcerr/scale
        t = (midtimes - midtimes[0])
        x, y = centroids.T[1], centroids.T[0]

        result_PLD, batman_params_PLD, PLD_params, Pns = fit_function_PLD(coeffs_dict_PLD, coeffs_tuple_PLD, fix_coeffs_PLD, t, timeseries, centroids, lc)
        popt_PLD = result_PLD.x

        # Calculate a new scale from after the initial first fit of the lightcurve
        Tdur = (batman_params_PLD.__dict__['per']/np.pi) * np.arcsin( np.sqrt( (1 + batman_params_PLD.__dict__['rp'])**2 - (batman_params_PLD.__dict__['a']*np.cos(batman_params_PLD.__dict__['inc']*np.pi/180.))**2 ) / batman_params_PLD.__dict__['a'] )
        t0 = batman_params_PLD.__dict__['t0']

        exclude0 = find_nearest(t,t0-Tdur/2.)
        exclude1 = find_nearest(t,t0+Tdur/2.)

        newlc = np.delete(lightcurve, xrange(exclude0,exclude1))
        scale = np.median(newlc)
        lc, lcerr = lightcurve/scale, np.sqrt(lightcurve)/scale

        print "\nNew scale PLD: {}".format(scale)

        # Least squares again
        result_PLD, batman_params_PLD, PLD_params, Pns = fit_function_PLD(coeffs_dict_PLD, coeffs_tuple_PLD, fix_coeffs_PLD, t, timeseries, centroids, lc)
        popt_PLD = result_PLD.x

        # Plot the least squares result
        plot_lightcurve(t,  lc, lcerr, popt_PLD, coeffs_dict_PLD, coeffs_tuple_PLD, fix_coeffs_PLD, batman_params_PLD, PLD_params,
                            Pns = Pns, errors = False, binsize = 50,
                            name = planet, channel = channel, orbit=AOR, savefile = True, TT_hjd = None,
                            method = method, color = 'r', scale = scale, filext = "lsq")

        # Perform a sigma clip of the photometry
        t, x, y, Pns, lc, lcerr, bkg, timeseries, centroids = sigma_clip_residuals(popt_PLD, t, background, lc, lcerr, coeffs_dict_poly,
                                                          coeffs_tuple_PLD, fix_coeffs_PLD, batman_params_PLD,
                                                          PLD_params, 4, 30, x=x,y=y,Pns=Pns, quiet = False,
                                                          planet=planet, AOR=AOR, channel=channel, method=method,
                                                          foldext=foldext, plot=True, timeseries=timeseries, centroids=centroids)

        # Least squares again
        result_PLD, batman_params_PLD, PLD_params, Pns = fit_function_PLD(coeffs_dict_PLD, coeffs_tuple_PLD, fix_coeffs_PLD, t, timeseries, centroids, lc)
        popt_PLD = result_PLD.x

        labels_PLD = [ key for key in coeffs_tuple_PLD if key not in fix_coeffs_PLD ]

        # Inflate the errors
        newlcerr = inflate_errs(popt_PLD, t, lc, lcerr, coeffs_dict_PLD, coeffs_tuple_PLD, fix_coeffs_PLD, batman_params_PLD, PLD_params, Pns=Pns, method = method)

        # Write stuff to lists for looping
        coeffs_dict_PLD_list.append(coeffs_dict_PLD)
        lc_list_PLD.append(lc)
        lcerr_list_PLD.append(newlcerr)
        t_list_PLD.append(t)
        centroids_list_PLD.append(centroids)
        timeseries_list_PLD.append(timeseries)
        bkg_list_PLD.append(bkg)
        T0_list_PLD.append(midtimes[0])
        scales_list_PLD.append(scale)

# polynomial
opt_poly, batman_params_list_poly, poly_params_list =  fit_function_poly_Ntransits(coeffs_dict_poly_list, coeffs_tuple_poly,
                                fix_coeffs_poly, fix_coeffs_channels_poly,
                                t_list_poly, x_list_poly, y_list_poly, lc_list_poly)

x1 = len([key for key in coeffs_tuple_poly[0:9] if key not in fix_coeffs_poly
           and key not in fix_coeffs_channels_poly])
x2 = len([key for key in coeffs_tuple_poly[0:9] if key in fix_coeffs_channels_poly])
x3 = len([key for key in coeffs_tuple_poly if key not in fix_coeffs_poly
                      and key not in fix_coeffs_channels_poly])

for i in range(len(coeffs_dict_poly_list)):

    coeffs_fit = np.concatenate((opt_poly[i*x3:i*x3 + x1],
                                   opt_poly[-x2:],
                                   opt_poly[i*x3 + x1 :(i+1)*x3 ]))

    plot_lightcurve(t_list_poly[i],  lc_list_poly[i], lcerr_list_poly[i], coeffs_fit, coeffs_dict_poly_list[i],
                    coeffs_tuple_poly, fix_coeffs_poly, batman_params_list_poly[i], poly_params_list[i],
                    x=x_list_poly[i], y=y_list_poly[i], errors = True, binsize = 50,
                    name = planet, channel = PP[i*2][2], orbit = PP[i*2][1], savefile = True, TT_hjd = None,
                    method = PP[i*2][0], color = 'b', scale = scale, filext = "lsq_joint")

# PLD least squares
opt_PLD,batman_params_list_PLD,PLD_params_list,Pns_list =  fit_function_PLD_Ntransits(coeffs_dict_PLD_list, coeffs_tuple_PLD,
                                fix_coeffs_PLD, fix_coeffs_channels_PLD,
                                t_list_PLD, timeseries_list_PLD, centroids_list_PLD, lc_list_PLD)

x1 = len([key for key in coeffs_tuple_PLD[0:9] if key not in fix_coeffs_PLD
           and key not in fix_coeffs_channels_PLD])
x2 = len([key for key in coeffs_tuple_PLD[0:9] if key in fix_coeffs_channels_PLD])
x3 = len([key for key in coeffs_tuple_PLD if key not in fix_coeffs_PLD
                      and key not in fix_coeffs_channels_PLD])

for i in range(len(coeffs_dict_PLD_list)):

    coeffs_fit = np.concatenate((opt_PLD[i*x3:i*x3 + x1],
                                   opt_PLD[-x2:],
                                   opt_PLD[i*x3 + x1 :(i+1)*x3 ]))

    plot_lightcurve(t_list_PLD[i],  lc_list_PLD[i], lcerr_list_PLD[i], coeffs_fit, coeffs_dict_PLD_list[i],
                    coeffs_tuple_PLD, fix_coeffs_PLD, batman_params_list_PLD[i], PLD_params_list[i],
                    Pns = Pns_list[i], errors = False, binsize = 50,
                    name = planet, channel = PP[i*2+1][2], orbit = PP[i*2+1][1], savefile = True, TT_hjd = None,
                    method = PP[i*2+1][0], color = 'r', scale = scale, filext = "lsq_joint")

# Make labels
labels_poly, labels_PLD = [],[]
for i in range(len(coeffs_dict_poly_list)):
    labels_poly += [ '{}_{}'.format(key,i) for key in coeffs_tuple_poly if key not in fix_coeffs_poly
                  and key not in fix_coeffs_channels_poly]
    labels_PLD += [ '{}_{}'.format(key,i) for key in coeffs_tuple_PLD if key not in fix_coeffs_PLD
                 and key not in fix_coeffs_channels_PLD]

labels_poly += [ '{}'.format(key) for key in coeffs_tuple_poly if key in fix_coeffs_channels_poly]
labels_PLD += [ '{}'.format(key) for key in coeffs_tuple_PLD if key in fix_coeffs_channels_PLD]


# polynomial mcmc
bounds_list = []
for i in range(len(coeffs_dict_poly_list)):
    bounds_list.append(make_bounds(coeffs_tuple_poly, fix_coeffs_poly, t_list_poly[i], fix_coeffs_channels_poly))
bounds_list.append(make_bounds(coeffs_tuple_poly, fix_coeffs_poly, t_list_poly[i], fix_coeffs_channels_poly, normal=False))
bounds_poly = np.concatenate([np.array(o) for o in bounds_list],axis=1).tolist()

data_poly = (t_list_poly, x_list_poly,y_list_poly, lc_list_poly, lcerr_list_poly, bounds_poly,
        coeffs_dict_poly_list, coeffs_tuple_poly, fix_coeffs_poly,fix_coeffs_channels_poly,
        batman_params_list_poly, poly_params_list)

sampler = mcmc_poly_Ntransits(opt_poly, data_poly, burnin_steps = burninsteps, production_steps = runsteps)



# PLD mcmc
bounds_list = []
for i in range(len(coeffs_dict_poly_list)):
    bounds_list.append(make_bounds(coeffs_tuple_PLD, fix_coeffs_PLD, t_list_PLD[i], fix_coeffs_channels_PLD))
bounds_list.append(make_bounds(coeffs_tuple_PLD, fix_coeffs_PLD, t_list_PLD[i], fix_coeffs_channels_PLD, normal=False))
bounds_PLD = np.concatenate([np.array(o) for o in bounds_list],axis=1).tolist()

data_PLD = (t_list_PLD, Pns_list, lc_list_PLD, lcerr_list_PLD, bounds_PLD,
        coeffs_dict_PLD_list, coeffs_tuple_PLD, fix_coeffs_PLD, fix_coeffs_channels_PLD,
        batman_params_list_PLD, PLD_params_list)

sampler_PLD = mcmc_PLD_Ntransits(opt_PLD, data_PLD, burnin_steps = burninsteps, production_steps = runsteps)



# Plot the results nicely
leastsquares, averages, medians, stdevs, rmss, chi2s, BICs, poserrs, negerrs = [],[],[],[],[],[],[],[],[]

for i in range(len(coeffs_dict_PLD_list)):
    avgs, stds, meds, pos, neg, rms, chi2, bic = mcmc_results(sampler, opt_poly,
                t_list_poly[i], lc_list_poly[i], lcerr_list_poly[i],
                x_list_poly[i], y_list_poly[i], None, bkg_list_poly[i],
                coeffs_dict_poly_list[i], coeffs_tuple_poly,
                fix_coeffs_poly, batman_params_list_poly[i], poly_params_list[i],
                scales_list_poly[i], labels_poly, planet,
                PP[i*2][1], PP[i*2][2], PP[i*2][0], T0_list_poly[i], saveplots = True, foldext = foldext,
                fix_coeffs_channels = fix_coeffs_channels_poly, AOR_No = i )

    leastsquares.append(opt_poly)
    averages.append(avgs)
    stdevs.append(stds)
    rmss.append(rms)
    chi2s.append(chi2)
    BICs.append(bic)
    medians.append(meds)
    poserrs.append(pos)
    negerrs.append(neg)

    avgs, stds, meds, pos, neg, rms, chi2, bic = mcmc_results(sampler_PLD, opt_PLD,
                t_list_PLD[i], lc_list_PLD[i], lcerr_list_PLD[i],
                centroids_list_PLD[i][:,0],centroids_list_PLD[i][:,1], Pns_list[i], bkg_list_PLD[i],
                coeffs_dict_PLD_list[i], coeffs_tuple_PLD,
                fix_coeffs_PLD, batman_params_list_PLD[i], PLD_params_list[i],
                scales_list_PLD[i], labels_PLD, planet,
                PP[i*2+1][1], PP[i*2+1][2], PP[i*2+1][0], T0_list_PLD[i], saveplots = True, foldext = foldext,
                fix_coeffs_channels = fix_coeffs_channels_PLD, AOR_No = i )

    leastsquares.append(opt_PLD)
    averages.append(avgs)
    stdevs.append(stds)
    rmss.append(rms)
    chi2s.append(chi2)
    BICs.append(bic)
    medians.append(meds)
    poserrs.append(pos)
    negerrs.append(neg)


resultsfilepath = "{2}/PhD/SpitzerTransits/{0}{1}/{0}_results.txt".format(planet,foldext, os.getenv('HOME'))
resultsfile = open(resultsfilepath,'w')

fitted_params = [key for key in coeffs_tuple_poly if key in batman_params.__dict__ and key not in fix_coeffs_poly]

p = [", {0}_lqs, {0}_mu, {0}_med, {0}_std, {0}_poserr, {0}_negerr".format(lab) for lab in fitted_params]
p = ["AOR, channel, method, chi2, rms, BIC"] + p
print >>resultsfile, ''.join(p)

for i in range(len(PP)):
    l = [", {0}, {1}, {2}, {3}, {4}, {5}".format(leastsquares[i][lab], averages[i][lab], medians[i][lab], stdevs[i][lab],poserrs[i][lab],negerrs[i][lab]) for lab in range(len(fitted_params))]
    l = [", {0}, {1}, {2}".format(chi2s[i], rmss[i], BICs[i])] + l
    l = ["{}, {}, {}".format(PP[i][1], PP[i][2], PP[i][0])] + l
    print >>resultsfile, ''.join(l)

resultsfile.close()

# Plot the final parameters
parameter_plots(resultsfilepath, fitted_params, "Median", planet, True,
                "{1}/PhD/SpitzerData/{0}/{0}_publishedTransit.txt".format(planet, os.getenv('HOME')),
                saveplot = True, foldext = foldext)


sigma_sec_PLD, nsigma_PLD = t0check('{2}/PhD/SpitzerTransits/{0}{1}/{0}_results.txt'.format(planet,foldext, os.getenv('HOME')), 'Mean','PLD', coeffs_dict_PLD['per'], coeffs_dict_PLD['per_err'])

print "\nChecking timing for PLD..."
print "\tSigma = {:.2f} seconds".format(sigma_sec_PLD)
print "\tNsigma between t0s = {:.2f}".format(nsigma_PLD)


sigma_sec_poly, nsigma_poly = t0check('{2}/PhD/SpitzerTransits/{0}{1}/{0}_results.txt'.format(planet,foldext, os.getenv('HOME')), 'Mean','poly', coeffs_dict_poly['per'], coeffs_dict_poly['per_err'])

print "\nChecking timing for poly..."
print "\tSigma = {:.2f} seconds".format(sigma_sec_poly)
print "\tNsigma between t0s = {:.2f}".format(nsigma_poly)
