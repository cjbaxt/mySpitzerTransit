# imports
from astropy.io import fits
import numpy as np
import glob, os, sys, time
import matplotlib.pyplot as plt
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

# Custom imports
sys.path.insert(0, '{}/PhD/code/mySpitzerTransit/'.format(os.getenv('HOME')))
from timeseries_routines import *
from transitFitting_routines import *
sys.path.insert(1, '{}/PhD/code/'.format(os.getenv('HOME')))
from ProgressBar import *
foldext = sys.argv[5]
filext=''

# Get input file
try:
    planetinFile = sys.argv[1]
    pipelineinFile = sys.argv[2]
    ncuts = int(sys.argv[3])
    cuttime = int(sys.argv[4]) # Minutes
except:
    raise ValueError('You need to provide input planet file and pipeline file as arguements followed by the ncuts and cuttime.')

print "Using planet data from {}".format(planetinFile)
print "Using pipeline information from {}".format(pipelineinFile)
print "Going to make {0} cuts of {1} minutes".format(ncuts, cuttime)

# Read in the planet information from a text file
inputData = np.genfromtxt(planetinFile, dtype=None, delimiter=': ', comments='#', encoding = None)
planet = inputData[0][1]
AORs = inputData[1][1].split(', ')
channels = inputData[2][1].split(', ')
eclipses = inputData[3][1].split(', ')
t0s = inputData[4][1].split(', ')
cutstarts = [float(x) for x in inputData[5][1].split(', ')]
cutends = [float(x) for x in inputData[6][1].split(', ')]
posGuess = [float(x) for x in inputData[7][1].split(', ')]

ldlaw = inputData[int(np.where(inputData.T[0]=='limb_dark')[0])][1]

PP = np.genfromtxt(pipelineinFile, dtype = None, skip_header = 1, delimiter = ', ', comments = '#', encoding = None)

# Loop over the AOR with each model
for m in range(len(PP)):

    if 'E' in eclipses[m/2]:
        eclipse = True
    else:
        eclipse = False

    star_params = {'Teff':0,'logg':0,'z':0,'Tefferr':0,'loggerr':0,'zerr':0}
    for key in star_params:
        star_params[key] = float(inputData[int(np.where(inputData.T[0]==key)[0])][1])

    # Create a dictionary of the polynomial parameters...
    if eclipse:
        coeffs_tuple_poly = ('t_secondary', 'fp','per', 'rp', 'a', 'inc', 'ecc', 'w', 'u', 'limb_dark',
                       'K1', 'K2', 'K3', 'K4', 'K5',
                       'f', 'g', 'h')
        fix_coeffs_poly = inputData[int(np.where(inputData.T[0]=='fixcoeffs_poly_E')[0])][1].split(', ')
    else:
        coeffs_tuple_poly = ('t0', 'per', 'rp', 'a', 'inc', 'ecc', 'w', 'u', 'limb_dark',
                       'K1', 'K2', 'K3', 'K4', 'K5',
                       'f', 'g', 'h')
        fix_coeffs_poly = inputData[int(np.where(inputData.T[0]=='fixcoeffs_poly')[0])][1].split(', ')

    coeffs_dict_poly = dict()
    for label in coeffs_tuple_poly:
        if label != 'u' and label != 't_secondary':
            try:
                coeffs_dict_poly[label] = float(inputData[int(np.where(inputData.T[0]==label)[0])][1].split(', ')[m/2])
            except:
                try: # If there is nothing to split
                    coeffs_dict_poly[label] = float(inputData[int(np.where(inputData.T[0]==label)[0])][1])
                except: # the limb darkening law
                    coeffs_dict_poly[label] = inputData[int(np.where(inputData.T[0]==label)[0])][1]


    # Create a dictionary of the PLD paramters...
    if eclipse:
        coeffs_tuple_PLD = ('t_secondary', 'fp','per', 'rp', 'a', 'inc', 'ecc', 'w', 'u', 'limb_dark',
                       'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9',
                       'g', 'h')
        fix_coeffs_PLD = inputData[int(np.where(inputData.T[0]=='fixcoeffs_PLD_E')[0])][1].split(', ')
    else:
        coeffs_tuple_PLD = ('t0', 'per', 'rp', 'a', 'inc', 'ecc', 'w', 'u', 'limb_dark',
                       'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9',
                       'g', 'h')
        fix_coeffs_PLD = inputData[int(np.where(inputData.T[0]=='fixcoeffs_PLD')[0])][1].split(', ')

    coeffs_dict_PLD = dict()
    for label in coeffs_tuple_PLD:
        if label != 'u' and label != 't_secondary':
            try:
                coeffs_dict_PLD[label] = float(inputData[int(np.where(inputData.T[0]==label)[0])][1].split(', ')[m/2])
            except:
                try: # If there is nothing to split
                    coeffs_dict_PLD[label] = float(inputData[int(np.where(inputData.T[0]==label)[0])][1])
                except: # the limb darkening law
                    coeffs_dict_PLD[label] = inputData[int(np.where(inputData.T[0]==label)[0])][1]

    # Prior coefficients
    try:
        prior_coeffs = inputData[int(np.where(inputData.T[0]=='prior_coeffs')[0])][1].split(', ')
    except:
        prior_coeffs = []

    if len(prior_coeffs) > 1.:
        gaussian_priors = True
    else:
        gaussian_priors = False

    method = PP[m][0]
    channel = PP[m][2]
    AOR = PP[m][1]
    bkg_method = PP[m][4]
    bkg_boxsize = None if PP[m][5] == 'None' else int(PP[m][5])
    bkg_annradius = None if PP[m][6] == 'None' else int(PP[m][6])
    bkg_annsize = None if PP[m][7] == 'None' else int(PP[m][7])
    cent_method = PP[m][8]
    cent_sizebary = None if PP[m][9] == 'None' else int(PP[m][9])
    photom_radius = float(PP[m][10])
    binsize = float(PP[m][11])

    # Get the interpolated limb darkening coefficients
    ldcoeffs, ldcoeffs_err = getldcoeffs(star_params['Teff'],star_params['logg'],star_params['z'],
                                         star_params['Tefferr'],star_params['loggerr'],star_params['zerr'],
                                         law = ldlaw, channel = channel)

    coeffs_dict_poly['u'], coeffs_dict_PLD['u'] = ldcoeffs, ldcoeffs # Must be list!
    #coeffs_dict_poly['t0'], coeffs_dict_PLD['t0'] = float(t0s[m]), float(t0s[m])
    # Path to get the data
    path = "{3}/PhD/SpitzerData/{0}/{1}/{2}/bcd/".format(planet,AOR,channel, os.getenv('HOME'))

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
    sigma_badpix = 4
    nframes = 30
    timeseries_badpix = fast_bad_pix_mask(timeseries, sigma_badpix, nframes)

    lightcurve_red, timeseries_red, centroids_red, midtimes_red, background_red =  runPipeline(timeseries_badpix, midtimes,
           method_bkg = bkg_method, method_cent = cent_method,
           ramp_time = 0., end_time = 0.,
           frametime = framtime,
           x0guess = posGuess[0], y0guess = posGuess[1],
           sigma_clip_cent = 4, iters_cent = 2, nframes_cent = 30, radius_photom = photom_radius,
           sigma_clip_phot = 4, iters_photom = 2, nframes_photom = 30,
           size_bkg_box = bkg_boxsize , radius_bkg_ann = bkg_annradius, size_bkg_ann = bkg_annsize,
           size_cent_bary = cent_sizebary, quiet = False, sysmethod = method, binsize_methods_params=binsize)

    cutstart_poly, avgs_poly, rmss_poly, chi2s_poly = [],[],[],[]
    cutstart_PLD, avgs_PLD, rmss_PLD, chi2s_PLD = [],[],[],[]

    #POLYNOMIAL
    if method == 'poly':
        # Make a plot of the lightcurve with lines where the cuts are

        for n in range(ncuts):
            print '\nCutting {} minutes from beginning of lightcurve'.format(n*cuttime)

            ncutframes = n*cuttime*60/framtime

            ncutframes = int(ncutframes)

            # Bin the lightcurves and timeseries
            #lightcurve_red = custom_bin(lightcurve_red, binsize_methods_params[l])
            timeseries_red = custom_bin(timeseries_red, binsize)
            centroids_red = custom_bin(centroids_red, binsize)
            midtimes_red = custom_bin(midtimes_red, binsize)
            background_red = custom_bin(background_red, binsize)

            timeseries, centroids, midtimes, background = timeseries_red[ncutframes/binsize:], centroids_red[ncutframes/binsize:], midtimes_red[ncutframes/binsize:], background_red[ncutframes/binsize:]


            # Don't bin the lightcruve right away, we need to get the errors first
            lc_unbinned = lightcurve_red*MJysr2lelectrons
            lcerr_unbinned = np.sqrt(lc_unbinned)

            # Bin the lightcurve and propagate the binning to the errors
            # Just taking the average of the errors would result in the errors being way too large for each of the datapoints
            # Shld actually check this by plotting it
            lc = custom_bin(lc_unbinned[ncutframes:], binsize_methods_params[l])
            lcerr = custom_bin(lcerr_unbinned[ncutframes:], binsize_methods_params[l], error = True)
            scale = np.median(lc[:int(100/binsize_methods_params[l])])
            lc, lcerr = lc/scale, lcerr/scale
            t = (midtimes - midtimes[0])
            x, y = centroids[:,1], centroids[:,0]


            if eclipse:
                coeffs_dict_poly['t_secondary'], coeffs_dict_PLD['t_secondary'] = float(t0s[m/2]), float(t0s[m/2])
            else:
                coeffs_dict_poly['t0'], coeffs_dict_PLD['t0'] = float(t0s[m/2]), float(t0s[m/2])

            result, batman_params_poly, poly_params = fit_function_poly(coeffs_dict_poly,
                                                        coeffs_tuple_poly, fix_coeffs_poly, t, x, y, lc, eclipse = eclipse)
            popt = result.x

            labels_poly = [ key for key in coeffs_tuple_poly if key not in fix_coeffs_poly ]
            fitted_params_poly = [key for key in batman_params_poly.__dict__ if key in labels_poly]

            # Inflate the errors
            newlcerr = inflate_errs(popt, t, lc, lcerr, coeffs_dict_poly, coeffs_tuple_poly,
            fix_coeffs_poly, batman_params_poly, poly_params,x=x,y=y, method = method, eclipse = eclipse)

            # Run MCMC
            bounds = make_bounds(coeffs_tuple_poly, fix_coeffs_poly, t)
            data = (t, x,y, lc, newlcerr, bounds, coeffs_dict_poly, coeffs_tuple_poly, fix_coeffs_poly, batman_params_poly, poly_params, gaussian_priors, prior_coeffs, eclipse)
            sampler = mcmc_poly(popt, data, nwalkers = 50, burnin_steps = 500, production_steps = 1000)

            samples = sampler.chain
            print ("\nGelman-Rubin diagnostics for polynomial")
            print ("\tParam\tlsq_mu\t\tmu\t\tsigma\tGelman-Rubin")
            print ("\t=============================================================")
            for var, lsq, chain in list(zip(labels_poly, popt, ([samples[:,:,i:i+1] for i in range(samples.shape[2])]))):
                print ("\t{0:}\t{1: 1.3e}\t{2: 1.3e}\t{3: 1.3e}\t{4:1.4g}".format(
                    var,
                    lsq,
                    chain.reshape(-1, chain.shape[-1]).mean(axis=0)[0],
                    chain.reshape(-1, chain.shape[-1]).std(axis=0)[0],
                    gelman_rubin(chain)[0]))

            if 'inc' not in fix_coeffs_poly:
                # We need this try/except for the cases when we have fixed the inclination...
                samples = fold_inc(sampler, "chain", labels_poly)
                samples_fc = fold_inc(sampler, "flatchain", labels_poly)
            else:
                samples = sampler.chain
                samples_fc = sampler.flatchain

            if eclipse:
                if 't_secondary' not in fix_coeffs_poly:
                    samples_fc = t0_Tinitial(samples_fc, "flatchain", labels_poly, n*cuttime/(60*24))
                    samples = t0_Tinitial(samples, "chain", labels_poly, n*cuttime/(60*24))
                else:
                    pass
            else:
                if 't0' not in fix_coeffs_poly:
                    samples_fc = t0_Tinitial(samples_fc, "flatchain", labels_poly, n*cuttime/(60*24))
                    samples = t0_Tinitial(samples, "chain", labels_poly, n*cuttime/(60*24))
                else:
                    pass

            # Calculate the median and standard deviation
            meds = np.median(samples_fc,axis=0)
            stds = np.std(samples_fc,axis=0)

            # Calculate the residuals and the rms
            optflux = model_poly(popt, t, x, y, coeffs_dict_poly, coeffs_tuple_poly, fix_coeffs_poly, batman_params_poly, poly_params, components = False, eclipse = eclipse)
            residuals = lc - optflux
            rms = np.sqrt(np.sum(residuals**2)/len(residuals))
            chi2 = chi(popt, t, lc, lcerr, coeffs_dict_poly, coeffs_tuple_poly, fix_coeffs_poly, batman_params_poly, poly_params, x=x,y=y, method = 'poly', eclipse = eclipse)/(len(lc)-len(popt))

            cutstart_poly.append(ncutframes)
            avgs_poly.append([meds[:len(fitted_params_poly)], stds[:len(fitted_params_poly)]])
            rmss_poly.append(rms)
            chi2s_poly.append(chi2)

            if n == 0:
                plot_lightcurve(t,  lc, lcerr, popt, coeffs_dict_poly, coeffs_tuple_poly, fix_coeffs_poly, batman_params_poly, poly_params,
                                x=x, y=y, errors = False, binsize = 50,
                                name = planet, channel = channel, orbit = AOR, savefile = True, TT_hjd = None,
                                method = "poly", color = 'b', scale = scale, filext = "cutStart", foldext=foldext,
                                showCuts = True, ncutstarts = ncuts, cutstartTime = cuttime, eclipse = eclipse)

        fig, ax = plt.subplots(1, len(fitted_params_poly)+1, figsize=(5*(len(fitted_params_poly)+1), 5))
        for k in range(len(fitted_params_poly)):
            print k
            ax[k].errorbar(np.array(cutstart_poly)*(framtime/(60)), np.array(avgs_poly)[:,0][:,k],
                                    yerr = np.array(avgs_poly)[:,1][:,k], linestyle='', color='b', marker='o')
            ax[k].set_title(labels_poly[k])
            ax[k].set_xlabel("Cut-time (min)")

        ax[-1].plot(np.array(cutstart_poly)*(framtime/(60)), chi2s_poly,  color='b', marker='o')
        ax2 = ax[-1].twinx()
        ax2.plot(np.array(cutstart_poly)*(framtime/(60)), rmss_poly,  color='c', marker='o')
        ax[-1].set_ylabel('Red Chi2', color='b')
        ax2.set_ylabel('rms', color='c')
        plt.tight_layout()
        plt.savefig("{5}/PhD/SpitzerTransits/{0}{4}/{0}_{1}_{2}_{3}_ParamsVScuttime.png".format(planet, AOR, channel, 'poly', foldext, os.getenv('HOME')))
        plt.close()

    #PLD
    elif method == 'PLD':
        for n in range(ncuts):
            print '\nCutting {} minutes from beginning of lightcurve'.format(n*cuttime)

            ncutframes = n*cuttime*60/framtime

            ncutframes = int(ncutframes)

            # Bin the lightcurves and timeseries
            #lightcurve_red = custom_bin(lightcurve_red, binsize_methods_params[l])
            timeseries_red = custom_bin(timeseries_red, binsize)
            centroids_red = custom_bin(centroids_red, binsize)
            midtimes_red = custom_bin(midtimes_red, binsize)
            background_red = custom_bin(background_red, binsize)

            timeseries, centroids, midtimes, background = timeseries_red[ncutframes/binsize:], centroids_red[ncutframes/binsize:], midtimes_red[ncutframes/binsize:], background_red[ncutframes/binsize:]

            # Don't bin the lightcruve right away, we need to get the errors first
            lc_unbinned = lightcurve_red*MJysr2lelectrons
            lcerr_unbinned = np.sqrt(lc_unbinned)

            # Bin the lightcurve and propagate the binning to the errors
            # Just taking the average of the errors would result in the errors being way too large for each of the datapoints
            # Shld actually check this by plotting it
            lc = custom_bin(lc_unbinned[ncutframes:], binsize_methods_params[l])
            lcerr = custom_bin(lcerr_unbinned[ncutframes:], binsize_methods_params[l], error = True)
            scale = np.median(lc[:int(100/binsize_methods_params[l])])
            lc, lcerr = lc/scale, lcerr/scale
            t = (midtimes - midtimes[0])
            x, y = centroids[:,1], centroids[:,0]

            if eclipse:
                coeffs_dict_poly['t_secondary'], coeffs_dict_PLD['t_secondary'] = float(t0s[m/2]), float(t0s[m/2])
            else:
                coeffs_dict_poly['t0'], coeffs_dict_PLD['t0'] = float(t0s[m/2]), float(t0s[m/2])

            result_PLD, batman_params_PLD, PLD_params, Pns = fit_function_PLD(coeffs_dict_PLD, coeffs_tuple_PLD, fix_coeffs_PLD, t, timeseries, centroids, lc, eclipse = eclipse)
            popt_PLD = result_PLD.x

            # Labels of all of the fitted parameters
            labels_PLD = [ key for key in coeffs_tuple_PLD if key not in fix_coeffs_PLD ]
            fitted_params_PLD = [key for key in batman_params_PLD.__dict__ if key in labels_PLD]

            # Inflate the errors
            newlcerr = inflate_errs(popt_PLD, t, lc, lcerr, coeffs_dict_PLD, coeffs_tuple_PLD, fix_coeffs_PLD, batman_params_PLD, PLD_params, Pns=Pns, method = method, eclipse = eclipse)

            # Run MCMC
            bounds = make_bounds(coeffs_tuple_PLD, fix_coeffs_PLD, t)
            data = (t, Pns, lc, newlcerr, bounds, coeffs_dict_PLD, coeffs_tuple_PLD,
            fix_coeffs_PLD, batman_params_PLD, PLD_params, gaussian_priors, prior_coeffs, eclipse)
            sampler = mcmc_PLD(popt_PLD, data, nwalkers = 50, burnin_steps = 500, production_steps = 1000)

            samples = sampler.chain
            print ("\nGelman-Rubin diagnostics for polynomial")
            print ("\tParam\tlsq_mu\t\tmu\t\tsigma\tGelman-Rubin")
            print ("\t=============================================================")
            for var, lsq, chain in list(zip(labels_PLD, popt_PLD, ([samples[:,:,i:i+1] for i in range(samples.shape[2])]))):
                print ("\t{0:}\t{1: 1.3e}\t{2: 1.3e}\t{3: 1.3e}\t{4:1.4g}".format(
                    var,
                    lsq,
                    chain.reshape(-1, chain.shape[-1]).mean(axis=0)[0],
                    chain.reshape(-1, chain.shape[-1]).std(axis=0)[0],
                    gelman_rubin(chain)[0]))

            if 'inc' not in fix_coeffs_PLD:
                # We need this try/except for the cases when we have fixed the inclination...
                samples = fold_inc(sampler, "chain", labels_PLD)
                samples_fc = fold_inc(sampler, "flatchain", labels_PLD)
            else:
                samples = sampler.chain
                samples_fc = sampler.flatchain

            if eclipse:
                if 't_secondary' not in fix_coeffs_PLD:
                    samples_fc = t0_Tinitial(samples_fc, "flatchain", labels_PLD, n*cuttime/(60*24))
                    samples = t0_Tinitial(samples, "chain", labels_PLD, n*cuttime/(60*24))
                else:
                    pass
            else:
                if 't0' not in fix_coeffs_PLD:
                    samples_fc = t0_Tinitial(samples_fc, "flatchain", labels_PLD, n*cuttime/(60*24))
                    samples = t0_Tinitial(samples, "chain", labels_PLD, n*cuttime/(60*24))
                else:
                    pass

            # Calculate the median and standard deviation
            meds = np.median(samples_fc,axis=0)
            stds = np.std(samples_fc,axis=0)

            # Find index of rp and save to an array

            # Calculate the residuals and the rms
            optflux = model_PLD(popt_PLD, t, Pns, coeffs_dict_PLD, coeffs_tuple_PLD, fix_coeffs_PLD, batman_params_PLD, PLD_params, components = False, eclipse = eclipse)
            residuals = lc - optflux
            rms = np.sqrt(np.sum(residuals**2)/len(residuals))
            chi2 = chi(popt_PLD, t, lc, lcerr, coeffs_dict_PLD, coeffs_tuple_PLD, fix_coeffs_PLD, batman_params_PLD, PLD_params, Pns=Pns, method = 'PLD', eclipse = eclipse)/(len(lc)-len(popt))

            cutstart_PLD.append(ncutframes)
            avgs_PLD.append([meds[:len(fitted_params_PLD)], stds[:len(fitted_params_PLD)]])
            rmss_PLD.append(rms)
            chi2s_PLD.append(chi2)

            # Plot the lightcurve with the sections cut out
            if n == 0:
                plot_lightcurve(t,  lc, lcerr, popt_PLD, coeffs_dict_PLD, coeffs_tuple_PLD, fix_coeffs_PLD, batman_params_PLD, PLD_params,
                                Pns = Pns, errors = False, binsize = 50,
                                name = planet, channel = channel, orbit=AOR, savefile = True, TT_hjd = None,
                                method = "PLD", color = 'r', scale = scale, filext = "cutStart", foldext=foldext,
                                showCuts = True, ncutstarts = ncuts, cutstartTime = cuttime, eclipse = eclipse)

        fig, ax = plt.subplots(1, len(fitted_params_PLD)+1, figsize=(5*(len(fitted_params_PLD)+1), 5))
        for k in range(len(fitted_params_PLD)):
            ax[k].errorbar(np.array(cutstart_PLD)*(framtime/(60)), np.array(avgs_PLD)[:,0][:,k],
                           yerr = np.array(avgs_PLD)[:,1][:,k], linestyle='', color='r', marker='o')
            ax[k].set_title(labels_PLD[k])
            ax[k].set_xlabel("Cut-time (min)")
        ax[-1].plot(np.array(cutstart_PLD)*(framtime/(60)), chi2s_PLD,  color='r', marker='o')
        ax2 = ax[-1].twinx()
        ax2.plot(np.array(cutstart_PLD)*(framtime/(60)), rmss_PLD,  color='m', marker='o')
        ax[-1].set_ylabel('Red Chi2', color='r')
        ax2.set_ylabel('rms', color='m')
        ax2.set_xlabel("Cut-time (min)")
        plt.tight_layout()
        plt.savefig("{5}/PhD/SpitzerTransits/{0}{4}/{0}_{1}_{2}_{3}_ParamsVScuttime.png".format(planet, AOR, channel, 'PLD', foldext, os.getenv('HOME')))
        plt.close()
