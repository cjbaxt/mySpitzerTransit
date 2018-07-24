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
import pickle

print(sys.version)
print "this is the eclipse development version "

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
    nwalkers = int(sys.argv[5])
    foldext = sys.argv[6]
except:
    raise ValueError('You need to provide input planetFile pipelineFile burning runsteps foldext.')

try:
    publishedDataFile = sys.argv[7]
    plotPublished = True
    print "Using literature planet data from {}".format(publishedDataFile)
except:
    publishedDataFile = None
    plotPublished = False

print "Using planet data from {}".format(planetinFile)
print "Using pipeline information from {}".format(pipelineinFile)

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
T0_bjd = float(inputData[int(np.where(inputData.T[0]=='T0_BJD')[0])][1]) - 2400000.5
period = float(inputData[int(np.where(inputData.T[0]=='per')[0])][1])

ldlaw = inputData[int(np.where(inputData.T[0]=='limb_dark')[0])][1]
if foldext == 'None':
    foldext = ''
else:
    pass

print type(foldext)

PP = np.genfromtxt(pipelineinFile, dtype = None, skip_header = 1, delimiter = ', ', comments = '#', encoding = None)

leastsquares, averages, medians, stdevs, rmss, chi2s, BICs, poserrs, negerrs = [],[],[],[],[],[],[],[],[]

#print inputData

for i in range(len(PP)):

    if 'E' in eclipses[i/2]:
        eclipse = True
    else:
        eclipse = False

    # Read in the planet information from a text file
    # Create dictionary and fill of the stellar parameters (for limb darkening)
    star_params = {'Teff':0,'logg':0,'z':0,'Tefferr':0,'loggerr':0,'zerr':0}
    for key in star_params:
        star_params[key] = float(inputData[int(np.where(inputData.T[0]==key)[0])][1])

    # Create a dictionary of the polynomial parameters...
    if eclipse:
        coeffs_tuple_poly = ('t_secondary', 'fp','per', 'rp', 'a', 'inc', 'ecc', 'w', 'u', 'limb_dark',
                       'K1', 'K2', 'K3', 'K4', 'K5',
                       'f', 'g', 'h')
        #fix_coeffs_poly = inputData[int(np.where(inputData.T[0]=='fixcoeffs_poly_E')[0])][1].split(', ')
    else:
        coeffs_tuple_poly = ('t0', 'per', 'rp', 'a', 'inc', 'ecc', 'w', 'u', 'limb_dark',
                       'K1', 'K2', 'K3', 'K4', 'K5',
                       'f', 'g', 'h')
        fix_coeffs_poly = inputData[int(np.where(inputData.T[0]=='fixcoeffs_poly')[0])][1].split(', ')

    coeffs_dict_poly = dict()
    for label in coeffs_tuple_poly:
        if label != 'u' and label != 't_secondary':
            try:
                if len(inputData[int(np.where(inputData.T[0]==label)[0])][1].split(', ')) == len(PP):
                    coeffs_dict_poly[label] = float(inputData[int(np.where(inputData.T[0]==label)[0])][1].split(', ')[i])
                elif len(inputData[int(np.where(inputData.T[0]==label)[0])][1].split(', ')) == len(PP)/2:
                    coeffs_dict_poly[label] = float(inputData[int(np.where(inputData.T[0]==label)[0])][1].split(', ')[i/2])
                else:
                    coeffs_dict_poly[label] = float(inputData[int(np.where(inputData.T[0]==label)[0])][1])
            except:
                try: # If there is nothing to split
                    coeffs_dict_poly[label] = float(inputData[int(np.where(inputData.T[0]==label)[0])][1])
                except: # the limb darkening law
                    coeffs_dict_poly[label] = inputData[int(np.where(inputData.T[0]==label)[0])][1]

    # and a list of polynomial parameters to fix
    fix_coeffs_poly = inputData[int(np.where(inputData.T[0]=='fixcoeffs_poly')[0])][1].split(', ')

    # Create a dictionary of the PLD paramters...
    if eclipse:
        coeffs_tuple_PLD = ('t_secondary', 'fp','per', 'rp', 'a', 'inc', 'ecc', 'w', 'u', 'limb_dark',
                       'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9',
                       'g', 'h')
        #fix_coeffs_PLD = inputData[int(np.where(inputData.T[0]=='fixcoeffs_PLD_E')[0])][1].split(', ')
    else:
        coeffs_tuple_PLD = ('t0', 'per', 'rp', 'a', 'inc', 'ecc', 'w', 'u', 'limb_dark',
                       'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9',
                       'g', 'h')
        fix_coeffs_PLD = inputData[int(np.where(inputData.T[0]=='fixcoeffs_PLD')[0])][1].split(', ')

    coeffs_dict_PLD = dict()
    for label in coeffs_tuple_PLD:
        if label != 'u' and label != 't_secondary':
            try:
                if len(inputData[int(np.where(inputData.T[0]==label)[0])][1].split(', ')) == len(PP):
                    coeffs_dict_PLD[label] = float(inputData[int(np.where(inputData.T[0]==label)[0])][1].split(', ')[i])
                elif len(inputData[int(np.where(inputData.T[0]==label)[0])][1].split(', ')) == len(PP)/2:
                    coeffs_dict_PLD[label] = float(inputData[int(np.where(inputData.T[0]==label)[0])][1].split(', ')[i/2])
                else:
                    coeffs_dict_PLD[label] = float(inputData[int(np.where(inputData.T[0]==label)[0])][1])
            except:
                try: # If there is nothing to split
                    coeffs_dict_PLD[label] = float(inputData[int(np.where(inputData.T[0]==label)[0])][1])
                except: # the limb darkening law
                    coeffs_dict_PLD[label] = inputData[int(np.where(inputData.T[0]==label)[0])][1]

    # and a list of PLD parameters to fix
    fix_coeffs_PLD = inputData[int(np.where(inputData.T[0]=='fixcoeffs_PLD')[0])][1].split(', ')

    # Prior coefficients
    try:
        prior_coeffs = inputData[int(np.where(inputData.T[0]=='prior_coeffs')[0])][1].split(', ')
    except:
        prior_coeffs = []

    # Get the errors of the parameters that are used for calculating the depth and b
    try:
        if 'a' in fix_coeffs_PLD or 'a' in prior_coeffs:
            if len(inputData[int(np.where(inputData.T[0]=='a_err')[0])][1].split(', ')) == len(PP):
                coeffs_dict_poly['a_err'] = float(inputData[int(np.where(inputData.T[0]=='a_err')[0])][1].split(', ')[i])
                coeffs_dict_PLD['a_err'] = float(inputData[int(np.where(inputData.T[0]=='a_err')[0])][1].split(', ')[i])
            elif len(inputData[int(np.where(inputData.T[0]=='a_err')[0])][1].split(', ')) == len(PP)/2:
                coeffs_dict_poly['a_err'] = float(inputData[int(np.where(inputData.T[0]=='a_err')[0])][1].split(', ')[i/2])
                coeffs_dict_PLD['a_err'] = float(inputData[int(np.where(inputData.T[0]=='a_err')[0])][1].split(', ')[i/2])
            else:
                coeffs_dict_poly['a_err'] = float(inputData[int(np.where(inputData.T[0]=='a_err')[0])][1])
                coeffs_dict_PLD['a_err'] = float(inputData[int(np.where(inputData.T[0]=='a_err')[0])][1])
    except:
        raise ValueError("Trying to fix a without giving an error!")
    try:
        if 'inc' in fix_coeffs_PLD or 'inc' in prior_coeffs:
            if len(inputData[int(np.where(inputData.T[0]=='inc_err')[0])][1].split(', ')) == len(PP):
                coeffs_dict_poly['inc_err'] = float(inputData[int(np.where(inputData.T[0]=='inc_err')[0])][1].split(', ')[i])
                coeffs_dict_PLD['inc_err'] = float(inputData[int(np.where(inputData.T[0]=='inc_err')[0])][1].split(', ')[i])
            elif len(inputData[int(np.where(inputData.T[0]=='inc_err')[0])][1].split(', ')) == len(PP)/2:
                coeffs_dict_poly['inc_err'] = float(inputData[int(np.where(inputData.T[0]=='inc_err')[0])][1].split(', ')[i/2])
                coeffs_dict_PLD['inc_err'] = float(inputData[int(np.where(inputData.T[0]=='inc_err')[0])][1].split(', ')[i/2])
            else:
                coeffs_dict_poly['inc_err'] = float(inputData[int(np.where(inputData.T[0]=='inc_err')[0])][1])
                coeffs_dict_PLD['inc_err'] = float(inputData[int(np.where(inputData.T[0]=='inc_err')[0])][1])
    except:
        raise ValueError("Trying to fix inclination without giving an error!")
    try:
        if 'per' in fix_coeffs_PLD or 'per' in prior_coeffs:
            coeffs_dict_PLD['per_err'] = float(inputData[int(np.where(inputData.T[0]=='per_err')[0])][1])
            coeffs_dict_poly['per_err'] = float(inputData[int(np.where(inputData.T[0]=='per_err')[0])][1])
    except:
        raise ValueError("Trying to fix period without giving an error!")


    if len(prior_coeffs) > 1.:
        gaussian_priors = True
    else:
        gaussian_priors = False

    # # Check prior coeffs are in fix_coeffs
    # if not set(prior_coeffs).issubset(fix_coeffs_poly):
    #     raise ValueError("Trying to set gaussian priors on {} but they are not in fix_coeffs_poly.".format(prior_coeffs))
    # if not set(prior_coeffs).issubset(fix_coeffs_PLD):
    #     raise ValueError("Trying to set gaussian priors on {} but they are not in fix_coeffs_PLD.".format(prior_coeffs))

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
    binsize = float(PP[i][11])

    datapath = "{3}/PhD/SpitzerData/{0}/{1}/{2}/bcd/".format(planet,AOR,channel, os.getenv('HOME'))

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
                   quiet = False, plot = True, AOR = AOR, planet = planet, channel = channel, sysmethod = method, foldext=foldext)

    # Bin the lightcurves and timeseries
    #lightcurve_red = custom_bin(lightcurve_red, binsize_methods_params[l])
    timeseries = custom_bin(timeseries, binsize)
    centroids = custom_bin(centroids, binsize)
    midtimes = custom_bin(midtimes, binsize)
    background = custom_bin(background, binsize)

    # Don't bin the lightcruve right away, we need to get the errors first
    lc_unbinned = lightcurve*MJysr2lelectrons
    lcerr_unbinned = np.sqrt(lc_unbinned)

    # Bin the lightcurve and propagate the binning to the errors
    # Just taking the average of the errors would result in the errors being way too large for each of the datapoints
    # Shld actually check this by plotting it
    lc = custom_bin(lc_unbinned, binsize)
    lcerr = custom_bin(lcerr_unbinned, binsize, error = True)
    scale = np.median(lc[:int(100/binsize)])
    lc, lcerr = lc/scale, lcerr/scale
    t = (midtimes - midtimes[0])
    x, y = centroids[:,1], centroids[:,0]

    if eclipse:
        # N_orbits = np.floor((midtimes[0] - T0_bjd)/period)
        # ET_bjd = T0_bjd + period*(N_orbits+0.5)
        # TT_bjd = T0_bjd + period*(N_orbits)
        # t = midtimes - midtimes[0]
        # coeffs_dict_poly['t_secondary'], coeffs_dict_PLD['t_secondary'] = ET_bjd- midtimes[0], ET_bjd- midtimes[0]#float(t0s[m]), float(t0s[m])
        #coeffs_dict_poly['t0'], coeffs_dict_PLD['t0'] = TT_bjd- midtimes[0], TT_bjd- midtimes[0]
        coeffs_dict_poly['t_secondary'], coeffs_dict_PLD['t_secondary'] = float(t0s[i/2]), float(t0s[i/2])
    else:
        # If I have given t0 in BJD as opposed to start of observations
        if float(t0s[0]) > 10.:
            print "converting from BJD to time frmo beginning of observations"
            coeffs_dict_poly['t0'], coeffs_dict_PLD['t0'] = float(t0s[i/2])-midtimes[0] - 2400000.5, float(t0s[i/2])-midtimes[0]- 2400000.5
        else:
            print "Time is from beginning of observations"
            coeffs_dict_poly['t0'], coeffs_dict_PLD['t0'] = float(t0s[i/2]), float(t0s[i/2])

    print coeffs_dict_poly
    print coeffs_dict_PLD
    # Get limb darkening coefficients
    ldcoeffs, ldcoeffs_err = getldcoeffs(star_params['Teff'],star_params['logg'],star_params['z'],
                                         star_params['Tefferr'],star_params['loggerr'],star_params['zerr'],
                                         law = ldlaw, channel = channel, quiet = False)

    coeffs_dict_poly['u'], coeffs_dict_PLD['u'] = ldcoeffs, ldcoeffs # Must be list!

    if method == 'poly':

        result, batman_params, poly_params = fit_function_poly(coeffs_dict_poly,
        coeffs_tuple_poly, fix_coeffs_poly, t, x, y, lc,
        gaussian_priors = gaussian_priors, prior_params =prior_coeffs,
        eclipse = eclipse)

        popt = result.x

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
        result, batman_params, poly_params = fit_function_poly(coeffs_dict_poly,
        coeffs_tuple_poly, fix_coeffs_poly, t, x, y, lc,
        gaussian_priors = gaussian_priors, prior_params = prior_coeffs,
        eclipse = eclipse)
        popt = result.x

        # Plot the least squares result
        plot_lightcurve(t,  lc, lcerr, popt, coeffs_dict_poly, coeffs_tuple_poly, fix_coeffs_poly, batman_params, poly_params,
                            x=x, y=y, errors = False, binsize = 50,
                            name = planet, channel = channel, orbit = AOR, savefile = True, TT_hjd = None,
                            method = method, color = 'b', scale = scale, filext = "lsq", foldext = foldext, eclipse = eclipse)

        # Perform a sigma clip of the photometry
        t, x, y, lc, lcerr, bkg = sigma_clip_residuals(popt, t, background, lc, lcerr, coeffs_dict_poly,
                                                          coeffs_tuple_poly, fix_coeffs_poly, batman_params,
                                                          poly_params, 4, 30, x=x, y=y, quiet = False,
                                                          planet=planet, AOR=AOR, channel=channel, method=method,
                                                          foldext=foldext, plot=True, eclipse = eclipse)

        # Least squares again
        result, batman_params, poly_params = fit_function_poly(coeffs_dict_poly,
        coeffs_tuple_poly, fix_coeffs_poly, t, x, y, lc,
        gaussian_priors =gaussian_priors, prior_params =prior_coeffs,
        eclipse = eclipse)

        popt = result.x

        # Prayer bead
        prayer_bead = prayer_bead_poly(popt, t, x, y, lc, coeffs_dict_poly,
        coeffs_tuple_poly, fix_coeffs_poly, batman_params, poly_params,
        100, planet, AOR, channel, method, foldext=foldext, plot = True, eclipse = eclipse)

        # labels
        labels_poly = [ key for key in coeffs_tuple_poly if key not in fix_coeffs_poly ]

        # Inflate the errors
        newlcerr = inflate_errs(popt, t, lc, lcerr, coeffs_dict_poly,
        coeffs_tuple_poly, fix_coeffs_poly, batman_params, poly_params,x=x,y=y,
        method = method, eclipse = eclipse)

        # Run MCMC
        bounds = make_bounds(coeffs_tuple_poly, fix_coeffs_poly, t)
        data = (t, x, y, lc, newlcerr, bounds, coeffs_dict_poly, coeffs_tuple_poly,
        fix_coeffs_poly, batman_params, poly_params, gaussian_priors, prior_coeffs, eclipse)
        sampler = mcmc_poly(popt, data, nwalkers = nwalkers, burnin_steps = burninsteps, production_steps = runsteps)

        avgs, stds, meds, pos, neg, rms, chi2, bic, fixedparameters, plotting_stuff = mcmc_results(sampler, popt, t, lc, newlcerr, x,y, None, bkg,
                                    coeffs_dict_poly, coeffs_tuple_poly, fix_coeffs_poly, batman_params, poly_params, scale, labels_poly,
                                    planet, AOR, channel, method, midtimes[0], saveplots=True, foldext=foldext, eclipse = eclipse, extraoutputs=True)

        leastsquares.append(popt)
        # Save outputs in format for paper
        pipelineparameters = {'Background Method':bkg_method,
                          'Background Params': [bkg_boxsize, bkg_annradius, bkg_annsize],
                          'Centroiding Method': cent_method,
                          'Centroiding Params': cent_sizebary,
                          'Aperture Size': photom_radius}

        datared = {'lc':lc, 'lcerr':newlcerr, 'centroids': centroids,
                    'midtimes':t, 'x':x, 'y':y, 'background':bkg,
                    'timeseries':timeseries}

        labels = [ key for key in coeffs_tuple_PLD if key not in fix_coeffs_PLD ]

        results = {"parameters":labels, "averages": avgs, "stddevs": stds, "medians":meds,
                    "poserr":pos, "negerr":neg, "rms":rms, "chi2":chi2,
                    "bic":bic}

        AORdata = {'Data Reduction': datared,
               'Results': results,
               'Fixed Parameters': fixedparameters,
               'Best Fit': plotting_stuff,
               'Pipeline Parameters': pipelineparameters}

        AORfilepath = "{2}/PhD/SpitzerTransits/{0}{1}/{0}_{3}_{4}_{5}_ResultsDict.npy".format(planet,foldext, os.getenv('HOME'), AOR, method, channel)
        np.save(AORfilepath, AORdata)
        # Draw from gaussian
        ld_coeffs = np.random.normal(ldcoeffs, ldcoeffs_err, 500)
        ldsamples_poly = np.zeros((len(ld_coeffs),len(coeffs_tuple_poly)- len(fix_coeffs_poly)))

        for i in range(len(ld_coeffs)):

            coeffs_dict_poly['u'] = [ld_coeffs[i]]# Must be list!

            #POLYNOMIAL
            result, batman_params_poly, poly_params = fit_function_poly(coeffs_dict_poly,
            coeffs_tuple_poly, fix_coeffs_poly, t, x, y, lc,
            gaussian_priors =gaussian_priors, prior_params =prior_coeffs, eclipse = eclipse)
            popt = result.x
            ldsamples_poly[i] = np.array(popt)

        index_rp = 1
        print "\nChecking the effect of varying limb darkening..."
        print "\t1 sigma change in ld produces {} sigma change in rp/r* for poly".format(np.std(ldsamples_poly[:,index_rp])/stds[index_rp])

        print avgs

    if method == 'PLD':

        print coeffs_dict_PLD

        result_PLD, batman_params_PLD, PLD_params, Pns = fit_function_PLD(coeffs_dict_PLD,
        coeffs_tuple_PLD, fix_coeffs_PLD, t, timeseries, centroids, lc,
        boxsize = (3,3), gaussian_priors =gaussian_priors,
        prior_params =prior_coeffs, eclipse = eclipse)
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
        result_PLD, batman_params_PLD, PLD_params, Pns = fit_function_PLD(coeffs_dict_PLD,
        coeffs_tuple_PLD, fix_coeffs_PLD, t, timeseries, centroids, lc,
        boxsize = (3,3),gaussian_priors =gaussian_priors,
        prior_params =prior_coeffs, eclipse = eclipse)
        popt_PLD = result_PLD.x

        # Plot the least squares result
        plot_lightcurve(t,  lc, lcerr, popt_PLD, coeffs_dict_PLD, coeffs_tuple_PLD, fix_coeffs_PLD, batman_params_PLD, PLD_params,
                            Pns = Pns, errors = False, binsize = 50,
                            name = planet, channel = channel, orbit=AOR, savefile = True, TT_hjd = None,
                            method = method, color = 'r', scale = scale, filext = "lsq", foldext=foldext, eclipse = eclipse)

        # Perform a sigma clip of the photometry
        t, x, y, Pns, lc, lcerr, bkg, timeseries, centroids = sigma_clip_residuals(popt_PLD, t, background, lc, lcerr, coeffs_dict_poly,
                                                          coeffs_tuple_PLD, fix_coeffs_PLD, batman_params_PLD,
                                                          PLD_params, 4, 30, x=x,y=y,Pns=Pns, quiet = False,
                                                          planet=planet, AOR=AOR, channel=channel, method=method,
                                                          foldext=foldext, plot=True, timeseries=timeseries, centroids=centroids,
                                                          eclipse = eclipse)

        # Least squares again
        result_PLD, batman_params_PLD, PLD_params, Pns = fit_function_PLD(coeffs_dict_PLD,
        coeffs_tuple_PLD, fix_coeffs_PLD, t, timeseries, centroids, lc,
        boxsize = (3,3),gaussian_priors =gaussian_priors,
        prior_params =prior_coeffs, eclipse = eclipse)
        popt_PLD = result_PLD.x
        print popt_PLD

        # Prayer bead
        prayer_bead = prayer_bead_PLD(popt_PLD, t, Pns, lc, timeseries, centroids, coeffs_dict_PLD, coeffs_tuple_PLD, fix_coeffs_PLD, batman_params_PLD, PLD_params,
                        10, planet, AOR, channel, method, foldext=foldext, plot = True, eclipse = eclipse)

        labels_PLD = [ key for key in coeffs_tuple_PLD if key not in fix_coeffs_PLD ]

        # Inflate the errors
        newlcerr = inflate_errs(popt_PLD, t, lc, lcerr, coeffs_dict_PLD, coeffs_tuple_PLD,
        fix_coeffs_PLD, batman_params_PLD, PLD_params, Pns=Pns, method = method, eclipse = eclipse)

        # Run MCMC
        bounds = make_bounds(coeffs_tuple_PLD, fix_coeffs_PLD, t)
        data = (t, Pns, lc, newlcerr, bounds, coeffs_dict_PLD, coeffs_tuple_PLD,
        fix_coeffs_PLD, batman_params_PLD, PLD_params, gaussian_priors, prior_coeffs, eclipse)
        sampler = mcmc_PLD(popt_PLD, data, nwalkers = nwalkers, burnin_steps = burninsteps, production_steps = runsteps)

        avgs, stds, meds, pos, neg, rms, chi2, bic, fixedparameters, plotting_stuff = mcmc_results(sampler, popt_PLD,t,lc,newlcerr,x,y,Pns,bkg,
                                    coeffs_dict_PLD, coeffs_tuple_PLD, fix_coeffs_PLD, batman_params_PLD, PLD_params,scale,
                                    labels_PLD, planet, AOR, channel, method, midtimes[0],
                                    saveplots=True, foldext=foldext, eclipse = eclipse, extraoutputs = True)

        # Save outputs in format for paper
        pipelineparameters = {'Background Method':bkg_method,
                          'Background Params': [bkg_boxsize, bkg_annradius, bkg_annsize],
                          'Centroiding Method': cent_method,
                          'Centroiding Params': cent_sizebary,
                          'Aperture Size': photom_radius}

        datared = {'lc':lc, 'lcerr':newlcerr, 'centroids': centroids,
                    'midtimes':t, 'Pns':Pns, 'background':bkg,
                    'timeseries':timeseries}

        labels = [ key for key in coeffs_tuple_PLD if key not in fix_coeffs_PLD ]

        results = {"parameters":labels, "averages": avgs, "stddevs": stds, "medians":meds,
                    "poserr":pos, "negerr":neg, "rms":rms, "chi2":chi2,
                    "bic":bic}

        AORdata = {'Data Reduction': datared,
               'Results': results,
               'Fixed Parameters': fixedparameters,
               'Best Fit': plotting_stuff,
               'Pipeline Parameters': pipelineparameters}

        AORfilepath = "{2}/PhD/SpitzerTransits/{0}{1}/{0}_{3}_{4}_{5}_ResultsDict.txt".format(planet,foldext, os.getenv('HOME'), AOR, method, channel)

        SaveDictionary(AORdata,AORfilepath)

        leastsquares.append(popt_PLD)

        # Draw from gaussian
        ld_coeffs = np.random.normal(ldcoeffs, ldcoeffs_err, 500)
        ldsamples_PLD = np.zeros((len(ld_coeffs),len(coeffs_tuple_PLD)- len(fix_coeffs_PLD)))

        for i in range(len(ld_coeffs)):
            coeffs_dict_PLD['u'] = [ld_coeffs[i]]# Must be list!

            #PLD
            result_PLD, batman_params_PLD, PLD_params, Pns = fit_function_PLD(coeffs_dict_PLD,
            coeffs_tuple_PLD, fix_coeffs_PLD, t, timeseries, centroids, lc,
            boxsize = (3,3),gaussian_priors =gaussian_priors,
            prior_params =prior_coeffs, eclipse = eclipse)
            popt_PLD = result_PLD.x
            ldsamples_PLD[i] = np.array(popt_PLD)

        index_rp = 1
        print "\nChecking the effect of varying limb darkening..."
        print "\t1 sigma change in ld produces {} sigma change in rp/r* for PLD".format(np.std(ldsamples_PLD[:,index_rp])/stds[index_rp])

        print avgs

    averages.append(avgs)
    stdevs.append(stds)
    rmss.append(rms)
    chi2s.append(chi2)
    BICs.append(bic)
    medians.append(meds)
    poserrs.append(pos)
    negerrs.append(neg)



# Save results to a file so that we can work with them
# Save results to a file so that we can work with them
resultsfilepath = "{2}/PhD/SpitzerTransits/{0}{1}/{0}_results.txt".format(planet,foldext, os.getenv('HOME'))
resultsfile = open(resultsfilepath,'w')

fitted_params = [key for key in labels_poly  if key in batman_params.__dict__]

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
parameter_plots(resultsfilepath, fitted_params, "Mean", planet, plotPublished, publishedDataFile, saveplot = True, foldext = foldext, eclipse = eclipse)

sigma_sec_PLD, nsigma_PLD = t0check(resultsfilepath, 'Mean', 'PLD', coeffs_dict_PLD['per'], coeffs_dict_PLD['per_err'])

print "\nChecking timing for PLD..."
print "\tSigma = {:.2f} seconds".format(sigma_sec_PLD)
print "\tNsigma between t0s = {:.2f}".format(nsigma_PLD)


sigma_sec_poly, nsigma_poly = t0check(resultsfilepath, 'Mean', 'poly', coeffs_dict_poly['per'], coeffs_dict_poly['per_err'])

print "\nChecking timing for poly..."
print "\tSigma = {:.2f} seconds".format(sigma_sec_poly)
print "\tNsigma between t0s = {:.2f}".format(nsigma_poly)
