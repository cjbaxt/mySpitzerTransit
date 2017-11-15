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

print "this is the eclipse development version"

# Get input file
try:
    inFile = sys.argv[1]
except:
    raise ValueError('You need to provide input planet file as arguement.')

print "Using planet data from {}".format(inFile)

#Get folder extension
foldext = raw_input("Provide folder extension (hit ENTER for None): ")

# Read in the planet information from a text file
inputData = np.genfromtxt(inFile, dtype=None, delimiter=': ', comments='#')
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

for m in range(len(AORs)):
    if 'E' in eclipses[m]:
        eclipse = True
    else:
        eclipse = False

    star_params = {'Teff':0,'logg':0,'z':0,'Tefferr':0,'loggerr':0,'zerr':0}
    for key in star_params:
        star_params[key] = float(inputData[int(np.where(inputData.T[0]==key)[0])][1])

    # Create a dictionary of the polynomial parameters...
    if eclipse:
        coeffs_tuple_poly = ('t0',  't_secondary', 'fp','per', 'rp', 'a', 'inc', 'ecc', 'w', 'u', 'limb_dark',
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
        if label != 'u' and label != 't_secondary' and label != 't0':
            try:
                coeffs_dict_poly[label] = float(inputData[int(np.where(inputData.T[0]==label)[0])][1].split(', ')[m])
            except:
                try: # If there is nothing to split
                    coeffs_dict_poly[label] = float(inputData[int(np.where(inputData.T[0]==label)[0])][1])
                except: # the limb darkening law
                    coeffs_dict_poly[label] = inputData[int(np.where(inputData.T[0]==label)[0])][1]

    # Create a dictionary of the PLD paramters...
    if eclipse:
        coeffs_tuple_PLD = ('t0',  't_secondary', 'fp','per', 'rp', 'a', 'inc', 'ecc', 'w', 'u', 'limb_dark',
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
        if label != 'u' and label != 't_secondary' and label != 't0':
            try:
                coeffs_dict_PLD[label] = float(inputData[int(np.where(inputData.T[0]==label)[0])][1].split(', ')[m])
            except:
                try: # If there is nothing to split
                    coeffs_dict_PLD[label] = float(inputData[int(np.where(inputData.T[0]==label)[0])][1])
                except: # the limb darkening law
                    coeffs_dict_PLD[label] = inputData[int(np.where(inputData.T[0]==label)[0])][1]

    AOR = AORs[m]
    channel = channels[m]

    # Get the interpolated limb darkening coefficients
    ldcoeffs, ldcoeffs_err = getldcoeffs(star_params['Teff'],star_params['logg'],star_params['z'],
                                         star_params['Tefferr'],star_params['loggerr'],star_params['zerr'],
                                         law = ldlaw, channel = channel)

    if np.nan not in ldcoeffs:
        coeffs_dict_poly['u'], coeffs_dict_PLD['u'] = ldcoeffs, ldcoeffs # Must be list!
    else:
        coeffs_dict_poly['u'] = [float(inputData[int(np.where(inputData.T[0]=='u')[0])][1].split('\ ')[m])]
        coeffs_dict_PLD['u'] = [float(inputData[int(np.where(inputData.T[0]=='u')[0])][1].split('\ ')[m])]
        ldcoeffs_err = [float(inputData[int(np.where(inputData.T[0]=='u_err')[0])][1].split('\ ')[m])]

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

    print "\t Exptime = {}, Readnoise = {}, Gain = {}, Fluxconv = {}, Framtime = {}".format(exptime, readnoise, gain, fluxconv, framtime)
    print "\t MJy/sr to electrons conversion factor = {}".format(MJysr2lelectrons)

    bkg_method = "Annulus"
    bkg_boxsize = None
    bkg_annradius = 7
    bkg_annsize = 4
    cent_method = "Barycenter"
    cent_sizebary = 5
    photom_radius = 2.5

    datapath = "{3}/PhD/SpitzerData/{0}/{1}/{2}/bcd/".format(planet,AOR,channel, os.getenv('HOME'))

    # If the AOR is the same as the one before do not run the pipeline again (deleted)
    lightcurve_red, timeseries_red, centroids_red, midtimes_red, background_red = runFullPipeline(datapath, 4, 30,
                   method_bkg = bkg_method, method_cent = cent_method, plotting_binsize=50,
                   ramp_time = cutstarts[m]*60, end_time = cutends[m]*60,
                   x0guess = posGuess[0], y0guess = posGuess[1],
                   sigma_clip_cent = 4, iters_cent = 2, nframes_cent=30,
                   radius_photom = photom_radius,
                   sigma_clip_phot = 4, iters_photom = 2, nframes_photom=30,
                   size_bkg_box = bkg_boxsize, radius_bkg_ann = bkg_annradius, size_bkg_ann = bkg_annsize,
                   size_cent_bary = cent_sizebary, passenger57 = True,
                   quiet = False, plot = True, AOR = AOR, planet = planet, channel = channel, foldext=foldext)

    lc = lightcurve_red
    lcerr = np.sqrt(lc)
    scale = np.median(lc[:100]) # Guess initial scale
    lc, lcerr = lc/scale, lcerr/scale
    x, y = centroids_red[:,1], centroids_red[:,0]
    t = (midtimes_red - midtimes_red[0])

    if eclipse:
        N_orbits = np.floor((midtimes_red[0] - T0_bjd)/period)
        ET_bjd = T0_bjd + period*(N_orbits+0.5)
        TT_bjd = T0_bjd + period*(N_orbits)
        t = midtimes_red - midtimes_red[0]
        coeffs_dict_poly['t_secondary'], coeffs_dict_PLD['t_secondary'] = ET_bjd- midtimes_red[0], ET_bjd- midtimes_red[0]#float(t0s[m]), float(t0s[m])
        coeffs_dict_poly['t0'], coeffs_dict_PLD['t0'] = TT_bjd- midtimes_red[0], TT_bjd- midtimes_red[0]
    else:
        coeffs_dict_poly['t0'], coeffs_dict_PLD['t0'] = float(t0s[m]), float(t0s[m])
    #POLYNOMIAL

    print coeffs_dict_poly
    print fix_coeffs_poly

    result, batman_params_poly, poly_params = fit_function_poly(coeffs_dict_poly, coeffs_tuple_poly, fix_coeffs_poly, t, x, y, lc, eclipse = eclipse)
    popt = result.x
    labels_poly = [ key for key in coeffs_tuple_poly if key not in fix_coeffs_poly ]
    print tabulate([labels_poly, popt])
    plot_lightcurve(t,  lc, lcerr, popt, coeffs_dict_poly, coeffs_tuple_poly, fix_coeffs_poly, batman_params_poly, poly_params,
                        x=x,y=y, errors = False, binsize = 70,
                        name = planet, channel = channel, orbit = AOR, savefile = True, TT_hjd = None,
                        method = "poly", color = 'b', scale = scale, filext = "lsq_prelim", foldext=foldext, eclipse = eclipse)

    #PLD
    result_PLD, batman_params_PLD, PLD_params, Pns = fit_function_PLD(coeffs_dict_PLD, coeffs_tuple_PLD, fix_coeffs_PLD, t, timeseries_red, centroids_red, lc, eclipse = eclipse)
    popt_PLD = result_PLD.x
    labels_PLD = [ key for key in coeffs_tuple_PLD if key not in fix_coeffs_PLD ]
    print tabulate([labels_PLD,popt_PLD])
    plot_lightcurve(t,  lc, lcerr, popt_PLD, coeffs_dict_PLD, coeffs_tuple_PLD, fix_coeffs_PLD, batman_params_PLD, PLD_params,
                        Pns = Pns, errors = False, binsize = 70,
                        name = planet, channel = channel, orbit=AOR, savefile = True, TT_hjd = None,
                        method = "PLD", color = 'r', scale = scale, filext = "lsq_prelim", foldext=foldext, eclipse = eclipse)

    #Create list of lists of parameters to FIX
    poly_Ks_fit = [['K1'], ['K3'], ['K1','K3'], ['K1', 'K2', 'K3'], ['K1', 'K3', 'K4'],
                ['K1', 'K2', 'K3', 'K4'], ['K1', 'K2', 'K3', 'K4', 'K5']]
    poly_Ks_fix = [['K2', 'K3', 'K4', 'K5'], ['K1', 'K2', 'K4', 'K5'], ['K2', 'K4', 'K5'], ['K4', 'K5'], ['K2','K5'], ['K5'], []]

    fix_coeffs_nosys = [p for p in fix_coeffs_poly if 'K' not in p]

    BICs, nparams, chi2s = [], [], []

    for j in range(len(poly_Ks_fit)):
        fix_coeffs_test = fix_coeffs_nosys + poly_Ks_fix[j]
        for key in poly_Ks_fix[j]:
            coeffs_dict_poly[key] = 0.
        result, batman_params_poly, poly_params = fit_function_poly(coeffs_dict_poly, coeffs_tuple_poly, fix_coeffs_test, t, x, y, lc, eclipse = eclipse)
        nparams.append(len(result.x))
        chi2s.append(chi(result.x, t, lc, lcerr, coeffs_dict_poly, coeffs_tuple_poly, fix_coeffs_test, batman_params_poly, poly_params,
                x=x, y=y, Pns=None, method = 'poly', eclipse = eclipse))
        BICs.append(BIC(result.x, t, lc, lcerr, coeffs_dict_poly, coeffs_tuple_poly, fix_coeffs_test, batman_params_poly, poly_params,
                x=x, y=y, Pns=None, method = 'poly', eclipse = eclipse))


    f = open("{4}/PhD/SpitzerTransits/{0}{1}/BICs_{2}_{3}.txt".format(planet,foldext, AOR, channel, os.getenv('HOME')), 'w')

    print >> f, tabulate(np.array([poly_Ks_fit, BICs, chi2s, nparams]).T, headers = ['coeffs', 'BIC', 'chi2', 'Nparams'])

    f.close()


    # Look at the BIC of changing number of parameters for

    # # Update coeffs_dict so that it uses values from previous iteration
    # for key in coeffs_dict_poly.keys():
    #     try:
    #         # Batman
    #          coeffs_dict_poly[key] = batman_params_poly.__dict__[key]
    #     except:
    #         # Normal dictionary
    #         coeffs_dict_poly[key] = poly_params[key]
    #
    # for key in coeffs_dict_PLD.keys():
    #     try:
    #         # Batman
    #          coeffs_dict_PLD[key] = batman_params_PLD.__dict__[key]
    #     except:
    #         # Normal dictionary
    #         coeffs_dict_PLD[key] = PLD_params[key]
    #
    # print coeffs_dict_poly
    #
    # print "\n"
    #
    # print coeffs_dict_PLD
