# imports
from astropy.io import fits
import numpy as np
import glob, os, sys, time
import matplotlib.pyplot as plt
from scipy import stats
from scipy.ndimage.measurements import center_of_mass
from scipy.optimize import minimize_scalar, leastsq, least_squares
from scipy.interpolate import UnivariateSpline
from photutils import CircularAperture, aperture_photometry, CircularAnnulus
import batman, emcee, corner
from tabulate import tabulate
from IPython.display import HTML
import traceback

print "This is the adding binning version"

# Custom imports
sys.path.insert(0, '{}/PhD/code/mySpitzerTransit/'.format(os.getenv('HOME')))
from timeseries_routines import *
from transitFitting_routines import *
sys.path.insert(1, '{}/PhD/code/'.format(os.getenv('HOME')))
from ProgressBar import *

# Get input file
try:
    inFile = sys.argv[1]
except:
    raise ValueError('You need to provide input planet file as arguement.')

print "Using planet data from {}".format(inFile)

# Read in the planet information from a text file
inputData = np.genfromtxt(inFile, dtype=None, delimiter=': ', comments='#', encoding = None)
planet = inputData[0][1]
AORs = inputData[1][1].split(', ')
channels = inputData[2][1].split(', ')
eclipses = inputData[3][1].split(', ')
t0s = inputData[4][1].split(', ')
cutstarts = [float(x) for x in inputData[5][1].split(', ')]
cutends = [float(x) for x in inputData[6][1].split(', ')]
posGuess = [float(x) for x in inputData[7][1].split(', ')]
ldlaw = inputData[int(np.where(inputData.T[0]=='limb_dark')[0])][1]
foldext = raw_input("Provide folder extension (hit ENTER for None): ")

# Open a file to save the best pipeline parameters to
f = open("{1}/PhD/SpitzerData/{0}/pipelineParams_{0}.txt".format(planet, os.getenv('HOME')), 'w')
f.write("method, AOR, channel, chi2, bgk_method, bkg_boxsize, bkg_annradius, bkg_annsize, cent_method, cent_params, photom_radius \n")

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
                coeffs_dict_poly[label] = float(inputData[int(np.where(inputData.T[0]==label)[0])][1].split(', ')[m])
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

    #Create timeseries, midtimes and maskseries
    timeseries = data_info.create_timeseries()
    midtimes = data_info.midtimes()

    if eclipse:
        # N_orbits = np.floor((midtimes[0] - T0_bjd)/period)
        # ET_bjd = T0_bjd + period*(N_orbits+0.5)
        # TT_bjd = T0_bjd + period*(N_orbits)
        # t = midtimes - midtimes[0]
        # coeffs_dict_poly['t_secondary'], coeffs_dict_PLD['t_secondary'] = ET_bjd- midtimes[0], ET_bjd- midtimes[0]#float(t0s[m]), float(t0s[m])
        #coeffs_dict_poly['t0'], coeffs_dict_PLD['t0'] = TT_bjd- midtimes[0], TT_bjd- midtimes[0]
        coeffs_dict_poly['t_secondary'], coeffs_dict_PLD['t_secondary'] = float(t0s[m]), float(t0s[m])
    else:
        # If I have given t0 in BJD as opposed to start of observations
        if float(t0s[0]) > 10.:
            print "converting from BJD to time frmo beginning of observations"
            coeffs_dict_poly['t0'], coeffs_dict_PLD['t0'] = float(t0s[m])-midtimes[0] - 2400000.5, float(t0s[m])-midtimes[0]- 2400000.5
        else:
            print "Time is from beginning of observations"
            coeffs_dict_poly['t0'], coeffs_dict_PLD['t0'] = float(t0s[m]), float(t0s[m])

    #Fix bad pixles
    sigma_badpix = 5
    nframes = 30
    timeseries_badpix = fast_bad_pix_mask(timeseries, sigma_badpix, nframes)

    #Background optimisation methods
    bkg_methods_labels = ["Hist", "Box_2", "Box_4"
                   ]
    bkg_methods = ["Histogram","Box", "Box"
                   ]
    bkg_methods_params = [[None,None,None], [2,None,None], [4,None,None]] #box_size, ann_radius, ann_size

    #Centroid optimisation methods
    cent_methods_labels = ["Bary_3", "Bary_5", "Gaussian", "Moffat"]
    cent_methods = ["Barycenter", "Barycenter", "Gaussian", "Moffat"]
    cent_methods_params = [3,5,None,None]

    #Aperture Photometry Optimisation methods
    if channel == 'ch1':
        photom_methods_params = np.arange(2.0, 2.75, 0.25).tolist()
    elif channel == 'ch2':
        photom_methods_params = np.arange(2.0, 2.75, 0.25).tolist()

    # Bin size Optimisation methoda
    binsize_methods_params = [1]#], 4, 16, 64]

    # #Background optimisation methods
    # bkg_methods_labels = ["Box_2"]
    # bkg_methods = ["Box"]
    # bkg_methods_params = [[2,None,None]] #box_size, ann_radius, ann_size
    #
    # #Centroid optimisation methods
    # cent_methods_labels = ["Bary_3"]
    # cent_methods = ["Barycenter"]
    # cent_methods_params = [3]
    # #Aperture Photometry Optimisation methods
    # photom_methods_params = np.arange(2, 2.5, 0.5).tolist()

    chi2_poly = np.zeros((len(bkg_methods_params), len(cent_methods_params), len(photom_methods_params), len(binsize_methods_params)))
    chi2_PLD = np.zeros((len(bkg_methods_params), len(cent_methods_params), len(photom_methods_params), len(binsize_methods_params)))

    nsteps = len(bkg_methods_params)*len(cent_methods_params)*len(photom_methods_params)*len(binsize_methods_params)
    nparams_poly = len(coeffs_tuple_poly) - len(fix_coeffs_poly)
    nparams_PLD = len(coeffs_tuple_PLD) - len(fix_coeffs_PLD)
    samples_poly = np.zeros((nsteps,nparams_poly))
    samples_PLD = np.zeros((nsteps,nparams_PLD))

    p = ProgressBar(nsteps)
    count = 0
    print "\nEntering pipeline optimisation loop..."
    for i in range(len(bkg_methods_params)):
        for j in range(len(cent_methods_params)):
            for k in range(len(photom_methods_params)):
                for l in range(len(binsize_methods_params)):
                    p.animate(count)

                    try:
                        # Run pipeline and create the lightcurve and timeseries
                        lightcurve_red, timeseries_red, centroids_red, midtimes_red, background_red =  runPipeline(timeseries_badpix, midtimes,
                               method_bkg = bkg_methods[i], method_cent = cent_methods[j],
                               ramp_time = (cutstarts[m])*60,
                               end_time = (cutends[m])*60,
                               frametime = framtime,
                               x0guess = posGuess[0], y0guess = posGuess[1],
                               sigma_clip_cent = 4, iters_cent = 2, nframes_cent = 30, radius_photom = photom_methods_params[k],
                               sigma_clip_phot = 4, iters_photom = 2, nframes_photom = 30,
                               size_bkg_box = bkg_methods_params[i][0] , radius_bkg_ann = bkg_methods_params[i][1], size_bkg_ann = bkg_methods_params[i][2],
                               size_cent_bary = cent_methods_params[j], quiet = True, foldext=foldext)

                        # Bin the lightcurves and timeseries
                        #lightcurve_red = custom_bin(lightcurve_red, binsize_methods_params[l])
                        timeseries_red = custom_bin(timeseries_red, binsize_methods_params[l])
                        centroids_red = custom_bin(centroids_red, binsize_methods_params[l])
                        midtimes_red = custom_bin(midtimes_red, binsize_methods_params[l])
                        background_red = custom_bin(background_red, binsize_methods_params[l])

                        # Don't bin the lightcruve right away, we need to get the errors first
                        lc_unbinned = lightcurve_red*MJysr2lelectrons
                        lcerr_unbinned = np.sqrt(lc_unbinned)

                        # Bin the lightcurve and propagate the binning to the errors
                        # Just taking the average of the errors would result in the errors being way too large for each of the datapoints
                        # Shld actually check this by plotting it
                        lc = custom_bin(lc_unbinned, binsize_methods_params[l])
                        lcerr = custom_bin(lcerr_unbinned, binsize_methods_params[l], error = True)
                        scale = np.median(lc[:int(100/binsize_methods_params[l])])
                        lc, lcerr = lc/scale, lcerr/scale
                        t = (midtimes_red - midtimes_red[0])
                        x, y = centroids_red[:,1], centroids_red[:,0]

                        if (planet == 'Wasp13b') and (channel == 'ch1'):
                            ind0 = find_nearest(t,0.190)
                            ind1 = find_nearest(t,0.215)

                            lcerr[ind0:ind1] = 1e6*lcerr[ind0:ind1]
                        else:
                            pass

                        print coeffs_dict_poly
                        print coeffs_tuple_poly

                        #POLYNOMIAL
                        result, batman_params_poly, poly_params = fit_function_poly(coeffs_dict_poly, coeffs_tuple_poly, fix_coeffs_poly, t, x, y, lc, eclipse = eclipse)
                        popt = result.x

                        # Update coeffs_dict so that it uses values from previous iteration
                        # for key in coeffs_dict_poly.keys():
                        #     try:
                        #         # Batman
                        #          coeffs_dict_poly[key] = batman_params_poly.__dict__[key]
                        #     except:
                        #         # Normal dictionary
                        #         coeffs_dict_poly[key] = poly_params[key]

                        plot_lightcurve(t,  lc, lcerr, popt, coeffs_dict_poly, coeffs_tuple_poly, fix_coeffs_poly, batman_params_poly, poly_params,
                                            x=x,y=y, errors = False, binsize = int(np.ceil((50./binsize_methods_params[l]))),
                                            name = planet, channel = channel, orbit = AOR, savefile = True, TT_hjd = None,
                                            method = "poly", color = 'b', scale = scale, filext = "pipelineOpt_{}".format(count), foldext=foldext, eclipse = eclipse)

                        # Save the least squares polynomial result to an array so we can plot a corer plot
                        for q in range(len(popt)):
                            samples_poly[count][q] = popt[q]

                        #PLD
                        result_PLD, batman_params_PLD, PLD_params, Pns = fit_function_PLD(coeffs_dict_PLD, coeffs_tuple_PLD,
                                                                            fix_coeffs_PLD, t, timeseries_red, centroids_red, lc, eclipse = eclipse)
                        popt_PLD = result_PLD.x
                        # for key in coeffs_dict_PLD.keys():
                        #     try:
                        #         # Batman
                        #          coeffs_dict_PLD[key] = batman_params_PLD.__dict__[key]
                        #     except:
                        #         # Normal dictionary
                        #         coeffs_dict_PLD[key] = PLD_params[key]

                        plot_lightcurve(t,  lc, lcerr, popt_PLD, coeffs_dict_PLD, coeffs_tuple_PLD, fix_coeffs_PLD, batman_params_PLD, PLD_params,
                                            Pns = Pns, errors = False, binsize = int(np.ceil((50./binsize_methods_params[l]))),
                                            name = planet, channel = channel, orbit=AOR, savefile = True, TT_hjd = None,
                                            method = "PLD", color = 'r', scale = scale, filext = "pipelineOpt_{}".format(count), foldext=foldext, eclipse = eclipse)

                        # Save the least squares polynomial result to an array so we can plot a corer plot
                        for q in range(len(popt_PLD)):
                            samples_PLD[count][q] = popt_PLD[q]

                        # Calculate the chi2
                        chi2_A = chi(popt, t, lc, lcerr, coeffs_dict_poly, coeffs_tuple_poly, fix_coeffs_poly,
                                                    batman_params_poly, poly_params,
                                                    x=x, y=y, method = "poly", eclipse = eclipse)/(len(lc)-len(popt))
                        chi2_poly[i][j][k][l] = chi2_A

                        chi2_B = chi(popt_PLD, t, lc, lcerr, coeffs_dict_PLD, coeffs_tuple_PLD, fix_coeffs_PLD,
                                                    batman_params_PLD, PLD_params,
                                                    Pns = Pns, method = "PLD", eclipse = eclipse)/(len(lc)-len(popt_PLD))
                        chi2_PLD[i][j][k][l] = chi2_B

                         # Here we could add in the test of the least squares between the rms expected vs real

                    except Exception as e:
                        print "Skipping pipeline iteration: {}-{}-{}-{}".format(bkg_methods_params[i], cent_methods_params[j], photom_methods_params[k], binsize_methods_params[l])
                        print(e)

                        chi2_poly[i][j][k][l] = np.nan
                        chi2_PLD[i][j][k][l] = np.nan
                        for q in range(len(popt)):
                            samples_poly[count][q] = np.nan
                        for q in range(len(popt_PLD)):
                            samples_PLD[count][q] = np.nan

                    count += 1

    np.save('{5}/PhD/SpitzerTransits/{0}{4}/{0}_{1}_{2}_{3}_pipelineOptSamples'.format(planet, AOR, channel,"poly",foldext, os.getenv('HOME')), samples_poly)
    np.save('{5}/PhD/SpitzerTransits/{0}{4}/{0}_{1}_{2}_{3}_pipelineOptSamples'.format(planet, AOR, channel,"PLD",foldext, os.getenv('HOME')), samples_PLD)

    print "\nSaving polynomial and results for AOR:{}...".format(AOR)

    # Save the chi2 polynomial grid to a numpy object
    np.save('{5}/PhD/SpitzerTransits/{0}{4}/{0}_{1}_{2}_{3}_Chi2grid'.format(planet, AOR, channel,"poly",foldext, os.getenv('HOME')), chi2_poly)

    index = np.where(chi2_poly == np.nanmin(chi2_poly))

    try:
        chi2_photom_poly = chi2_poly[index[0][0]][index[1][0][index[2][0]]]
        plt.plot(photom_methods_params, chi2_photom_poly)
        plt.xlabel("Aperture radius")
        plt.ylabel("Chi2")
        plt.savefig("{5}/PhD/SpitzerTransits/{0}{4}/{0}_{1}_{2}_{3}_RadiusvsChi2.png".format(planet, AOR, channel,"poly",foldext, os.getenv('HOME')))
        plt.close()
    except:
        pass

    f.write("{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {} \n".format("poly", AOR, channel,
                                                    chi2_poly[index[0][0]][index[1][0]][index[2][0]],
                                                    bkg_methods[index[0][0]], bkg_methods_params[index[0][0]][0],
                                                    bkg_methods_params[index[0][0]][1],bkg_methods_params[index[0][0]][2],
                                                    cent_methods[index[1][0]], cent_methods_params[index[1][0]],
                                                    photom_methods_params[index[2][0]],
                                                    binsize_methods_params[index[3][0]]))

    # Save the values that the pipeline was explored over
    np.save('{4}/PhD/SpitzerTransits/{0}{3}/{0}_{1}_{2}_bkgMethodsParams'.format(planet, AOR, channel,foldext, os.getenv('HOME')), np.array(bkg_methods_labels))
    np.save('{4}/PhD/SpitzerTransits/{0}{3}/{0}_{1}_{2}_centMethodsParams'.format(planet, AOR, channel,foldext, os.getenv('HOME')), np.array(cent_methods_labels))
    np.save('{4}/PhD/SpitzerTransits/{0}{3}/{0}_{1}_{2}_photomMethodsParams'.format(planet, AOR, channel,foldext, os.getenv('HOME')), np.array(photom_methods_params))
    np.save('{4}/PhD/SpitzerTransits/{0}{3}/{0}_{1}_{2}_binsizeMethodsParams'.format(planet, AOR, channel,foldext, os.getenv('HOME')), np.array(binsize_methods_params))

    print "\nSaving PLD and results for AOR:{}...".format(AOR)

    # Save the chi2 PLD grid to a numpy object
    np.save('{5}/PhD/SpitzerTransits/{0}{4}/{0}_{1}_{2}_{3}_Chi2grid'.format(planet, AOR, channel,"PLD",foldext, os.getenv('HOME')), chi2_PLD)

    index_PLD = np.where(chi2_PLD == np.nanmin(chi2_PLD))

    try:
        chi2_photom_PLD = chi2_PLD[index_PLD[0][0]][index_PLD[1][0][index_PLD[2][0]]]
        print len(chi2_photom_PLD), len(photom_methods_params)
        plt.plot(photom_methods_params, chi2_photom_PLD)
        plt.xlabel("Aperture radius")
        plt.ylabel("Chi2")
        plt.savefig("{5}/PhD/SpitzerTransits/{0}{4}/{0}_{1}_{2}_{3}_RadiusvsChi2.png".format(planet, AOR, channel,"PLD",foldext, os.getenv('HOME')))
        plt.close()
    except:
        pass

    f.write("{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {} \n".format("PLD", AOR, channel,
                                                    chi2_PLD[index_PLD[0][0]][index_PLD[1][0]][index_PLD[2][0]],
                                                    bkg_methods[index_PLD[0][0]], bkg_methods_params[index_PLD[0][0]][0],
                                                    bkg_methods_params[index_PLD[0][0]][1],bkg_methods_params[index_PLD[0][0]][2],
                                                    cent_methods[index_PLD[1][0]], cent_methods_params[index_PLD[1][0]],
                                                    photom_methods_params[index_PLD[2][0]],
                                                    binsize_methods_params[index_PLD[3][0]]))


f.close()

labels_PLD = [key for key in coeffs_tuple_PLD if key not in fix_coeffs_PLD]
labels_poly = [ key for key in coeffs_tuple_poly if key not in fix_coeffs_poly]

# Make diagnostic plots of the pipeline parameters
# for m in range(len(AORs)):
#     AOR = AORs[m]
#     channel = channels[m]
#     pipelineOptPlots(planet, channel, 'poly', AOR, labels_poly, saveplots = True, foldext = foldext)
#     pipelineOptPlots(planet, channel, 'PLD', AOR, labels_PLD, saveplots = True, foldext = foldext)
