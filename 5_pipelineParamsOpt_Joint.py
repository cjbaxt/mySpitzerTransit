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

# Custom imports
sys.path.insert(0, '{}/PhD/code/mySpitzerTransit/'.format(os.getenv('HOME')))
from timeseries_routines import *
from transitFitting_routines import *
sys.path.insert(1, '{}/PhD/code/'.format(os.getenv('HOME')))
from ProgressBar import *

# Get input file
try:
    inFile = sys.argv[1]
    #inFile = '{}/PhD/SpitzerData/Kepler9b/Kepler9b_eccentric.txt'.format(os.getenv('HOME'))
except:
    raise ValueError('You need to provide input planet file as arguement.')

print "Using planet data from {}".format(inFile)

# Read in the planet information from a text file
inputData = np.genfromtxt(inFile, dtype=None, delimiter=': ', comments='#')
planet = inputData[0][1]
AORs = inputData[1][1].split(', ')
channels = inputData[2][1].split(', ')
t0s = inputData[3][1].split(', ')
cutstarts = [float(x) for x in inputData[4][1].split(', ')]
cutends = [float(x) for x in inputData[5][1].split(', ')]
posGuess = [float(x) for x in inputData[6][1].split(', ')]
print posGuess
ldlaw = inputData[int(np.where(inputData.T[0]=='limb_dark')[0])][1]
foldext = raw_input("Provide folder extension (hit ENTER for None): ")

# Open a file to save the best pipeline parameters to
f = open("{2}/PhD/SpitzerData/{0}/pipelineParams_Joint_{0}{1}.txt".format(planet,foldext, os.getenv('HOME')), 'w')
f.write("method, AOR, channel, chi2, bgk_method, bkg_boxsize, bkg_annradius, bkg_annsize, cent_method, cent_params, photom_radius \n")

#Background optimisation methods
bkg_methods_labels = ["Box_2", "Box_4",
              "Ann_6/2", "Ann_8/2",
              "Ann_6/4", "Hist"]
bkg_methods = ["Box", "Box",
              "Annulus", "Annulus",
              "Annulus", "Histogram"]
bkg_methods_params = [[2,None,None], [4,None,None],
              [None,6,2], [None,8,2],
              [None,6,4], [None,None,None]] #box_size, ann_radius, ann_size

#Centroid optimisation methods
cent_methods_labels = ["Bary_3", "Bary_5", "Bary_7", "Gaussian", "Moffat"]
cent_methods = ["Barycenter", "Barycenter", "Barycenter", "Gaussian", "Moffat"]
cent_methods_params = [3,5,7,None,None]

#Aperture Photometry Optimisation methods
photom_methods_params = np.arange(2.5, 4, 0.25).tolist()

coeffs_dict_poly = dict()
coeffs_dict_PLD = dict()

# Create a dictionary of the polynomial parameters...
coeffs_tuple_poly = ('t0', 'per', 'rp', 'a', 'inc', 'ecc', 'w', 'u', 'limb_dark',
               'K1', 'K2', 'K3', 'K4', 'K5',
               'f', 'g', 'h')


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

# Create a dictionary of the PLD paramters...
coeffs_tuple_PLD = ('t0', 'per', 'rp', 'a', 'inc', 'ecc', 'w', 'u', 'limb_dark',
               'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9',
               'g', 'h')

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

# Read in the planet information from a text file
# Create dictionary and fill of the stellar parameters (for limb darkening)
star_params = {'Teff':0,'logg':0,'z':0,'Tefferr':0,'loggerr':0,'zerr':0}
for key in star_params:
    star_params[key] = float(inputData[int(np.where(inputData.T[0]==key)[0])][1])

chi2_poly = np.zeros((len(bkg_methods_params), len(cent_methods_params), len(photom_methods_params)))
chi2_PLD = np.zeros((len(bkg_methods_params), len(cent_methods_params), len(photom_methods_params)))

nsteps = len(bkg_methods_params)*len(cent_methods_params)*len(photom_methods_params)
nparams_poly = len(coeffs_tuple_poly) - len(fix_coeffs_poly)
nparams_PLD = len(coeffs_tuple_PLD) - len(fix_coeffs_PLD)
samples_poly = np.zeros((nsteps,nparams_poly))
samples_PLD = np.zeros((nsteps,nparams_PLD))

midtimes_list, timeseries_badpix_list = [],[]
for m in range(len(AORs)):

    channel = channels[m]
    AOR = AORs[m]

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

    #Fix bad pixles
    sigma_badpix = 4
    nframes = 30
    timeseries_badpix = fast_bad_pix_mask(timeseries, sigma_badpix, nframes)

    #Sve to a list so that we only have to do this step once.
    midtimes_list.append(midtimes)
    timeseries_badpix_list.append(timeseries_badpix)



coeffs_dict_poly_list, coeffs_dict_PLD_list = [],[]
lc_list_poly, lcerr_list_poly, t_list_poly, x_list_poly, y_list_poly, bkg_list_poly, scales_list_poly, T0_list_poly = [],[],[],[],[],[],[],[]
lc_list_PLD, lcerr_list_PLD, t_list_PLD, centroids_list_PLD, timeseries_list_PLD, bkg_list_PLD, scales_list_PLD, T0_list_PLD = [],[],[],[],[],[],[],[]

p = ProgressBar(nsteps)
count = 0
print "\nEntering pipeline optimisation loop..."
for i in range(len(bkg_methods_params)):
    for j in range(len(cent_methods_params)):
        for k in range(len(photom_methods_params)):
            p.animate(count)
            # Create lightcurve lists

            for m in range(len(AORs)): # loop over the AORs

                # This can change with each AOR
                for label in coeffs_tuple_poly:
                    if label != 'u':
                        try:
                            coeffs_dict_poly[label] = float(inputData[int(np.where(inputData.T[0]==label)[0])][1].split(', ')[m])
                        except:
                            try: # If there is nothing to split
                                coeffs_dict_poly[label] = float(inputData[int(np.where(inputData.T[0]==label)[0])][1])
                            except: # the limb darkening law
                                coeffs_dict_poly[label] = inputData[int(np.where(inputData.T[0]==label)[0])][1]

                for label in coeffs_tuple_PLD:
                    if label != 'u':
                        try:
                            coeffs_dict_PLD[label] = float(inputData[int(np.where(inputData.T[0]==label)[0])][1].split(', ')[m])
                        except:
                            try: # If there is nothing to split
                                coeffs_dict_PLD[label] = float(inputData[int(np.where(inputData.T[0]==label)[0])][1])
                            except: # the limb darkening law
                                coeffs_dict_PLD[label] = inputData[int(np.where(inputData.T[0]==label)[0])][1]

                channel = channels[m]
                AOR = AORs[m]

                # Get limb darkening coefficients
                ldcoeffs, ldcoeffs_err = getldcoeffs(star_params['Teff'],star_params['logg'],star_params['z'],
                                                     star_params['Tefferr'],star_params['loggerr'],star_params['zerr'],
                                                     law = ldlaw, channel = channel, quiet = True)

                # Figure out which cutstarts and cutends to use
                cutstart = cutstarts[np.where(np.array(AORs) == AOR)[0][0]]
                cutend = cutends[np.where(np.array(AORs) == AOR)[0][0]]
                t0guess = t0s[np.where(np.array(AORs) == AOR)[0][0]]

                #datapath = "{3}/PhD/SpitzerData/{0}/{1}/{2}/bcd/".format(planet,AOR,channel, os.getenv('HOME'))

                coeffs_dict_poly['t0'] = t0guess
                coeffs_dict_poly['u'] = ldcoeffs # Must be list!
                coeffs_dict_PLD['t0'] = t0guess
                coeffs_dict_PLD['u'] = ldcoeffs # Must be list!

                # Run pipeline and create the lightcurve and timeseries
                lightcurve_red, timeseries_red, centroids_red, midtimes_red, background_red =  runPipeline(timeseries_badpix_list[m], midtimes_list[m],
                       method_bkg = bkg_methods[i], method_cent = cent_methods[j],
                       ramp_time = (cutstarts[m])*60,
                       end_time = (cutends[m])*60,
                       frametime = framtime,
                       x0guess = posGuess[0], y0guess = posGuess[1],
                       sigma_clip_cent = 4, iters_cent = 2, nframes_cent = 30, radius_photom = photom_methods_params[k],
                       sigma_clip_phot = 4, iters_photom = 2, nframes_photom = 30,
                       size_bkg_box = bkg_methods_params[i][0] , radius_bkg_ann = bkg_methods_params[i][1], size_bkg_ann = bkg_methods_params[i][2],
                       size_cent_bary = cent_methods_params[j], quiet = True, foldext=foldext)

                lc = lightcurve_red*MJysr2lelectrons
                lcerr = np.sqrt(lc)
                scale = np.median(lc[:100])
                lc, lcerr = lc/scale, lcerr/scale
                t = (midtimes_red - midtimes_red[0])
                x, y = centroids_red[:,1], centroids_red[:,0]

                # Initial fit to get a new scale and increase the errors
                result, batman_params, poly_params = fit_function_poly(coeffs_dict_poly, coeffs_tuple_poly,
                                                                       fix_coeffs_poly, t, x, y, lc)
                popt = result.x

                # Plot the least squares result
                plot_lightcurve(t,  lc, lcerr, popt, coeffs_dict_poly, coeffs_tuple_poly, fix_coeffs_poly, batman_params, poly_params,
                                    x=x, y=y, errors = False, binsize = 50,
                                    name = planet, channel = channel, orbit = AOR, savefile = True, TT_hjd = None,
                                    method = 'poly', color = 'b', scale = scale, filext = "lsq", foldext = foldext)


                coeffs_dict_poly_list.append(coeffs_dict_poly)
                lc_list_poly.append(lc)
                lcerr_list_poly.append(lcerr)
                t_list_poly.append(t)
                x_list_poly.append(x)
                y_list_poly.append(y)
                bkg_list_poly.append(background_red)
                T0_list_poly.append(midtimes_red[0])
                scales_list_poly.append(scale)

                #PLD
                lc = lightcurve_red*MJysr2lelectrons
                lcerr = np.sqrt(lc)
                scale = np.median(lc[:100])
                lc, lcerr = lc/scale, lcerr/scale
                t = (midtimes_red - midtimes_red[0])
                x, y = centroids_red[:,1], centroids_red[:,0]

                result_PLD, batman_params_PLD, PLD_params, Pns = fit_function_PLD(coeffs_dict_PLD, coeffs_tuple_PLD, fix_coeffs_PLD, t, timeseries_red, centroids_red, lc)
                popt_PLD = result_PLD.x

                # Plot the least squares result
                plot_lightcurve(t,  lc, lcerr, popt_PLD, coeffs_dict_PLD, coeffs_tuple_PLD, fix_coeffs_PLD, batman_params_PLD, PLD_params,
                                    Pns = Pns, errors = False, binsize = 50,
                                    name = planet, channel = channel, orbit=AOR, savefile = True, TT_hjd = None,
                                    method = 'PLD', color = 'r', scale = scale, filext = "lsq", foldext=foldext)

                # Write stuff to lists for looping
                coeffs_dict_PLD_list.append(coeffs_dict_PLD)
                lc_list_PLD.append(lc)
                lcerr_list_PLD.append(lcerr)
                t_list_PLD.append(t)
                centroids_list_PLD.append(centroids_red)
                timeseries_list_PLD.append(timeseries_red)
                bkg_list_PLD.append(background_red)
                T0_list_PLD.append(midtimes_red[0])
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
            chi2_A = 0.
            chi2_B = 0.

            for n in range(len(coeffs_dict_poly_list)):

                coeffs_fit = np.concatenate((opt_poly[n*x3:n*x3 + x1],
                                               opt_poly[-x2:],
                                               opt_poly[n*x3 + x1 :(n+1)*x3 ]))


                plot_lightcurve(t_list_poly[n],  lc_list_poly[n], lcerr_list_poly[n], coeffs_fit, coeffs_dict_poly_list[n],
                                coeffs_tuple_poly, fix_coeffs_poly, batman_params_list_poly[n], poly_params_list[n],
                                x=x_list_poly[n], y=y_list_poly[n], errors = True, binsize = 50,
                                name = planet, channel = channel, orbit = AOR, savefile = True, TT_hjd = None,
                                method = 'poly', color = 'b', scale = scales_list_poly[n], foldext=foldext, filext = "pipelineOpt_Joint_{}".format(count))

                chi2_A += chi(coeffs_fit, t_list_poly[n], lc_list_poly[n], lcerr_list_poly[n], coeffs_dict_poly_list[n],
                                coeffs_tuple_poly, fix_coeffs_poly, batman_params_list_poly[n], poly_params_list[n],
                                            x=x_list_poly[n], y=y_list_poly[n], method = "poly")/(len(lc_list_poly[n])-len(coeffs_fit))

            chi2_poly[i][j][k] = chi2_A

            print chi2_A

            # PLD least squares
            opt_PLD,batman_params_list_PLD,PLD_params_list,Pns_list =  fit_function_PLD_Ntransits(coeffs_dict_PLD_list, coeffs_tuple_PLD,
                                            fix_coeffs_PLD, fix_coeffs_channels_PLD,
                                            t_list_PLD, timeseries_list_PLD, centroids_list_PLD, lc_list_PLD)

            x1 = len([key for key in coeffs_tuple_PLD[0:9] if key not in fix_coeffs_PLD
                       and key not in fix_coeffs_channels_PLD])
            x2 = len([key for key in coeffs_tuple_PLD[0:9] if key in fix_coeffs_channels_PLD])
            x3 = len([key for key in coeffs_tuple_PLD if key not in fix_coeffs_PLD
                                  and key not in fix_coeffs_channels_PLD])

            for n in range(len(coeffs_dict_PLD_list)):

                coeffs_fit = np.concatenate((opt_PLD[n*x3:n*x3 + x1],
                                               opt_PLD[-x2:],
                                               opt_PLD[n*x3 + x1 :(n+1)*x3 ]))

                plot_lightcurve(t_list_PLD[n],  lc_list_PLD[n], lcerr_list_PLD[n], coeffs_fit, coeffs_dict_PLD_list[n],
                                coeffs_tuple_PLD, fix_coeffs_PLD, batman_params_list_PLD[n], PLD_params_list[n],
                                Pns = Pns_list[n], errors = False, binsize = 50,
                                name = planet, channel = channel, orbit = AOR, savefile = True, TT_hjd = None,
                                method = 'PLD', color = 'r', scale = scales_list_PLD[n], filext = "pipelineOpt_Joint_{}".format(count),
                                foldext=foldext)

                chi2_B += chi(coeffs_fit, t_list_PLD[n],  lc_list_PLD[n], lcerr_list_PLD[n], coeffs_dict_PLD,
                                coeffs_tuple_PLD, fix_coeffs_PLD, batman_params_list_PLD[n], PLD_params_list[n],
                                            Pns = Pns_list[n], method = "PLD")/(len(lc_list_PLD[n])-len(coeffs_fit))

            chi2_PLD[i][j][k] = chi2_B

            print chi2_B

            count += 1



np.save('{5}/PhD/SpitzerTransits/{0}{4}/{0}_{1}_{2}_{3}_pipelineOptSamples_Joint'.format(planet, AOR, channel,"poly",foldext, os.getenv('HOME')), samples_poly)
np.save('{5}/PhD/SpitzerTransits/{0}{4}/{0}_{1}_{2}_{3}_pipelineOptSamples_Joint'.format(planet, AOR, channel,"PLD",foldext, os.getenv('HOME')), samples_PLD)

print "\nSaving polynomial and results for AOR:{}...".format(AOR)

# Save the chi2 polynomial grid to a numpy object
np.save('{5}/PhD/SpitzerTransits/{0}{4}/{0}_{1}_{2}_{3}_Chi2grid_Joint'.format(planet, AOR, channel,"poly",foldext, os.getenv('HOME')), chi2_poly)

index = np.where(chi2_poly == chi2_poly.min())

try:
    chi2_photom_poly = chi2_poly[index[0][0]][index[1][0]]
    plt.plot(photom_methods_params, chi2_photom_poly)
    plt.xlabel("Aperture radius")
    plt.ylabel("Chi2")
    plt.savefig("{5}/PhD/SpitzerTransits/{0}{4}/{0}_{1}_{2}_{3}_RadiusvsChi2_Joint.png".format(planet, AOR, channel,"poly",foldext, os.getenv('HOME')))
    plt.close()
except:
    pass

f.write("{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {} \n".format("poly", AOR, channel,
                                                chi2_poly[index[0][0]][index[1][0]][index[2][0]],
                                                bkg_methods[index[0][0]], bkg_methods_params[index[0][0]][0],
                                                bkg_methods_params[index[0][0]][1],bkg_methods_params[index[0][0]][2],
                                                cent_methods[index[1][0]], cent_methods_params[index[1][0]],
                                                photom_methods_params[index[2][0]]))

# Save the values that the pipeline was explored over
np.save('{4}/PhD/SpitzerTransits/{0}{3}/{0}_{1}_{2}_bkgMethodsParams_Joint'.format(planet, AOR, channel,foldext, os.getenv('HOME')), np.array(bkg_methods_labels))
np.save('{4}/PhD/SpitzerTransits/{0}{3}/{0}_{1}_{2}_centMethodsParams_Joint'.format(planet, AOR, channel,foldext, os.getenv('HOME')), np.array(cent_methods_labels))
np.save('{4}/PhD/SpitzerTransits/{0}{3}/{0}_{1}_{2}_photomMethodsParams_Joint'.format(planet, AOR, channel,foldext, os.getenv('HOME')), np.array(photom_methods_params))

print "\nSaving PLD and results for AOR:{}...".format(AOR)

# Save the chi2 PLD grid to a numpy object
np.save('{4}/PhD/SpitzerTransits/{0}{4}/{0}_{1}_{2}_{3}_Chi2grid_Joint'.format(planet, AOR, channel,"PLD",foldext, os.getenv('HOME')), chi2_PLD)

index_PLD = np.where(chi2_PLD == chi2_PLD.min())

try:
    chi2_photom_PLD = chi2_PLD[index_PLD[0][0]][index_PLD[1][0]]
    print len(chi2_photom_PLD), len(photom_methods_params)
    plt.plot(photom_methods_params, chi2_photom_PLD)
    plt.xlabel("Aperture radius")
    plt.ylabel("Chi2")
    plt.savefig("{5}/PhD/SpitzerTransits/{0}{4}/{0}_{1}_{2}_{3}_RadiusvsChi2_Joint.png".format(planet, AOR, channel,"PLD",foldext, os.getenv('HOME')))
    plt.close()
except:
    pass

f.write("{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {} \n".format("PLD", AOR, channel,
                                                chi2_PLD[index_PLD[0][0]][index_PLD[1][0]][index_PLD[2][0]],
                                                bkg_methods[index_PLD[0][0]], bkg_methods_params[index_PLD[0][0]][0],
                                                bkg_methods_params[index_PLD[0][0]][1],bkg_methods_params[index_PLD[0][0]][2],
                                                cent_methods[index_PLD[1][0]], cent_methods_params[index_PLD[1][0]],
                                                photom_methods_params[index_PLD[2][0]]))



#########################################
f.close()

labels_PLD = [key for key in coeffs_tuple_PLD if key not in fix_coeffs_PLD]
labels_poly = [ key for key in coeffs_tuple_poly if key not in fix_coeffs_poly]

# Make diagnostic plots of the pipeline parameters
for m in range(len(AORs)):
    AOR = AORs[m]
    channel = channels[m]
    pipelineOptPlots(planet, channel, 'poly', AOR, labels_poly, saveplots = True, foldext = foldext)
    pipelineOptPlots(planet, channel, 'PLD', AOR, labels_PLD, saveplots = True, foldext = foldext)
