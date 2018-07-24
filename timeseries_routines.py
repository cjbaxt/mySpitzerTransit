# File contraining classes and functions used for the reduction of Spitzer
# lightcurves.

# Imports
from astropy.io import fits
import numpy as np
import glob
import scipy
import warnings, sys, time, os
from astropy.modeling.functional_models import Gaussian2D
from astropy.modeling.functional_models import Gaussian1D
from astropy.modeling.functional_models import Moffat2D
from astropy.modeling import fitting
from photutils import CircularAperture
from photutils import aperture_photometry
from photutils import CircularAnnulus
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

#This is a test edit of a new branch

# Class for calculating the standard deviation of the aperture photometry for
# finding the best radius value. This function currently just assumes that the
# lightcurve would be a straight line, this is not right and thus this feature
# is still to be implemented properly.
class stdPhotom():
    def __init__(self, data):
        """Constructor with data"""
        self.data = data

    def evaluate(self, radius):
        "Method to evaluate the weighted standard deviation of the aperture photometry"
        positions = [(15.,15.)]
        apertures = CircularAperture(positions, r=radius)
        photoms = [aperture_photometry(frame, apertures, method='subpixel', subpixels=5)[0][0] for frame in self.data]
        return np.std(photoms)/(np.mean(photoms))

# Class to read the files, create timeseries and get infotmation from header
class read_files():
    """Class to read the headers of spitzer IRAC data."""

    def __init__(self, path):
        """Constructor to initialise path containing fits files."""
        self.path = path

    def create_timeseries(self):
        """path = path to folder containing the fits images for one epoch in one band"""

        print  "\nReading data from '{}' ... ".format(self.path)
        files = sorted(glob.glob(self.path + '*bcd.fits'))

        if len(files) < 1:
            raise ValueError("No files found, check directory path!")

        test = fits.open(files[0]) #get image dimensions from first frame

        naxis = test[0].header['naxis']
        naxis1 = test[0].header['naxis1']
        naxis2 = test[0].header['naxis2']

        #Create timeseries for subarray mode
        if naxis == 3:
            naxis3 = test[0].header['naxis3']
            data = np.ndarray((len(files)*naxis3,naxis2,naxis1))
            times = np.ndarray(data.shape)

            for i in range(len(files)):
                hdulist = fits.open(files[i])
                scidata = hdulist[0].data
                for j in range(len(scidata)):
                    data[j+i*len(scidata)] = scidata[j]

        #Create timeseries for full mode
        elif naxis == 2:
            data = np.ndarray(len(files),naxis2,naxis1)
            times = np.ndarray(data.shape)

            for i in range(len(files)):
                hdulist = fits.open(files[i])
                scidata = hdulist[0].data
                data[i] = scidata

        print "\t Timeseries created with shape {}".format(data.shape)
        return np.nan_to_num(data)

    def create_mask_timeseries(self):
        """path = path to folder containing the fits images for one epoch in one band"""

        files = sorted(glob.glob(self.path + '*imsk.fits'))
        test = fits.open(files[0]) #get image dimensions from first frame

        naxis = test[0].header['naxis']
        naxis1 = test[0].header['naxis1']
        naxis2 = test[0].header['naxis2']

        #Create timeseries for subarray mode
        if naxis == 3:
            naxis3 = test[0].header['naxis3']
            data = np.ndarray((len(files)*naxis3,naxis2,naxis1))
            times = np.ndarray(data.shape)

            for i in range(len(files)):
                hdulist = fits.open(files[i])
                scidata = hdulist[0].data
                for j in range(len(scidata)):
                    data[j+i*len(scidata)] = scidata[j]

        #Create timeseries for full mode
        elif naxis == 2:
            data = np.ndarray(len(files),naxis2,naxis1)
            times = np.ndarray(data.shape)

            for i in range(len(files)):
                hdulist = fits.open(files[i])
                scidata = hdulist[0].data
                data[i] = scidata

        print "\t Maskseries created with shape {}".format(data.shape)
        return data

    def calculate(self):
        """Calculate and create arrays containing the information from the headers.
        midtimes = Array containing the midtimes of each frame. Subarray & Full array mode"""
        files = sorted(glob.glob(self.path + '*bcd.fits'))

        if len(files) < 1:
            raise ValueError("No files found, check directory path! \n \t{}".format(self.path))

        nfits = len(files)

        hdulist = fits.open(files[0])
        header = hdulist[0].header
        naxis = header['naxis']

        if naxis == 3:
            naxis3 = header['naxis3']
        else:
            naxis3 = 1

        midtimes = np.zeros( nfits*naxis3 )
        exptimes = np.zeros( nfits*naxis3 )
        readnoises = np.zeros( nfits*naxis3 )
        gains = np.zeros( nfits*naxis3 )
        fluxconvs = np.zeros( nfits*naxis3 )
        framtimes = np.zeros( nfits*naxis3 )

        for i in range(nfits):

            hdulist = fits.open(files[i])
            header = hdulist[0].header
            naxis = header['naxis']
            frametime = header['FRAMTIME']/(24.*60.*60.) #Convert to days

            if naxis == 3:
                naxis3 = header['naxis3']
            else:
                naxis3 = 1

            for j in range(naxis3):

                midtimes[ j+i*naxis3 ] = header['BMJD_OBS'] + (0.5 + j)*frametime
                exptimes[ j+i*naxis3 ] = header['EXPTIME']
                readnoises[ j+i*naxis3 ] = header['RONOISE']
                gains[ j+i*naxis3 ] = header['GAIN']
                fluxconvs[ j+i*naxis3 ] = header['FLUXCONV']
                framtimes[ j+i*naxis3 ] = header['FRAMTIME']

        return (midtimes, exptimes, readnoises, gains, fluxconvs, framtimes)

    def midtimes(self):
        midtimes = self.calculate()[0]
        print "\t Midtimes created with shape {}".format(midtimes.shape)
        return midtimes

    def exptime(self):

        exptimes = self.calculate()[1]
        if np.diff( exptimes ).max()==0:
            exptime = exptimes[0]
        else:
            warnings.warn( 'exptimes not all identical - taking median' )
            exptime = np.median( exptimes )

        return exptime

    def readnoise(self):

        readnoises = self.calculate()[2]
        if np.diff( readnoises ).max()==0:
            readnoise = readnoises[0]
        else:
            warnings.warn( 'readnoises not all identical - taking median' )
            readnoise = np.median( readnoises )

        return readnoise

    def gain(self):

        gains = self.calculate()[3]
        if np.diff( gains ).max()==0:
            gain = gains[0]
        else:
            warnings.warn( 'gains not all identical - taking median' )
            gain = np.median( gains )

        return gain

    def fluxconv(self):

        fluxconvs = self.calculate()[4]
        if np.diff( fluxconvs ).max()==0:
            fluxconv = fluxconvs[0]
        else:
            warnings.warn( 'fluxconvs not all identical - taking median' )
            fluxconv = np.median( fluxconvs )

        return fluxconv

    def framtime(self):

        framtimes = self.calculate()[5]
        if np.diff( framtimes ).max()==0:
            framtime = framtimes[0]
        else:
            warnings.warn( 'framtimes not all identical - taking median' )
            framtime = np.median( framtimes )

        return framtime

# Functions for data reduction
def pix_timeseries(timeseries, el1, el2):
    #Create empty array to append pixels
    pixseries = np.zeros(timeseries.shape[0])

    #Loop over all the frames in time series
    for i in range(len(pixseries)):
        pixseries[i] = timeseries[i][el1][el2]

    return pixseries

def correct_bad_pix(timeseries, bad_index, nframes):
    """Fucntion that returns a corrected time series given the location of a
    transient bad pixel"""

    frameno = bad_index[0]
    el1 = bad_index[1]
    el2 = bad_index[2]

    pix_flux = pix_timeseries(timeseries,el1,el2)
    length = nframes/2
    start = frameno - length
    end = frameno + length + 1
    if start < 0:
        start = 0
    pix_flux[frameno] = np.nan
    section = pix_flux[start: end]
    no_nans = np.isnan
    median = np.nanmedian(section)
    timeseries[frameno][el1][el2] = median
    return timeseries

def bad_pix_mask(timeseries, sigmathresh, quiet = False):
    """Mask the bad pixels over a certain threshold.
    timeseries = N x 2dframeshape array for bad pixel correction
    sigmathresh = number of sigmas to create the cut for bad pixels"""

    if not quiet: print "\nMasking bad pixels..."
    # Find bad pixels
    bad_indices = []
    counter = 0
    for i in range(timeseries.shape[1]):
        for j in range(timeseries.shape[2]):
            pix_flux = pix_timeseries(timeseries,i,j)
            #Create sliding median
            for k in range(timeseries.shape[0]):
                idx_start = k-30
                idx_end = k+1+30
                if idx_start < 0:
                    idx_start=0
                pix_flux_range = np.append( pix_flux[idx_start : k], pix_flux[k+1 : idx_end])
                median = np.median(pix_flux_range)
                stddev = np.std(pix_flux_range)
                thresh_u = median+sigmathresh*stddev
                thresh_l = median-sigmathresh*stddev
                if pix_flux[k] > thresh_u or pix_flux[k] < thresh_l:
                    bad_index = [k,i,j]
                    timeseries = correct_bad_pix(timeseries, bad_index)
                    counter += 1
                    pix_flux = pix_timeseries(timeseries,i,j)
                else:
                    pass

    if not quiet: print "\t {0} out of {1} bad pixels masked".format(counter, timeseries.shape[0]*timeseries.shape[1]*timeseries.shape[2])
    return timeseries

def fast_bad_pix_mask(timeseries, sigmathresh, nframes, quiet = False, foldext=None):
    """Function that looks for transient bad pixels, outliers in the data and corrects
    them to the median of the surrounding pixels in the timeseries"""

    if not quiet: print "\nMasking bad pixels at {} sigma...".format(sigmathresh)
    startidx = 0
    counter = 0

    # Loop over the whole timeseries looking for bad pixels
    while startidx < timeseries.shape[0]:
        endidx = startidx + nframes
        timeseries_snip = timeseries[startidx:endidx]
        median = np.median(timeseries_snip, axis=0)
        stddev = np.std(timeseries_snip, axis=0)
        diff = timeseries_snip
        thresh_u = median + sigmathresh*stddev
        thresh_l = median - sigmathresh*stddev

        # Mask the bad pixels greater than the threshold
        bad_index = np.where(timeseries_snip > thresh_u)
        counter += len(bad_index[0])

        for i in range(len(bad_index[0])):
            bad = [bad_index[0][i]+startidx, bad_index[1][i], bad_index[2][i]]
            timeseries = correct_bad_pix(timeseries, bad, nframes)

        # Mask the bad pixels lower than the thre
        bad_index = np.where(timeseries_snip < thresh_l)
        counter += len(bad_index[0])

        for i in range(len(bad_index[0])):
            bad = [bad_index[0][i]+startidx, bad_index[1][i], bad_index[2][i]]
            timeseries = correct_bad_pix(timeseries, bad, nframes)

        startidx += nframes

    if not quiet: print "\t {0} out of {1} bad pixels masked ({2:.3f}%)".format(counter, timeseries.shape[0]*timeseries.shape[1]*timeseries.shape[2],
                                                                            float(counter*100.)/float(timeseries.shape[0]*timeseries.shape[1]*timeseries.shape[2]))

    return timeseries

##def old_bad_pix_mask(timeseries, sigmathresh, quiet = False):
##    """Mask the bad pixels over a certain threshold.
##    timeseries = N x 2dframeshape array for bad pixel correction
##    sigmathresh = number of sigmas to create the cut for bad pixels"""
##
##    if not quiet: print "\nMasking bad pixels..."
##    # Find bad pixels
##    bad_indices = []
##    for i in range(timeseries.shape[1]):
##        for j in range(timeseries.shape[2]):
##            pix_flux = pix_timeseries(timeseries,i,j)
##            thresh = np.mean(pix_flux)+sigmathresh*np.std(pix_flux)
##            pix_flux[pix_flux > thresh] = np.nan
##            nans = np.where(np.isnan(pix_flux))
##            if len(nans[0]) == 0:
##                pass
##            else:
##                for h in range(len(nans[0])):
##                    bad_indices.append([nans[0][h], i, j])
##
##    #Mask bad pixels
##    for k in range(len(bad_indices)):
##        frameno = bad_indices[k][0]
##        el1 = bad_indices[k][1]
##        el2 = bad_indices[k][2]
##        boxlen = 3
##        while (boxlen < 10):
##            #define box params
##            side1, side2 = boxlen, boxlen
##            start1, start2 = el1 - side1/2, el2 - side2/2
##            if start1 < 0: #Change the square if pixel at edge of image
##                start1, side1 = 0, side1/2 + 1
##            if start2 < 0:
##                start2, side2 = 0, side2/2 + 1
##            frame = timeseries[frameno]
##            square = frame[ start1 : start1+side1 , start2 : start2+side2 ] #Slice a square by the right dimensions
##            no_nans = np.isnan(square).sum()
##            fraction = no_nans/float(square.size)
##            if fraction <= 1./3.:
##                median = np.nanmedian(square.flatten()) # Calculate median of the square, ignoring nan
##                timeseries[frameno][el1][el2]=median
##                break
##            else:
##                boxlen = boxlen + 2
##                if boxlen > 9:
##                    warnings.warn( 'Bad pixel mask maximum box size reached.' )
##
##    if not quiet: print "\t {0} out of {1} bad pixels masked".format(len(bad_indices), timeseries.shape[0]*timeseries.shape[1]*timeseries.shape[2])
##    return timeseries

def bck_subtract(timeseries, method, positions=None, radius=None, size=None, boxsize = None, quiet = False, plot = False,
                plotting_binsize = None, AOR = None, planet = None, channel=None, sysmethod = None, foldext=None):

    if sysmethod == 'poly': c, c2 = 'b', '#ff7f0e'
    elif sysmethod == 'PLD': c, c2 = 'r', 'c'
    else: c, c2 = 'k', 'g'

    if not quiet: print "\nSubtracting background..."
    newtimeseries = np.ndarray(timeseries.shape)
    bkg = []

    if method == 'Annulus':
        """Positions = Pixel co-ordinates of the center of annulus
            radius = inner radius of annulus in pixels
            size = size of annulus in pixels"""

        for i in range(len(timeseries)):
            positions = [15,15]
            annulus = CircularAnnulus(positions, r_in = radius, r_out = radius+size)
            photom_bkg = aperture_photometry(timeseries[i], annulus, method='subpixel', subpixels=5)[0][0]
            area = np.pi*((radius+size)**2 - (radius)**2)
            photomperpix = photom_bkg/area
            bkg.append(photomperpix)
            newtimeseries[i] = timeseries[i] - photomperpix
        if not quiet: print "\t Background subtracted using {0} method with radius {1} and size {2}".format(method, radius, size)

    elif method == 'Box':
        """boxsize = number of pixels for the size of boxes in each corner"""

        for i in range(len(timeseries)):
            frame = timeseries[i]

            box1 = frame[ 0 : boxsize , 0 : boxsize ]
            box2 = frame[ frame.shape[0]-boxsize : frame.shape[0] , frame.shape[1]-boxsize : frame.shape[1] ]
            box3 = frame[ 0 : boxsize , frame.shape[1]-boxsize : frame.shape[1]]
            box4 = frame[ frame.shape[0]-boxsize : frame.shape[0] , 0 : boxsize ]

            pixelvalues = np.concatenate((box1.flatten(),box2.flatten(),box3.flatten(),box4.flatten()),axis=0)

            median = np.median(pixelvalues)
            bkg.append(median)
            newtimeseries[i] = timeseries[i] - median
        if not quiet: print "\t Background subtracted using {0} method with size {1}".format(method, boxsize)

    # elif method == 'Histogram':
    #     """Creates a hisogram of the pixel values in a frame, converts to data points and fits a gaussian to
    #     get the mean and the standard deviation of the background."""
    #
    #     for i in range(len(timeseries)):
    #         frame = timeseries[i]
    #         array = frame.flatten()
    #         array = array[array<5.] # This is a bit arbitrary...
    #
    #         # Create histogram
    #         counts, bins = np.histogram(array, bins=len(array)/10)
    #         xdata = np.ones(len(bins)-1)
    #         for i in range(len(bins)-1):
    #             xdata[i] = (bins[i+1]-bins[i])/2. + bins[i]
    #
    #         # Find the x position of the maximum in the histogram
    #         maxindex = np.where(counts == counts.max())
    #         if len(maxindex[0]) > 1:
    #             maxindex = maxindex[0][0]
    #
    #         # Fit a gaussian to the histogram
    #         g_init = Gaussian1D(amplitude=counts.max(), mean=xdata[maxindex], stddev=2.)
    #         fitter = fitting.LevMarLSQFitter()
    #         g = fitter(g_init, xdata, counts)
    #
    #         # # Create new clipped array
    #         # array = array[array<np.median(array)+2.*g.stddev]
    #         #
    #         # # Create second histogram
    #         # counts, bins = np.histogram(array, bins=len(array)/10)
    #         # xdata = np.ones(len(bins)-1)
    #         # for i in range(len(bins)-1):
    #         #     xdata[i] = (bins[i+1]-bins[i])/2. + bins[i]
    #         #
    #         # # Find the x position of the maximum in the histogram
    #         # maxindex = np.where(counts == counts.max())
    #         # if len(maxindex[0]) > 1:
    #         #     maxindex = maxindex[0][0]
    #         #
    #         # # Fit a gaussian to the second histogram
    #         # g_init = Gaussian1D(amplitude=counts.max(), mean=xdata[maxindex], stddev=g.stddev)
    #         # fitter = fitting.LevMarLSQFitter()
    #         # g = fitter(g_init, xdata, counts)
    #
    #         newtimeseries[i] = timeseries[i] - g.mean*1.
    #         bkg.append(g.mean*1.)
    #
    #     if not quiet: print "\t Background subtracted using {0} method".format(method)

    elif method == 'Histogram':

        for i in range(len(timeseries)):
            frame = timeseries[i]
            array = frame.flatten()
            iters = 5

            for j in range(iters):
                std_dev = np.std(array)
                median = np.median(array)

                array = array[array<median+std_dev] # This is a bit arbitrary...
                array = array[array>median-std_dev]

                median = np.median(array)

            newtimeseries[i] = timeseries[i] - median
            bkg.append(median)

        if not quiet: print "\t Background subtracted using {0} method".format(method)
    else:
        raise ValueError('Incorrect value for keyword method in background subtraction.')

    if plot:
        # Plot of all the background at each timestep
        fig, ax = plt.subplots(figsize=(20,5))
        ax.plot(bkg, label = method, c=c)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Background Flux (image units)")
        plt.savefig("{5}/PhD/SpitzerTransits/{0}{4}/{0}_{1}_{2}_{3}_bkg.png".format(planet, AOR, channel,sysmethod,foldext, os.getenv('HOME')))
        plt.close()

        # Plot of all the background at each timestep binned.
        binsize = plotting_binsize
        start, end = 0, binsize
        binned_bkg = []
        while end < len(bkg):
            binned_bkg.append( np.mean(bkg[start:end]) )
            start += binsize
            end += binsize
        fig, ax = plt.subplots(figsize=(20,5))
        ax.plot(binned_bkg, label = method, c=c)
        ax.set_xlabel("Frame (Binned x {})".format(binsize))
        ax.set_ylabel("Background Flux (image units)")
        plt.savefig("{5}/PhD/SpitzerTransits/{0}{4}/{0}_{1}_{2}_{3}_BinnedBkg.png".format(planet, AOR, channel,sysmethod,foldext,os.getenv('HOME')))
        plt.close()

        # This doesn't work because we have discarded some of the data before this step...
        fig3, ax3 = plt.subplots()
        composite_list = [bkg[k:k+64] for k in range(0,len(bkg), 64)]
        summed_list = np.median(composite_list, axis=0)
        ax3.plot(np.arange(len(summed_list)),summed_list, 'o', c=c)
        ax3.plot(57, summed_list[57], 'o', c = c2 )
        ax3.set_xlabel("Frame")
        ax3.set_ylabel("Flux (image units)")
        plt.savefig("{5}/PhD/SpitzerTransits/{0}{4}/{0}_{1}_{2}_{3}_bkgMedian.png".format(planet, AOR, channel, sysmethod, foldext, os.getenv('HOME')))
        plt.close()

        # if method == 'Histogram':
        #     fig2, ax2 = plt.subplots()
        #     ax2.hist(array,bins,c=c)
        #     ax2.plot(xdata, g(xdata), linewidth = 3, label='Fitted Gaussian',c=c2)
        #     ax2.legend()
        #     ax2.set_xlabel("Pixel Flux (image units)")
        #     ax2.set_ylabel("Counts")
        #     plt.savefig("{}/PhD/SpitzerTransits/{0}{4}/{0}_{1}_{2}_{3}_bkgHist.png".format(planet, AOR, channel, sysmethod, foldext))
        #     plt.close()
        #fig2, ax2 = plt.subplots(figsize=(10, 5))
        #composite_list = [bkg_box[x:x+64] for x in range(0, len(bkg_box),64)]
        #summed_list = np.median(composite_list, axis=0)
        #ax2.plot(np.arange(len(summed_list)),summed_list*MJysr2lelectrons, 'bo')
        #ax2.set_xlabel("Frame number")
        #ax2.set_ylabel("Flux (electrons)")
    return newtimeseries, np.array(bkg)

def discard_ramp(timeseries, midtimes, background, time, end_time, framtime, quiet = False, passenger57 = False, foldext=None):

    if not quiet: print "\nDiscarding ramp at beginning of observations and discarding passenger 57..."
    counts=0

    if passenger57:
        counts = len(timeseries)/64
        timeseries = np.delete(timeseries, slice(57,None,64), axis=0)
        midtimes = np.delete(midtimes, slice(57,None,64))
        background = np.delete(background, slice(57,None,64))
    else: pass

    if not quiet: print "\t {} passenger 57 frames discarded".format(int(counts))

    #Time and frametime must be in seconds
    nframes = int(float(time)/framtime)
    if end_time == 0.:
        nframesend = 0
        timeseries = timeseries[nframes:,]
        midtimes = midtimes[nframes:,]
        background = background[nframes:,]
    else:
        nframesend = int(float(end_time)/framtime)
        timeseries = timeseries[nframes:-nframesend,]
        midtimes = midtimes[nframes:-nframesend,]
        background = background[nframes:-nframesend,]

    if not quiet: print "\t {} frames discarded from beginning of observation".format(int(nframes))
    if not quiet: print "\t {} frames discarded from end of observation".format(int(nframesend))

    return timeseries, midtimes, background

def centroid(timeseries, method, boxsize = None, quiet = False, plot = False, AOR = None, planet = None, channel = None, sysmethod = None, foldext=None, x0guess=None, y0guess=None):

    if sysmethod == 'poly': c, c2 ='b', '#ff7f0e'
    elif sysmethod == 'PLD': c, c2 = 'r', 'c'
    else: c, c2 = 'k','g'

    if not quiet: print "\n Centroiding using {} method...".format(method)

    #Create array for storing the centroid positions of each frame
    centroids = np.zeros((len(timeseries),2))

    if method == 'Gaussian':

        y, x = np.mgrid[:timeseries.shape[1], :timeseries.shape[2]]

        for i in range(len(timeseries)):

            if i == 0:
                if x0guess == None and y0guess == None:
                    x0guess = np.where(timeseries[0] == timeseries[0].max())[1]
                    y0guess = np.where(timeseries[0] == timeseries[0].max())[0]
                    if len(x0guess) > 1:
                        if not quiet: "\t Warning - timeseries[0] had multiple maxima setting xguess to middle of array"
                        x0guess = timeseries.shape[1]/2
                    if len(y0guess) > 1:
                        if not quiet: "\t Warning - timeseries[0] had multiple maxima setting yguess to middle of array"
                        y0guess = timeseries.shape[1]/2
                else:
                    pass
            else:
                x0guess = centroids[i-1][0]
                y0guess = centroids[i-1][1]
            z = timeseries[i]

            #### GET RID OF THESE NUMBERS
            gaussian = Gaussian2D(amplitude=timeseries[i].max(), x_mean=x0guess, y_mean=y0guess, x_stddev=2., y_stddev=2.)
            fitter = fitting.LevMarLSQFitter()
            params = fitter(gaussian, x, y, z)
            centroids[i][0] = params.x_mean*1. + 0.5
            centroids[i][1] = params.y_mean*1. + 0.5


    elif method == 'Moffat':

        y, x = np.mgrid[:32, :32]

        for i in range(len(timeseries)):

            if i == 0:
                if x0guess == None and y0guess == None:
                    x0guess = np.where(timeseries[0] == timeseries[0].max())[1]
                    y0guess = np.where(timeseries[0] == timeseries[0].max())[0]
                    if len(x0guess) > 1:
                        if not quiet: "\t Warning - timeseries[0] had multiple maxima setting xguess to middle of array"
                        x0guess = timeseries.shape[1]/2
                    if len(y0guess) > 1:
                        if not quiet: "\t Warning - timeseries[0] had multiple maxima setting yguess to middle of array"
                        y0guess = timeseries.shape[1]/2
                else:
                    pass
            else:
                x0guess = centroids[i-1][0]
                y0guess = centroids[i-1][1]

            z = timeseries[i]
            ####GET RID OF THESE NUMBERS

            moffat = Moffat2D(amplitude=timeseries[i].max(), x_0=x0guess, y_0=y0guess)
            fitter = fitting.LevMarLSQFitter()
            params = fitter(moffat, x, y, z)
            centroids[i][0] = params.x_0*1. + 0.5
            centroids[i][1] = params.y_0*1. + 0.5

    elif method == 'Barycenter':

        if not quiet: print "\t Using boxsize {}...".format(boxsize)

        for i in range(len(timeseries)):
            if i == 0:
                if x0guess == None and y0guess == None:
                    x0guess = np.where(timeseries[0] == timeseries[0].max())[1]
                    y0guess = np.where(timeseries[0] == timeseries[0].max())[0]
                    if len(x0guess) > 1:
                        if not quiet: "\t Warning - timeseries[0] had multiple maxima setting xguess to middle of array"
                        x0guess = timeseries.shape[1]/2
                    if len(y0guess) > 1:
                        if not quiet: "\t Warning - timeseries[0] had multiple maxima setting yguess to middle of array"
                        y0guess = timeseries.shape[1]/2
                else:
                    pass
            else:
                x0guess = centroids[i-1][0]
                y0guess = centroids[i-1][1]

            ####GET RID OF THESE NUMBERS
            coordinates = fluxweight_centroid( *cut_subarray(timeseries[i], x0guess, y0guess, boxsize ) )

            centroids[i][0] = coordinates[0]
            centroids[i][1] = coordinates[1]

    else:
        raise ValueError('Incorrect value for keyword method in centroiding.')

    if not quiet: print "\t Centroiding complete: \n\t x_mean: {0:.4f} x_stddev: {1:.4f} \n\t y_mean: {2:.4f} y_stddev: {3:.4f}".format(np.mean(centroids[:,0]),
                                                                                                                   np.std(centroids[:,0]),
                                                                                                                   np.mean(centroids[:,1]),
                                                                                                                   np.std(centroids[:,1]))
    if plot:
        fig, ax = plt.subplots(2,1, figsize = (20,5))
        ax[0].plot(centroids[:,1], c=c)#, 'd', markersize = 5)
        ax[0].set_xlabel("Frame")
        ax[0].set_ylabel("x-pos")
        ax[1].plot(centroids[:,0], c=c)#, 'd', markersize = 5)
        ax[1].set_xlabel("Frame")
        ax[1].set_ylabel("y-pos")
        plt.savefig("{5}/PhD/SpitzerTransits/{0}{4}/{0}_{1}_{2}_{3}_centroids.png".format(planet, AOR, channel,sysmethod,foldext, os.getenv('HOME')))
        plt.close()

        fig2, ax2 = plt.subplots(1,1, figsize = (10,10))
        ax2.scatter(centroids[:,1], centroids[:,0], c = np.arange(len(centroids)), s = 4, alpha = 0.5)
        ax2.set_xlabel('x-pos')
        ax2.set_ylabel('y-pos')
        plt.savefig("{5}/PhD/SpitzerTransits/{0}{4}/{0}_{1}_{2}_{3}_centroidsMap.png".format(planet, AOR, channel,sysmethod,foldext, os.getenv('HOME')))
        plt.close()


    return centroids

def cut_subarray( fullarray, xcent, ycent, boxwidth ):
    """
    Cuts a subarray from a 2D array.
    """
    naxis1 = np.shape( fullarray )[1]
    naxis2 = np.shape( fullarray )[0]

    # Left-hand edges of x pixels, where 0 is
    # the left-hand edge of the first column:
    xpixs = np.arange( naxis1 )

    # Lower edges of y pixels, where 0 is the
    # lower edge of the first row:
    ypixs = np.arange( naxis2 )

    # Number of pixels either side of
    # central pixel in subarray:
    delpix = int( 0.5*( boxwidth - 1 ) )

    # Cut out the subarray:
    xixs = ( xpixs>=np.floor( xcent )-delpix )*\
           ( xpixs<np.floor( xcent )+1+delpix )
    yixs = ( ypixs>=np.floor( ycent )-delpix )*\
           ( ypixs<np.floor( ycent )+1+delpix )
    subarray = fullarray[yixs,:][:,xixs]

    # Convert coordinates from pixel edges to
    # pixel centers before returning:
    xsub = xpixs[xixs] + 0.5
    ysub = ypixs[yixs] + 0.5

    return subarray, xsub, ysub

def fluxweight_centroid( subarray, xsub, ysub ):
    """
    Used to also have fluxweight2d, but realised that
    is mathematically identical and nearly 50% slower.
    """

    # Negative flux values bias the output, so
    # set minimum value to be zero:
    subarray -= subarray.min()
    # Calculate the flux-weighted mean:
    marginal_x = np.sum( subarray, axis=0 )
    marginal_y = np.sum( subarray, axis=1 )
    fluxsumx = np.sum( marginal_x )
    x0 = np.sum( xsub*marginal_x )/fluxsumx
    fluxsumy = np.sum( marginal_y )
    y0 = np.sum( ysub*marginal_y )/fluxsumy

    return x0, y0

def sigma_clip_centroid(timeseries, centroids, midtimes, background, sigmaclip, iters, nframes, quiet = False, plot = False, AOR = None, planet = None, channel=None,sysmethod=None, foldext=None):
    """Perform 4 or 5 sigma clipping to the data and corresponding arrays """

    if sysmethod == 'poly': c, c2 ='b', '#ff7f0e'
    elif sysmethod == 'PLD': c, c2 = 'r', 'c'
    else: c, c2 = 'k','g'

    if not quiet: print "\nClipping frames >{0} sigma away from median centroid position...".format(sigmaclip)
    origmidtimes, origcentroids = midtimes, centroids
    for iter in range(iters):

        flagged_frames = np.full(timeseries.shape[0], True, dtype=bool)

        for i in range(len(centroids)):

            idx_start = i-(nframes/2)
            idx_end = i+1+(nframes/2)

            if idx_start < 0:
                idx_start=0

            xmedian = np.median( np.append( centroids[:,0][idx_start : i], centroids[:,0][i+1 : idx_end]) )
            ymedian = np.median(np.append( centroids[:,1][idx_start : i], centroids[:,1][i+1 : idx_end]))
            fullxmedian, fullymedian = np.median(centroids[:,0]), np.median(centroids[:,1])
            xstd = np.std( np.append( centroids[:,0][idx_start : i], centroids[:,0][i+1 : idx_end]) )
            ystd = np.std(np.append( centroids[:,1][idx_start : i], centroids[:,1][i+1 : idx_end]))
            fullxstd, fullystd = np.std(centroids[:,0]), np.std(centroids[:,1])

            if (abs(centroids[i][0] - xmedian) > sigmaclip*xstd) or (abs(centroids[i][1]- ymedian) > sigmaclip*ystd):
                flagged_frames[i] = False
            if (abs(centroids[i][0] - fullxmedian) > sigmaclip*fullxstd) or (abs(centroids[i][1]- fullymedian) > sigmaclip*fullystd):
                flagged_frames[i] = False
            else:
                pass

        timeseries = timeseries[flagged_frames,::]
        centroids = centroids[flagged_frames,:]
        midtimes = midtimes[flagged_frames,]
        background = background[flagged_frames,]

        nflagged = len(flagged_frames) - np.sum(flagged_frames)

        if not quiet: print "\t Clipped {0} frames in iteration {1}".format(nflagged, iter+1)

        if plot and iter == 0:
            fig, ax = plt.subplots(2,1, figsize = (20,5))
            ax[0].plot(origmidtimes[~flagged_frames], origcentroids[:,1][~flagged_frames], 'o', c=c2, markersize = 8)
            ax[0].plot(origmidtimes, origcentroids[:,1], 'd', c = c, markersize = 4)
            ax[0].set_xlabel("Time (BJD)")
            ax[0].set_ylabel("x-pos")
            ax[1].plot(origmidtimes[~flagged_frames], origcentroids[:,0][~flagged_frames], 'o', c=c2, markersize = 8)
            ax[1].plot(origmidtimes, origcentroids[:,0], 'd', c = c, markersize = 4)
            ax[1].set_xlabel("Time (BJD)")
            ax[1].set_ylabel("y-pos")
            plt.savefig("{5}/PhD/SpitzerTransits/{0}{4}/{0}_{1}_{2}_{3}_centroidsClip.png".format(planet, AOR, channel,sysmethod,foldext,os.getenv('HOME')))
            plt.close()

    return timeseries, centroids, midtimes, background

def aperture_photom(timeseries, centroids, radius, quiet = False, foldext=None):
    """Finds the optimum aperture for calculating aperture photometry
        by minimising the standard deviaion.
        Then calculates the apeture photometry for the best radius."""

    lightcurve = np.zeros(timeseries.shape[0])
    ##    if not quiet: print "\nFinding optimum radius for aperture photometry..."
    ##
    ##    bounds = [0,timeseries.shape[1]/2-1]
    ##    #Calls on class that calculates standard deviation of the photometry
    ##    fun = stdPhotom(timeseries)
    ##    res = minimize_scalar(fun.evaluate, bounds=bounds, method='Bounded', tol=1e-12)
    ##
    ##
    ##    radius = res.x
    ##
    ##    if not quiet: print "\t Optimum radius found to be {}".format(radius)

    if not quiet: print "\nPerforming aperture photometry..."

    if not quiet: print "\t Using radius of {} pixels.".format(radius)

    #Calculates aperture photometry for the timeseries
    for i in range(len(timeseries)):
        positions = centroids[i] - 0.5
        apertures = CircularAperture(positions, r=radius)
        photom = aperture_photometry(timeseries[i], apertures, method='subpixel', subpixels=5)
        lightcurve[i] = photom[0][3]

    if not quiet: print "\t Success"

    return lightcurve

def sigma_clip_photom(photom, timeseries, centroids, midtimes, background, sigmaclip, iters, nframes, quiet = False, plot = False, AOR = None, planet = None, channel=None,sysmethod=None, foldext=None):
    """Perform 4 or 5 sigma clipping to the data and corresponding arrays """
    if sysmethod == 'poly': c, c2 ='b', '#ff7f0e'
    elif sysmethod == 'PLD': c, c2 = 'r', 'c'
    else: c, c2 = 'k','g'

    if not quiet: print "\nClipping frames >{0} sigma away from median photometry...".format(sigmaclip)

    origmidtimes, origphotom = midtimes, photom
    for iter in range(iters):

        flagged_frames = np.full(photom.shape, True, dtype=bool)

        for i in range(len(photom)):
            idx_start = i-(nframes/2)
            idx_end = i+1+(nframes/2)
            if idx_start < 0:
                idx_start=0
            fullmedian = np.median(photom)
            fullstd = np.std(photom)
            median = np.median(np.append(photom[idx_start:i],photom[i+1:idx_end]))
            std = np.std(np.append(photom[idx_start:i],photom[i+1:idx_end]))
            if (abs(photom[i] - median) > sigmaclip*std) or (abs(photom[i] - fullmedian) > sigmaclip*fullstd):
                flagged_frames[i] = False
            else:
                pass

        photom = photom[flagged_frames,]
        timeseries = timeseries[flagged_frames,::]
        midtimes = midtimes[flagged_frames,]
        background = background[flagged_frames,]
        centroids = centroids[flagged_frames,:]

        nflagged = len(flagged_frames) - np.sum(flagged_frames)

        if not quiet: print "\t Clipped {0} frames in iteration {1}".format(nflagged, iter+1)

        if plot and iter == 0:
            fig, ax = plt.subplots(figsize = (20,5))
            ax.plot(origmidtimes[~flagged_frames], origphotom[~flagged_frames], 'o', c=c2, markersize = 8)
            ax.plot(origmidtimes, origphotom, 'd', markersize = 4, c=c)
            ax.set_xlabel("Time (BJD)")
            ax.set_ylabel("Flux (image units)")
            plt.savefig("{5}/PhD/SpitzerTransits/{0}{4}/{0}_{1}_{2}_{3}_photomClip.png".format(planet, AOR, channel,sysmethod,foldext,os.getenv('HOME')))
            plt.close()

    return photom, timeseries, centroids, midtimes, background
