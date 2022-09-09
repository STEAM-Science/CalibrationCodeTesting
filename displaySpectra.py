import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from scipy.optimize import curve_fit
import numpy as np
import glob
import cv2
import argparse
from scipy.stats import norm
from mpl_point_clicker import clicker

### Read spectrum file and return calibration data and spectrum
def readSpectrumFile(filePath):

	### Open the spectrum file
	with open(filePath) as file:

		## Read file
		lines = file.read().splitlines()

		## Lists to store lines from file
		calibrationData = []
		spectralData = []

		## Flag to check whether current line is part of spectrum
		spectrum = False

		## Iterate through all lines
		for line in lines:

			# Check if we are currently in <<DATA> section
			if line == '<<DATA>>':

				spectrum = True

				continue

			# Check if we are at the end of spectrum
			if line == '<<END>>':

				spectrum = False

				continue

			# If neither, add to required list
			else:

				# If line part of calibration data
				if not spectrum:

					calibrationData.append(line)

				# If line part of spectrum
				if spectrum:

					spectralData.append(line)

	return calibrationData, np.asarray(spectralData)



### Calibrate spectrum with 
def calibrateSpectrum(calibrationData, spectralData):

	## Flag to keep track of where in the calibration data we are
	flag = 0

	## List to store calibration 'coordinates'
	calibrationCoords = []

	## Go through every line in calibration data
	for line in calibrationData:

		# If we are in the calibration section
		if line == "<<CALIBRATION>>":

			# Set correct flag
			flag = 1

		# If we are in ROI section
		if line == "<<ROI>>":

			# Set in correct flag
			flag = 0

		# If flag is greater than 1
		if flag >= 1:

			# If flag is 2 we are looking at energy units
			if flag == 2:

				# Get units (either keV or eV)
				energyUnit = line.split(' ')[-1]

			# If flag is 3, we are looking at calibration 'coordinates'
			if flag >= 3:

				# Obtain current 'coordinate'
				coordinate = np.asarray(line.split(' ')).astype(float)

				# Add current 'coordinate' to list
				calibrationCoords.append(coordinate)

			# Keep incrementing flag to keep track of the line
			flag = flag + 1

			continue

	## Convert calibration coordinates to NumPy array
	calibrationCoords = np.asarray(calibrationCoords)

	## Finding slope from coordinates. Note that the first two coordinates are used
	m = (calibrationCoords[1][1] - calibrationCoords[0][1])/(calibrationCoords[1][0] - calibrationCoords[0][0])

	## Check units for energy. Rescale if in eV
	if energyUnit == 'eV':
		m = m/1000

	## Get required number of bins
	binCount = len(spectralData)

	## Create calibrated bins
	bins = np.linspace(0*m, (binCount-1)*m, binCount)

	## Convert spectrum to integer type
	calibratedSpectrum = spectralData.astype(int)

	## Output bins and spectrum
	return bins, calibratedSpectrum



### Defining a Gaussian distribution of total counts
def gaussian(xs, A, sigma, mu):

	temp = (xs - mu)/sigma
	temp = -0.5 * temp**2
	temp = np.exp(temp)
	return (A/(sigma*np.sqrt(2*np.pi)))*temp



### Function to fit pre-defined ROIS 
def fit_ROIs(filePath, clip=0):

	## Use readSpectrumFile() to read file
	calibrationData, spectralData = readSpectrumFile(filePath)

	## Calibrate output of file with calibration data
	bins, calibratedSpectrum = calibrateSpectrum(calibrationData, spectralData)

	## If user wanted to clip the data
	if clip != 0:

		# Clip the spectrum to remove noise (optional)
		spectralData = np.clip(spectralData, a_min = 0, a_max = clip)

	## Flag to keep track of whether the line is ROI info
	ROIFlag = False

	## List to store ROIs
	ROIs = []

	## Go through every line in calibration data
	for line in calibrationData:

		# Check if we are currently in <<ROI>>
		if line == '<<ROI>>':

			ROIFlag = True

			continue

		# Check if we are at the end of <<ROI>>
		if line == '<<DATA>>':

			break

		# If in ROI, add current ROI to list
		if ROIFlag:

			# Split the line
			temp = line.split()

			# Get ROI x-values
			ROIStart = int(temp[0])
			ROIEnd = int(temp[1])

			# Add it to list of ROIs
			ROIs.append([ROIStart, ROIEnd])

	## Make array of ROIs
	ROIs = np.asarray(ROIs)

	print(f'Found {len(ROIs)} region(s) of interest.\n')

	## Go through every listed ROI
	for i, ROIIndices in enumerate(ROIs):

		print(f'ROI {i+1} of {len(ROIs)}.')

		# Get bins and counts in the ROI
		ROIBins = bins[ROIIndices[0]:ROIIndices[1]]
		ROICounts = calibratedSpectrum[ROIIndices[0]:ROIIndices[1]]

		# Get Gaussian image
		popt, pcov = getGaussFit(ROIBins, ROICounts)

		# Get counts for the fit curve
		FitCounts = gaussian(ROIBins, *popt)

		# Plot the data in ROI
		plt.plot(ROIBins, ROICounts, color='black', label='data')

		# Plot gaussian fit of ROI
		plt.plot(ROIBins, FitCounts, color='red', label='Gaussian fit')

		# Add title
		titleText = filePath.split('/')[-1].split('.')[0]
		plt.title(titleText)

		# Add axes labels
		plt.xlabel('keV')
		plt.ylabel('counts')

		# Add legend
		plt.legend()

		# Display the plot
		plt.show()

		# Clear plots
		plt.clf()



### Get Gaussian fit to data
def getGaussFit(xs, ys):

	## Find peak value of line
	maxCount = np.amax(ys)

	## Find half the peak
	halfMax = maxCount//2

	## Find all points above half the peak
	halfMaxMask = ys >= halfMax

	## Find indices inside half max
	halfMaxIndices = np.where(halfMaxMask == True)

	## Find half maxes
	leftHalfMax = halfMaxIndices[0][0]
	rightHalfMax = halfMaxIndices[0][-1]

	## Estimate standard deviation with FWHM
	stdEstimate = (rightHalfMax - leftHalfMax)/2.4

	## Find energy of spectral line
	peakEnergy = xs[np.where(ys == maxCount)[0][0]]

	print('Estimated fit parameters:')
	print(f'A = {5*maxCount}')
	print(f'σ = {stdEstimate}')
	print(f'μ = {peakEnergy}\n')

	## Fit the gaussian to the ROI
	popt, pcov = curve_fit(gaussian, xs, ys, p0=[5*maxCount, stdEstimate, peakEnergy])

	print('Computed fit parameters:')
	print(f'A = {popt[0]}')
	print(f'σ = {popt[1]}')
	print(f'μ = {popt[2]}\n')

	## Return fit parameters
	return popt, pcov



### Fit regions manually
def fit_manual(filePath, clip=0):

	## Function to easily plot data
	def plotData(xs, ys, plotArgs):

		# Plot whole spectrum
		plt.plot(xs, ys, color=plotArgs['color'], label=plotArgs['label'])

		# Add title
		plt.title(plotArgs['title'])

		# Add axes labels
		plt.xlabel(plotArgs['xlabel'])
		plt.ylabel(plotArgs['ylabel'])

		# If user wants to show a legend
		if plotArgs['legend']:

			# Add legend
			plt.legend()

	## Use readSpectrumFile() to read file
	calibrationData, spectralData = readSpectrumFile(filePath)

	## Calibrate output of file with calibration data
	bins, calibratedSpectrum = calibrateSpectrum(calibrationData, spectralData)

	## If user wants to clip the data
	if clip != 0:

		# Clip the spectrum to remove noise (optional)
		calibratedSpectrum = np.clip(calibratedSpectrum, a_min = 0, a_max = clip)

	## Title text
	titleText = filePath.split('/')[-1].split('.')[0] + ' select points to fit.'

	## Create dictionary to store plotting parameters
	plotArgs = {
		'color': 'k',
		'label': 'data',
		'xlabel': 'keV',
		'ylabel': 'counts',
		'title': titleText,
		'legend': True
	}

	## Plot data
	plotData(bins, calibratedSpectrum, plotArgs)

	## Get point inputs
	points = np.asarray(plt.ginput(2))

	## Get start and end 'x' coordinates
	startX = points[0][0]
	endX = points[1][0]

	## Width from start to end
	width = endX - startX

	## Extra region to plot (in percent)
	extraR = 0.2

	print(f'Fitting region from {startX} keV to {endX} keV.\n')

	## Close the plot
	plt.close()

	## Title text
	plotArgs['title'] = filePath.split('/')[-1].split('.')[0] + ' fit'

	## Plot data
	plotData(bins, calibratedSpectrum, plotArgs)

	## Plot the correct ranges
	plt.xlim(startX - width*(extraR/2), endX + width*(extraR/2))

	## Create mask for certain values
	mask = (bins > startX) & (bins < endX)

	## Mask the bins and calibrated spectrum
	maskedBins = bins[mask]
	maskedSpectrum = calibratedSpectrum[mask]

	## Fit the gaussian to the ROI
	popt, pcov = getGaussFit(maskedBins, maskedSpectrum)

	## Get counts for the fit curve
	FitCounts = gaussian(maskedBins, *popt)

	## Plot gaussian fit of ROI
	plt.plot(maskedBins, FitCounts, color='red', label='Gaussian fit')

	## Show plots
	plt.show()



### Main functioning of script
def main(args):

	## Check if user wants to fit ROIs automatically
	if args.ROIs:

		print(f"Analyzing ROIs in {args.src}.")

		# Run fit_ROIs
		fit_ROIs(args.src, int(args.clip))

		return

	## If the user wants to do it manually instead
	else:

		print(f"Analyzing {args.src} manually.")

		# Run fit manual
		fit_manual(args.src, int(args.clip))
		return

	return


### Run if file is run directly
if __name__ == '__main__':

	## Create new parser
	parser = argparse.ArgumentParser(description='Process inputs to calibrate spectra.')

	## Choose spectrum sourse
	parser.add_argument('--src', action='store', nargs='?', type=str, default='spectra/55Fe_CdTe.txt', help='Spectrum source file.')

	## Choose whether or not to fit automatic ROIs
	parser.add_argument('--ROIs', action='store_true', help='Choose whether to automatically fit ROIs.')

	## Choose whether or not to clip data
	parser.add_argument('--clip', action='store', default=0, help='Decide clipping value.')

	## Parse arguments
	args = parser.parse_args()

	## Call main
	main(args)