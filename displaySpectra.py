import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import glob
import cv2


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
def fit_ROIs(filePath):

	## Use readSpectrumFile() to read file
	calibrationData, spectralData = readSpectrumFile(filePath)

	## Calibrate output of file with calibration data
	bins, calibratedSpectrum = calibrateSpectrum(calibrationData, spectralData)

	## Clip the spectrum to remove noise (optional)
	# spectralData = np.clip(spectralData, a_min = 0, a_max = 6000)

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

		# Fit the gaussian to the ROI
		popt, pcov = curve_fit(gaussian, ROIBins, ROICounts)

		print('Computed fit parameters:')
		print(f'A = {popt[0]}')
		print(f'σ = {popt[1]}')
		print(f'μ = {popt[2]}\n')

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


### Run if file is run directly
if __name__ == '__main__':
	
	fit_ROIs('Spectra/55Fe CdTe.txt')
	# fit_ROIs('Spectra/133Ba CdTe.txt')
	# fit_ROIs('Spectra/241Am CdTe.txt')