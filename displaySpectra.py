import matplotlib.pyplot as plt
import numpy as np
import glob
import cv2

def readSpectrumFile(filePath):

	### Open the spectrum file
	with open(filePath) as file:

		## Read file
		lines = file.read().splitlines()
		calibrationData = []
		spectralData = []

		## Flag to check whether current line is part of spectrum
		spectrum = False

		## Iterate through all lines
		for line in lines:

			if line == '<<DATA>>':

				spectrum = True

				continue

			if line == '<<END>>':

				spectrum = False

				continue

			else:

				if spectrum == False:

					calibrationData.append(line)

				if spectrum == True:

					spectralData.append(line)

	return calibrationData, np.asarray(spectralData)

def calibrateSpectrum(filePath):

	calibrationData, spectralData = readSpectrumFile(filePath)

	flag = 0

	calibrationCoords = []

	for line in calibrationData:

		# print(line)

		if line == "<<CALIBRATION>>":

			flag = 1

		if line == "<<ROI>>":

			flag = 0

		if flag >= 1:

			if flag == 2:

				energyUnit = line.split(' ')[-1]
				print(energyUnit)

			if flag >= 3:

				coordinate = np.asarray(line.split(' ')).astype(float)

				calibrationCoords.append(coordinate)

			flag = flag + 1

			continue

	calibrationCoords = np.asarray(calibrationCoords)

	# print(calibrationCoords)

	# Finding slope
	m = (calibrationCoords[1][1] - calibrationCoords[0][1])/(calibrationCoords[1][0] - calibrationCoords[0][0])

	if energyUnit == 'eV':

		m = m/1000

	binCount = len(spectralData)

	bins = np.linspace(0*m, (binCount-1)*m, binCount)

	spectralData = spectralData.astype(int)

	# Clip the spectrum to remove noise (optional)
	# spectralData = np.clip(spectralData, a_min = 0, a_max = 6000)

	plt.plot(bins, spectralData)
	plt.xlabel('keV')
	plt.ylabel('counts')
	plt.show()


if __name__ == '__main__':
	
	calibrateSpectrum('Spectra/133Ba CdTe.txt')