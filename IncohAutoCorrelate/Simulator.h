#pragma once

#include <string>

#include "Detector.h"
#include "Settings.h"
#include "Crystal.h"
#include "Settings.h"

class Simulator
{
private:

	void WriteIntensityToH5(Detector & Det, std::string Filename, std::string Dataset);

public:
	Simulator();
	~Simulator();


	struct SimulationSettings
	{
		unsigned int NumberOfSimulations = 1;

		unsigned int Modes=1; //averaged exposures

		unsigned int SubSampling = 0; // (SS+1)^2 subpixels

		float AveragePhotonesPerEmitterOnDetector = 1.0f;

		bool PoissonSample = true; //apply Poisson distribution to obtain discrete Photons
		float Value_per_Photon = 6.4f; //factor to be applied (after Poisson sampling)

		bool AutoPixelOrientation = true; //if true guesses orientation from Pixel-map
		bool AutoPixelSize = true; //if true guesses size from Pixel-map
		double PixelOrientationVectors[6]; // ignored if (AutoPixelOrientation), otherwise in format [x1,x2,x3,y1,y2,y3]
		double PixelSize[2]; // ignored if(AutoPixelSize)

		Crystal::EmittingCrystSettings CrystSettings; //Settings for emitting Crystal
	};

	struct SimulationOutput
	{
		std::vector<Settings::HitEvent> HitEvents;
		std::vector<std::vector<float>> Intensities;
	};

	void Simulate(Crystal EmitterCrystal,Detector &Det, SimulationSettings SimSettings, SimulationOutput & Output, Settings & Options); //Detector needs a PixleMap

};

