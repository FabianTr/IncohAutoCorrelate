#pragma once

#include <string>

#include "Detector.h"
#include "Settings.h"
#include "Crystal.h"
#include "Settings.h"

class Simulator
{
private:
	inline double Drand();
	std::uniform_real_distribution<double> rnd;
	std::mt19937_64 mt;
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
		float ADUNoise = 0.0f; //additional white noise in +/- Value_per_Photon

		bool AutoPixelOrientation = true; //if true guesses orientation from Pixel-map
		bool AutoPixelSize = true; //if true guesses size from Pixel-map
		double PixelOrientationVectors[6]; // ignored if (AutoPixelOrientation), otherwise in format [x1,x2,x3,y1,y2,y3]
		double PixelSize[2]; // ignored if(AutoPixelSize)

		double Wavelength = 1.94; //needs to be in the same unit as the emitter positions => Crystal recom.: use A
		Crystal::EmittingCrystSettings CrystSettings; //Settings for emitting Crystal

		//output stuff
		bool SaveResults = false;
		std::string Dataset = "data";
		std::string Filename_Intensity = "";
		std::string Filename_XML = "";

		unsigned int UnitCells[3] = { 1,1,1 };
		float CrystalSize[3] = { 0,0,0 };
	};

	struct SimulationOutput 
	{
		//must not contain pointers
		std::vector<Settings::HitEvent> HitEvents;
		std::vector<std::vector<float>> Intensities;

		unsigned int DetectorSize[2] = { 0, 0 };
	};

	void Simulate(Crystal EmitterCrystal,Detector &Det, SimulationSettings SimSettings, SimulationOutput & Output, Settings & Options); //Detector needs a PixleMap
	void ParSimulate(Crystal EmitterCrystal, Detector & Det, SimulationSettings SimSettings, SimulationOutput & Output, Settings & Options);

	void SaveSimulationOutput(SimulationOutput &Output, std::string HDF5_Path, std::string XML_Path, SimulationSettings SimSettings);


	void GeneratePixelMap(std::string Filename, std::string Dataset,int SizeA, int SizeB, float PixelSize, std::array<float,3> Center, std::array<float, 3> VecA, std::array<float, 3> VecB);

	void DisturbePixelMap(Detector &Det, double Translation, double Roatation); //Rotation in Degree [0, 360)

private: 
	static void SimulatePart(Crystal  EmitterCrystal, Detector & Det, SimulationSettings  SimSettings, SimulationOutput & Output, Settings & Options, int ThreadNum);
	void PrintSimInfos(const SimulationSettings & SimSettings);
};

