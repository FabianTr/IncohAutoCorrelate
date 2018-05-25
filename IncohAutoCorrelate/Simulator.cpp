#include <hdf5.h>
#include <iostream>
#include <thread>

#include "Simulator.h"
#include "ProfileTime.h"




void Simulator::WriteIntensityToH5(Detector & Det, std::string Filename, std::string Dataset)
{
	#pragma omp critical
	{
		H5::H5File file(Filename, H5F_ACC_RDWR);
		//H5::DataSet dataset = file.openDataSet(DataSet);

		std::cerr << "To be implemented\n";
		file.close();
	}
}

Simulator::Simulator()
{
}


Simulator::~Simulator()
{
}

void Simulator::Simulate(Crystal EmitterCrystal, Detector & Det, SimulationSettings SimSettings, SimulationOutput & Output, Settings & Options)
{
	ProfileTime Profiler;

	unsigned int N = SimSettings.NumberOfSimulations;
	Output.Intensities.clear();
	Output.Intensities.reserve(N);
	Output.HitEvents.clear();
	Output.HitEvents.reserve(N);


	//
	if (SimSettings.AutoPixelOrientation) //Guess PixelOrientation under assumption that all pixel are of same size and orientations are always parallel
	{
		SimSettings.PixelOrientationVectors[0] = Det.PixelMap[0 + 3] - Det.PixelMap[0 + 0];
		SimSettings.PixelOrientationVectors[1] = Det.PixelMap[1 + 3] - Det.PixelMap[1 + 0];
		SimSettings.PixelOrientationVectors[2] = Det.PixelMap[2 + 3] - Det.PixelMap[2 + 0];

		SimSettings.PixelOrientationVectors[3] = Det.PixelMap[0 + 3 * Det.DetectorSize[1]] - Det.PixelMap[0 + 0];
		SimSettings.PixelOrientationVectors[4] = Det.PixelMap[1 + 3 * Det.DetectorSize[1]] - Det.PixelMap[1 + 0];
		SimSettings.PixelOrientationVectors[5] = Det.PixelMap[2 + 3 * Det.DetectorSize[1]] - Det.PixelMap[2 + 0];

		double Norm[2];

		Norm[0] = sqrt(SimSettings.PixelOrientationVectors[0] * SimSettings.PixelOrientationVectors[0] + SimSettings.PixelOrientationVectors[1] * SimSettings.PixelOrientationVectors[1] + SimSettings.PixelOrientationVectors[2] * SimSettings.PixelOrientationVectors[2]);
		Norm[1] = sqrt(SimSettings.PixelOrientationVectors[3] * SimSettings.PixelOrientationVectors[3] + SimSettings.PixelOrientationVectors[4] * SimSettings.PixelOrientationVectors[4] + SimSettings.PixelOrientationVectors[5] * SimSettings.PixelOrientationVectors[5]);

		if (SimSettings.AutoPixelSize)
		{
			SimSettings.PixelSize[0] = Norm[0];
			SimSettings.PixelSize[1] = Norm[1];
		}

		SimSettings.PixelOrientationVectors[0] = SimSettings.PixelOrientationVectors[0] / Norm[0];
		SimSettings.PixelOrientationVectors[1] = SimSettings.PixelOrientationVectors[1] / Norm[0];
		SimSettings.PixelOrientationVectors[2] = SimSettings.PixelOrientationVectors[2] / Norm[0];

		SimSettings.PixelOrientationVectors[3] = SimSettings.PixelOrientationVectors[3] / Norm[1];
		SimSettings.PixelOrientationVectors[4] = SimSettings.PixelOrientationVectors[4] / Norm[1];
		SimSettings.PixelOrientationVectors[5] = SimSettings.PixelOrientationVectors[5] / Norm[1];
	}
	else
	{
		if (SimSettings.AutoPixelSize)
		{
			double tmpPixelOrient[6];

			tmpPixelOrient[0] = Det.PixelMap[0 + 3] - Det.PixelMap[0 + 0];
			tmpPixelOrient[1] = Det.PixelMap[1 + 3] - Det.PixelMap[1 + 0];
			tmpPixelOrient[2] = Det.PixelMap[2 + 3] - Det.PixelMap[2 + 0];

			tmpPixelOrient[3] = Det.PixelMap[0 + 3 * Det.DetectorSize[1]] - Det.PixelMap[0 + 0];
			tmpPixelOrient[4] = Det.PixelMap[1 + 3 * Det.DetectorSize[1]] - Det.PixelMap[1 + 0];
			tmpPixelOrient[5] = Det.PixelMap[2 + 3 * Det.DetectorSize[1]] - Det.PixelMap[2 + 0];

			double Norm[2];

			Norm[0] = sqrt(tmpPixelOrient[0] * tmpPixelOrient[0] + tmpPixelOrient[1] * tmpPixelOrient[1] + tmpPixelOrient[2] * tmpPixelOrient[2]);
			Norm[1] = sqrt(tmpPixelOrient[3] * tmpPixelOrient[3] + tmpPixelOrient[4] * tmpPixelOrient[4] + tmpPixelOrient[5] * tmpPixelOrient[5]);

			SimSettings.PixelSize[0] = Norm[0];
			SimSettings.PixelSize[1] = Norm[1];
		}
	}


	//Debug Bullshit
	std::cout << "Pixel Orientation a: " << SimSettings.PixelOrientationVectors[0] << ", " << SimSettings.PixelOrientationVectors[1] << ", " << SimSettings.PixelOrientationVectors[2] << "\n";
	std::cout << "Pixel Orientation b: " << SimSettings.PixelOrientationVectors[3] << ", " << SimSettings.PixelOrientationVectors[4] << ", " << SimSettings.PixelOrientationVectors[5] << "\n";

	std::cout << "Pixel Size a: " << SimSettings.PixelSize[0] << "\n";
	std::cout << "Pixel Size b: " << SimSettings.PixelSize[1] << "\n";
	//


	//reserve OpenCL Device
	int OpenCLDeviceNumber = -1;
	cl_int err;

	while ((OpenCLDeviceNumber = Options.OCL_ReserveDevice()) == -1)
	{
		std::this_thread::sleep_for(std::chrono::microseconds(Options.ThreadSleepForOCLDev));
	}

	//Start Loop
	for (unsigned int i = 0; i < N; i++)
	{
		Settings::HitEvent curr_Event;
		std::vector<float> curr_Intensity;
		curr_Intensity.resize(Det.DetectorSize[0] * Det.DetectorSize[1]);

		std::vector<Crystal::Emitter> EmitterList;

		//Obtain EmitterList
		std::array<float, 9> RotMat;
		EmitterList = EmitterCrystal.GetEmitters(SimSettings.CrystSettings, RotMat);
		unsigned int NumEM = EmitterList.size();

		for (int i = 0; i < 9; i++)//Store Rotation Matrix of current Crystal
		{
			curr_Event.RotMatrix[i] = RotMat[i];
		}

		double * EM = new double(4 * NumEM);
		for (unsigned int i = 0; i < NumEM; i++)
		{
			EM[0] = EmitterList[i].Position[0];
			EM[1] = EmitterList[i].Position[1];
			EM[2] = EmitterList[i].Position[2];
			EM[3] = EmitterList[i].Phase;
		}

		double * Intensity = new double(Det.DetectorSize[0] * Det.DetectorSize[1]);
		double * Params = new double[2]();
		Params[0] = (double)NumEM; // number of 


		
		


		delete[] Params;
		delete[] EM;
		delete[] Intensity;
	}

	//free OpenCL Device
	Options.OCL_FreeDevice(OpenCLDeviceNumber);
}
