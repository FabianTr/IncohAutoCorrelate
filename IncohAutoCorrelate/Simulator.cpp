#include <hdf5.h>
#include <iostream>
#include <thread>

#include "Simulator.h"
#include "ProfileTime.h"



inline double Simulator::Drand()
{
	return rnd(mt);
}

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

Simulator::Simulator() :mt(std::random_device{}())
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
		Params[0] = (double)NumEM; // number of Emitters
		
		Params[1] = (double)SimSettings.PoissonSample; 

		Params[2] = (double)SimSettings.SubSampling; //Subsampling is only possible if the orientation and size of a pixel is known! 
		//Pixels are within the plane given by u and v. u and v also represents the orientation (their edges). Here it is assumed, that all pixels are orientated in parallel
		Params[3] = SimSettings.PixelOrientationVectors[0]; //u1
		Params[4] = SimSettings.PixelOrientationVectors[1]; //u2
		Params[5] = SimSettings.PixelOrientationVectors[2]; //u3
		Params[6] = SimSettings.PixelOrientationVectors[3]; //v1
		Params[7] = SimSettings.PixelOrientationVectors[4]; //v2
		Params[8] = SimSettings.PixelOrientationVectors[5]; //v3
		//Also the pixel size is required, as the "PixelMap" only gives one coordinate, which we interprete as the center of the pixel 
		// => SubSampling (SuSa) is therefore performed on a grid with the central spot given by the PixelMap.
		// A SuSa = 1 for example means one Point on the Pixel center plus eight points lieing at the center +/- 0.33*Size in each dimension (1/3) lines
		Params[9] = SimSettings.PixelSize[0]; // Size in u (Su) direction SuSa step is Su/(2*SuSa + 1)
		Params[10] = SimSettings.PixelSize[1]; // Size in v (Sv) direction SuSa step is Sv/(2*SuSa + 1)


		for (unsigned int ModeRun = 0; ModeRun < SimSettings.Modes; ModeRun++)
		{
			if (ModeRun > 0 )//Roll new Phases if ModeRun != 0 TODO: Implement new roll of random emitters for same rotation!
			{
				#pragma omp parallel for
				for (unsigned int i = 0; i < EmitterList.size(); i++)
				{
					EmitterList[i].Phase = 2 * M_PI*Drand();
				}
			}




		}


		delete[] Params;
		delete[] EM;
		delete[] Intensity;
	}

	//free OpenCL Device
	Options.OCL_FreeDevice(OpenCLDeviceNumber);
}
