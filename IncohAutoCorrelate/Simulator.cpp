#include <hdf5.h>
#include <iostream>
#include <thread>

#include "Simulator.h"
#include "ProfileTime.h"
#include "ArrayOperators.h"



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

	Profiler.Tic();
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

		for (int j = 0; j < 9; j++)//Store Rotation Matrix of current Crystal
		{
			curr_Event.RotMatrix[j] = RotMat[j];
		}

		double * EM = new double[4 * NumEM];
		for (unsigned int j = 0; j < NumEM; j++)
		{
			EM[j + 0] = EmitterList[j].Position[0];
			EM[j + 1] = EmitterList[j].Position[1];
			EM[j + 2] = EmitterList[j].Position[2];
			EM[j + 3] = EmitterList[j].Phase;
		}

		//Calculate steps for u and v SuSa: each pixel is divided in each direction by (2 * SuSa + 1) stripes => total of (2*SuSa + 1)^2 
		//Subpixel. The pixel size is Su analog Sv with the vectors u_Step & v_Step.
		//Fist step is to normalize the vectors to unity, then multyply by PixelSize and finally divide by (2 * SuSa + 1).
		double t_Norm = 0;
		//normalize N*u
		t_Norm = 1.0 / sqrt(SimSettings.PixelOrientationVectors[0] * SimSettings.PixelOrientationVectors[0] + SimSettings.PixelOrientationVectors[1] * SimSettings.PixelOrientationVectors[1] + SimSettings.PixelOrientationVectors[2] * SimSettings.PixelOrientationVectors[2]);
		//N' = (N*Su) / (2*SuSa + 1)
		t_Norm = (t_Norm / ((double)(2 * SimSettings.SubSampling + 1)))*SimSettings.PixelSize[0];
		double u_Step[3]; 
		u_Step[0] = SimSettings.PixelOrientationVectors[0] * t_Norm;
		u_Step[1] = SimSettings.PixelOrientationVectors[1] * t_Norm;
		u_Step[2] = SimSettings.PixelOrientationVectors[2] * t_Norm;
		//analog for v:
		t_Norm = 1.0 / sqrt(SimSettings.PixelOrientationVectors[3] * SimSettings.PixelOrientationVectors[3] + SimSettings.PixelOrientationVectors[4] * SimSettings.PixelOrientationVectors[4] + SimSettings.PixelOrientationVectors[5] * SimSettings.PixelOrientationVectors[5]);
		t_Norm = (t_Norm / ((double)(2 * SimSettings.SubSampling + 1)))*SimSettings.PixelSize[1];
		double v_Step[3];
		v_Step[0] = SimSettings.PixelOrientationVectors[3] * t_Norm;
		v_Step[1] = SimSettings.PixelOrientationVectors[4] * t_Norm;
		v_Step[2] = SimSettings.PixelOrientationVectors[5] * t_Norm;


		double * Intensity = new double[Det.DetectorSize[0] * Det.DetectorSize[1]]();
		double * Params = new double[10]();
		Params[0] = (double)NumEM; // number of Emitters
		
		Params[1] = (double)SimSettings.PoissonSample; 

		Params[2] = (double)SimSettings.SubSampling; //Subsampling is only possible if the orientation and size of a pixel is known! 
		//Pixels are within the plane given by u and v. u and v also represents the orientation (their edges). Here it is assumed, that all pixels are orientated in parallel
		
		Params[3] = u_Step[0]; //u1 
		Params[4] = u_Step[1]; //u2
		Params[5] = u_Step[2]; //u3
		Params[6] = v_Step[0]; //v1
		Params[7] = v_Step[1]; //v2
		Params[8] = v_Step[2]; //v3
		
		Params[9] = SimSettings.Wavelength;//Wavelength (needed to calculate k)


		for (unsigned int ModeRun = 0; ModeRun < SimSettings.Modes; ModeRun++)
		{
			double * t_Intensity = new double[Det.DetectorSize[0] * Det.DetectorSize[1]]();
			if (ModeRun > 0 )//Roll new Phases if ModeRun != 0 (and keep rotation matrix)
			{
				EmitterList.clear();
				EmitterList = EmitterCrystal.GetEmitters(SimSettings.CrystSettings, RotMat, true);

				NumEM = EmitterList.size();
				Params[0] = (double)NumEM;

				EM = new double[4 * NumEM];
				for (unsigned int j = 0; j < NumEM; j++)
				{
					EM[j + 0] = EmitterList[j].Position[0];
					EM[j + 1] = EmitterList[j].Position[1];
					EM[j + 2] = EmitterList[j].Position[2];
					EM[j + 3] = EmitterList[j].Phase;
				}

			}
			//obtain Device
			cl::Device CL_Device = Options.CL_devices[OpenCLDeviceNumber];
			//Setup Queue
			cl::CommandQueue queue(Options.CL_context, CL_Device, 0, &err);
			Options.checkErr(err, "Setup CommandQueue in Simulator::Simulate() ");
			cl::Event cl_event;
			

			//Output 
			size_t IntSize = sizeof(double) * Det.DetectorSize[0] * Det.DetectorSize[1];
			cl::Buffer CL_Intensity(Options.CL_context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, IntSize, t_Intensity, &err);
			//Input
			size_t PixMapsize = sizeof(float) * 3 * Det.DetectorSize[0] * Det.DetectorSize[1];
			cl::Buffer CL_PixMap(Options.CL_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, PixMapsize, Det.PixelMap, &err);
			size_t EMsize = sizeof(double) * 4 * NumEM;
			cl::Buffer CL_EM(Options.CL_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, EMsize, EM, &err);
			cl::Buffer CL_Params(Options.CL_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(Params), &Params, &err);

			
			//Setup Kernel
			cl::Kernel kernel(Options.CL_Program, "SimulateCrystal", &err);
			Options.checkErr(err, "Setup AutoCorr_CQ in Simulator::Simulate() ");

			//Set Arguments
			kernel.setArg(0, CL_PixMap);
			kernel.setArg(1, CL_EM);
			kernel.setArg(2, CL_Params);
			kernel.setArg(3, CL_Intensity);
			const size_t &global_size = Det.DetectorSize[0] * Det.DetectorSize[1];

			//launch Kernel

			err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(global_size), cl::NullRange, NULL, &cl_event);

			Options.checkErr(err, "Launch Kernel in Simulator::Simulate()");
			cl_event.wait();


			//Read Results
			err = queue.enqueueReadBuffer(CL_Intensity, CL_TRUE, 0, IntSize, t_Intensity);
			Options.checkErr(err, "OpenCL kernel, launched in Simulator::Simulate()");




			//add up intensity (incoherent for mode simulation)
			ArrayOperators::ParAdd(Intensity, t_Intensity, Det.DetectorSize[0] * Det.DetectorSize[1]);

			//free memory of temps
			delete[] t_Intensity;
			delete[] EM;
		}

		//PostProcess 

		//Rescale for expected number of photons
		double ExpNumOfPhotones = double(SimSettings.AveragePhotonesPerEmitterOnDetector * SimSettings.CrystSettings.FlYield * EmitterCrystal.AtomPositions.size());
		double IntegratedIntensity = 0;
		for (unsigned int l = 0; l < Det.DetectorSize[0]* Det.DetectorSize[1]; l++)
		{
			IntegratedIntensity += Intensity[l];
		}
		double t_IntFactor = ExpNumOfPhotones / IntegratedIntensity;
		ArrayOperators::ParMultiplyScalar(Intensity, t_IntFactor, Det.DetectorSize[0] * Det.DetectorSize[1]);

		//TODO: Poisson Sample
		if (SimSettings.PoissonSample)
		{

		}


		//TODO: add noise

		//TODO: Multiply with Photon value



		//Push back patern Intensity to Output Vector
		for (unsigned int j = 0; j < Det.DetectorSize[0] * Det.DetectorSize[1]; j++)
		{//convert Intensity of pattern to float
			curr_Intensity[j] = (float)Intensity[j];
		}
		Output.Intensities.push_back(curr_Intensity);

		//Event Data
		{
			//RotMatrix is already stored in curr_Event
			curr_Event.SerialNumber = i;
			curr_Event.Event = i;
			curr_Event.Filename = SimSettings.Filename_Intensity;
			curr_Event.Dataset = SimSettings.Dataset;
			
		}


		Output.HitEvents.push_back(curr_Event);



		//print status
		if ((i+1) % (N / 100) == 0)
		{
			std::cout << "Pattern " << (i+1) << "/" << N << " ^= " << ((i+1) * 100 / N) << "\% \t in: " << Profiler.Toc(false) << "s\n";
		}
		delete[] Params;
		delete[] Intensity;
	}

	//free OpenCL Device
	Options.OCL_FreeDevice(OpenCLDeviceNumber);


	//ToImplement Save stuff



}
