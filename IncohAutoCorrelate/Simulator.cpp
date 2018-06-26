#include <hdf5.h>
#include <iostream>
#include <thread>
#include "H5Cpp.h"
#include <unordered_map>
#include <thread>
#include <mutex>

#include "Simulator.h"
#include "ProfileTime.h"
#include "ArrayOperators.h"
#include "Settings.h"






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

	//Check requirements
	if (!Det.Checklist.PixelMap)
	{
		std::cerr << "ERROR: Detector needs a PixelMap (use Detector::LoadPixelMask())\n";
		std::cerr << "   -> in Simulator::Simulate()\n";
		throw;
	}

	//


	unsigned int N = SimSettings.NumberOfSimulations;
	Output.Intensities.clear();
	Output.Intensities.reserve(N);
	Output.HitEvents.clear();
	Output.HitEvents.reserve(N);

	Output.DetectorSize[0] = Det.DetectorSize[0];
	Output.DetectorSize[1] = Det.DetectorSize[1];

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

	std::cout << "Pixel Size (a x b): " << SimSettings.PixelSize[0] <<" x " << SimSettings.PixelSize[1]<<"\n";
	
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

		float * EM = new float[4 * NumEM];
		for (unsigned int j = 0; j < NumEM; j++)
		{
			EM[4 * j + 0] = (float)EmitterList[j].Position[0];
			EM[4 * j + 1] = (float)EmitterList[j].Position[1];
			EM[4 * j + 2] = (float)EmitterList[j].Position[2];
			EM[4 * j + 3] = (float)EmitterList[j].Phase;
			//std::cout << "r = (" << EM[j + 0] << ", " << EM[j + 1] << ", " << EM[j + 2] << ") \t phi = " << EM[j + 3] << "\n";
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


		float * Intensity = new float[Det.DetectorSize[0] * Det.DetectorSize[1]]();
		double Params[10];
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
			float * t_Intensity = new float[Det.DetectorSize[0] * Det.DetectorSize[1]]();
			if (ModeRun > 0 )//Roll new Phases if ModeRun != 0 (and keep rotation matrix)
			{
				EmitterList.clear();
				EmitterList = EmitterCrystal.GetEmitters(SimSettings.CrystSettings, RotMat, true);

				NumEM = EmitterList.size();
				Params[0] = (double)NumEM;

				EM = new float[4 * NumEM];
				for (unsigned int j = 0; j < NumEM; j++)
				{
					EM[4 * j + 0] = (float)EmitterList[j].Position[0];
					EM[4 * j + 1] = (float)EmitterList[j].Position[1];
					EM[4 * j + 2] = (float)EmitterList[j].Position[2];
					EM[4 * j + 3] = (float)EmitterList[j].Phase;
				}

			}


		//	std::cout << "Number of Emitter: " << NumEM << "\n";

			//obtain Device
			cl::Device CL_Device = Options.CL_devices[OpenCLDeviceNumber];
			//Setup Queue
			cl::CommandQueue queue(Options.CL_context, CL_Device, 0, &err);
			Options.checkErr(err, "Setup CommandQueue in Simulator::Simulate() ");
			cl::Event cl_event;
			

			//Output 
			size_t IntSize = sizeof(float) * Det.DetectorSize[0] * Det.DetectorSize[1];
			cl::Buffer CL_Intensity(Options.CL_context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, IntSize, t_Intensity, &err);
			//Input
			size_t PixMapsize = sizeof(float) * 3 * Det.DetectorSize[0] * Det.DetectorSize[1];
			cl::Buffer CL_PixMap(Options.CL_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, PixMapsize, Det.PixelMap, &err);
			size_t EMsize = sizeof(float) * 4 * NumEM;
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


			////Debug
			//double sumInt = 0;
			//sumInt = ArrayOperators::Sum(Intensity, Det.DetectorSize[0] * Det.DetectorSize[1]);
			//std::cout << "SumInt: " << sumInt << ";    tIntSum: " << ArrayOperators::Sum(t_Intensity, Det.DetectorSize[0] * Det.DetectorSize[1]) << "\n";
			//sumInt = 0;

			////



			//free memory of temps
			delete[] t_Intensity;
			delete[] EM;
		}

		//PostProcess 



		//Rescale for expected number of photons
		float ExpNumOfPhotones = (float)(SimSettings.AveragePhotonesPerEmitterOnDetector * SimSettings.CrystSettings.FlYield * EmitterCrystal.AtomPositions.size());
		float IntegratedIntensity = 0.0f;
		for (unsigned int l = 0; l < Det.DetectorSize[0]* Det.DetectorSize[1]; l++)
		{ //Reminder: don't even think about to parallelize this!
			IntegratedIntensity += Intensity[l];
		}
		float t_IntFactor = ExpNumOfPhotones / IntegratedIntensity;
		ArrayOperators::ParMultiplyScalar(Intensity, t_IntFactor, Det.DetectorSize[0] * Det.DetectorSize[1]);

		//Poisson Sample (if required)
		if (SimSettings.PoissonSample)
		{
			ArrayOperators::ParPoissonSampling(Intensity, Det.DetectorSize[0] * Det.DetectorSize[1]);
		}

		//Add noise (if required)
		if (SimSettings.ADUNoise != 0)
		{
			ArrayOperators::ParAddWhiteNoise(Intensity, SimSettings.ADUNoise, Det.DetectorSize[0] * Det.DetectorSize[1]);
		}

		//Multiply with Photon value
		ArrayOperators::MultiplyScalar(Intensity, SimSettings.Value_per_Photon, Det.DetectorSize[0] * Det.DetectorSize[1]);





		//Push back pattern Intensity to Output Vector
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

			//Calculate Mean value and Photon count
			double mean = 0;
			int PhotonCount = 0;
			mean = ArrayOperators::Sum(Intensity, Det.DetectorSize[0] * Det.DetectorSize[1]);
			PhotonCount = (int)floor((mean / ((double)SimSettings.Value_per_Photon)) + 0.5);
			mean = mean / ((double)(Det.DetectorSize[0] * Det.DetectorSize[1]));
			curr_Event.MeanIntensity = (float)mean;
			curr_Event.PhotonCount = PhotonCount;
		}


		Output.HitEvents.push_back(curr_Event);



		//print status

		if (N >= 100) {
			if ((i + 1) % (N / 100) == 0)
			{
				std::cout << "Pattern " << (i + 1) << "/" << N << " ^= " << ((i + 1) * 100 / N) << "\% \t in: " << Profiler.Toc(false) << "s\n";
				std::cout << "Current Intensity: " << ArrayOperators::Sum(curr_Intensity.data(), Det.DetectorSize[0] * Det.DetectorSize[1])
					<< " =^ " << (int)(ArrayOperators::Sum(curr_Intensity.data(), Det.DetectorSize[0] * Det.DetectorSize[1]) / SimSettings.Value_per_Photon)
					<< " Photons\n";
			}
		}
		else
		{
			std::cout << "Pattern " << (i + 1) << "/" << N << " ^= " << ((i + 1) * 100 / N) << "\% \t in: " << Profiler.Toc(false) << "s\n";
			std::cout << "Current Intensity: " << ArrayOperators::Sum(curr_Intensity.data(), Det.DetectorSize[0] * Det.DetectorSize[1])
				<< " =^ " << (int)(ArrayOperators::Sum(curr_Intensity.data(), Det.DetectorSize[0] * Det.DetectorSize[1]) / SimSettings.Value_per_Photon)
				<< " Photons\n";
		}
		delete[] Intensity;
	}

	//free OpenCL Device
	Options.OCL_FreeDevice(OpenCLDeviceNumber);


	// Save stuff
	if (SimSettings.SaveResults)
	{
		SaveSimulationOutput(Output, SimSettings.Filename_Intensity, SimSettings.Filename_XML, SimSettings);
	}

	if (Options.echo)
		PrintSimInfos(SimSettings);
		
}

void Simulator::SaveSimulationOutput(SimulationOutput & Output, std::string HDF5_Path, std::string XML_Path, SimulationSettings SimSettings)
{
	H5::H5File file(HDF5_Path, H5F_ACC_TRUNC); //H5F_ACC_TRUNC => overwerite or create if not existing

	hsize_t dims[3];
	dims[0] = Output.Intensities.size();
	dims[1] = Output.DetectorSize[0];
	dims[2] = Output.DetectorSize[1];
	H5::DataSpace dataspace(3, dims);

	//Annotation: uses only constant Dataset name (Output.HitEvents[0].Dataset), not the most beautiful solution, maybe improve! 
	H5::DataSet dataset = file.createDataSet(Output.HitEvents[0].Dataset, H5::PredType::NATIVE_FLOAT, dataspace);

	hsize_t start[3] = {0, 0, 0};  // Start of hyperslab, offset
	hsize_t stride[3] = {1, 1, 1}; // Stride of hyperslab
	hsize_t count[3] = {1, 1, 1};  // Block count
	hsize_t block[3] = {1, dims[1], dims[2]}; // Block sizes


	H5::DataSpace mspace(3, block);
	for (unsigned int i = 0; i < Output.Intensities.size(); i++)
	{
		start[0] = i;
		dataspace.selectHyperslab(H5S_SELECT_SET, count, start, stride, block);
		dataset.write(Output.Intensities[i].data(), H5::PredType::NATIVE_FLOAT, mspace, dataspace);
	}

	mspace.close();
	dataspace.close();
	dataset.close();
	file.close();


	//Save XML File

	//additional informations
	std::unordered_map<std::string, std::string> SimInfo;
	SimInfo["Simulation.ADUNoise"] = std::to_string(SimSettings.ADUNoise);
	SimInfo["Simulation.AutoPixelOrientation"] = std::to_string(SimSettings.AutoPixelOrientation);
	SimInfo["Simulation.AutoPixelSize"] = std::to_string(SimSettings.AutoPixelSize);
	SimInfo["Simulation.AveragePhotonesPerEmitterOnDetector"] = std::to_string(SimSettings.AveragePhotonesPerEmitterOnDetector);

	SimInfo["Simulation.CrystSettings.FlYield"] = std::to_string(SimSettings.CrystSettings.FlYield);
	SimInfo["Simulation.CrystSettings.Incoherent"] = std::to_string(SimSettings.CrystSettings.Incoherent);
	SimInfo["Simulation.CrystSettings.Isotropie"] = std::to_string(SimSettings.CrystSettings.Isotropie);
	SimInfo["Simulation.CrystSettings.RandOrientation"] = std::to_string(SimSettings.CrystSettings.RandOrientation);

	SimInfo["Simulation.Dataset"] = SimSettings.Dataset;
	SimInfo["Simulation.Filename_Intensity"] = SimSettings.Filename_Intensity;
	SimInfo["Simulation.Modes"] = std::to_string(SimSettings.Modes);
	SimInfo["Simulation.NumberOfSimulations"] = std::to_string(SimSettings.NumberOfSimulations);
	SimInfo["Simulation.PixelOrientationVectors"] = "(" + std::to_string(SimSettings.PixelOrientationVectors[0]) 
		+", "+ std::to_string(SimSettings.PixelOrientationVectors[1]) + ", " + std::to_string(SimSettings.PixelOrientationVectors[2]) +
		");  (" + std::to_string(SimSettings.PixelOrientationVectors[3]) + ", " + std::to_string(SimSettings.PixelOrientationVectors[4]) + ", " + std::to_string(SimSettings.PixelOrientationVectors[4]) + ")";
	SimInfo["Simulation.PixelSize"] = std::to_string(SimSettings.PixelSize[0]) + " x " + std::to_string(SimSettings.PixelSize[1]);
	SimInfo["Simulation.PoissonSample"] = std::to_string(SimSettings.PoissonSample);
	SimInfo["Simulation.SubSampling"] = std::to_string(SimSettings.SubSampling);
	SimInfo["Simulation.Value_per_Photon"] = std::to_string(SimSettings.Value_per_Photon);
	SimInfo["Simulation.Wavelength"] = std::to_string(SimSettings.Wavelength);

	SimInfo["Simulation.CrystalShape.NumberUC_X"] = std::to_string(SimSettings.UnitCells[0]);
	SimInfo["Simulation.CrystalShape.NumberUC_Y"] = std::to_string(SimSettings.UnitCells[1]);
	SimInfo["Simulation.CrystalShape.NumberUC_Z"] = std::to_string(SimSettings.UnitCells[2]);
	SimInfo["Simulation.CrystalShape.Size_X"] = std::to_string(SimSettings.CrystalSize[0]);
	SimInfo["Simulation.CrystalShape.Size_Y"] = std::to_string(SimSettings.CrystalSize[1]);
	SimInfo["Simulation.CrystalShape.Size_Z"] = std::to_string(SimSettings.CrystalSize[2]);



	Settings tmp_Options;
	tmp_Options.SafeHitEventListToFile(XML_Path, Output.HitEvents,true, SimInfo);


}


void Simulator::PrintSimInfos(const SimulationSettings & SimSettings)
{
	std::cout << "Simulation & crystal parameter:\n-----------------------------\n";
	std::cout << "Number of unitcells: " << SimSettings.UnitCells[0] << " x " << SimSettings.UnitCells[1] << " x " << SimSettings.UnitCells[2] << "\n";
	std::cout << "Crystal size       : " << SimSettings.CrystalSize[0] << " x " << SimSettings.CrystalSize[1] << " x " << SimSettings.CrystalSize[2] << "\n";
}

//Parallel Sim (3 Threads)
std::mutex g_echo_mutex_Sim;
void Simulator::SimulatePart(Crystal  EmitterCrystal, Detector & Det, SimulationSettings  SimSettings, SimulationOutput & Output, Settings & Options, int ThreadNum)
{
	ProfileTime Profiler;

	//Check requirements
	if (!Det.Checklist.PixelMap)
	{
		std::cerr << "ERROR: Detector needs a PixelMap (use Detector::LoadPixelMask())\n";
		std::cerr << "   -> in Simulator::Simulate()\n";
		throw;
	}

	//


	unsigned int N = SimSettings.NumberOfSimulations;
	Output.Intensities.clear();
	Output.Intensities.reserve(N);
	Output.HitEvents.clear();
	Output.HitEvents.reserve(N);

	Output.DetectorSize[0] = Det.DetectorSize[0];
	Output.DetectorSize[1] = Det.DetectorSize[1];

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

	std::cout << "Pixel Size (a x b): " << SimSettings.PixelSize[0] << " x " << SimSettings.PixelSize[1] << "\n";



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

		float * EM = new float[4 * NumEM];
		for (unsigned int j = 0; j < NumEM; j++)
		{
			EM[4 * j + 0] = (float)EmitterList[j].Position[0];
			EM[4 * j + 1] = (float)EmitterList[j].Position[1];
			EM[4 * j + 2] = (float)EmitterList[j].Position[2];
			EM[4 * j + 3] = (float)EmitterList[j].Phase;
			//std::cout << "r = (" << EM[j + 0] << ", " << EM[j + 1] << ", " << EM[j + 2] << ") \t phi = " << EM[j + 3] << "\n";
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


		float * Intensity = new float[Det.DetectorSize[0] * Det.DetectorSize[1]]();
		double Params[10];
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
			float * t_Intensity = new float[Det.DetectorSize[0] * Det.DetectorSize[1]]();
			if (ModeRun > 0)//Roll new Phases if ModeRun != 0 (and keep rotation matrix)
			{
				EmitterList.clear();
				EmitterList = EmitterCrystal.GetEmitters(SimSettings.CrystSettings, RotMat, true);

				NumEM = EmitterList.size();
				Params[0] = (double)NumEM;

				EM = new float[4 * NumEM];
				for (unsigned int j = 0; j < NumEM; j++)
				{
					EM[4 * j + 0] = (float)EmitterList[j].Position[0];
					EM[4 * j + 1] = (float)EmitterList[j].Position[1];
					EM[4 * j + 2] = (float)EmitterList[j].Position[2];
					EM[4 * j + 3] = (float)EmitterList[j].Phase;
				}

			}


			//reserve OpenCL Device
			int OpenCLDeviceNumber = -1;
			cl_int err;

			while ((OpenCLDeviceNumber = Options.OCL_ReserveDevice()) == -1)
			{
				std::this_thread::sleep_for(std::chrono::microseconds(Options.ThreadSleepForOCLDev));
			}
			{
				//	std::cout << "Number of Emitter: " << NumEM << "\n";

				//obtain Device
				cl::Device CL_Device = Options.CL_devices[OpenCLDeviceNumber];
				//Setup Queue
				cl::CommandQueue queue(Options.CL_context, CL_Device, 0, &err);
				Options.checkErr(err, "Setup CommandQueue in Simulator::SimulatePart() ");
				cl::Event cl_event;


				//Output 
				size_t IntSize = sizeof(float) * Det.DetectorSize[0] * Det.DetectorSize[1];
				cl::Buffer CL_Intensity(Options.CL_context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, IntSize, t_Intensity, &err);
				//Input
				size_t PixMapsize = sizeof(float) * 3 * Det.DetectorSize[0] * Det.DetectorSize[1];
				cl::Buffer CL_PixMap(Options.CL_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, PixMapsize, Det.PixelMap, &err);
				size_t EMsize = sizeof(float) * 4 * NumEM;
				cl::Buffer CL_EM(Options.CL_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, EMsize, EM, &err);
				cl::Buffer CL_Params(Options.CL_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(Params), &Params, &err);


				//Setup Kernel
				cl::Kernel kernel(Options.CL_Program, "SimulateCrystal", &err);
				Options.checkErr(err, "Setup AutoCorr_CQ in Simulator::SimulatePart() ");

				//Set Arguments
				kernel.setArg(0, CL_PixMap);
				kernel.setArg(1, CL_EM);
				kernel.setArg(2, CL_Params);
				kernel.setArg(3, CL_Intensity);
				const size_t &global_size = Det.DetectorSize[0] * Det.DetectorSize[1];

				//launch Kernel

				err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(global_size), cl::NullRange, NULL, &cl_event);

				Options.checkErr(err, "Launch Kernel in Simulator::SimulatePart()");
				cl_event.wait();



				//Read Results
				err = queue.enqueueReadBuffer(CL_Intensity, CL_TRUE, 0, IntSize, t_Intensity);
				Options.checkErr(err, "OpenCL kernel, launched in Simulator::SimulatePart()");

			}
			//free OpenCL Device
			Options.OCL_FreeDevice(OpenCLDeviceNumber);

			//add up intensity (incoherent for mode simulation)
			ArrayOperators::ParAdd(Intensity, t_Intensity, Det.DetectorSize[0] * Det.DetectorSize[1]);



			delete[] t_Intensity;
			delete[] EM;
		}

		//PostProcess 



		//Rescale for expected number of photons
		float ExpNumOfPhotones = (float)(SimSettings.AveragePhotonesPerEmitterOnDetector * SimSettings.CrystSettings.FlYield * EmitterCrystal.AtomPositions.size());
		float IntegratedIntensity = 0.0f;
		for (unsigned int l = 0; l < Det.DetectorSize[0] * Det.DetectorSize[1]; l++)
		{ //Reminder: don't even think about to parallelize this!
			IntegratedIntensity += Intensity[l];
		}
		float t_IntFactor = ExpNumOfPhotones / IntegratedIntensity;
		ArrayOperators::ParMultiplyScalar(Intensity, t_IntFactor, Det.DetectorSize[0] * Det.DetectorSize[1]);

		//Poisson Sample (if required)
		if (SimSettings.PoissonSample)
		{
			ArrayOperators::ParPoissonSampling(Intensity, Det.DetectorSize[0] * Det.DetectorSize[1]);
		}

		//Add noise (if required)
		if (SimSettings.ADUNoise != 0)
		{
			ArrayOperators::ParAddWhiteNoise(Intensity, SimSettings.ADUNoise, Det.DetectorSize[0] * Det.DetectorSize[1]);
		}

		//Multiply with Photon value
		ArrayOperators::MultiplyScalar(Intensity, SimSettings.Value_per_Photon, Det.DetectorSize[0] * Det.DetectorSize[1]);





		//Push back pattern Intensity to Output Vector
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

			//Calculate Mean value and Photon count
			double mean = 0;
			int PhotonCount = 0;
			mean = ArrayOperators::Sum(Intensity, Det.DetectorSize[0] * Det.DetectorSize[1]);
			PhotonCount = (int)floor((mean / ((double)SimSettings.Value_per_Photon)) + 0.5);
			mean = mean / ((double)(Det.DetectorSize[0] * Det.DetectorSize[1]));
			curr_Event.MeanIntensity = (float)mean;
			curr_Event.PhotonCount = PhotonCount;
		}


		Output.HitEvents.push_back(curr_Event);



		//print status

		if (N >= 100) {
			
			if ((i + 1) % (N / 100) == 0)
			{
				g_echo_mutex_Sim.lock();
				std::cout << "Thread " << ThreadNum << ": Pattern " << (i + 1) << "/" << N << " ^= " << ((i + 1) * 100 / N) << "\% \t in: " << Profiler.Toc(false) << "s\n";
				//std::cout << "Current Intensity: " << ArrayOperators::Sum(curr_Intensity.data(), Det.DetectorSize[0] * Det.DetectorSize[1])
				//	<< " =^ " << (int)(ArrayOperators::Sum(curr_Intensity.data(), Det.DetectorSize[0] * Det.DetectorSize[1]) / SimSettings.Value_per_Photon)
				//	<< " Photons\n";
				g_echo_mutex_Sim.unlock();
			}
		}
		else
		{
			g_echo_mutex_Sim.lock();
			std::cout << "Thread " << ThreadNum << ": Pattern " << (i + 1) << "/" << N << " ^= " << ((i + 1) * 100 / N) << "\% \t in: " << Profiler.Toc(false) << "s\n";
			g_echo_mutex_Sim.unlock();
			//std::cout << "Current Intensity: " << ArrayOperators::Sum(curr_Intensity.data(), Det.DetectorSize[0] * Det.DetectorSize[1])
			//	<< " =^ " << (int)(ArrayOperators::Sum(curr_Intensity.data(), Det.DetectorSize[0] * Det.DetectorSize[1]) / SimSettings.Value_per_Photon)
			//	<< " Photons\n";
		}
		delete[] Intensity;
	}


}


void Simulator::ParSimulate(Crystal EmitterCrystal, Detector & Det, SimulationSettings SimSettings, SimulationOutput & Output, Settings & Options)
{
	Options.Echo(" ");
	ProfileTime Profiler;

	//Check requirements
	if (!Det.Checklist.PixelMap)
	{
		std::cerr << "ERROR: Detector needs a PixelMap (use Detector::LoadPixelMask())\n";
		std::cerr << "   -> in Simulator::Simulate()\n";
		throw;
	}

	//


	unsigned int N = SimSettings.NumberOfSimulations;
	Output.Intensities.clear();
	Output.Intensities.reserve(N);
	Output.HitEvents.clear();
	Output.HitEvents.reserve(N);

	Output.DetectorSize[0] = Det.DetectorSize[0];
	Output.DetectorSize[1] = Det.DetectorSize[1];


	//
	SimulationOutput OutputPart[3];
	OutputPart[0] = Output;
	OutputPart[1] = Output;
	//OutputPart[2] = Output;

	Detector DetPart[2]{ Detector(Det, true), Detector(Det, true) };

		 
	SimulationSettings SimSettingsPart[2] = { SimSettings ,SimSettings  };
	SimSettingsPart[0].NumberOfSimulations = (unsigned int)(SimSettings.NumberOfSimulations / 2);
	SimSettingsPart[1].NumberOfSimulations = SimSettings.NumberOfSimulations - SimSettingsPart[0].NumberOfSimulations;
	//SimSettingsPart[2].NumberOfSimulations = SimSettings.NumberOfSimulations - SimSettingsPart[0].NumberOfSimulations - SimSettingsPart[1].NumberOfSimulations;

	//launch threads
	std::cout << "Launch 2 Threads\n";
	Profiler.Tic();

	std::thread Thread1(SimulatePart, EmitterCrystal, std::ref(DetPart[0]), SimSettingsPart[0], std::ref(OutputPart[0]), std::ref(Options), 1);
	std::thread Thread2(SimulatePart, EmitterCrystal, std::ref(DetPart[1]), SimSettingsPart[1], std::ref(OutputPart[1]), std::ref(Options), 2);
	//std::thread Thread3(SimulatePart, EmitterCrystal, std::ref(DetPart[2]), SimSettingsPart[2], std::ref(OutputPart[2]), std::ref(Options), 3);
	
	Thread1.join();
	Thread2.join();
	//Thread3.join();

	std::cout << "**************\n" << SimSettings.NumberOfSimulations <<" patterns done in ";
	Profiler.Toc(true);

	Output.HitEvents = OutputPart[0].HitEvents;
	Output.HitEvents.insert(Output.HitEvents.end(), OutputPart[1].HitEvents.begin(), OutputPart[1].HitEvents.end());
	//Output.HitEvents.insert(Output.HitEvents.end(), OutputPart[2].HitEvents.begin(), OutputPart[2].HitEvents.end());

	Output.Intensities = OutputPart[0].Intensities;
	Output.Intensities.insert(Output.Intensities.end(), OutputPart[1].Intensities.begin(), OutputPart[1].Intensities.end());
	//Output.Intensities.insert(Output.Intensities.end(), OutputPart[2].Intensities.begin(), OutputPart[2].Intensities.end());


	// Save stuff
	if (SimSettings.SaveResults)
	{
		SaveSimulationOutput(Output, SimSettings.Filename_Intensity, SimSettings.Filename_XML, SimSettings);
	}

	if (Options.echo)
		PrintSimInfos(SimSettings);

}

