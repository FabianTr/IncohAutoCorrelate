#include <cstdio>
#include <iostream>
#include <fstream>
#include <algorithm>

#include "ArrayMaths.h"

#include "hdf5Handle.h"
#include "ProfileTime.h"
#include "IniParser.h"
#include "DePhStSi_Settings.h"
#include "PhStatSimulator.h"


void Test()
{
	ProfileTime profiler;

	profiler.Tic();

	float * M = new float[1024*1024*16*16]();
	profiler.Toc(true);
	
	ArrayMaths::GetNegativeBinomialArray(M, 1024 * 1024 * 16 * 16, (0.01f / (16.0f*16.0f)), (1.0f / (16.0f*16.0f)));

	profiler.Toc(true);

	float * Kernel = new float[31 * 31];

	ArrayMaths::CreateGaussKernel(Kernel, 31 , 0.22 * 16, true);

	ArrayMaths::Convolve2D(M, { 1024 * 16 , 1024 * 16 }, Kernel, { 31,31 });

	profiler.Toc(true);

	float * PixM = new float[1024 * 1024]();
	ArrayMaths::Pixelize2DArray(M, { 1024 * 16, 1024 * 16 }, PixM, { 16,16 });

	profiler.Toc(true);

	//std::vector<hsize_t> Shape;
	//Shape.push_back(9);
	//Shape.push_back(9);


	//for (size_t i = 0; i < 25; i++)
	//{
	//	for (size_t j = 0; j < 25; j++)
	//	{
	//		std::cout << PixM[j + 25 * i] << "\t";
	//	}
	//	std::cout << std::endl;
	//}

	hdf5Handle::H5Quicksave(Kernel, { 31,31 }, "/home/trostfab/kernel.h5", "data");
	hdf5Handle::H5Quicksave(M, {1024*16,1024*16}, "/home/trostfab/M.h5", "data");
	hdf5Handle::H5Quicksave(PixM, { 1024,1024}, "/home/trostfab/PixM.h5", "data");

	return;
	while (true)
	{

		float * M = new float[100];


		float Modes = 0;

		std::cin >> Modes;

		ArrayMaths::GetNegativeBinomialArray(M, 100, 1, Modes);


		for (int i = 0; i < 10; i++)
		{
			for (int j = 0; j < 10; j++)
			{
				std::cout << M[j + 10 * i] << "\t";
			}
			std::cout << std::endl;
		}
		std::cout << "\n"<< std::endl;

	}
}

void TestINIParse()
{
	IniParser ini_pars;

	ini_pars.SetValue("Key1", "Val1");
	ini_pars.SetValue("Key2", "10");
	ini_pars.SetValue("Key3", "0.5");

	ini_pars.SafeFile("/home/trostfab/projects/IncohAutoCorrelate/bin/x64/Debug/Test.ini");

	ini_pars.Clear();

	std::cout << ini_pars.GetValue<std::string>("Key1", "empty")<< std::endl;

	ini_pars.LoadFile("/home/trostfab/projects/IncohAutoCorrelate/bin/x64/Debug/Test.ini");
	std::cout << ini_pars.GetValue<std::string>("Key1", "empty") << std::endl;
	std::cout << ini_pars.GetValue<int>("Key2", 0) << std::endl;
	std::cout << ini_pars.GetValue<float>("Key3", 0.0) << std::endl;

}


void TestArrayMath_Convolve()
{
	float* Kernel = new float[21 * 21];
	ArrayMaths::CreateGaussKernel(Kernel, 21, 5, true);
	hdf5Handle::H5Quicksave(Kernel, { 21,21 }, "/gpfs/cfel/cxi/scratch/user/trostfab/LR17/PhotonStatistics/ChargeSharingFit/SimTests/kernel.h5", "data");

	float* M = new float[250 * 250]();
	M[250 * 125 + 25] = 1;
	M[250 * 125 + 100] = 1;
	M[250 * 25 + 125] = 1;
	M[250 * 100 + 125] = 1;

	hdf5Handle::H5Quicksave(M, { 250,250 }, "/gpfs/cfel/cxi/scratch/user/trostfab/LR17/PhotonStatistics/ChargeSharingFit/SimTests/M.h5", "data");

	ArrayMaths::Convolve2D(M, { 250,250 }, Kernel, { 21,21 });

	hdf5Handle::H5Quicksave(M, { 250,250 }, "/gpfs/cfel/cxi/scratch/user/trostfab/LR17/PhotonStatistics/ChargeSharingFit/SimTests/Conv.h5", "data");

	delete[] Kernel;
	delete[] M;
}

int main(int argc, char** argv)
{
	std::cout << "Detector Photon Statistics Simulation\n";
	std::cout << "*************************************\n" << std::endl;

	DePhStSi_Settings Options;

	
	if (argc < 2)
	{
		std::cout << "Need 1 argument \"config.ini\"\n";
		std::cout << "To generate an example config file please enter filename: (empty for exit only)" << std::endl;
		std::string PathOut;
		std::cin >> PathOut;
		Options.SafeExampleDePhStSi_Settings(PathOut);
		std::cout << "Example DePhStSi_Settings as \"" << PathOut << "\"" << std::endl;
		return 0;
	}

	PhStatSimulator Simulator(argv[1]);
	Simulator.Simulate();

	std::cout << "\nthe end." << std::endl;
    return 0;
}