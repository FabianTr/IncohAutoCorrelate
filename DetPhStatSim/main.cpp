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
#include <fftw3.h>


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
	//hdf5Handle::H5Quicksave(Kernel, { 21,21 }, "/home/trostfab/scratch/Simulation/DetSimProfiler//kernel.h5", "data");

	float* M = new float[250 * 250]();
	M[250 * 125 + 25] = 1;
	M[250 * 125 + 100] = 1;
	M[250 * 25 + 125] = 1;
	M[250 * 100 + 125] = 1;

	//hdf5Handle::H5Quicksave(M, { 250,250 }, "/home/trostfab/scratch/Simulation/DetSimProfiler//M.h5", "data");

	ArrayMaths::Convolve2D(M, { 250,250 }, Kernel, { 21,21 });

	hdf5Handle::H5Quicksave(M, { 250,250 }, "/home/trostfab/scratch/Simulation/DetSimProfiler/slowConv.h5", "data");

	delete[] Kernel;
	delete[] M;
}

void TestArrayMath_FastConvolve()
{
	//float* Kernel = new float[250 * 250];
	//ArrayMaths::CreateGaussKernel(Kernel, 250, 5, true);

	//fftw_complex* FtKernel;
	//FtKernel = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * (250 * 250));
	//ArrayMaths::fft2(Kernel, FtKernel, 250, 250);
	//
	//float* M = new float[250 * 250]();
	//M[250 * 125 + 25] = 1;
	//M[250 * 125 + 100] = 1;
	//M[250 * 25 + 125] = 1;
	//M[250 * 100 + 125] = 1;
	//hdf5Handle::H5Quicksave(M, { 250,250 }, "/home/trostfab/scratch/Simulation/DetSimProfiler/M.h5", "data");

	//ArrayMaths::Convolve2DFast(M, { 250,250 }, FtKernel);

	//hdf5Handle::H5Quicksave(Kernel, { 250,250 }, "/home/trostfab/scratch/Simulation/DetSimProfiler/kernel.h5", "data");
	//
	//hdf5Handle::H5Quicksave(M, { 250,250 }, "/home/trostfab/scratch/Simulation/DetSimProfiler/fastConv.h5", "data");

	//delete[] Kernel;
	//delete[] M;
	//fftw_free(FtKernel);
}

void TestFFT()
{
	//float* Kernel = new float[1024 * 1024];
	//ArrayMaths::CreateGaussKernel(Kernel, 1024, 4.0, true);

	//hdf5Handle::H5Quicksave(Kernel, { 1024,1024 }, "/home/trostfab/scratch/Simulation/DetSimProfiler/Kernel.h5", "data");

	//fftw_complex* FT_Kernel;
	//FT_Kernel = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * (1024 * 1024));

	//ArrayMaths::fft2(Kernel, FT_Kernel, 1024, 1024);
	//float* FTA_Kernel = new float[1024 * 1024];
	//for (size_t i = 0; i < (1024*1024); i++)
	//{
	//	FTA_Kernel[i] = std::sqrt(FT_Kernel[i][0]* FT_Kernel[i][0] + FT_Kernel[i][1] * FT_Kernel[i][1]);
	//}
	//hdf5Handle::H5Quicksave(FTA_Kernel, { 1024,1024 }, "/home/trostfab/scratch/Simulation/DetSimProfiler/FT_Kernel.h5", "data");

	////Inverse FFT
	//fftw_complex* iFtFT_Kernel;
	//iFtFT_Kernel = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * (1024 * 1024));
	//ArrayMaths::ifft2(FT_Kernel, iFtFT_Kernel, 1024, 1024);
	//float* iFtFTA_Kernel = new float[1024 * 1024];
	//for (size_t i = 0; i < (1024 * 1024); i++)
	//{
	//	iFtFTA_Kernel[i] = std::sqrt(iFtFT_Kernel[i][0] * iFtFT_Kernel[i][0] + iFtFT_Kernel[i][1] * iFtFT_Kernel[i][1]);
	//}
	//hdf5Handle::H5Quicksave(iFtFTA_Kernel, { 1024,1024 }, "/home/trostfab/scratch/Simulation/DetSimProfiler/iFtFT_Kernel.h5", "data");


	//fftw_free(FT_Kernel);
	//fftw_free(iFtFT_Kernel);
	//delete[] FTA_Kernel;
	//delete[] iFtFTA_Kernel;
	//delete[] Kernel;
	//std::cout << "done." << std::endl;
}

int main(int argc, char** argv)
{
	//std::cout << "Test fft" << std::endl;
	//TestArrayMath_Convolve();
	//TestArrayMath_FastConvolve();
	////
	//std::cout << "done." << std::endl;
	//return 0;
	std::cout << "Detector Photon Statistics Simulation\n";
	std::cout << "*************************************\n" << std::endl;

	DePhStSi_Settings Options;

	
	if (argc < 2)
	{
		std::cout << "Need 1 argument \"config.ini\"\n";
		std::cout << "To generate an example config file please enter filename: (space for exit only)" << std::endl;
		std::string PathOut;
		std::cin >> PathOut;
		if (PathOut == " ")
			return 0;

		Options.SafeExampleDePhStSi_Settings(PathOut);
		std::cout << "Example DePhStSi_Settings as \"" << PathOut << "\"" << std::endl;
		return 0;
	}

	PhStatSimulator Simulator;
	if (IniParser::FileExists(argv[1]))
	{
		std::string AddArgStream = "";
		for (int i = 2; i < argc; i++)
		{
			AddArgStream = AddArgStream + ";" + argv[i];
		}
		Simulator.Options.LoadDetectorDePhStSi_Settings(argv[1], AddArgStream);
	}
	else
	{
		std::string AddArgStream = "";
		for (int i = 1; i < argc; i++)
		{
			AddArgStream = AddArgStream + ";" + argv[i];
		}
		Simulator.Options.LoadDetectorDePhStSi_Settings("", AddArgStream);
	}

	Simulator.Simulate();

	std::cout << "\nthe end." << std::endl;
    return 0;
}