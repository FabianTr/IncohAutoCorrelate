#include "PhStatSimulator.h"
#include <cstdio>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <random> 
#include <thread>
#include <mutex>

#include "ArrayMaths.h"

#include "hdf5Handle.h"
#include "ProfileTime.h"
#include "IniParser.h"
#include "Settings.h"




PhStatSimulator::PhStatSimulator()
{
}

PhStatSimulator::PhStatSimulator(std::string SettingsPath)
{
	Options.LoadDetectorSettings(SettingsPath);
}


std::mutex g_echo_mutex_Sim;
void PhStatSimulator::SimulatePart(std::vector<std::vector<float>> & DetImage,Settings & Options, unsigned int Loops, int ThreadNum, std::atomic<int>& counter)
{
	std::mt19937_64 mt(std::random_device{}() * ThreadNum); //random device

	size_t KernelSize = ((size_t)(Options.SuSa * 4 * Options.ChargeSharingSigma) + 1);
	float* Kernel = new float[KernelSize * KernelSize];
	ArrayMaths::CreateGaussKernel(Kernel, KernelSize, Options.ChargeSharingSigma * Options.SuSa, true); //create kernel for charge-sharing


		////Debug Stuff
	//g_echo_mutex_Sim.lock();
	//hdf5Handle::H5Quicksave(Kernel, { KernelSize,KernelSize }, "/gpfs/cfel/cxi/scratch/user/trostfab/LR17/PhotonStatistics/ChargeSharingFit/SimTests/kernel.h5", Options.OutputDataset);
	//g_echo_mutex_Sim.unlock();
	//// \Debug Stuff


	unsigned int FullSize = Options.DetSize * Options.DetSize * Options.SuSa * Options.SuSa;
	float* M = new float[FullSize]();
	for (unsigned int i = 0; i < Loops; i++)
	{
		ArrayMaths::GetNegativeBinomialArray(M, FullSize,Options.MeanIntensity / ((float)(Options.SuSa * Options.SuSa)), Options.Modes / ((float)(Options.SuSa * Options.SuSa)),mt); //Negative Binomial distribution (oversampled)
		
			////Debug Stuff
			//g_echo_mutex_Sim.lock();
			//hdf5Handle::H5Quicksave(M, { Options.DetSize * Options.SuSa,Options.DetSize * Options.SuSa }, "/gpfs/cfel/cxi/scratch/user/trostfab/LR17/PhotonStatistics/ChargeSharingFit/SimTests/M.h5", Options.OutputDataset);
			//g_echo_mutex_Sim.unlock();
			//// \Debug Stuff
		
		ArrayMaths::Convolve2D(M, { Options.DetSize * Options.SuSa, Options.DetSize * Options.SuSa }, Kernel, { KernelSize,KernelSize }); //simulate charge sharing

			////Debug Stuff
			//g_echo_mutex_Sim.lock();
			//hdf5Handle::H5Quicksave(M, { Options.DetSize * Options.SuSa,Options.DetSize * Options.SuSa }, "/gpfs/cfel/cxi/scratch/user/trostfab/LR17/PhotonStatistics/ChargeSharingFit/SimTests/GFM.h5", Options.OutputDataset);
			//g_echo_mutex_Sim.unlock();
			//// \Debug Stuff

		ArrayMaths::Pixelize2DArray(M, { Options.DetSize*Options.SuSa , Options.DetSize * Options.SuSa }, DetImage[i].data(), { Options.SuSa, Options.SuSa });

			////Debug Stuff
			//g_echo_mutex_Sim.lock();
			//hdf5Handle::H5Quicksave(DetImage[i].data(), { Options.DetSize ,Options.DetSize}, "/gpfs/cfel/cxi/scratch/user/trostfab/LR17/PhotonStatistics/ChargeSharingFit/SimTests/Pix.h5", Options.OutputDataset);
			//g_echo_mutex_Sim.unlock();
			//// \Debug Stuff

		if (Options.DarkNoise > 0.0) //add Noise
		{
			ArrayMaths::AddGaussianNoise(DetImage[i].data(), Options.DetSize * Options.DetSize, (double)Options.DarkNoise, mt);
		}
		
		counter++;

		if (Options.Pattern >= 100)
		{
			if ((counter) % (Options.Pattern / 100) == 0)
			{
				g_echo_mutex_Sim.lock();
				std::cout << counter << " / " << Options.Pattern << " ^= " << round(counter * 100 / Options.Pattern) << "\%" << std::endl;
				g_echo_mutex_Sim.unlock();
			}
		}
		else
		{
			g_echo_mutex_Sim.lock();
			std::cout << counter << " / " << Options.Pattern << " ^= " << round(counter * 100 / Options.Pattern) << "\%" << std::endl;
			g_echo_mutex_Sim.unlock();
		}

	}
	delete[] M;
	delete[] Kernel;
}


void PhStatSimulator::Simulate()
{
	std::cout << "Start simulation with\n"
		<< "Pattern  : " << Options.Pattern << "\n"
		<< "MeanInten: " << Options.MeanIntensity << "\n"
		<< "Modes    : " << Options.Modes << "\n"
		<< "DetSiz   : " << Options.DetSize << "\n"
		<< "SuSa     : " << Options.SuSa << "\n"
		<< "CS-Sigma : " << Options.ChargeSharingSigma << "\n"
		<< "DarkNoise: " << Options.DarkNoise << "\n" << std::endl;

	ProfileTime profiler;

	profiler.Tic();

	unsigned int NumOfThreads = std::thread::hardware_concurrency();
	if (NumOfThreads < 1)
		NumOfThreads = 1;

	unsigned int MainThreadLoops = Options.Pattern / NumOfThreads;
	unsigned int LastThreadLoops = Options.Pattern % NumOfThreads;

	if (MainThreadLoops == 0) //more cores than patterns
	{
		NumOfThreads = LastThreadLoops; //number of threads = number of pattern
		LastThreadLoops = 0; // no additional thread
		MainThreadLoops = 1; // 1 Loop per thread
	}

	std::vector<unsigned int> Loops;
	for (unsigned int i = 0; i < NumOfThreads; i++)
	{
		Loops.push_back(MainThreadLoops);
	}


	if (LastThreadLoops != 0) // if Patern%Threads != 0 => create additional thread
	{
		NumOfThreads += 1;
		Loops.push_back(LastThreadLoops);
	}
	   
	unsigned int DetSize = Options.DetSize * Options.DetSize;

	std::vector<std::vector<std::vector<float>>> DetImages(NumOfThreads);//space for results
	std::vector<std::thread> Threads;
	std::atomic<int>  counter = 0;

	std::cout << "Launch " << NumOfThreads << " threads ...\n" << std::endl;
	for (unsigned int i = 0; i < NumOfThreads; i++)
	{
		DetImages[i].resize(Loops[i]);
		for (unsigned int j = 0; j < Loops[i]; j++)
		{
			DetImages[i][j].resize(DetSize);

			//std::cout << DetImages.size() << " :: " << DetImages[i].size() << " :: " << DetImages[i][j].size() << std::endl;
		}

		Threads.push_back(std::thread(SimulatePart, std::ref(DetImages[i]), std::ref(Options), Loops[i], (int)i, std::ref(counter)));
	}
	

	//Join threads
	for (unsigned int i = 0; i < Threads.size(); i++)
	{
		Threads[i].join();
	}


	//Reduce
	float* Result = new float[Options.DetSize * Options.DetSize * Options.Pattern];
	size_t ind = 0;
	//std::cout << Options.DetSize << " ; " << Options.Pattern << " ; " << Options.DetSize * Options.DetSize * Options.Pattern << std::endl;
	//std::cout << DetImages.size() << std::endl;


	for (size_t i = 0; i < DetImages.size(); i++)
	{
		//std::cout << i <<"\t " << DetImages[i].size() <<" ; " << DetImages[i][0].size() <<   std::endl;

		for (size_t j = 0; j < DetImages[i].size(); j++)
		{
			for (size_t k = 0; k < DetImages[i][j].size(); k++)
			{
				Result[ind] = DetImages[i][j][k];
				ind ++;
			}
		}
	}

	//Save Results
	hdf5Handle::H5Quicksave(Result, { Options.Pattern,Options.DetSize ,Options.DetSize }, Options.OutputPath, Options.OutputDataset);

	std::cout << "Results saved as \"" << Options.OutputPath << "\" in H5-Dataset \"" << Options.OutputDataset << "\"\ndone within ";
	profiler.Toc(true);
	std::cout << std::endl;

	delete[] Result;
}
