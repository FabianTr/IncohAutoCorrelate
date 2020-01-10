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
#include "DePhStSi_Settings.h"



PhStatSimulator::PhStatSimulator()
{
}

PhStatSimulator::PhStatSimulator(std::string DePhStSi_SettingsPath)
{
	Options.LoadDetectorDePhStSi_Settings(DePhStSi_SettingsPath);
}


std::mutex g_echo_mutex_Sim;
void PhStatSimulator::SimulatePart(std::vector<std::vector<float>> & DetImage, std::vector<std::vector<float>>& GroundTruth,DePhStSi_Settings & Options, unsigned int Loops, int ThreadNum, std::atomic<int>& counter)
{
	std::mt19937_64 mt(std::random_device{}() * ThreadNum); //random device

	size_t KernelSize = ((size_t)(Options.SuSa * 4.5 * Options.ChargeSharingSigma) );
	bool ChargeSharing = true;
	if (KernelSize < 1) // no charge sharing
	{
		ChargeSharing = false;
		KernelSize = 1;
	}
	if (KernelSize % 2 == 0)
		KernelSize++;
	float* Kernel = new float[KernelSize * KernelSize];
	if (ChargeSharing)
		ArrayMaths::CreateGaussKernel(Kernel, KernelSize, Options.ChargeSharingSigma * (float)Options.SuSa, true); //create kernel for charge-sharing


	unsigned int FullSize = Options.DetSize * Options.DetSize * Options.SuSa * Options.SuSa;
	float* M = new float[FullSize]();
	for (unsigned int i = 0; i < Loops; i++)
	{
		if (Options.Dark || Options.MeanIntensity == 0)
		{//set all pixels to zero
			for (size_t j = 0; j < Options.DetSize * Options.DetSize; j++)
			{
				DetImage[i].data()[j] = 0.0f;
				GroundTruth[i].data()[j] = 0.0f;
			}
		}
		else
		{//get distribution and charge sharing
			//Get negative binomial distribution
			ArrayMaths::GetNegativeBinomialArray(M, FullSize, Options.MeanIntensity / ((float)(Options.SuSa * Options.SuSa)), Options.Modes / ((float)(Options.SuSa * Options.SuSa)), mt); //Negative Binomial distribution (oversampled)
			

			//bin (sub-pixel) to pixel for ground truth
			ArrayMaths::Pixelize2DArray(M, { (size_t)(Options.DetSize * Options.SuSa) , (size_t)(Options.DetSize * Options.SuSa) }, GroundTruth[i].data(), { (size_t)Options.SuSa,  (size_t)Options.SuSa });

			if (ChargeSharing)//Charge Sharing
				ArrayMaths::Convolve2D(M, { (size_t)(Options.DetSize * Options.SuSa), (size_t)(Options.DetSize * Options.SuSa) }, Kernel, { KernelSize,KernelSize }); //simulate charge sharing

			//bin (sub-pixel) to pixel
			ArrayMaths::Pixelize2DArray(M, { (size_t)(Options.DetSize * Options.SuSa) , (size_t)(Options.DetSize * Options.SuSa) }, DetImage[i].data(), { (size_t)Options.SuSa,  (size_t)Options.SuSa });
		}


		if (Options.DarkNoise > 0.0) //add Noise
		{
			ArrayMaths::AddGaussianNoise<float>(DetImage[i].data(), Options.DetSize * Options.DetSize, Options.DarkNoise, mt);
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
	if (Options.ChargeSharingSigma > 0)
	{
		if ((float)Options.SuSa * Options.ChargeSharingSigma < 4.0f)
		{
			std::cout << "\nWARNING: Sub-sampling might be insufficient. \" Sub-sampling * Charge-sharing sigma > 4 \" is recommanded.\n" << std::endl;
		}
	}


	std::cout << "Start simulation with\n"
		<< "Pattern  : " << Options.Pattern << "\n"
		<< "MeanInten: " << Options.MeanIntensity << "\n"
		<< "Modes    : " << Options.Modes << "\n"
		<< "DetSiz   : " << Options.DetSize << "\n"
		<< "SuSa     : " << Options.SuSa << "\n"
		<< "CS-Sigma : " << Options.ChargeSharingSigma << "\n"
		<< "DarkNoise: " << Options.DarkNoise << "\n" 
		<< "OutputFile: " << Options.OutputPath << "; Dataset: " << Options.OutputDataset <<"; GroundTruth Dataset: " << Options.GroundTruthDataset << "\n" << std::endl;

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
	std::vector<std::vector<std::vector<float>>> GroundTruthDetImages(NumOfThreads);//space for results (ground Truth
	std::vector<std::thread> Threads;
	std::atomic<int>  counter = 0;

	std::cout << "Launch " << NumOfThreads << " threads ...\n" << std::endl; //Launch
	for (unsigned int i = 0; i < NumOfThreads; i++)
	{
		DetImages[i].resize(Loops[i]);
		GroundTruthDetImages[i].resize(Loops[i]);
		for (unsigned int j = 0; j < Loops[i]; j++)
		{
			DetImages[i][j].resize(DetSize);
			GroundTruthDetImages[i][j].resize(DetSize);
			//std::cout << DetImages.size() << " :: " << DetImages[i].size() << " :: " << DetImages[i][j].size() << std::endl;
		}

		Threads.push_back(std::thread(SimulatePart, std::ref(DetImages[i]), std::ref(GroundTruthDetImages[i]), std::ref(Options), Loops[i], (int)i, std::ref(counter)));
	}
	

	//Join threads
	for (unsigned int i = 0; i < Threads.size(); i++)
	{
		Threads[i].join();
	}


	//Reduce
	float* Result = new float[(size_t)Options.DetSize * (size_t)Options.DetSize * (size_t)Options.Pattern]();

	float* GroundTruth = new float[(size_t)Options.DetSize * (size_t)Options.DetSize * (size_t)Options.Pattern]();
	
	size_t ind = 0;
	for (size_t i = 0; i < DetImages.size(); i++)
	{
		//std::cout << i <<"\t " << DetImages[i].size() <<" ; " << DetImages[i][0].size() <<   std::endl;

		for (size_t j = 0; j < DetImages[i].size(); j++)
		{
			for (size_t k = 0; k < DetImages[i][j].size(); k++)
			{
				Result[ind] = DetImages[i][j][k];
				GroundTruth[ind] = GroundTruthDetImages[i][j][k];
				ind ++;
			}
		}
	}


	//Save Results
	hdf5Handle::H5Quicksave(Result, { (hsize_t)Options.Pattern,(hsize_t)Options.DetSize, (hsize_t)Options.DetSize }, Options.OutputPath, Options.OutputDataset);
	hdf5Handle::H5Quicksave(GroundTruth, { (hsize_t)Options.Pattern,(hsize_t)Options.DetSize, (hsize_t)Options.DetSize }, Options.OutputPath, Options.GroundTruthDataset);



	std::cout << "\nResults saved as \"" << Options.OutputPath << "\" in H5-Dataset \"" << Options.OutputDataset << "\"\ndone within ";
	profiler.Toc(true);
	std::cout << std::endl;

	delete[] Result;
	delete[] GroundTruth;
}
