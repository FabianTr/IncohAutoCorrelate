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

ProfileTime profiler;
std::mutex fft_mutex;
std::mutex g_echo_mutex_Sim;
void PhStatSimulator::SimulatePart(std::vector<std::vector<double>> & DetImage, std::vector<std::vector<double>>& GroundTruth,DePhStSi_Settings & Options, unsigned int Loops, int ThreadNum, std::atomic<int>& counter)
{
	const auto seed = std::random_device{}() * (ThreadNum+1);
	//std::cout << seed << std::endl;;
	std::mt19937_64 mt(seed); //random device

	int FullSize = Options.DetSize * Options.DetSize * Options.SuSa * Options.SuSa;
	double* M = new double[FullSize]();

	//size_t KernelSize = ((size_t)(Options.SuSa * 4.5 * Options.ChargeSharingSigma) );
	size_t KernelSize = ((size_t)(Options.DetSize * Options.SuSa));
	bool ChargeSharing = true;

	double* Kernel = new double[KernelSize * KernelSize]();
	ArrayMaths::CreateGaussKernel(Kernel, KernelSize, Options.ChargeSharingSigma * (float)Options.SuSa, true); //create kernel for charge-sharing
	
	fft_mutex.lock();
	fftw_complex* FtKernel;
	FtKernel = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * (KernelSize * KernelSize));
	fftw_complex* Array;
	Array = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * (KernelSize * KernelSize));
	fftw_complex* Buffer;
	Buffer = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * (KernelSize * KernelSize));

	fftw_plan P_fft2;
	P_fft2 = fftw_plan_dft_2d((int)KernelSize, (int)KernelSize, Array, Buffer, FFTW_FORWARD, FFTW_MEASURE); //FFTW_MEASURE

	fftw_plan P_ifft2;
	P_ifft2 = fftw_plan_dft_2d((int)KernelSize, (int)KernelSize, Array, Buffer, FFTW_BACKWARD, FFTW_MEASURE); //FFTW_ESTIMATE
	fft_mutex.unlock();

	ArrayMaths::fft2(Kernel, FtKernel, KernelSize, KernelSize, fft_mutex);
	delete[] Kernel;
	
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
			
//			profiler.Tic();
			//bin (sub-pixel) to pixel for ground truth
			ArrayMaths::Pixelize2DArray(M, { (size_t)(Options.DetSize * Options.SuSa) , (size_t)(Options.DetSize * Options.SuSa) }, GroundTruth[i].data(), { (size_t)Options.SuSa,  (size_t)Options.SuSa });
//			time_1 += profiler.Toc(false, false);

//			profiler.Tic();
			if (ChargeSharing)//Charge Sharing
			{
				for (size_t k = 0; k < KernelSize* KernelSize; k++)
				{
					Array[k][0] = M[k]; //real
					Array[k][1] = 0.0;  //imag
				}

				//ArrayMaths::Convolve2D(M, { (size_t)(Options.DetSize * Options.SuSa), (size_t)(Options.DetSize * Options.SuSa) }, Kernel, { KernelSize,KernelSize }); //simulate charge sharing
				
				//ArrayMaths::Convolve2DFast(M, { (size_t)(Options.DetSize * Options.SuSa), (size_t)(Options.DetSize * Options.SuSa) }, FtKernel, fft_mutex);
				
				ArrayMaths::Convolve2DFast(Array, Buffer, M, { KernelSize,KernelSize }, FtKernel, P_fft2, P_ifft2);
			}
//			time_2 += profiler.Toc(false, false);

//			profiler.Tic();
			//bin (sub-pixel) to pixel
			ArrayMaths::Pixelize2DArray(M, { (size_t)(Options.DetSize * Options.SuSa) , (size_t)(Options.DetSize * Options.SuSa) }, DetImage[i].data(), { (size_t)Options.SuSa,  (size_t)Options.SuSa });
//			time_3 += profiler.Toc(false, false);
		}


		if (Options.DarkNoise > 0.0) //add Noise
		{
//			profiler.Tic();
			ArrayMaths::AddGaussianNoise<double>(DetImage[i].data(), Options.DetSize * Options.DetSize, Options.DarkNoise, mt);
//			time_4 += profiler.Toc(false, false);
		}
		
		counter++;

		if (Options.Pattern >= 100)
		{
			if ((counter) % (Options.Pattern / 100) == 0)
			{
				g_echo_mutex_Sim.lock();
				std::cout << counter << " / " << Options.Pattern << " ^= " << round(counter * 100 / Options.Pattern) << "\% in ";
				profiler.Toc(true, true);
				g_echo_mutex_Sim.unlock();
			}
		}
		else
		{
			g_echo_mutex_Sim.lock();
			std::cout << counter << " / " << Options.Pattern << " ^= " << round(counter * 100 / Options.Pattern) << "\% in ";
			profiler.Toc(true, true);
			g_echo_mutex_Sim.unlock();
		}

	}
	delete[] M;

	fft_mutex.lock();
	fftw_free(FtKernel);
	fftw_free(Array);
	fftw_free(Buffer);
	fft_mutex.unlock();

	//std::cout << "Time consumed by getting random numbers:\n";
	//ProfileTime::PrintTime(time_1);

	//std::cout << "\nTime consumed by convolution with CS-kernel:\n";
	//ProfileTime::PrintTime(time_2);

	//std::cout << "\nTime consumed by Pixel chunking:\n";
	//ProfileTime::PrintTime(time_3);

	//std::cout << "\nTime consumed by Noise simulation:\n";
	//ProfileTime::PrintTime(time_4);

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
		<< "OutputFile: " << Options.OutputPath << "; Dataset: " << Options.OutputDataset <<"; GroundTruth Dataset: " << Options.GroundTruthDataset << "\n" 
		<< "H5Compression: " << Options.compression
		<< std::endl;

	
	profiler.Tic();

	
	unsigned int NumOfThreads = std::thread::hardware_concurrency();
	if (NumOfThreads < 1)
		NumOfThreads = 1;

	if (Options.MaxThreads != -1 && Options.MaxThreads < NumOfThreads)
		NumOfThreads = Options.MaxThreads;

	//NumOfThreads = 3; //For profiling purpose

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

	std::vector<std::vector<std::vector<double>>> DetImages(NumOfThreads);//space for results
	std::vector<std::vector<std::vector<double>>> GroundTruthDetImages(NumOfThreads);//space for results (ground Truth
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
		}

		Threads.push_back(std::thread(SimulatePart, std::ref(DetImages[i]), std::ref(GroundTruthDetImages[i]), std::ref(Options), Loops[i], (int)i, std::ref(counter)));
	}
	

	//Join threads
	for (unsigned int i = 0; i < Threads.size(); i++)
	{
		Threads[i].join();
	}

	fftw_cleanup();

	//Reduce
	float* Result      = new float[(size_t)Options.DetSize * (size_t)Options.DetSize * (size_t)Options.Pattern]();
	float* GroundTruth = new float[(size_t)Options.DetSize * (size_t)Options.DetSize * (size_t)Options.Pattern]();
	
	//Fill result and groundtruth in arrays
	size_t ind = 0; 
	for (size_t i = 0; i < DetImages.size(); i++)
	{
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
	hdf5Handle::H5Quicksave(Result, { (hsize_t)Options.Pattern,(hsize_t)Options.DetSize, (hsize_t)Options.DetSize }, Options.OutputPath, Options.OutputDataset,Options.compression);
	hdf5Handle::H5Quicksave(GroundTruth, { (hsize_t)Options.Pattern,(hsize_t)Options.DetSize, (hsize_t)Options.DetSize }, Options.OutputPath, Options.GroundTruthDataset, Options.compression);



	std::cout << "\nResults saved as \"" << Options.OutputPath << "\" in H5-Dataset \"" << Options.OutputDataset << "\"\ndone within ";
	profiler.Toc(true);
	std::cout << " ^= " << Options.Pattern / profiler.Toc(false) << "Hz";
	std::cout << std::endl;

	delete[] Result;
	delete[] GroundTruth;
}
