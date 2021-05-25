#pragma once

#include <omp.h>
#include <fstream>
#include <string>
#include <math.h>
#include <random> 
#include <array>
#include <iostream>
#include "ProfileTime.h"
#include <fftw3.h>
#include <string.h>
#include <mutex>


namespace
{
	std::mt19937_64 mt(std::random_device{}());

	std::uniform_real_distribution<double> rnd;

	std::default_random_engine lce(mt());
}


namespace ArrayMaths
{
	inline double Drand(std::mt19937_64& MT)
	{
		return rnd(MT);
	}

	inline unsigned int ScalarPoissonSampling(double mean, std::mt19937_64& MT)
	{
		double ret = 0;
		//Poisson random number for small means 
		//Inversion by sequential search
		if (mean < 25)
		{
			int i = 0;
			double p, q, r;
			r = exp(-mean);
			p = r;
			q = Drand(MT);
			while (p < q) {
				i++;
				r *= mean / i;
				p += r;
			}
			ret = i;
		}
		else
			// Gaussian number by Box-Muller method for large means
		{
			double u = 1 - Drand(MT);
			double v = 1 - Drand(MT);
			ret = (mean + sqrt(mean) * sqrt(-2. * log(u)) * cos(2. * M_PI * v));
			if (ret < 0)
				ret = 0;
		}

		return (unsigned int)ret;
	}

	template<typename T>
	void GetNegativeBinomialArray(T* Array, size_t ArraySize, float Mean, float Modes, std::mt19937_64& MT)
	{
		ProfileTime profiler;
		//std::negative_binomial_distribution<int> NB(Modes, Modes / (Modes + Mean));
		std::gamma_distribution<float> Gamma(Modes, (Mean / Modes));

		//profiler.Tic();
		//#pragma omp parallel for
		for (size_t i = 0; i < ArraySize; i++)
		{
			Array[i] = (T)ScalarPoissonSampling(Gamma(MT), MT);
			//Array[i] = (T)ScalarPoissonSampling(Drand());
			//Array[i] = (T)Gamma(mt);
		}
	}

	template<typename T>
	void AddGaussianNoise(T* Array, size_t ArraySize, T Sigma, std::mt19937_64& MT)
	{
		std::normal_distribution<T> Gauss((T)0.0, Sigma);
		for (size_t i = 0; i < ArraySize; i++)
		{
			Array[i] += Gauss(MT);
		}


	}

	//think of a more accurate description for the noise than Gaussian

	template<typename T>
	void Convolve2D(T* Array, std::array<size_t, 2> ArraySize, const T* Kernel, std::array<size_t, 2> KernelSize) //Sizees in ss,fs
	{
		if (KernelSize[0] % 2 != 1 || KernelSize[1] % 2 != 1)
		{
			std::cerr << "´WARNING: Convolution kernel should be of uneven size" << std::endl;
		}

		T* ret = new T[ArraySize[0] * ArraySize[1]]; //temporary return array

		for (size_t Ass = 0; Ass < ArraySize[0]; Ass++)
		{
			for (size_t Afs = 0; Afs < ArraySize[1]; Afs++)
			{
				//apply kernel
				T Sum = 0;
				for (size_t Kss = 0; Kss < KernelSize[0]; Kss++)
				{
					for (size_t Kfs = 0; Kfs < KernelSize[1]; Kfs++)
					{
						size_t ss = Ass + Kss - KernelSize[0] / 2;
						size_t fs = Afs + Kfs - KernelSize[1] / 2;

						//check for boundaries
						if (ss < 0 || fs < 0 || ss >= ArraySize[0] || fs >= ArraySize[1])
							continue;

						Sum += Array[fs + ss * ArraySize[1]] * Kernel[Kfs + Kss * KernelSize[1]];
					}
				}
				ret[Afs + Ass * ArraySize[1]] = Sum;
			}
		}

		for (size_t i = 0; i < ArraySize[0] * ArraySize[1]; i++)
		{
			Array[i] = ret[i];
		}
		delete[] ret;
	}

	template<typename T>
	inline void fft2(T* Input, fftw_complex* Output, int Size_FS, int Size_SS, std::mutex & fft_mutex)
	{
		fftw_complex* DataInput;
		DataInput = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * (Size_FS * Size_SS));

		for (int i = 0; i < (Size_FS * Size_SS); i++)
		{
			DataInput[i][0] = Input[i]; //real
			DataInput[i][1] = 0.0; //imag
		}
		
		fft_mutex.lock();
		fftw_plan p;
		p = fftw_plan_dft_2d(Size_FS, Size_SS, DataInput, Output, FFTW_FORWARD, FFTW_ESTIMATE);
		fft_mutex.unlock();

		fftw_execute(p);

		fft_mutex.lock();
		fftw_destroy_plan(p);
		fftw_free(DataInput);
		fft_mutex.unlock();
	}

	inline void ifft2(fftw_complex* Input, fftw_complex* Output, int Size_FS, int Size_SS, std::mutex & fft_mutex)
	{
		
		fft_mutex.lock();
		fftw_plan p;
		p = fftw_plan_dft_2d(Size_FS, Size_SS, Input, Output, FFTW_BACKWARD, FFTW_ESTIMATE);
		fft_mutex.unlock();

		fftw_execute(p);

		for (int i = 0; i < (Size_FS * Size_SS); i++)
		{
			Output[i][0] /= (double)(Size_FS * Size_SS);
			Output[i][1] /= (double)(Size_FS * Size_SS);
		}

		fft_mutex.lock();
		fftw_destroy_plan(p);
		fft_mutex.unlock();
	}
	//void fftshift2(double* Array, std::array<size_t, 2> ArraySize);

	template<typename T>
	inline void fftshift2D(T* Input, T* Output, int n_fs, int n_ss)// only even n_ss && n_fs
	{
		for (int i = 0; i < n_ss/2; i++)
		{
			int o = i * n_fs;
			int o2 = o + (n_fs * n_ss/2);

			memcpy(Output + (o)          , Input + (o2 + n_fs/2), sizeof(T) * (n_fs/2));
			memcpy(Output + (o2 + n_fs/2), Input + (o)          , sizeof(T) * (n_fs/2));

			memcpy(Output + (o2)          , Input + (o + n_fs / 2), sizeof(T) * (n_fs / 2));
			memcpy(Output + (o + n_fs / 2), Input + (o2)          , sizeof(T) * (n_fs / 2));
		}
	}

	template<typename T>
	inline void Convolve2DFast(T* Array, std::array<size_t, 2> ArraySize, const fftw_complex* FTKernel, std::mutex & fft_mutex) //Size in ss,fs
	{
		if (ArraySize[0] % 2 != 0 || ArraySize[1] % 2 != 0)
		{
			std::cout << "Use even sized Detectors please..." << std::endl;
			throw;
		}
		fft_mutex.lock();
		fftw_complex* FTArray;
		FTArray = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * (ArraySize[0] * ArraySize[1]));
		fft_mutex.unlock();
		//Fourier transform input
		fft2(Array, FTArray, (int)ArraySize[0], (int)ArraySize[1], fft_mutex);
		//multiply with (FT)Kernel
		for (size_t i = 0; i < (ArraySize[0] * ArraySize[1]); i++)
		{
			//multiplication of complex numbers
			double a = FTArray[i][0];
			double b = FTArray[i][1];
			double x = FTKernel[i][0];
			double y = FTKernel[i][1];
			FTArray[i][0] = a * x - b * y;
			FTArray[i][1] = a * y + b * x;
		}
		fft_mutex.lock();
		fftw_complex* COutput;
		COutput = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * (ArraySize[0] * ArraySize[1]));
		fft_mutex.unlock();
		//inverse Fourier transform product
		ifft2(FTArray, COutput, (int)ArraySize[0], (int)ArraySize[1], fft_mutex);
		//in principal there should be no imag contribution
		T* tmp = new T[ArraySize[0] * ArraySize[1]];
		for (size_t i = 0; i < (ArraySize[0] * ArraySize[1]); i++)
		{
			tmp[i] = (T)std::sqrt(COutput[i][0] * COutput[i][0] + COutput[i][1] * COutput[i][1]);
		}
		fftshift2D(tmp, Array, ArraySize[0], ArraySize[1]);
		fft_mutex.lock();
		fftw_free(FTArray);
		fftw_free(COutput);
		fft_mutex.unlock();
		delete[] tmp;
	}

	void Convolve2DFast(fftw_complex* Array, fftw_complex* Buffer, double* Output, std::array<size_t, 2> ArraySize, const fftw_complex* FTKernel, fftw_plan & fft2P, fftw_plan & ifft2P); //Size in ss,fs
	


	template<typename T>
	void CreateGaussKernel(T* Kernel, size_t KernelSize, float Sigma, bool Norm = true)//kernel size is edge-size 
	{
		for (size_t ss = 0; ss < KernelSize; ss++)
		{
			for (size_t fs = 0; fs < KernelSize; fs++)
			{
				float XL = (float)ss - 0.5f * (float)(KernelSize - 1) - 0.5f;
				float XH = (float)ss - 0.5f * (float)(KernelSize - 1) + 0.5f;
				float YL = (float)fs - 0.5f * (float)(KernelSize - 1) - 0.5f;
				float YH = (float)fs - 0.5f * (float)(KernelSize - 1) + 0.5f;

				T Val = 0.25f * (std::erf(XH / (std::sqrt(2.0f) * Sigma)) - std::erf(XL / (std::sqrt(2.0f) * Sigma))) *
					(std::erf(YH / (std::sqrt(2.0f) * Sigma)) - std::erf(YL / (std::sqrt(2.0f) * Sigma)));

				Kernel[fs + ss * KernelSize] = Val;
			}
		}

		//normalize (e.g. required if Sigma is not << KernelSize)
		if (Norm)
		{
			T Sum = 0;
			for (size_t i = 0; i < KernelSize * KernelSize; i++)
				Sum += Kernel[i];

			for (size_t i = 0; i < KernelSize * KernelSize; i++)
				Kernel[i] /= Sum;
		}
	}


	template<typename T>
	void Pixelize2DArray(T* Input, std::array<size_t, 2> InputSize, T* Return, std::array<size_t, 2> PixelSize) //sizes in {slow,fast}
	{
		if (InputSize[0] % PixelSize[0] != 0 || InputSize[1] % PixelSize[1] != 0)
		{
			std::cerr << "ERROR: Inputsize must be modulo PixelSize" << std::endl;
			std::cerr << "   -> in Pixelize2DArray(...)" << std::endl;
			throw;
		}

		for (size_t ss = 0; ss < InputSize[0] / PixelSize[0]; ss++)
		{
			for (size_t fs = 0; fs < InputSize[1] / PixelSize[1]; fs++)
			{
				T sum = 0.0;
				for (size_t k = 0; k < PixelSize[0]; k++)
				{
					size_t ss_t = ss * PixelSize[0] + k;
					for (size_t l = 0; l < PixelSize[1]; l++)
					{
						size_t fs_t = fs * PixelSize[1] + l;
						sum += Input[fs_t + ss_t * InputSize[1]];
					}
				}

				Return[fs + ss * (InputSize[1] / PixelSize[1])] = sum;
			}
		}
	}


}