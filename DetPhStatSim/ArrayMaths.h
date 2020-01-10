#pragma once

#include <omp.h>
#include <fstream>
#include <string>
#include <math.h>
#include <random> 
#include <array>
#include <iostream>
#include "ProfileTime.h"

namespace
{
	std::mt19937_64 mt(std::random_device{}());

	std::uniform_real_distribution<double> rnd;

	std::default_random_engine lce(mt());
}

namespace ArrayMaths
{

	inline double Drand(std::mt19937_64 & MT)
	{
		return rnd(MT);
	}

	inline unsigned int ScalarPoissonSampling(double mean, std::mt19937_64 & MT )
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
			ret = (mean + sqrt(mean)*sqrt(-2. * log(u)) * cos(2. * M_PI * v));
			if (ret < 0)
				ret = 0;
		}

		return (unsigned int)ret;
	}

	template<typename T>
	void GetNegativeBinomialArray(T * Array, size_t ArraySize, float Mean, float Modes, std::mt19937_64 & MT)
	{
		ProfileTime profiler;
		//std::negative_binomial_distribution<int> NB(Modes, Modes / (Modes + Mean));
		std::gamma_distribution<float> Gamma(Modes, (Mean / Modes));
		
		//profiler.Tic();
		//#pragma omp parallel for
		for (size_t i = 0; i < ArraySize; i++)
		{
			Array[i] = (T)ScalarPoissonSampling(Gamma(MT),MT);
			//Array[i] = (T)ScalarPoissonSampling(Drand());
			//Array[i] = (T)Gamma(mt);
		}
	}

	template<typename T>
	void AddGaussianNoise(T* Array, size_t ArraySize, T Sigma, std::mt19937_64 & MT)
	{
		std::normal_distribution<T> Gauss((T)0.0, Sigma);
		for (size_t i = 0; i < ArraySize; i++)
		{
			Array[i] += Gauss(MT);
		}
	}

	//think of a more accurate description for the noise than Gaussian

	template<typename T>
	void Convolve2D(T * Array, std::array<size_t, 2> ArraySize, const T * Kernel, std::array<size_t, 2> KernelSize) //Sizees in ss,fs
	{
		if (KernelSize[0] % 2 != 1 || KernelSize[1] % 2 != 1)
		{
			std::cerr << "´WARNING: Convolution kernel should be of uneven size" << std::endl;
		}

		T * ret = new T[ArraySize[0] * ArraySize[1]]; //temporary return array

		for (size_t Ass = 0; Ass < ArraySize[0]; Ass++)
		{
			for (size_t Afs = 0; Afs < ArraySize[1]; Afs++)
			{
				//apply kernel
				T Sum = 0;
				for (size_t Kss	 = 0; Kss < KernelSize[0]; Kss++)
				{
					for (size_t Kfs = 0; Kfs < KernelSize[1]; Kfs++)
					{
						size_t ss = Ass + Kss - KernelSize[0] / 2;
						size_t fs = Afs + Kfs - KernelSize[1] / 2;

						//check for boundaries
						if (ss < 0 || fs < 0 || ss >= ArraySize[0] || fs >= ArraySize[1])
							continue;

						Sum += Array[fs + ss * ArraySize[1]] * Kernel[Kfs + Kss* KernelSize[1]];
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
	void CreateGaussKernel(T * Kernel, size_t KernelSize, float Sigma, bool Norm = true)//kernel size is edge-size 
	{
		for (size_t ss = 0; ss < KernelSize; ss++)
		{
			for (size_t fs = 0; fs < KernelSize; fs++)
			{
				float XL = (float)ss - 0.5f * (float)(KernelSize - 1) - 0.5f;
				float XH = (float)ss - 0.5f * (float)(KernelSize - 1) + 0.5f;
				float YL = (float)fs - 0.5f * (float)(KernelSize - 1) - 0.5f;
				float YH = (float)fs - 0.5f * (float)(KernelSize - 1) + 0.5f;

				T Val = 0.25f * (std::erf(XH / (std::sqrt(2.0f)*Sigma)) - std::erf(XL / (std::sqrt(2.0f)*Sigma)))*
					(std::erf(YH / (std::sqrt(2.0f)*Sigma)) - std::erf(YL / (std::sqrt(2.0f)*Sigma)));

				Kernel[fs + ss * KernelSize] = Val;
			}
		}

		//normalize (e.g. required if Sigma is not << KernelSize)
		if (Norm)
		{
			T Sum = 0;
			for (size_t i = 0; i < KernelSize*KernelSize; i++)
				Sum += Kernel[i];

			for (size_t i = 0; i < KernelSize*KernelSize; i++)
				Kernel[i] /= Sum;
		}
	}


	template<typename T>
	void Pixelize2DArray(T * Input, std::array<size_t, 2> InputSize, T * Return, std::array<size_t, 2> PixelSize) //sizes in {slow,fast}
	{
		if (InputSize[0] % PixelSize[0] != 0 || InputSize[1] % PixelSize[1] != 0)
		{
			std::cerr << "ERROR: Inputsize must be modulo PixelSize" << std::endl;
			std::cerr << "   -> in Pixelize2DArray(...)" << std::endl;
			throw;
		}

		for (size_t ss = 0; ss < InputSize[0]/PixelSize[0]; ss++)
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

				Return[fs + ss* (InputSize[1] / PixelSize[1]) ] = sum;
			}
		}
	}

}

