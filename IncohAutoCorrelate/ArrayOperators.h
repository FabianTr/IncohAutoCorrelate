#pragma once

#include <omp.h>
#include <fstream>
#include <string>
#include <math.h>
#include <random> 

#include "Settings.h"

namespace 
{
    std::mt19937_64 mt(std::random_device{}());
	std::uniform_real_distribution<double> rnd;
}

namespace ArrayOperators
{
	// Scalar functions
	inline double Drand()
	{
		return rnd(mt);
	}
	inline unsigned int ScalarPoissonSampling(double mean)
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
			q = Drand();
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
			double u = 1 - Drand();
			double v = 1 - Drand();
			ret = (mean + sqrt(mean)*sqrt(-2. * log(u)) * cos(2. * M_PI * v));
			if (ret < 0)
				ret = 0;
		}

		return (unsigned int)ret;
	}

	
	//par functions
	inline void ParAdd(float* Array, float* Summand, int Size)
	{
		#pragma omp parallel for
		for (int i = 0; i < Size; i++)
		{
			Array[i] += Summand[i];
		}
	}
	inline void ParAdd(double* Array, double* Summand, int Size)
	{
		#pragma omp parallel for
		for (int i = 0; i < Size; i++)
		{
			Array[i] += Summand[i];
		}
	}
	inline void ParAdd(float* Array, float* Summand, int Size, float Threshold)
	{
		#pragma omp parallel for
		for (int i = 0; i < Size; i++)
		{
			if (Summand[i] >= Threshold)
				Array[i] += Summand[i];
		}
	}

	inline void ParMultiplyScalar(float* Array, float Factor, int Size)
	{
		#pragma omp parallel for
		for (int i = 0; i < Size; i++)
		{
			Array[i] *= Factor;
		}
	}
	inline void ParMultiplyScalar(double* Array, double Factor, int Size)
	{
#pragma omp parallel for
		for (int i = 0; i < Size; i++)
		{
			Array[i] *= Factor;
		}
	}
	inline void ParMultiplyElementwise(float* Array, float* Factor, unsigned int Size)
	{
		#pragma omp parallel for
		for (unsigned int i = 0; i < Size; i++)
		{
			Array[i] = Array[i] * Factor[i];
		}
	}
	inline void ParMultiplyElementwise(float* Array, int* Factor, unsigned int Size)
	{
		#pragma omp parallel for
		for (unsigned int i = 0; i < Size; i++)
		{
			Array[i] = Array[i] * (float)Factor[i];
		}
	}


	inline void MultiplyElementwise(float* Array, float* Factor, unsigned int Size)
	{
		for (unsigned int i = 0; i < Size; i++)
		{
			Array[i] = Array[i] * Factor[i];
		}
	}
	inline void MultiplyElementwise(float* Array, int* Factor, unsigned int Size)
	{
		for (unsigned int i = 0; i < Size; i++)
		{
			Array[i] = Array[i] * (float)Factor[i];
		}
	}

	inline void DiscretizeToPhotons(float* Array, float Threshold, float PhotonSamplingStep, unsigned int Size)
	{
		//Simple Threshold Method
		#pragma omp parallel for
		for (unsigned int i = 0; i < Size; i++)
		{
			float t = ceilf((Array[i] - Threshold) / PhotonSamplingStep)*(Array[i] >= Threshold);
			
			Array[i] = t;
		} 
	}

	inline void ThresholdValues(float* Array, float Threshold, int Size)
	{
		#pragma omp parallel for
		for (int i = 0; i < Size; i++)
		{
			if (Array[i] < Threshold)
				Array[i] = 0;
		}
	}

	inline void Min_Max_Mean_Value(float* Array, int Size, float& MinValue, float& MaxValue, float& MeanValue)
	{
		double t_mean=0;
		MinValue = 1e38f;
		MaxValue = -1e38f;
		for (int i = 0; i < Size; i++)
		{
			if (Array[i] > MaxValue)
				MaxValue = Array[i];
			if (Array[i] < MinValue)
				MinValue = Array[i];
			t_mean += Array[i];
		}
		t_mean = t_mean / Size;
		MeanValue = (float)t_mean;
	}

	inline void Min_Max_Mean_Value(double* Array, int Size, double& MinValue, double& MaxValue, double& MeanValue)
	{
		double t_mean = 0;
		MinValue = 1e38f;
		MaxValue = -1e38f;
		for (int i = 0; i < Size; i++)
		{
			if (Array[i] > MaxValue)
				MaxValue = Array[i];
			if (Array[i] < MinValue)
				MinValue = Array[i];
			t_mean += Array[i];
		}
		t_mean = t_mean / Size;
		MeanValue = (double)t_mean;
	}

	//Poisson sampl is serial because of copied seed problem!!, FIX!!!
	inline void ParPoissonSampling(double* Array, unsigned int Size)
	{
		//#pragma omp parallel for
		for (unsigned int i = 0; i < Size; i++)
		{
			Array[i] = (double)ScalarPoissonSampling(Array[i]);
		}
	}
	inline void ParPoissonSampling(float* Array, unsigned int Size)
	{
		//#pragma omp parallel for
		for (unsigned int i = 0; i < Size; i++)
		{
			Array[i] = (float)ScalarPoissonSampling(Array[i]);
		}
	}
	inline void ParAddWhiteNoise(double* Array,double level, unsigned int Size) //Noise is +/- level (spread: 2x level, mean = 0, StAbw ~= 0.577 level)
	{
		#pragma omp parallel for
		for (unsigned int i = 0; i < Size; i++)
		{
			Array[i] = Array[i] + ((2*Drand()-1)*level);
		}
	}
	inline void ParAddWhiteNoise(float* Array, float level, unsigned int Size) //Noise is +/- level (spread: 2x level, mean = 0, StAbw ~= 0.577 level)
	{
		#pragma omp parallel for
		for (unsigned int i = 0; i < Size; i++)
		{
			Array[i] = Array[i] + ((2 * Drand() - 1)*level);
		}
	}

	//serialfunctions
	inline void MultiplyScalar(float* Array, float Factor, int Size)
	{
		for (int i = 0; i < Size; i++)
		{
			Array[i] *= Factor;
		}
	}
	inline void MultiplyScalar(double* Array, double Factor, int Size)
	{
		for (int i = 0; i < Size; i++)
		{
			Array[i] *= Factor;
		}
	}
	inline void Rotate(float* Vector, float RotationMatrix[9])
	{
		//V = M * q_local = { { V[0] * M[0] + V[1] * M[1] + V[2] * M[2] },{ V[0] * M[3] + V[1] * M[4] +... },{V[0] * M[6] ... } }
		float r[3];
		r[0] = Vector[0] * RotationMatrix[0] + Vector[1] * RotationMatrix[1] + Vector[2] * RotationMatrix[2];
		r[1] = Vector[0] * RotationMatrix[3] + Vector[1] * RotationMatrix[4] + Vector[2] * RotationMatrix[5];
		r[2] = Vector[0] * RotationMatrix[6] + Vector[1] * RotationMatrix[7] + Vector[2] * RotationMatrix[8];
		Vector[0] = r[0];
		Vector[1] = r[1];
		Vector[2] = r[2];
	}
	inline int Sum(int* Array, int Size)
	{//Annotation: parallelisation dosn't bring any measurable effect
		int sum = 0;
		for (int i = 0; i <Size; i++)
		{
			sum += Array[i];
		}
		return sum;
	}
	inline float Sum(float* Array, int Size)
	{
		double sum = 0.0;
		for (int i = 0; i <Size; i++)
		{
			sum += Array[i];
		}
		return (float)sum;
	}
	inline double Sum(double* Array, int Size)
	{
		double sum = 0;
		for (int i = 0; i <Size; i++)
		{
			sum += Array[i];
		}
		return sum;
	}

	



	//Functions for File and storage Stuff
	enum FileType{Binary=1, HDF5=2};
	void SafeArrayToFile(char* Filename, double* Array, unsigned int Size, FileType Type);
	void SafeArrayToFile(char* Filename, float* Array, unsigned int Size, FileType Type);
	void SafeArrayToFile(char* Filename, unsigned int* Array, unsigned int Size, FileType Type);

	void SafeArrayToFile(std::string Filename, double* Array, unsigned int Size, FileType Type);
	void SafeArrayToFile(std::string Filename, float* Array, unsigned int Size, FileType Type);
	void SafeArrayToFile(std::string Filename, unsigned int* Array, unsigned int Size, FileType Type);
	void SafeArrayToFile(std::string Filename, unsigned long * Array, unsigned int Size, FileType Type);



	
	template<typename T>
	void LoadArrayFromFile(char * Filename, T * Array, unsigned int Size)
	{
		std::ifstream FILE(Filename, std::ios::in | std::ofstream::binary);
		FILE.read(reinterpret_cast<char*>(Array), Size * sizeof(T));
		FILE.close();
	}

	template<typename T>
	void LoadArrayFromFile(std::string Filename, T * Array, unsigned int Size)
	{
		std::ifstream FILE(Filename, std::ios::in | std::ofstream::binary);
		FILE.read(reinterpret_cast<char*>(Array), Size * sizeof(T));
		FILE.close();
	}
}