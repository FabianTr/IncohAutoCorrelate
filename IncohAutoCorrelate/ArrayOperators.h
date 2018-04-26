#pragma once

#include <omp.h>

namespace ArrayOperators
{
	//inline void ParAdd(float* Array, float* Summand, int Size);

	inline void ParAdd(float* Array, float* Summand, int Size)
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
	inline void ThresholdValues(float* Array, float Threshold, int Size)
	{
		#pragma omp parallel for
		for (int i = 0; i < Size; i++)
		{
			if (Array[i] < Threshold)
				Array[i] = 0;
		}
	}


	//temporary exercise stuff

	void FunWithThreads();
}