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

	inline void ParMultiplyScalar(float* Array, float Factor, int Size)
	{
		#pragma omp parallel for
		for (int i = 0; i < Size; i++)
		{
			Array[i] *= Factor;
		}
	}

}