#pragma once

#include <omp.h>

namespace ArrayOperators
{
	//inline void ParAdd(float* Array, float* Summand, int Size);

	//par functions
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



	//serialfunctions
	inline void MultiplyScalar(float* Array, float Factor, int Size)
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
		float sum = 0;
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
}