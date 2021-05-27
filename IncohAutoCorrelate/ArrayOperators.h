#pragma once

#include <omp.h>
#include <fstream>
#include <string>
#include <math.h>
#include <random> 
#include "H5Cpp.h"

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

	template <typename T>
	inline void MultiplyScalar(T* Array, T Factor, size_t Size)
	{
		for (int i = 0; i < Size; i++)
		{
			Array[i] *= Factor;
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

	template <typename T0, typename T1>
	inline void MultiplyElementwise(T0* Array, T1* Factor, size_t Size)
	{
		for (unsigned int i = 0; i < Size; i++)
		{
			Array[i] = Array[i] * (T0)Factor[i];
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
	inline void RotateMatrix(float* Matrix, float RotationMatrix[9])
	{
		//V = M * q_local = { { V[0] * M[0] + V[1] * M[1] + V[2] * M[2] },{ V[0] * M[3] + V[1] * M[4] +... },{V[0] * M[6] ... } }
		float m[9];
		m[0] = RotationMatrix[0] * Matrix[0] + RotationMatrix[1] * Matrix[3] + RotationMatrix[2] * Matrix[6];
		m[1] = RotationMatrix[0] * Matrix[1] + RotationMatrix[1] * Matrix[4] + RotationMatrix[2] * Matrix[7];
		m[2] = RotationMatrix[0] * Matrix[2] + RotationMatrix[1] * Matrix[5] + RotationMatrix[2] * Matrix[8];

		m[3] = RotationMatrix[3] * Matrix[0] + RotationMatrix[4] * Matrix[3] + RotationMatrix[5] * Matrix[6];
		m[4] = RotationMatrix[3] * Matrix[1] + RotationMatrix[4] * Matrix[4] + RotationMatrix[5] * Matrix[7];
		m[5] = RotationMatrix[3] * Matrix[2] + RotationMatrix[4] * Matrix[5] + RotationMatrix[5] * Matrix[8];

		m[6] = RotationMatrix[6] * Matrix[0] + RotationMatrix[7] * Matrix[3] + RotationMatrix[8] * Matrix[6];
		m[7] = RotationMatrix[6] * Matrix[1] + RotationMatrix[7] * Matrix[4] + RotationMatrix[8] * Matrix[7];
		m[8] = RotationMatrix[6] * Matrix[2] + RotationMatrix[7] * Matrix[5] + RotationMatrix[8] * Matrix[8];

		for (unsigned int i = 0; i < 9; i++)
			Matrix[i] = m[i];
	}//

	template<typename T>
	inline T Sum(T* Array, size_t Size)
	{//Annotation: parallelisation dosn't bring any measurable effect
		T sum = (T)0.0;
		for (size_t i = 0; i <Size; i++)
		{
			sum += Array[i];
		}
		return sum;
	}

	template<typename T>
	inline double Mean(T* Array, size_t Size)
	{
		T sum = Sum(Array, Size);
		return (double(sum) / (double)Size);
	}

	template<typename T>
	inline double MaskedMean(T* Array,int* Mask , size_t Size)
	{
		T sum = 0;
		int maskSum = 0;
		for (size_t i = 0; i < Size; i++)
		{
			sum += Array[i] * Mask[i];
			maskSum += Mask[i];
		}
		return (double(sum) / (double)maskSum);
	}


	template <typename T>
	void KabschRotationMatrixRetrieval3x3(T * Input, T * Reference, T * RotationMatrix) 
	{
		// Returns rotation matrix, that rotates the Reference column vectors onto the Input column vectors.
		// The input and output, however, is row-major!! ( a(0,0), a(0,1), a(0,2), , a(1,0), ...  ) or in vectors: (V1_x , V2_x, V3_x, V1_y, V2_y, ...)
		//
		// Kabsch algorithm, see https://en.wikipedia.org/wiki/Kabsch_algorithm 

		Eigen::Matrix<T, 3, 3> M_in;
		M_in << Input[0], Input[1], Input[2], Input[3], Input[4], Input[5], Input[6], Input[7], Input[8];

		Eigen::Matrix<T, 3, 3> MReference;
		MReference << Reference[0], Reference[1], Reference[2], Reference[3], Reference[4], Reference[5], Reference[6], Reference[7], Reference[8];



		Eigen::JacobiSVD<Eigen::Matrix<T, 3, 3>> svd(MReference *  M_in.transpose(), Eigen::ComputeFullU | Eigen::ComputeFullV);

		T det;
		det = (svd.matrixV()*svd.matrixU().transpose()).determinant();

		Eigen::Matrix<T, 3, 3> Diag;
		Diag << 1, 0, 0, 0, 1, 0, 0, 0, det;

		Eigen::Matrix<T, 3, 3> Rot;
		Rot = svd.matrixV()*Diag*svd.matrixU().transpose();

		for (int k = 0; k < 3; k++)
		{
			for (int l = 0; l < 3; l++)
				RotationMatrix[l + 3 * k] = Rot(k, l);
		}


	}//




	//Functions for File and storage Stuff
	enum FileType{Binary=1, HDF5=2};

	template<typename T>
	void SafeArrayToFile(std::string Filename, T * Array, size_t Size, FileType Type = std::ofstream::binary)
	{
		switch (Type)
		{
		case Binary:
		{
			std::ofstream FILE(Filename, std::ios::out | std::ofstream::binary);
			FILE.write(reinterpret_cast<char*>(Array), Size * sizeof(T));
			FILE.flush();
			FILE.close();
			if (!FILE) {
				std::cerr << "ERROR: can't write to file: \"" << Filename << "\"\n";
				std::cerr << "    -> in ArrayOperators::SafeArrayToFile()\n";
				throw;
			}
		}
		break;
		case HDF5:
		{
			std::cerr << "ERROR: HDF5 storage is not implemented\n";
			std::cerr << "    -> in ArrayOperators::SafeArrayToFile()\n";
			throw;
		}
		break;
		}
	}


	template<typename T>
	void LoadArrayFromFile(std::string Filename, T * Array, unsigned int Size)
	{
		std::ifstream FILE(Filename, std::ios::in | std::ofstream::binary);
		FILE.read(reinterpret_cast<char*>(Array), Size * sizeof(T));
		FILE.close();
	}


	struct H5Infos
	{
		std::vector<int> Dimensions;
		//H5T_class_t DataType;

	};
	H5Infos GetH5FileInformation(std::string Filename, std::string Dataset);


	inline void H5Quicksave(float* Array, std::vector<hsize_t> ArrayShape, std::string Filename, std::string Dataset, int compression)
	{
		if (ArrayShape.size() == 0)
		{
			std::cerr << "ArrayShape is not defined!" << std::endl;
			throw;
		}

		H5::H5File file;
		if (std::ifstream(Filename).is_open())
		{
			file = H5::H5File(Filename, H5F_ACC_RDWR);
		}
		else
		{
			file = H5::H5File(Filename, H5F_ACC_TRUNC);
		}

		hsize_t* block = new hsize_t(ArrayShape.size());
		block[0] = 1;
		for (size_t i = 1; i < ArrayShape.size(); i++)
		{
			block[i] = ArrayShape[i];
		}
		H5::DSetCreatPropList plist = H5::DSetCreatPropList();
		plist.setChunk((int)ArrayShape.size(), block);
		plist.setDeflate(compression); //compression

		H5::DataSpace dataspace((int)ArrayShape.size(), ArrayShape.data());

		H5::DataSet dataset = file.createDataSet(Dataset, H5::PredType::NATIVE_FLOAT, dataspace, plist); // todo: think how to solve that with datatype
		dataset.write(Array, H5::PredType::NATIVE_FLOAT);
		file.close();

		delete[] block;
	}
}

