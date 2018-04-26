#include "Detector.h"
#include <math.h>
#include <limits>
#include <iostream>
#include <vector>
#include <omp.h>
#include <stdlib.h>
#include <array>
#include <string>
#include "H5Cpp.h"


#include "ArrayOperators.h"





Detector::Detector()
{
}


Detector::~Detector()
{
}

 


//Helpfunctions

//PixelMap
inline float Detector::GetPixelPos(int dimension, int i_x, int i_y)
{
	return PixelMap[dimension + 3 * i_x + 3 * DetectorSize[0] * i_y];
}
inline float Detector::GetkVal(int dimension, int i_x, int i_y)
{
	return kMap[dimension + 3 * i_x + 3 * DetectorSize[0] * i_y];
}
void Detector::Calc_PixelMapExtremeValues()
{
	PixelMapExtend.max_x = -9999999;
	PixelMapExtend.max_y = -9999999;
	PixelMapExtend.max_z = -9999999;

	PixelMapExtend.min_x = 9999999;
	PixelMapExtend.min_y = 9999999;
	PixelMapExtend.min_z = 9999999;
	for (int i = 0; i < DetectorSize[0] * DetectorSize[1]; i++)
	{
		int ind = i * 3;
		if (PixelMap[ind + 0] > PixelMapExtend.max_x)
			PixelMapExtend.max_x = PixelMap[ind + 0];
		if (PixelMap[ind + 0] < PixelMapExtend.min_x)
			PixelMapExtend.min_x = PixelMap[ind + 0];

		if (PixelMap[ind + 1] > PixelMapExtend.max_y)
			PixelMapExtend.max_y = PixelMap[ind + 1];
		if (PixelMap[ind + 1] < PixelMapExtend.min_y)
			PixelMapExtend.min_y = PixelMap[ind + 1];

		if (PixelMap[ind + 2] > PixelMapExtend.max_z)
			PixelMapExtend.max_z = PixelMap[ind + 2];
		if (PixelMap[ind + 2] < PixelMapExtend.min_z)
			PixelMapExtend.min_z = PixelMap[ind + 2];
	}
}

void Detector::GetSliceOutOfHDFCuboid(float* data, H5std_string Path, H5std_string DataSet, int SlicePosition)
{
	H5::H5File file(Path, H5F_ACC_RDONLY);
	H5::DataSet dataset = file.openDataSet(DataSet);

	if (dataset.getTypeClass() != H5T_FLOAT)
	{
		std::cerr << "ERROR: Intensity data is not stored as floating point numbers.\n";
		throw;
	}
	H5::DataSpace DS = dataset.getSpace();
	//std::cout << "Array shape: " << DS.getSimpleExtentNdims() << "\n";

	if (DS.getSimpleExtentNdims() != 3) //check if shape is [3][nx][ny] or [ny][nx][3]
	{
		std::cerr << "ERROR: Intensity data dimension is not 3 => shape is not (N, nx, ny)\n";
		throw;
	}
	hsize_t dims[3];
	DS.getSimpleExtentDims(dims, NULL);
//	std::cout << "Intensity data shape: " << dims[0] << " x " << dims[1] << " x " << dims[2] << "\n";

	if (dims[2] != DetectorSize[1] || dims[1] != DetectorSize[0])
	{
		std::cerr << "ERROR: Intensity size does not match pixle-map size.\n";
		throw;
	}
	//data = new float[dims[2] * dims[1]];

	//Get Subset 
	hsize_t offset[3], count[3], stride[3], block[3];
	hsize_t dimsm[3];

	offset[0] = (SlicePosition);
	offset[1] = 0;
	offset[2] = 0;

	count[0] = 1;
	count[1] = DetectorSize[0];
	count[2] = DetectorSize[1];

	block[0] = 1;
	block[1] = 1;
	block[2] = 1;

	stride[0] = 1;
	stride[1] = 1;
	stride[2] = 1;

	dimsm[0] = 1;
	dimsm[1] = DetectorSize[0];
	dimsm[2] = DetectorSize[1];

	H5::DataSpace mspace(3, dimsm, NULL);
	DS.selectHyperslab(H5S_SELECT_SET, count, offset, stride, block);

	dataset.read(data, H5::PredType::NATIVE_FLOAT, mspace, DS);


	//DetectorEvent->SerialNumber



	file.close();

}

//k-Map
void Detector::Calc_kMap()
{
	delete kMap;
	kMap = new float[3 * DetectorSize[0] * DetectorSize[1]];

	Max_k[0] = - std::numeric_limits<float>::max();
	Max_k[1] = - std::numeric_limits<float>::max();
	Max_k[2] = - std::numeric_limits<float>::max();
	Min_k[0] = std::numeric_limits<float>::max();
	Min_k[1] = std::numeric_limits<float>::max();
	Min_k[2] = std::numeric_limits<float>::max();

	for (int i_x = 0; i_x < DetectorSize[0]; i_x++)
	{
		for (int i_y = 0; i_y < DetectorSize[1]; i_y++)
		{
			float x, y, z, r;
			x = GetPixelPos(0, i_x, i_y);
			y = GetPixelPos(1, i_x, i_y);
			z = GetPixelPos(2, i_x, i_y);

			r = sqrtf(x*x + y*y + z*z);
			
			kMap[0 + 3 * i_x + 3 * DetectorSize[0] * i_y] = x / r;
			kMap[1 + 3 * i_x + 3 * DetectorSize[0] * i_y] = y / r;
			kMap[2 + 3 * i_x + 3 * DetectorSize[0] * i_y] = z / r;

			if (x / r > Max_k[0])
				Max_k[0] = x / r;
			if (y / r > Max_k[1])
				Max_k[1] = y / r;
			if (z / r > Max_k[2])
				Max_k[2] = z / r;

			if (x / r < Min_k[0])
				Min_k[0] = x / r;
			if (y / r < Min_k[1])
				Min_k[1] = y / r;
			if (z / r < Min_k[2])
				Min_k[2] = z / r;
			
		}
	}
	//Calculate maximal q-range for this detector
	Max_q[0] = fabsf(Max_k[0] - Min_k[0]);
	Max_q[1] = fabsf(Max_k[1] - Min_k[1]);
	Max_q[2] = fabsf(Max_k[2] - Min_k[2]);

}

void Detector::LoadPixelMap(H5std_string Path, H5std_string DataSet)
{


	H5::H5File file(Path, H5F_ACC_RDONLY);
	H5::DataSet dataset = file.openDataSet(DataSet);

	if (dataset.getTypeClass() != H5T_FLOAT)
	{
		std::cerr << "ERROR: PixelMap is not stored as floating point numbers.\n";
		throw;
	}
	std::cout <<"DataSize: " << dataset.getFloatType().getSize() << "\n";
	std::cout <<"float: "<< sizeof(float) << "\n";

	H5::DataSpace DS = dataset.getSpace();
	std::cout << "Array shape: " << DS.getSimpleExtentNdims() << "\n";

	if (DS.getSimpleExtentNdims() != 3) //check if shape is [3][nx][ny] or [ny][nx][3]
	{
		std::cerr << "ERROR: PixelMap dimension is not 3 => shape is not (3, nx, ny) or (ny, nx, 3)\n";
		throw;
	}
	hsize_t dims[3];
	DS.getSimpleExtentDims(dims, NULL);
	std::cout << "PixelMap shape: " << dims[0] << " x " << dims[1] << " x " << dims[2] << "\n";
	
	if (dims[0] != 3 && dims[2] != 3)
	{
		std::cerr << "ERROR: PixelMap first or third dimensions extend != 3  => shape is not (3, nx, ny) or (ny, nx, 3)\n";
		throw;
	}

	delete PixelMap;
	PixelMap = new float[dims[0] * (int)dims[1] * (int)dims[2]];


	if (dims[2] == 3)
	{
		DetectorSize[0] = (int)dims[1];//ss
		DetectorSize[1] = (int)dims[0];//fs

		H5::DataSpace mspace(3, dims);
		dataset.read(PixelMap, H5::PredType::NATIVE_FLOAT, mspace, DS);
	}
	else
	{
		DetectorSize[0] = (int)dims[1];//ss
		DetectorSize[1] = (int)dims[2];//fs

		float* TmpPixleMap = new float[dims[0] * dims[1] * dims[2]];
		H5::DataSpace mspace(3, dims);
		dataset.read(TmpPixleMap, H5::PredType::NATIVE_FLOAT, mspace, DS);

		for (int i_y = 0; i_y < dims[2]; i_y++)
		{
			for (int i_x = 0; i_x < dims[1]; i_x++)
			{
				for (int i_d = 0; i_d < 3; i_d++)
				{
					PixelMap[i_d + 3*i_x + 3*dims[1]*i_y] = TmpPixleMap[i_y + dims[2]*i_x + dims[2]*dims[1]*i_d];
				}
			}
		}
		delete[] TmpPixleMap;

	}
	file.close();
	//for (int t =0; t < 100; t++)
	//{
	//	std::cout << PixelMap[t]<< "\n";
	//}

	
}

void Detector::LoadAndAverageIntensity(std::vector<Settings::HitEvent> Events, float Threshold, int LowerBound, int UpperBound)
{
	if (Events.size() == 0)
	{
		std::cerr << "WARNING: Event list is empty, no intensity integration/averaging can be performed.\n";
		return;
	}

	Intensity = new float[DetectorSize[1] * DetectorSize[0]];
	{
		float* tmpIntensity = new float[DetectorSize[1] * DetectorSize[0]];
		//get first slide
		GetSliceOutOfHDFCuboid(tmpIntensity, Events[LowerBound].Filename, Events[LowerBound].Dataset, Events[LowerBound].Event);
		ArrayOperators::ThresholdValues(tmpIntensity, DetectorSize[1] * DetectorSize[0], Threshold); //treshold first slide

		Intensity = new float[DetectorSize[1] * DetectorSize[0]]();
		for (int i = LowerBound +1; i < UpperBound; i++)//get other slides
		{
			GetSliceOutOfHDFCuboid(tmpIntensity, Events[i].Filename, Events[i].Dataset, Events[i].Event);
			ArrayOperators::ParAdd(Intensity, tmpIntensity, DetectorSize[1] * DetectorSize[0], Threshold); //add with threshold

		}
		delete[] tmpIntensity;
	}
	ArrayOperators::ParMultiplyScalar(Intensity, 1.0 / (UpperBound - LowerBound ), DetectorSize[1] * DetectorSize[0]);
}
void Detector::LoadAndAverageIntensity(std::vector<Settings::HitEvent> Events, float Threshold) //Load Intensity of all Events and average them
{
	LoadAndAverageIntensity(Events, Threshold, 0, Events.size());
}
void Detector::LoadIntensityData(Settings::HitEvent * Event)//Load Intensity for one event
{
	DetectorEvent = Event;
	LoadIntensityData();

} 
void Detector::LoadIntensityData()
{
	H5std_string Path = DetectorEvent->Filename;
	H5std_string DataSet = DetectorEvent->Dataset;

	Intensity = new float[DetectorSize[1] * DetectorSize[0]];
	GetSliceOutOfHDFCuboid(Intensity, Path, DataSet, DetectorEvent->Event);
}

 

void Detector::CreateSparseHitList(float Threshold)
{
//	std::cout << "DEBUG: Start create SparseHitList\n";

	SparseHitList.clear();
	SparseHitList.reserve(1000);

	struct TmpSparseVecStr { std::vector<std::array<float,4>> Vec; };

	TmpSparseVecStr* TmpSparseVec = new TmpSparseVecStr[DetectorSize[1]];

	for (int i_y = 0; i_y < DetectorSize[1]; i_y++)
	{
	//	#pragma omp parallel for
		for (int i_x = 0; i_x < DetectorSize[0]; i_x++)
		{
			if (Intensity[i_x + DetectorSize[0]* i_y] >= Threshold)
			{
				std::array< float,4> TmpEntry;

				TmpEntry[0] = GetkVal(0, i_x, i_y);
				TmpEntry[1] = GetkVal(1, i_x, i_y);
				TmpEntry[2] = GetkVal(2, i_x, i_y);
				TmpEntry[3] = Intensity[i_x + DetectorSize[0] * i_y];

				TmpSparseVec[i_y].Vec.push_back(TmpEntry);
			}
		}
	}

	for (int i_y = 0; i_y < DetectorSize[1]; i_y++)
	{
		SparseHitList.insert(SparseHitList.end(), TmpSparseVec[i_y].Vec.begin(), TmpSparseVec[i_y].Vec.end());
	}

}
float Detector::CalculateMeanIntensity(bool FromSparse)
{
	double IntInt = 0;
	if (FromSparse)
	{
		for (int i = 0; i < SparseHitList.size(); i++)
		{
			IntInt += SparseHitList[i][3];
		}

	}
	else
	{
		for (int i = 0; i < DetectorSize[0]* DetectorSize[1]; i++)
		{
			IntInt += Intensity[i];
		}
	}
	IntInt = IntInt / (DetectorSize[0] * DetectorSize[1]);
	DetectorEvent->MeanIntensity = (float)IntInt;
	return (float)IntInt;
}

void Detector::InitializeDetector(H5std_string PixelMap_Path, H5std_string PixelMap_DataSet, float Pixel_Threshold)
{
	LoadPixelMap(PixelMap_Path, PixelMap_DataSet);
	Calc_kMap();
	CreateSparseHitList(Pixel_Threshold);
}

void Detector::AutoCorrelateSparseList(ACMesh & BigMesh,ACMesh &C_of_q, AutoCorrFlags Flags)
{
	//Implementation for CPU
	


}
