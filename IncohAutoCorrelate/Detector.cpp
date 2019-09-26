#include "Detector.h"
#include <math.h>
#include <limits>
#include <iostream>
#include <vector>
#include <omp.h>
#include <stdlib.h>
#include <array>
#include <string>
#include <algorithm>

#include "H5Cpp.h"

#include "ArrayOperators.h"
#include <thread>         
#include <chrono> 
#include "ProfileTime.h"




Detector::Detector()
{
	if (Intensity == nullptr)
		Intensity = new float[1];
	if (PixelMask == nullptr)
		PixelMask = new int[1];
	if (PixelMap == nullptr)
		PixelMap = new float[1];
	if (kMap == nullptr)
		kMap = new float[1];
}

Detector::Detector(const Detector &RefDet, bool DeepCopy) :Detector(RefDet)
{
	PixelMap = new float[DetectorSize[0] * DetectorSize[1] * 3];
	if (RefDet.Checklist.PixelMap)
	{
		for (unsigned int i = 0; i < DetectorSize[0] * DetectorSize[1] * 3; i++)
		{
			PixelMap[i] = RefDet.PixelMap[i];
		}
		//std::copy(RefDet.PixelMap, RefDet.PixelMap + DetectorSize[0] * DetectorSize[1] * 3, PixelMap);
	}

	kMap = new float[DetectorSize[0] * DetectorSize[1] * 3];
	if (RefDet.Checklist.KMap)
	{
		for (unsigned int i = 0; i < DetectorSize[0] * DetectorSize[1] * 3; i++)
		{
			kMap[i] = RefDet.kMap[i];
		}
		//std::copy(RefDet.kMap, RefDet.kMap + DetectorSize[0] * DetectorSize[1] * 3, kMap);
	}

	PixelMask = new int[DetectorSize[0] * DetectorSize[1]];
	if (RefDet.Checklist.PixelMask)
	{
		for (unsigned int i = 0; i < DetectorSize[0] * DetectorSize[1]; i++)
		{
			PixelMask[i] = RefDet.PixelMask[i];
		}
		//std::copy(RefDet.PixelMask, RefDet.PixelMask + DetectorSize[0] * DetectorSize[1], PixelMask);
	}

	Intensity = new float[DetectorSize[0] * DetectorSize[1]];
	if (RefDet.Checklist.Intensity)
	{
		for (unsigned int i = 0; i < DetectorSize[0] * DetectorSize[1]; i++)
		{
			Intensity[i] = RefDet.Intensity[i];
		}
		//std::copy(RefDet.Intensity, RefDet.Intensity + DetectorSize[0] * DetectorSize[1], Intensity);
	}
}

Detector::~Detector()
{
	delete[] Intensity;
	delete[] PixelMask;
	delete[] PixelMap;
	delete[] kMap;
}


//Helpfunctions


inline float Detector::DiscretizeToPhotones(float Value, float Threshold, float PhotonSamplingStep) //create single Photon counting by simple thresholding
{
	float val = Value;
	ArrayOperators::DiscretizeToPhotons(&val, Threshold, PhotonSamplingStep, 1);
	return val;
	//return ceilf((Value - Threshold) / PhotonSamplingStep)*(Value >= Threshold);
}

////PixelMap
//inline float Detector::GetPixelPos(int dimension, int i_x, int i_y)
//{
//	return PixelMap[dimension + 3 * i_x + 3 * DetectorSize[0] * i_y];
//}
//inline float Detector::GetkVal(int dimension, int i_x, int i_y)
//{
//	return kMap[dimension + 3 * i_x + 3 * DetectorSize[0] * i_y];
//}
void Detector::Calc_PixelMapExtremeValues()
{
	PixelMapExtend.max_x = -9999999;
	PixelMapExtend.max_y = -9999999;
	PixelMapExtend.max_z = -9999999;

	PixelMapExtend.min_x = 9999999;
	PixelMapExtend.min_y = 9999999;
	PixelMapExtend.min_z = 9999999;
	for (unsigned int i = 0; i < DetectorSize[0] * DetectorSize[1]; i++)
	{
		unsigned int ind = i * 3;
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
		std::cerr << "     -> in Detector::GetSliceOutOfHDFCuboid()\n";
		throw;
	}
	//auto TEST = dataset.getTypeClass();
	//std::cout << "DataSize: " << dataset.getFloatType().getSize() << "\n";

	H5::DataSpace DS = dataset.getSpace();

	//std::cout << "Array shape: " << DS.getSimpleExtentNdims() << "\n";

	if (DS.getSimpleExtentNdims() != 3) //check if shape is [nE][nx][ny] or [ny][nx][nE]  nE =^ Number of Slices(Events)
	{
		std::cerr << "ERROR: Intensity data dimension is not 3, but " << DS.getSimpleExtentNdims() << " => shape is not (N, nx, ny)\n";
		std::cerr << "     -> in Detector::GetSliceOutOfHDFCuboid()\n";
		throw;
	}
	hsize_t dims[3];
	DS.getSimpleExtentDims(dims, NULL);
	//	std::cout << "Intensity data shape: " << dims[0] << " x " << dims[1] << " x " << dims[2] << "\n";

	if (dims[1] != DetectorSize[0] || dims[2] != DetectorSize[1])
	{
		std::cerr << "ERROR: Intensity size does not match pixle-map size.\n";
		std::cerr << "     -> in Detector::GetSliceOutOfHDFCuboid()\n";
		throw;
	}

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

	H5::PredType type = H5::PredType::NATIVE_FLOAT;
	dataset.read(data, type, mspace, DS);


	//DetectorEvent->SerialNumber

	DS.close();
	dataset.close();
	mspace.close();

	//dataset.vlenReclaim(type.getId(), DS.getId(), mspace.getId(), data);

	file.close();

	Checklist.Intensity = true;
}

void Detector::LoadIntensityData_EPIX(float* data, H5std_string Path, H5std_string DataSet, int SlicePosition)
{
	std::cerr << "WARNING: old function, use GetSliceOutOfHDFCuboid() instead.\n";
	std::cerr << "    -> in LoadIntensityData_EPIX()\n";
	GetSliceOutOfHDFCuboid(data, Path, DataSet, SlicePosition);
	{
		//H5::H5File file(Path, H5F_ACC_RDONLY);
		//H5::DataSet dataset = file.openDataSet(DataSet);

		//if (dataset.getTypeClass() != H5T_FLOAT)
		//{
		//	std::cerr << "ERROR: Intensity data is not stored as floating point numbers.\n";
		//	std::cerr << "    -> in Detector::LoadIntensityData_EPIX()\n";
		//	throw;
		//}
		//H5::DataSpace DS = dataset.getSpace();
		////std::cout << "Array shape: " << DS.getSimpleExtentNdims() << "\n";

		//if (DS.getSimpleExtentNdims() != 3) //check if shape is [3][nx][ny] or [ny][nx][3]
		//{
		//	std::cerr << "ERROR: Intensity data dimension is not 3 => shape is not (N, nx, ny)\n";
		//	std::cerr << "    -> in Detector::LoadIntensityData_EPIX()\n";
		//	throw;
		//}
		//hsize_t dims[3];
		//DS.getSimpleExtentDims(dims, NULL);
		////	std::cout << "Intensity data shape: " << dims[0] << " x " << dims[1] << " x " << dims[2] << "\n";

		//if (dims[2] != DetectorSize[1] || dims[1] != DetectorSize[0])
		//{
		//	std::cerr << "ERROR: Intensity size does not match pixle-map size.\n";
		//	std::cerr << "    -> in Detector::LoadIntensityData_EPIX()\n";
		//	throw;
		//}

		////Get Subset 
		//hsize_t offset[3], count[3], stride[3], block[3];
		//hsize_t dimsm[3];

		//offset[0] = (SlicePosition);
		//offset[1] = 0;
		//offset[2] = 0;

		//count[0] = 1;
		//count[1] = DetectorSize[0];
		//count[2] = DetectorSize[1];

		//block[0] = 1;
		//block[1] = 1;
		//block[2] = 1;

		//stride[0] = 1;
		//stride[1] = 1;
		//stride[2] = 1;

		//dimsm[0] = 1;
		//dimsm[1] = DetectorSize[0];
		//dimsm[2] = DetectorSize[1];

		//H5::DataSpace mspace(3, dimsm, NULL);
		//DS.selectHyperslab(H5S_SELECT_SET, count, offset, stride, block);

		//dataset.read(data, H5::PredType::NATIVE_FLOAT, mspace, DS);


		////DetectorEvent->SerialNumber
		//DS.close();
		//dataset.close();
		//mspace.close();



		//file.close();
		//Checklist.Intensity = true;
	}
}

//k-Map
void Detector::Calc_kMap()
{
	delete[] kMap;
	kMap = new float[3 * DetectorSize[0] * DetectorSize[1]]();

	Max_k[0] = -100.0f;
	Max_k[1] = -100.0f;
	Max_k[2] = -100.0f;
	Min_k[0] = 100.0f;
	Min_k[1] = 100.0f;
	Min_k[2] = 100.0f;


	for (unsigned int i = 0; i < DetectorSize[0] * DetectorSize[1]; i++)
	{
		float x, y, z, r;
		x = PixelMap[3 * i + 0];
		y = PixelMap[3 * i + 1];
		z = PixelMap[3 * i + 2];

		r = sqrtf(x*x + y*y + z*z);

		x = x / r;
		y = y / r;
		z = z / r;

		kMap[3 * i + 0] = x;
		kMap[3 * i + 1] = y;
		kMap[3 * i + 2] = z;

		if (x > Max_k[0])
			Max_k[0] = x;
		if (y > Max_k[1])
			Max_k[1] = y;
		if (z > Max_k[2])
			Max_k[2] = z;

		if (x < Min_k[0])
			Min_k[0] = x;
		if (y < Min_k[1])
			Min_k[1] = y;
		if (z < Min_k[2])
			Min_k[2] = z;

	}

	Max_q[0] = fabsf(Max_k[0] - Min_k[0]);
	Max_q[1] = fabsf(Max_k[1] - Min_k[1]);
	Max_q[2] = fabsf(Max_k[2] - Min_k[2]);

	Checklist.KMap = true;
}

void Detector::CreateEmptyPixelMap(int FsSize, int SsSize)
{
	//creates empty PixelMap for run modes where an actual PixelMap is not required. 

	delete[] PixelMap;
	PixelMap = new float[FsSize * SsSize * 3]();

	DetectorSize[0] = SsSize;
	DetectorSize[1] = FsSize;

	Checklist.PixelMap = true;
}

void Detector::LoadPixelMap(H5std_string Path, H5std_string DataSet)
{
	bool swaped = true; //swap fs and ss
#pragma omp critical
	{
		H5::H5File file(Path, H5F_ACC_RDONLY);
		H5::DataSet dataset = file.openDataSet(DataSet);

		if (dataset.getTypeClass() != H5T_FLOAT)
		{
			std::cerr << "ERROR: PixelMap is not stored as floating point numbers.\n";
			throw;
		}
		std::cout << "Data Size : " << dataset.getFloatType().getSize() << "\n";
		std::cout << "float Size: " << sizeof(float) << "\n";

		//if (dataset.getFloatType().getSize() != sizeof(float))
		//{
		//	std::cerr << "ERROR: Pixelmap format is not float\n";
		//	std::cerr << "    -> in Detector::LoadPixelMap()\n";
		//	throw;
		//}

		H5::DataSpace DS = dataset.getSpace();
		std::cout << "Array shape: " << DS.getSimpleExtentNdims() << "\n";

		if (DS.getSimpleExtentNdims() != 3) //check if shape is [3][nx][ny] or [ny][nx][3]
		{
			std::cerr << "ERROR: PixelMap dimension is not 3 => shape is not (3, nx, ny) or (ny, nx, 3)\n";
			std::cerr << "    -> in Detector::LoadPixelMap()\n";
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

		delete[] PixelMap;
		PixelMap = new float[dims[0] * (int)dims[1] * (int)dims[2]];


		if (dims[2] == 3)//[ny][nx][3] //TODO: Check wether this is a possible and reasonable case
		{
			DetectorSize[0] = (int)dims[1];//ss
			DetectorSize[1] = (int)dims[0];//fs

			H5::DataSpace mspace(3, dims);
			dataset.read(PixelMap, H5::PredType::NATIVE_FLOAT, mspace, DS);


			float Pmax[3] = { -1e9, -1e9, -1e9 };
			float Pmin[3] = { 1e9, 1e9, 1e9 };

			for (unsigned int i_y = 0; i_y < dims[1]; i_y++)
			{
				for (unsigned int i_x = 0; i_x < dims[0]; i_x++)
				{
					for (unsigned int i_d = 0; i_d < 3; i_d++)
					{
						if (PixelMap[i_d + 3 * i_y * dims[0] + 3 * i_x] > Pmax[i_d])
							Pmax[i_d] = PixelMap[i_d + 3 * i_y * dims[0] + 3 * i_x];
						if (PixelMap[i_d + 3 * i_y * dims[0] + 3 * i_x] < Pmin[i_d])
							Pmin[i_d] = PixelMap[i_d + 3 * i_y * dims[0] + 3 * i_x];
					}
				}
			}

			std::cout << "PixMap_max = [" << Pmax[0] << "; " << Pmax[1] << "; " << Pmax[2] << "]\n";
			std::cout << "PixMap_min = [" << Pmin[0] << "; " << Pmin[1] << "; " << Pmin[2] << "]\n";

		}
		else// [3][nx][ny] //To be considerd as standeart
		{
			DetectorSize[0] = (int)dims[1];//ss   y'
			DetectorSize[1] = (int)dims[2];//fs   x'

			float* TmpPixleMap = new float[dims[0] * dims[1] * dims[2]];
			H5::DataSpace mspace(3, dims);
			dataset.read(TmpPixleMap, H5::PredType::NATIVE_FLOAT, mspace, DS);

			float Pmax[3] = { -1e9, -1e9, -1e9 };
			float Pmin[3] = { 1e9, 1e9, 1e9 };
			if (!swaped)//[ny][nx][3]
			{
				for (unsigned int i_y = 0; i_y < dims[2]; i_y++)
				{
					for (unsigned int i_x = 0; i_x < dims[1]; i_x++)
					{
						for (unsigned int i_d = 0; i_d < 3; i_d++)
						{
							PixelMap[i_d + 3 * i_x + 3 * dims[1] * i_y] = TmpPixleMap[i_y + dims[2] * i_x + dims[2] * dims[1] * i_d];
							if (TmpPixleMap[i_y + dims[2] * i_x + dims[2] * dims[1] * i_d] > Pmax[i_d])
								Pmax[i_d] = TmpPixleMap[i_y + dims[2] * i_x + dims[2] * dims[1] * i_d];
							if (TmpPixleMap[i_y + dims[2] * i_x + dims[2] * dims[1] * i_d] < Pmin[i_d])
								Pmin[i_d] = TmpPixleMap[i_y + dims[2] * i_x + dims[2] * dims[1] * i_d];
						}
					}
				}
			}
			else//[nx][ny][3]
			{
				for (unsigned int i_x = 0; i_x < dims[1]; i_x++)
				{
					for (unsigned int i_y = 0; i_y < dims[2]; i_y++)
					{
						for (unsigned int i_d = 0; i_d < 3; i_d++)
						{
							//	PixelMap[i_d + 3 * i_x + 3 * dims[1] * i_y] = TmpPixleMap[i_y + dims[2] * i_x + dims[2] * dims[1] * i_d];


							PixelMap[i_d + 3 * i_y + 3 * dims[2] * i_x] = TmpPixleMap[i_y + i_x * dims[2] + i_d * dims[2] * dims[1]];

							if (TmpPixleMap[i_y + dims[2] * i_x + dims[2] * dims[1] * i_d] > Pmax[i_d])
								Pmax[i_d] = TmpPixleMap[i_y + dims[2] * i_x + dims[2] * dims[1] * i_d];
							if (TmpPixleMap[i_y + dims[2] * i_x + dims[2] * dims[1] * i_d] < Pmin[i_d])
								Pmin[i_d] = TmpPixleMap[i_y + dims[2] * i_x + dims[2] * dims[1] * i_d];
						}
					}
				}
			}

			std::cout << "PixMap_max = [" << Pmax[0] << "; " << Pmax[1] << "; " << Pmax[2] << "]\n";
			std::cout << "PixMap_min = [" << Pmin[0] << "; " << Pmin[1] << "; " << Pmin[2] << "]\n";

			delete[] TmpPixleMap;

		}
		Checklist.PixelMap = true;
		file.close();

	}
}

void Detector::LoadPixelMask(std::string Path, std::string DataSet )
{
	delete[] PixelMask;
	PixelMask = new int[DetectorSize[0] * DetectorSize[1]]();
	if (Path != "")
	{
		if (DataSet != "")
		{
			std::cout << "Load pixel - mask from H5-file\n";

			H5::H5File file(Path, H5F_ACC_RDONLY);
			H5::DataSet dataset = file.openDataSet(DataSet);


			if (dataset.getTypeClass() != H5T_INTEGER)
			{
				std::cerr << "ERROR: Pixel-mask is not stored as integer array.\n";
				std::cerr << "     -> in Detector::LoadPixelMask()\n";
				throw;
			}

			H5::DataSpace DS = dataset.getSpace();
			if (DS.getSimpleExtentNdims() != 2) //check if shape is [nE][nx][ny] or [ny][nx][nE]  nE =^ Number of Slices(Events)
			{
				std::cerr << "ERROR: Pixel-mask data dimension is not 2, but " << DS.getSimpleExtentNdims() << " => shape is not (nx, ny)\n";
				std::cerr << "     -> in Detector::LoadPixelMask()\n";
				throw;
			}
			hsize_t dims[2];
			DS.getSimpleExtentDims(dims, NULL);
			//	std::cout << "Intensity data shape: " << dims[0] << " x " << dims[1] << " x " << dims[2] << "\n";

			if (dims[0] != DetectorSize[0] || dims[1] != DetectorSize[1])
			{
				std::cerr << "ERROR: Intensity size does not match pixle-map size.\n";
				std::cerr << "     -> in Detector::LoadPixelMask()\n";
				throw;
			}

			//Get Subset 
			hsize_t offset[2], count[2], stride[2], block[2];
			hsize_t dimsm[2];

			offset[0] = 0;
			offset[1] = 0;

			count[0] = DetectorSize[0];
			count[1] = DetectorSize[1];

			block[0] = 1;
			block[1] = 1;

			stride[0] = 1;
			stride[1] = 1;

			dimsm[0] = DetectorSize[0];
			dimsm[1] = DetectorSize[1];


			H5::DataSpace mspace(2, dimsm, NULL);
			DS.selectHyperslab(H5S_SELECT_SET, count, offset, stride, block);

			H5::PredType type = H5::PredType::NATIVE_INT;
			dataset.read(PixelMask, type, mspace, DS);

			//DetectorEvent->SerialNumber

			DS.close();
			dataset.close();
			mspace.close();



			//dataset.vlenReclaim(type.getId(), DS.getId(), mspace.getId(), data);



			file.close();
		}
		else
		{
			std::cout << "Load pixel - mask from binary-file\n";
			ArrayOperators::LoadArrayFromFile<int>(Path, PixelMask, DetectorSize[0] * DetectorSize[1]);
		}
	}
	else
	{
		// No Pixelmask given => set every entry to one
#pragma omp parallel for
		for (unsigned int i = 0; i < DetectorSize[0] * DetectorSize[1]; i++)
		{
			PixelMask[i] = 1;
		}
	}

	//Check Pixelmask
	bool BadPixelMask = false;
	#pragma omp parallel for 
	for (unsigned int i = 0; i < DetectorSize[0] * DetectorSize[1]; i++)
	{
		if(PixelMask[i] != 0 && PixelMask[i] != 1 )
		{
			BadPixelMask = true;//race conditions are irrelevant here
		}
	}

	if (BadPixelMask)
	{
		std::cerr << "\nWARNING:" << std::endl;
		std::cerr << "PixelMask contains values different than 0 and 1." << std::endl;
		std::cerr << "This could cause manifold malfunctions!\n" << std::endl;
	}

	Checklist.PixelMask = true;
}

void Detector::GenerateFlatOnesPixelMask()
{
	delete[] PixelMask;
	PixelMask = new int[DetectorSize[0] * DetectorSize[1]];
	for (unsigned int i = 0; i < DetectorSize[0] * DetectorSize[1]; i++)
	{
		PixelMask[i] = 1;
	}
	Checklist.PixelMask = true;
}

void Detector::ApplyPixelMask()
{
	ArrayOperators::ParMultiplyElementwise(Intensity, PixelMask, DetectorSize[0] * DetectorSize[1]);
}

void Detector::LoadAndAverageIntensity(std::vector<Settings::HitEvent>& Events, float Threshold, int LowerBound, int UpperBound)
{
	LoadAndAverageIntensity(Events, Threshold, -1.0f, LowerBound, UpperBound);
}
void Detector::LoadAndAverageIntensity(std::vector<Settings::HitEvent>& Events, float Threshold) //Load Intensity of all Events and average them
{
	LoadAndAverageIntensity(Events, Threshold, 0, Events.size());
}
void Detector::LoadAndAverageIntensity(std::vector<Settings::HitEvent>& Events, float Threshold, float PhotonSamplingStep)
{
	LoadAndAverageIntensity(Events, Threshold, PhotonSamplingStep, 0, Events.size());
}
void Detector::LoadAndAverageIntensity(std::vector<Settings::HitEvent>& Events, float Threshold, float PhotonSamplingStep, bool Pixelmask)
{
	LoadAndAverageIntensity(Events, Threshold, PhotonSamplingStep, 0, Events.size(), Pixelmask);
}
void Detector::LoadAndAverageIntensity(std::vector<Settings::HitEvent>& Events, float Threshold, float PhotonSamplingStep, int LowerBound, int UpperBound)
{
	LoadAndAverageIntensity(Events, Threshold, PhotonSamplingStep, LowerBound, UpperBound, false);
}
void Detector::LoadAndAverageIntensity(std::vector<Settings::HitEvent>& Events, float Threshold, float PhotonSamplingStep, int LowerBound, int UpperBound, bool Pixelmask)
{
	if (Events.size() == 0)
	{
		std::cerr << "WARNING: Event list is empty, no intensity integration/averaging can be performed.\n";
		return;
	}

	if (!Checklist.PixelMask)
	{
		GenerateFlatOnesPixelMask();
	}


	long* IntensityPhotonDiscr = NULL;
	delete[] Intensity;
	Intensity = new float[DetectorSize[1] * DetectorSize[0]]();
	{
		float* tmpIntensity = new float[DetectorSize[1] * DetectorSize[0]]();
		if (PhotonSamplingStep > 0)
			IntensityPhotonDiscr = new long[DetectorSize[1] * DetectorSize[0]]();

		//std::cout << " \n ";
		int t_prog = -1; //for progress indicator only
		for (int i = LowerBound; i < UpperBound; i++)//get  slides
		{
			if (UpperBound - LowerBound > 1)
			{
				if ((((i - LowerBound) * 100) / (UpperBound - LowerBound - 1)) > t_prog)
				{
					std::cout << "Load and Average: " << ((i - LowerBound) * 100) / (UpperBound - LowerBound - 1) << "%" << std::endl;
					t_prog++;
				}
			}

			GetSliceOutOfHDFCuboid(tmpIntensity, Events[i].Filename, Events[i].Dataset, Events[i].Event);


			if (Checklist.PixelMask)
			{
				//TODO IMPLEMENT Checklist for PixelMask and a LoadPixelmask() method
				ArrayOperators::ParMultiplyElementwise(tmpIntensity, PixelMask, DetectorSize[0] * DetectorSize[1]);
			}

			if (PhotonSamplingStep <= 0)// No Photon discretising
			{
				ArrayOperators::ParAdd(Intensity, tmpIntensity, DetectorSize[1] * DetectorSize[0], Threshold); //add with threshold

				//update Event
				Events[i].MeanIntensity = ArrayOperators::Sum(tmpIntensity, DetectorSize[1] * DetectorSize[0]) / ((float)(DetectorSize[1] * DetectorSize[0]));

			}
			else// Photon discretising
			{
				#pragma omp parallel for
				for (unsigned int j = 0; j < DetectorSize[1] * DetectorSize[0]; j++)
				{
					if (tmpIntensity[j] >= Threshold)
					{
						tmpIntensity[j] = DiscretizeToPhotones(tmpIntensity[j], Threshold, PhotonSamplingStep);
						IntensityPhotonDiscr[j] += (long)floor(tmpIntensity[j]+0.5);
					}
					else
					{
						tmpIntensity[j] = 0;
					}
				}

				//update Event
				Events[i].PhotonCount = ArrayOperators::Sum(tmpIntensity, DetectorSize[1] * DetectorSize[0]);
				Events[i].MeanIntensity = ArrayOperators::Sum(tmpIntensity, DetectorSize[1] * DetectorSize[0]) / ((float)(DetectorSize[1] * DetectorSize[0]));
			}
		}
		delete[] tmpIntensity;
	}
	if (PhotonSamplingStep > 0)
	{
#pragma omp parallel for
		for (unsigned int i = 0; i < DetectorSize[1] * DetectorSize[0]; i++)
		{
			Intensity[i] = (float)IntensityPhotonDiscr[i];
		}
	}
	ArrayOperators::ParMultiplyScalar(Intensity, 1.0f / ((float)(UpperBound - LowerBound)), DetectorSize[1] * DetectorSize[0]);
	Checklist.Intensity = true;
}


void Detector::LoadIntensityData(Settings::HitEvent * Event)//Load Intensity for one event
{
	DetectorEvent = Event;
	Checklist.Event = true;
	LoadIntensityData();
}
void Detector::LoadIntensityData()
{
	if (!Checklist.Event)
	{
		std::cerr << "ERROR: no Event set.\n ";
		std::cerr << "   ->: Detector::LoadIntensityData()\n ";
		throw;
	}
	H5std_string Path = DetectorEvent->Filename;
	H5std_string DataSet = DetectorEvent->Dataset;


	//	std::cerr << Intensity  << " " << *(reinterpret_cast<size_t*>(Intensity-2)) << "  "<< 4 * DetectorSize[1] * DetectorSize[0]<< std::endl;
		//if (Intensity != NULL)
	delete[] Intensity;

	Intensity = new float[DetectorSize[1] * DetectorSize[0]]();

	GetSliceOutOfHDFCuboid(Intensity, Path, DataSet, DetectorEvent->Event);


	////Debug stuff
	//for (int i = 0; i < DetectorSize[1] * DetectorSize[0]; i++)
	//{
	//	if (Intensity[i] < 0)
	//	{
	//		std::cout << "ERROR: I<0 : " << Intensity[i] << "  WTF\n";
	//	}
	//}

	//

}

void Detector::LoadIntensityData_PSANA_StyleJungfr(H5std_string Path, H5std_string DataSet, unsigned int Index)
{
	if (Intensity != nullptr)
	{
		delete[] Intensity;
		Intensity = nullptr;
	}

	H5::H5File file(Path, H5F_ACC_RDONLY);
	H5::DataSet dataset = file.openDataSet(DataSet);

	if (dataset.getTypeClass() != H5T_FLOAT)
	{
		std::cerr << "ERROR: Intensity data is not stored as floating point numbers.\n";
		std::cerr << "    -> Detector::LoadIntensityData_PSANA_Style()";
		throw;
	}


	H5::DataSpace DS = dataset.getSpace();
	//std::cout << "Array shape: " << DS.getSimpleExtentNdims() << "\n";

	if (DS.getSimpleExtentNdims() != 4) //check if shape is [N][2][nx][ny] or [ny][nx][3]
	{
		std::cerr << "ERROR: Intensity data dimension is not 4 => shape is not (N, 2, nx, ny)\n";
		std::cerr << "    -> Detector::LoadIntensityData_PSANA_Style()";
		throw;
	}

	//Temporary Det Parts
	float * SubDet1 = new float[(DetectorSize[0] / 2) * DetectorSize[1]]();
	float * SubDet2 = new float[(DetectorSize[0] / 2) * DetectorSize[1]]();

	//Get Subset 1
	hsize_t offset[4], count[4], stride[4], block[4];
	hsize_t dimsm[4];

	offset[0] = (Index);
	offset[1] = 0;
	offset[2] = 0;
	offset[3] = 0;

	count[0] = 1;
	count[1] = 1;
	count[2] = DetectorSize[0] / 2;
	count[3] = DetectorSize[1];

	block[0] = 1;
	block[1] = 1;
	block[2] = 1;
	block[3] = 1;

	stride[0] = 1;
	stride[1] = 1;
	stride[2] = 1;
	stride[3] = 1;

	dimsm[0] = 1;
	dimsm[1] = 1;
	dimsm[2] = DetectorSize[0] / 2;
	dimsm[3] = DetectorSize[1];

	H5::DataSpace mspace(4, dimsm, NULL);
	DS.selectHyperslab(H5S_SELECT_SET, count, offset, stride, block);
	dataset.read(SubDet1, H5::PredType::NATIVE_FLOAT, mspace, DS);

	offset[1] = 1;
	DS.selectHyperslab(H5S_SELECT_SET, count, offset, stride, block);
	dataset.read(SubDet2, H5::PredType::NATIVE_FLOAT, mspace, DS);


	DS.close();
	dataset.close();
	mspace.close();

	file.close();


	Intensity = new float[DetectorSize[0] * DetectorSize[1]]();

	for (unsigned int i = 0; i < (DetectorSize[0] / 2) * DetectorSize[1]; i++)
	{
		Intensity[i] = (float)SubDet1[i];
		Intensity[i + ((DetectorSize[0] / 2) * DetectorSize[1])] = (float)SubDet2[i];


	}

	delete[] SubDet1;
	delete[] SubDet2;
	Checklist.Intensity = true;
}

void Detector::CreateSparseHitList(float Threshold)
{
	SparseHitList.clear();
	SparseHitList.reserve(1000);

	for (unsigned int i = 0; i < DetectorSize[0] * DetectorSize[1]; i++)
	{
		float I = Intensity[i];

		if (I >= Threshold && I > 0)
		{
			std::array< float, 4> TmpEntry;

			TmpEntry[0] = kMap[3 * i + 0];
			TmpEntry[1] = kMap[3 * i + 1];
			TmpEntry[2] = kMap[3 * i + 2];
			TmpEntry[3] = I;
			SparseHitList.push_back(TmpEntry);
		}
	}


	Checklist.SparseHitList = true;
}
void Detector::CreateSparseHitList(float Threshold, float PhotonSamplingStep)
{
	CreateSparseHitList(Threshold);
#pragma omp parallel for
	for (unsigned int i = 0; i < SparseHitList.size(); i++)
	{
		SparseHitList[i][3] = DiscretizeToPhotones(SparseHitList[i][3], Threshold, PhotonSamplingStep);
	}
}
void Detector::CreateSparseHitList(float Threshold, float PhotonSamplingStep, bool Par)
{
	CreateSparseHitList(Threshold);
	if (Par)
	{
#pragma omp parallel for
		for (unsigned int i = 0; i < SparseHitList.size(); i++)
		{
			SparseHitList[i][3] = DiscretizeToPhotones(SparseHitList[i][3], Threshold, PhotonSamplingStep);
		}
	}
	else
	{
		for (unsigned int i = 0; i < SparseHitList.size(); i++)
		{
			SparseHitList[i][3] = DiscretizeToPhotones(SparseHitList[i][3], Threshold, PhotonSamplingStep);
		}
	}
}


float Detector::CalculateMeanIntensity(bool FromSparse)
{
	double IntInt = 0;
	if (FromSparse)
	{
		for (unsigned int i = 0; i < SparseHitList.size(); i++)
		{
			IntInt += SparseHitList[i][3];
		}

	}
	else
	{
		for (unsigned int i = 0; i < DetectorSize[0] * DetectorSize[1]; i++)
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

int Detector::AutoCorrelateSparseList(ACMesh & BigMesh, AutoCorrFlags FlagsFirstMap, AutoCorrFlags FlagsSecondMap, bool DoubleMapping, Settings & Options, int CpuGpu) //0: CPU; 1: GPU; -1: Auto 
{
	if (!Checklist.SparseHitList)
	{
		std::cerr << "ERROR: no sparse hit list available. Use Detector::CreateSparseHitList() first.\n ";
		std::cerr << "   ->: Detector::AutoCorrelateSparseList()\n ";
		throw;
	}
	if (!Checklist.Event)
	{
		std::cerr << "ERROR: no Event set.\n ";
		std::cerr << "   ->: Detector::AutoCorrelateSparseList()\n ";
		throw;
	}

	float SHLsizeQuot = ((float)SparseHitList.size()) / ((float)(DetectorSize[0] * DetectorSize[1]));
	if ( ( (SHLsizeQuot < 0.01f) && (CpuGpu == -1) ) || CpuGpu == 0) // (switch for SparseHitList.size / DetSize > p(0.01) [old p(0.0075)])
	{ //Implementation for CPU
		size_t sz_t = SparseHitList.size();

#pragma omp parallel for
		for (unsigned int i = 0; i < SparseHitList.size(); i++)
		{
			for (unsigned int j = i; j < SparseHitList.size(); j++)
			{
				float q[3];
				//float RM[9] = { 1,0,0,0,1,0,0,0,1 };//TODO IMPLEMENT ROTATION MATRIX -> THIS IS A DUMMY
				q[0] = SparseHitList[i][0] - SparseHitList[j][0];
				q[1] = SparseHitList[i][1] - SparseHitList[j][1];
				q[2] = SparseHitList[i][2] - SparseHitList[j][2];
				BigMesh.Atomic_Add_q_Entry(q, DetectorEvent->RotMatrix, SparseHitList[i][3] * SparseHitList[j][3], FlagsFirstMap.InterpolationMode, FlagsSecondMap.InterpolationMode, DoubleMapping); // DetectorEvent->RotMatrix
				//std::cout << SparseHitList[i][3] * SparseHitList[j][3] << ", ";
				q[0] = SparseHitList[j][0] - SparseHitList[i][0];
				q[1] = SparseHitList[j][1] - SparseHitList[i][1];
				q[2] = SparseHitList[j][2] - SparseHitList[i][2];
				if(i != j)
					BigMesh.Atomic_Add_q_Entry(q, DetectorEvent->RotMatrix, SparseHitList[i][3] * SparseHitList[j][3], FlagsFirstMap.InterpolationMode, FlagsSecondMap.InterpolationMode, DoubleMapping); // DetectorEvent->RotMatrix
			}
		}

		return 0; //For statistics: CPU -> 0
	}
	else
	{ //Implementation for GPU 

		double Multiplicator = (double)(1.0/Options.F_I_Conversion.Step); //1 should be sufficient for photon discretised values (only integer possible) for NN ???CHECK???
		if (Multiplicator > 1)
			Multiplicator = round(Multiplicator);

		//set Parameter
		double Params[9];
		Params[0] = SparseHitList.size();

		Params[1] = BigMesh.Shape.Voxel_per_dq; //Voxel per dq
		Params[2] = BigMesh.Shape.Size_AB; // Size perp
		Params[3] = BigMesh.Shape.Size_C; // Size C

		Params[4] = BigMesh.Shape.Max_Q;

		Params[5] = (double)DoubleMapping;

		Params[6] = Multiplicator; //Multiplicator for conversion to long
		Params[7] = FlagsFirstMap.InterpolationMode; //Not implementet, only nearest neighbours
		Params[8] = FlagsSecondMap.InterpolationMode;
								   //reserve OpenCL Device
		int OpenCLDeviceNumber = -1;
		cl_int err;

		while ((OpenCLDeviceNumber = Options.OCL_ReserveDevice()) == -1)
		{
			std::this_thread::sleep_for(std::chrono::microseconds(Options.ThreadSleepForOCLDev));
		}

		//obtain Device
		cl::Device CL_Device = Options.CL_devices[OpenCLDeviceNumber];

		//Setup Queue
		cl::CommandQueue queue(Options.CL_context, CL_Device, 0, &err);
		Options.checkErr(err, "Setup CommandQueue in Detector::AutoCorrelateSparseList() ");
		cl::Event cl_event;

		//profiler stuff
		ProfileTime Profiler;

		//Buffers
		//Output
		uint64_t * TempBigMesh; //Half Mesh
		TempBigMesh = new uint64_t[BigMesh.Shape.Size_AB * BigMesh.Shape.Size_AB *  ( (BigMesh.Shape.Size_AB + 1)/2)]();

		size_t ACsize = sizeof(uint64_t) * (BigMesh.Shape.Size_AB * BigMesh.Shape.Size_AB * ( (BigMesh.Shape.Size_AB + 1)/2));
		cl::Buffer CL_AC(Options.CL_context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, ACsize, TempBigMesh, &err);



		//Input:
		size_t SparseListSize = sizeof(float) * 4 * SparseHitList.size();

		cl::Buffer CL_SparseList(Options.CL_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, SparseListSize, (void*)SparseHitList.data(), &err);

		float RM[9];
		for (int i = 0; i < 9; i++)
			RM[i] = DetectorEvent->RotMatrix[i];


		cl::Buffer CL_RotM(Options.CL_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(RM), &RM, &err);
		cl::Buffer CL_Params(Options.CL_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(Params), &Params, &err);

		//Setup Kernel
		cl::Kernel kernel(Options.CL_Program, "Autocor_sparseHL", &err);
		Options.checkErr(err, "Setup AutoCorr_CQ in Detector::AutoCorrelateSparseList() ");

		//Set Arguments
		kernel.setArg(0, CL_SparseList);
		kernel.setArg(1, CL_Params);
		kernel.setArg(2, CL_RotM);
		kernel.setArg(3, CL_AC);
		const size_t &global_size = SparseHitList.size();

		//launch Kernel
		

		//Profiler.Tic();

		err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(global_size), cl::NullRange, NULL, &cl_event);

		//	Options.checkErr(err, "Launch Kernel in Detector::AutoCorrelateSparseList() ");
		cl_event.wait();
		//Profiler.Toc(true);

		err = queue.enqueueReadBuffer(CL_AC, CL_TRUE, 0, ACsize, TempBigMesh);
		Options.checkErr(err, "OpenCL kernel, launched in Detector::AutoCorrelateSparseList() ");



		//Free Device
		Options.OCL_FreeDevice(OpenCLDeviceNumber);

		//Point reflection and add to Bigmesh
		unsigned int MeshCenter = (BigMesh.Shape.Size_AB - 1) / 2; 
		#pragma omp parallel for
		for (unsigned int ss  = 0; ss < (BigMesh.Shape.Size_AB + 1) / 2; ss ++)
		{
			for (unsigned int ms = 0; ms < BigMesh.Shape.Size_AB; ms++)
			{
				for (unsigned int fs = 0; fs < BigMesh.Shape.Size_AB; fs++)
				{
					double t_val = (double)TempBigMesh[fs + ms * BigMesh.Shape.Size_AB + ss * BigMesh.Shape.Size_AB * BigMesh.Shape.Size_AB] / (double)Multiplicator;
					unsigned long val;
					val = (unsigned long)Options.FloatToInt(t_val);

					//add first
					BigMesh.Mesh[fs + ms * BigMesh.Shape.Size_AB + (ss + MeshCenter) * BigMesh.Shape.Size_AB * BigMesh.Shape.Size_AB] += val;

					//mirrow
					if (ss != 0)
					{
						BigMesh.Mesh[(2 * MeshCenter - fs) + (2 * MeshCenter - ms) * BigMesh.Shape.Size_AB + (MeshCenter - ss) * BigMesh.Shape.Size_AB * BigMesh.Shape.Size_AB] += val;
					}
				}
			}
		}

		//Check zero:



		//clean up
		delete[] TempBigMesh; 

		return 1; //For statistics: GPU -> 1
	}
}


void Detector::AutoCorrelate_CofQ_SmallMesh(ACMesh & SmallMesh, AutoCorrFlags Flags, Settings & Options)
{
	if (!SmallMesh.Checklist.CofQSmallMesh) //check if it is the right Mesh
	{
		std::cerr << "ERROR: SmallMesh argument is not a CofQ-Mesh\n";
		std::cerr << "   ->  in Detector::AutoCorrelate_CofQ_SMallMesh()\n";
		throw;
	}


	//reserve OpenCL Device
	int OpenCLDeviceNumber = -1;
	cl_int err;

	while ((OpenCLDeviceNumber = Options.OCL_ReserveDevice()) == -1)
	{
		std::this_thread::sleep_for(std::chrono::microseconds(Options.ThreadSleepForOCLDev));
	}

	//obtain Device
	cl::Device CL_Device = Options.CL_devices[OpenCLDeviceNumber];


	//Setup temporary Mesh (uint64_t*)

	uint64_t * TempMesh = new uint64_t[SmallMesh.Shape.Size_AB * SmallMesh.Shape.Size_AB *  SmallMesh.Shape.Size_C]();

	float Min_I = 0, Max_I = 0, Mean_I = 0;
	ArrayOperators::Min_Max_Mean_Value(Intensity, DetectorSize[0] * DetectorSize[1], Min_I, Max_I, Mean_I);


	if (Mean_I <= 0)
	{
		std::cerr << "ERROR: No intensity > 0 found in averaged intensity file, check if the file exist and if it is not corrupted\n";
		std::cerr << "   -> in Detector::AutoCorrelate_CofQ_SmallMesh()\n";
		throw;
	}

	double Multiplicator = 1e-10;
	for (; 1 > Mean_I*Mean_I * Multiplicator; )
	{
		Multiplicator *= 10;
	}
	Multiplicator = Multiplicator * 100000;

	//set Parameter
	double Params[10];
	Params[0] = DetectorSize[0] * DetectorSize[1]; //Numer of pixels (size[0]*size[1])
	Params[1] = SmallMesh.Shape.Voxel_per_dq; // Voxel per dq
	Params[2] = SmallMesh.Shape.Size_AB; // Size perp
	Params[3] = SmallMesh.Shape.Size_C; // Size C
	Params[4] = SmallMesh.Shape.k_A; // Dimension Alignment
	Params[5] = SmallMesh.Shape.k_B; // Dimension Alignment 
	Params[6] = SmallMesh.Shape.k_C; // Dimension Alignment 
	Params[7] = Flags.InterpolationMode;
	Params[8] = SmallMesh.Shape.Max_Q; // for q-zoom
	Params[9] = Multiplicator; //Multiplicator for conversion to long

	//DEBUG BULLSHIT
	std::cout << "Params:\n";
	for (int i = 0; i < 10; i++)
	{
		std::cout << Params[i] << "\n";
	}


	//Setup Queue
	cl::CommandQueue queue(Options.CL_context, CL_Device, 0, &err);
	Options.checkErr(err, "Setup CommandQueue in Detector::AutoCorrelate_CofQ_SmallMesh() ");
	cl::Event cl_event;



	//profiler stuff
	ProfileTime Profiler;
	//
	size_t ACsize = sizeof(uint64_t) * SmallMesh.Shape.Size_AB * SmallMesh.Shape.Size_AB * SmallMesh.Shape.Size_C;

	cl::Buffer CL_CQ_Small(Options.CL_context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, ACsize, TempMesh, &err);
	//Input:
	size_t Intsize = sizeof(float) * DetectorSize[0] * DetectorSize[1];
	cl::Buffer CL_Intensity(Options.CL_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Intsize, Intensity, &err);
	size_t KMapsize = sizeof(float) * 3 * DetectorSize[0] * DetectorSize[1];
	cl::Buffer CL_kMap(Options.CL_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, KMapsize, kMap, &err);
	cl::Buffer CL_Params(Options.CL_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(Params), &Params, &err);


	//Setup Kernel
	cl::Kernel kernel(Options.CL_Program, "AutoCorr_CQ_small", &err);
	Options.checkErr(err, "Setup AutoCorr_CQ in Detector::AutoCorrelate_CofQ_SmallMesh() ");

	//Set Arguments
	kernel.setArg(0, CL_Intensity);
	kernel.setArg(1, CL_kMap);
	kernel.setArg(2, CL_Params);
	kernel.setArg(3, CL_CQ_Small);
	const size_t &global_size = DetectorSize[0] * DetectorSize[1];

	//launch Kernel
	Options.Echo("Launch kernel ... \n");

	Profiler.Tic();

	err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(global_size), cl::NullRange, NULL, &cl_event);

	Options.checkErr(err, "Launch Kernel in Detector::AutoCorrelate_CofQ_SmallMesh() ");
	cl_event.wait();
	Options.Echo("C(q)-small kernel finished in");
	Profiler.Toc(true);

	err = queue.enqueueReadBuffer(CL_CQ_Small, CL_TRUE, 0, ACsize, TempMesh);
	Options.checkErr(err, "OpenCL kernel, launched in Detector::AutoCorrelate_CofQ_SmallMesh() ");

	//Free Device
	Options.OCL_FreeDevice(OpenCLDeviceNumber);


	//convert to Double
	double mean = 0;
	//#pragma omp parallel for
	for (unsigned int i = 0; i < SmallMesh.Shape.Size_AB* SmallMesh.Shape.Size_AB* SmallMesh.Shape.Size_C; i++)
	{
		SmallMesh.CQMesh[i] = (double)TempMesh[i] / Multiplicator;
		mean += SmallMesh.CQMesh[i];
	}
	std::cout << "Mean cq: " << mean / SmallMesh.Shape.Size_AB* SmallMesh.Shape.Size_AB* SmallMesh.Shape.Size_C << "\n";


	//Free Memory
	delete[] TempMesh;

}

void Detector::Merge_smallCofQ(ACMesh & BigMesh, ACMesh & SmallMesh, std::vector<Settings::HitEvent>& Events, Settings & Options, AutoCorrFlags Flags)
{
	Merge_smallCofQ(BigMesh, SmallMesh, Events, 0, Events.size(), Options, Flags);
}
void Detector::Merge_smallCofQ(ACMesh & BigMesh, ACMesh & SmallMesh, std::vector<Settings::HitEvent>& Events, unsigned int LowerBound, unsigned int UpperBound, Settings & Options, AutoCorrFlags Flags)
{
	{//Test stuff
		if (SmallMesh.Shape.Size_AB != BigMesh.Shape.Size_AB)
		{
			std::cerr << "ERROR: Perpendicular Mesh size does not fit.\n ";
			std::cerr << "   ->: Detector::Merge_smallCofQ()\n ";
			throw;
		}
		if (UpperBound > Events.size())
		{
			std::cerr << "ERROR: Upperbound exceeds number of Events.\n ";
			std::cerr << "   ->: Detector::Merge_smallCofQ()\n ";
			throw;
		}
	}//Test stuff


	double MaxWeight = 0.0;
	float* Rot_and_Weight = new float[10 * (UpperBound - LowerBound)];
	unsigned int ind = 0;
	for (unsigned int i = LowerBound; i < UpperBound; i++)
	{
		//rotation matrix
		Rot_and_Weight[ind + 0] = Events[i].RotMatrix[0];
		Rot_and_Weight[ind + 1] = Events[i].RotMatrix[1];
		Rot_and_Weight[ind + 2] = Events[i].RotMatrix[2];
		Rot_and_Weight[ind + 3] = Events[i].RotMatrix[3];
		Rot_and_Weight[ind + 4] = Events[i].RotMatrix[4];
		Rot_and_Weight[ind + 5] = Events[i].RotMatrix[5];
		Rot_and_Weight[ind + 6] = Events[i].RotMatrix[6];
		Rot_and_Weight[ind + 7] = Events[i].RotMatrix[7];
		Rot_and_Weight[ind + 8] = Events[i].RotMatrix[8];


		if (fabs((Events[i].RotMatrix[0] * Events[i].RotMatrix[0] + Events[i].RotMatrix[3] * Events[i].RotMatrix[3] + Events[i].RotMatrix[6] * Events[i].RotMatrix[6]) - 1.0) > 0.00001 ||
			fabs((Events[i].RotMatrix[1] * Events[i].RotMatrix[1] + Events[i].RotMatrix[4] * Events[i].RotMatrix[4] + Events[i].RotMatrix[7] * Events[i].RotMatrix[7]) - 1.0) > 0.00001 ||
			fabs((Events[i].RotMatrix[2] * Events[i].RotMatrix[2] + Events[i].RotMatrix[5] * Events[i].RotMatrix[5] + Events[i].RotMatrix[8] * Events[i].RotMatrix[8]) - 1.0) > 0.00001)
		{
			std::cout << "Invalid rotation matrix: \n";
			for (int M = 0; M < 9; M++)
			{
				std::cout << Events[i].RotMatrix[M] << "\t ";
				if ((M + 1) % 3 == 0)
					std::cout << "\n";
			}
			std::cout << "=>\n"
				<< Events[i].RotMatrix[0] * Events[i].RotMatrix[0] + Events[i].RotMatrix[3] * Events[i].RotMatrix[3] + Events[i].RotMatrix[6] * Events[i].RotMatrix[6]
				<< " \t " <<
				Events[i].RotMatrix[1] * Events[i].RotMatrix[1] + Events[i].RotMatrix[4] * Events[i].RotMatrix[4] + Events[i].RotMatrix[7] * Events[i].RotMatrix[7]
				<< " \t " <<
				Events[i].RotMatrix[2] * Events[i].RotMatrix[2] + Events[i].RotMatrix[5] * Events[i].RotMatrix[5] + Events[i].RotMatrix[8] * Events[i].RotMatrix[8]
				<< "\n";
		}


		//weight (mean intensity)
		Rot_and_Weight[ind + 9] = (float)((double)Events[i].MeanIntensity*(double)Events[i].MeanIntensity);

		if (Events[i].MeanIntensity > MaxWeight)
			MaxWeight = Events[i].MeanIntensity;

		ind += 10;
	}

	std::cout << "Max weight: " << MaxWeight << "\n";


	double Min_Cq, Mean_Cq, Max_Cq;
	ArrayOperators::Min_Max_Mean_Value(SmallMesh.CQMesh, SmallMesh.Shape.Size_AB*SmallMesh.Shape.Size_AB*SmallMesh.Shape.Size_C, Min_Cq, Max_Cq, Mean_Cq);


	std::cout << "Small C(q):: Min = " << Min_Cq << "; Max = " << Max_Cq << "; Mean = " << Mean_Cq << "\n";

	double Multiplicator = 1e-12;
	for (; 1 > MaxWeight*MaxWeight*Mean_Cq*Multiplicator;) //Order of magnitude of largest weight //=> goto largest weight^2
	{
		Multiplicator *= 10;
	}
	uint64_t OOM = 1; //Order of Magnitude (+1) entries in Events
	for (; OOM < (UpperBound - LowerBound);)
	{
		OOM *= 10;
	}
	Multiplicator = Multiplicator / OOM;
	Multiplicator *= 1e10;//1e16;
	if (Multiplicator > 0.6)
		Multiplicator = round(Multiplicator);
	if (Multiplicator < 1 && Multiplicator > 0.1)
		Multiplicator = 1;
	if (Multiplicator < 1e-12)
		Multiplicator = 1e-12;
	std::cout << "Multiplicator: " << Multiplicator << std::endl;

	//adjust accuracy of weights according to the multiplicator
	ind = 0;
	for (unsigned int i = LowerBound; i < UpperBound; i++)
	{
		float ReducedMultipl = (float)Multiplicator;
		if(Multiplicator > 0.6)
			ReducedMultipl = roundf((float) Multiplicator);// / Options.F_I_Conversion.Step;
		float t_weight = Rot_and_Weight[ind + 9];
		t_weight *= ReducedMultipl;
		t_weight = floorf(t_weight + 0.5);
		Rot_and_Weight[ind + 9] = t_weight / ReducedMultipl;
		ind += 10;
	}


	uint64_t * TempBigMesh = new uint64_t[SmallMesh.Shape.Size_AB * SmallMesh.Shape.Size_AB *SmallMesh.Shape.Size_AB]();

	
	//set Parameter
	double Params[9];
	Params[0] = SmallMesh.Shape.Voxel_per_dq; //dq per Voxel
	Params[1] = SmallMesh.Shape.Size_AB; // Size perp
	Params[2] = SmallMesh.Shape.Size_C; // Size C
	Params[3] = SmallMesh.Shape.k_A; // Dimension Alignment
	Params[4] = SmallMesh.Shape.k_B; // Dimension Alignment 
	Params[5] = SmallMesh.Shape.k_C; // Dimension Alignment 
	Params[6] = Flags.InterpolationMode;
	Params[7] = Multiplicator; //Multiplicator for conversion to long
	Params[8] = UpperBound - LowerBound; //Number of Rotations

	//DEBUG BULLSHIT
	std::cout << "\n***************\nParams:\n";
	for (int i = 0; i < 9; i++)
	{
		std::cout << Params[i] << "\n";
	}
	std::cout << "**************\n";


	//reserve OpenCL Device
	int OpenCLDeviceNumber = -1;
	cl_int err;

	while ((OpenCLDeviceNumber = Options.OCL_ReserveDevice()) == -1)
	{
		std::this_thread::sleep_for(std::chrono::microseconds(Options.ThreadSleepForOCLDev));
	}

	//obtain Device
	cl::Device CL_Device = Options.CL_devices[OpenCLDeviceNumber];


	//Setup Queue
	cl::CommandQueue queue(Options.CL_context, CL_Device, 0, &err);
	Options.checkErr(err, "Setup CommandQueue in Detector::AutoCorrelate_CofQ_SmallMesh() ");
	cl::Event cl_event;


	//profiler stuff
	ProfileTime Profiler;

	// KernelBuffer
	//size_t ACsize = sizeof(uint64_t) * SmallMesh.Shape.Size_AB * SmallMesh.Shape.Size_AB * ((SmallMesh.Shape.Size_AB + 1) / 2);
	size_t ACsize = sizeof(uint64_t) * (SmallMesh.Shape.Size_AB * SmallMesh.Shape.Size_AB * SmallMesh.Shape.Size_AB);
	cl::Buffer CL_CQ(Options.CL_context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, ACsize, TempBigMesh, &err);
	//Input:
	size_t ACsizeSmall = sizeof(double) * SmallMesh.Shape.Size_AB * SmallMesh.Shape.Size_AB * SmallMesh.Shape.Size_C;
	cl::Buffer CL_CQ_Small(Options.CL_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, ACsizeSmall, SmallMesh.CQMesh, &err);

	size_t RotWeightSize = sizeof(float) * 10 * (UpperBound - LowerBound);
	cl::Buffer CL_RW(Options.CL_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, RotWeightSize, Rot_and_Weight, &err);
	cl::Buffer CL_Params(Options.CL_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(Params), &Params, &err);

	if (Options.echo)
		std::cout << "Memory needed by kernel: " << (((double)(ACsize + ACsizeSmall + RotWeightSize + sizeof(Params))) / (1024.0*1024.0*1024.0)) << "Gb\n";

	//Setup Kernel
	cl::Kernel kernel(Options.CL_Program, "Merge_CQ", &err);
	Options.checkErr(err, "Setup AutoCorr_CQ in Detector::AutoCorrelate_CofQ_SmallMesh() ");

	//Set Arguments
	kernel.setArg(0, CL_CQ_Small);
	kernel.setArg(1, CL_RW);
	kernel.setArg(2, CL_Params);
	kernel.setArg(3, CL_CQ);
	const size_t &global_size = SmallMesh.Shape.Size_AB * SmallMesh.Shape.Size_AB * SmallMesh.Shape.Size_C;


	//launch Kernel
	Options.Echo("Launch kernel (Merge C(q)) ... \n");

	Profiler.Tic();

	err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(global_size), cl::NullRange, NULL, &cl_event);

	Options.checkErr(err, "Launch Kernel in Detector::Merge_smallCofQ() ");
	cl_event.wait();
	Options.Echo("C(q)-Merge kernel finished in");
	Profiler.Toc(true);

	err = queue.enqueueReadBuffer(CL_CQ, CL_TRUE, 0, ACsize, TempBigMesh);
	Options.checkErr(err, "OpenCL kernel, launched in Detector::Merge_smallCofQ() ");

	//Free Device
	Options.OCL_FreeDevice(OpenCLDeviceNumber);

	//*****************PostProcessing***********************

	//double * DoubleBigMesh = new double[SmallMesh.Shape.Size_AB * SmallMesh.Shape.Size_AB * ((SmallMesh.Shape.Size_AB + 1) / 2)];

#pragma omp parallel for
	for (unsigned int i = 0; i < BigMesh.Shape.Size_AB * BigMesh.Shape.Size_AB * BigMesh.Shape.Size_AB; i++)
	{
		BigMesh.CQMesh[i] = ((double)TempBigMesh[i] / Multiplicator);

		//if(TempBigMesh[i] != 0)
		//	std::cout << TempBigMesh[i] << " -> " << BigMesh.CQMesh[i] << std::endl;
		//if (TempBigMesh[i] > 0)
		//	std::cout << TempBigMesh[i] << "   " << BigMesh.CQMesh[i] << std::endl;
	}



	//test by comparing to small c(q)
	if(false)
	{
		double t = 0.0;
		int SZ = SmallMesh.Shape.Size_AB;
		for (unsigned int ss = 0; ss < SmallMesh.Shape.Size_C; ss++)
		{
			for (unsigned int ms = 0; ms < SmallMesh.Shape.Size_AB; ms++)
			{
				for (unsigned int fs = 0; fs < SmallMesh.Shape.Size_AB; fs++)
				{
					int scan[3];
					scan[SmallMesh.Shape.k_A] = fs;
					scan[SmallMesh.Shape.k_B] = ms;
					scan[SmallMesh.Shape.k_C] = ss - ((SmallMesh.Shape.Size_C-1)/2) + ((SmallMesh.Shape.Size_AB - 1) / 2);

					long indBig = scan[0] + scan[1] * SZ + scan[2] * SZ*SZ;
					long indSmall = fs + ms * SZ + ss * SZ*SZ;

					if (BigMesh.CQMesh[indBig] != 0)
					{
						if(BigMesh.CQMesh[indBig] / SmallMesh.CQMesh[indSmall] != t)
							std::cout << BigMesh.CQMesh[indBig] << " / " << SmallMesh.CQMesh[indSmall] << " = " << BigMesh.CQMesh[indBig] / SmallMesh.CQMesh[indSmall] << " :: "<<(BigMesh.CQMesh[indBig] / SmallMesh.CQMesh[indSmall])/t << std::endl;
						t = BigMesh.CQMesh[indBig] / SmallMesh.CQMesh[indSmall];
					}

				}
			}
		}

	}
	
	//Free memory
	delete[] Rot_and_Weight;
	delete[] TempBigMesh;
}




//Outdated?
void Detector::AutoCorrelate_CofQ(ACMesh & BigMesh, AutoCorrFlags Flags, std::vector<Settings::HitEvent>& Events, unsigned int LowerBound, unsigned int UpperBound, Settings& Options)
{
	if (!BigMesh.Checklist.CofQBigMesh) //check if it is the right Mesh
	{
		std::cerr << "ERROR: BigMesh argument is not a CofQ-Mesh\n";
		std::cerr << "   ->  in Detector::AutoCorrelate_CofQ()\n";
		throw;
	}

	//reserve OpenCL Device
	int OpenCLDeviceNumber = -1;

	while ((OpenCLDeviceNumber = Options.OCL_ReserveDevice()) == -1)
	{
		std::this_thread::sleep_for(std::chrono::microseconds(Options.ThreadSleepForOCLDev));
	}

	Options.Echo("AC C(q) - function:\n************************\n");
	if (Options.echo)
		std::cout << "reserved OpenCl device number: " << OpenCLDeviceNumber << "\n";
	cl_int err;

	//LocalEventList
	float* Rot_and_Weight = new float[10 * (UpperBound - LowerBound)];
	unsigned int ind = 0;
	for (unsigned int i = LowerBound; i < UpperBound; i++)
	{
		//rotation matrix
		Rot_and_Weight[ind + 0] = Events[i].RotMatrix[0];
		Rot_and_Weight[ind + 1] = Events[i].RotMatrix[1];
		Rot_and_Weight[ind + 2] = Events[i].RotMatrix[2];
		Rot_and_Weight[ind + 3] = Events[i].RotMatrix[3];
		Rot_and_Weight[ind + 4] = Events[i].RotMatrix[4];
		Rot_and_Weight[ind + 5] = Events[i].RotMatrix[5];
		Rot_and_Weight[ind + 6] = Events[i].RotMatrix[6];
		Rot_and_Weight[ind + 7] = Events[i].RotMatrix[7];
		Rot_and_Weight[ind + 8] = Events[i].RotMatrix[8];

		//std::cout << Events[i].RotMatrix[0] << "\t" << Events[i].RotMatrix[1] << "\t" << Events[2].RotMatrix[1] << "\n";
		//std::cout << Events[i].RotMatrix[3] << "\t" << Events[i].RotMatrix[4] << "\t" << Events[5].RotMatrix[1] << "\n";
		//std::cout << Events[i].RotMatrix[6] << "\t" << Events[i].RotMatrix[7] << "\t" << Events[2].RotMatrix[8] << "\n \n";

		//int wa;
		//std::cin >> wa;

		//weight (mean Intensity)
		Rot_and_Weight[ind + 9] = Events[i].MeanIntensity;
		ind += 10;
	}

	//obtain Device
	cl::Device CL_Device = Options.CL_devices[OpenCLDeviceNumber];

	//set Parameter
	double Params[5];
	Params[0] = DetectorSize[0] * DetectorSize[1]; //Numer of pixels (size[0]*size[1])
	Params[1] = BigMesh.Shape.Voxel_per_dq; // Voxel per dq
	Params[2] = BigMesh.Shape.Size_AB; // a  (V = a*a*(a+1)/2)
	Params[3] = (UpperBound - LowerBound); // how many events, for roataion and weight loop
	Params[4] = Flags.InterpolationMode;

	//Half AC-Buffer
	double* CQ_Half = new double[BigMesh.Shape.Size_AB * BigMesh.Shape.Size_AB * ((BigMesh.Shape.Size_AB + 1) / 2)]();

	//OpenCL Buffer:
	//AutoCorr_CQ(const float *IntensityData, const float *KMap, const float *Rotations_and_Weights, const double *Params, double *CQ)
	//Output:
	size_t ACsize = sizeof(double) * BigMesh.Shape.Size_AB * BigMesh.Shape.Size_AB * ((BigMesh.Shape.Size_AB + 1) / 2);
	cl::Buffer CL_CQ_Half(Options.CL_context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, ACsize, CQ_Half, &err);
	//Input:
	size_t Intsize = sizeof(float) * DetectorSize[0] * DetectorSize[1];
	cl::Buffer CL_Intensity(Options.CL_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Intsize, Intensity, &err);
	size_t KMapsize = sizeof(float) * 3 * DetectorSize[0] * DetectorSize[1];
	cl::Buffer CL_kMap(Options.CL_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, KMapsize, kMap, &err);
	size_t RaWsize = sizeof(float) * 10 * (UpperBound - LowerBound);
	cl::Buffer CL_RaW(Options.CL_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, RaWsize, Rot_and_Weight, &err);
	cl::Buffer CL_Params(Options.CL_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(Params), &Params, &err);

	//Setup Kernel
	cl::Kernel kernel(Options.CL_Program, "AutoCorr_CQ", &err);
	Options.checkErr(err, "Setup AutoCorr_CQ in Detector::AutoCorrelate_CofQ() ");

	//Set Arguments
	kernel.setArg(0, CL_Intensity);
	kernel.setArg(1, CL_kMap);
	kernel.setArg(2, CL_RaW);
	kernel.setArg(3, CL_Params);
	kernel.setArg(4, CL_CQ_Half);

	//Setup Queue
	cl::CommandQueue queue(Options.CL_context, CL_Device, 0, &err);
	Options.checkErr(err, "Setup CommandQueue in Detector::AutoCorrelate_CofQ() ");
	cl::Event cl_event;

	//profiler stuff
	ProfileTime Profiler;
	//

	const size_t &global_size = DetectorSize[0] * DetectorSize[1];
	//launch Kernel
	Options.Echo("Launch kernel ... \n");
	std::cout << "for " << Params[3] << "Events\n";
	std::cout << "on a " << Params[0] << " pixel detector" << std::endl;
	Profiler.Tic();
	err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(global_size), cl::NullRange, NULL, &cl_event);

	Options.checkErr(err, "Launch Kernel in Detector::AutoCorrelate_CofQ() ");
	cl_event.wait();
	Options.Echo("C(q) kernel finished in");
	Profiler.Toc(true);

	//read Buffer
	err = queue.enqueueReadBuffer(CL_CQ_Half, CL_TRUE, 0, ACsize, CQ_Half);
	Options.checkErr(err, "OpenCL kernel, launched in Detector::AutoCorrelate_CofQ() ");
	//ToImplement

	std::cout << "Ret[0]: " << CQ_Half[0] << "\tRet[1]: " << CQ_Half[1] << "\tRet[2]: " << CQ_Half[2] << "\n";


	//Free Device
	Options.OCL_FreeDevice(OpenCLDeviceNumber);

	//Free Memory
	delete[] Rot_and_Weight;

	// Mirrow and Postprocess Data:
	std::cerr << "\nTO IMPLEMENT IN Detector::AutoCorrelate_CofQ()\n\n";
}