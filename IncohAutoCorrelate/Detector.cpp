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
#include <thread>         
#include <chrono> 
#include "ProfileTime.h"




Detector::Detector()
{
}


Detector::~Detector()
{
}


//Helpfunctions


inline float Detector::DiscretizeToPhotones(float Value, float Threshold, float PhotonSamplingStep) //create single Photon counting by simple thresholding
{
	return ceilf((Value - Threshold) / PhotonSamplingStep)*(Value >= Threshold);
}

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

	for (unsigned int i_x = 0; i_x < DetectorSize[0]; i_x++)
	{
		for (unsigned int i_y = 0; i_y < DetectorSize[1]; i_y++)
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

		for (unsigned int i_y = 0; i_y < dims[2]; i_y++)
		{
			for (unsigned int i_x = 0; i_x < dims[1]; i_x++)
			{
				for (unsigned int i_d = 0; i_d < 3; i_d++)
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
void Detector::LoadAndAverageIntensity(std::vector<Settings::HitEvent>& Events, float Threshold, float PhotonSamplingStep, int LowerBound, int UpperBound)
{
	if (Events.size() == 0)
	{
		std::cerr << "WARNING: Event list is empty, no intensity integration/averaging can be performed.\n";
		return;
	}
	int* IntensityPhotonDiscr = NULL;
	Intensity = new float[DetectorSize[1] * DetectorSize[0]];
	{	
		float* tmpIntensity = new float[DetectorSize[1] * DetectorSize[0]]();
		if (PhotonSamplingStep > 0)
			IntensityPhotonDiscr = new int[DetectorSize[1] * DetectorSize[0]]();
		Intensity = new float[DetectorSize[1] * DetectorSize[0]]();
		for (int i = LowerBound; i < UpperBound; i++)//get  slides
		{
			GetSliceOutOfHDFCuboid(tmpIntensity, Events[i].Filename, Events[i].Dataset, Events[i].Event);
			if (PhotonSamplingStep <= 0)// No Photon discretising
			{
				ArrayOperators::ParAdd(Intensity, tmpIntensity, DetectorSize[1] * DetectorSize[0], Threshold); //add with threshold
				
				//update Event
				Events[i].MeanIntensity = ArrayOperators::Sum(tmpIntensity, DetectorSize[1] * DetectorSize[0]) / (DetectorSize[1] * DetectorSize[0]);

			}
			else// Photon discretising
			{
				#pragma omp parallel for
				for (unsigned int i = 0; i < DetectorSize[1] * DetectorSize[0]; i++)
				{
					if (tmpIntensity[i] >= Threshold)
					{
						tmpIntensity[i] = DiscretizeToPhotones(tmpIntensity[i], Threshold, PhotonSamplingStep);
						IntensityPhotonDiscr[i] += (int)tmpIntensity[i];
					}
					else
					{ 
						tmpIntensity[i] = 0;
					}
				}

				//update Event
				Events[i].PhotonCount = ArrayOperators::Sum(tmpIntensity, DetectorSize[1] * DetectorSize[0]);
				Events[i].MeanIntensity = (float)Events[i].PhotonCount / ((float) (DetectorSize[1] * DetectorSize[0]));

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
	ArrayOperators::ParMultiplyScalar(Intensity, 1.0 / (UpperBound - LowerBound), DetectorSize[1] * DetectorSize[0]);

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

	for (unsigned int i_y = 0; i_y < DetectorSize[1]; i_y++)
	{
		for (unsigned int i_x = 0; i_x < DetectorSize[0]; i_x++)
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

	for (unsigned int i_y = 0; i_y < DetectorSize[1]; i_y++)
	{
		SparseHitList.insert(SparseHitList.end(), TmpSparseVec[i_y].Vec.begin(), TmpSparseVec[i_y].Vec.end());
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
		for (unsigned int i = 0; i < DetectorSize[0]* DetectorSize[1]; i++)
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

void Detector::AutoCorrelateSparseList(ACMesh & BigMesh, AutoCorrFlags Flags)
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
	//Implementation for CPU
	#pragma omp parallel for
	for (unsigned int i = 0; i < SparseHitList.size(); i++)
	{
		for (unsigned int j = i; j < SparseHitList.size(); j++)
		{
			if (j == i)
				continue;
			float q[3];
			float RM[9] = { 1,0,0,0,1,0,0,0,1 };//TODO IMPLEMENT ROTATION MATRIX -> THIS IS A DUMMY

			q[0] = SparseHitList[i][0] - SparseHitList[j][0];
			q[1] = SparseHitList[i][1] - SparseHitList[j][1];
			q[2] = SparseHitList[i][2] - SparseHitList[j][2];
			BigMesh.Atomic_Add_q_Entry(q, RM, SparseHitList[i][3] * SparseHitList[j][3], Flags.InterpolationMode); // DetectorEvent->RotMatrix
			//std::cout << SparseHitList[i][3] * SparseHitList[j][3] << ", ";
			q[0] = SparseHitList[j][0] - SparseHitList[i][0];
			q[1] = SparseHitList[j][1] - SparseHitList[i][1];
			q[2] = SparseHitList[j][2] - SparseHitList[i][2];
			BigMesh.Atomic_Add_q_Entry(q,RM, SparseHitList[i][3] * SparseHitList[j][3], Flags.InterpolationMode); // DetectorEvent->RotMatrix
		}
	}


}

void Detector::AutoCorrelate_CofQ(ACMesh & BigMesh, AutoCorrFlags Flags, std::vector<Settings::HitEvent>& Events, unsigned int LowerBound, unsigned int UpperBound, Settings& Options)
{
	if (!BigMesh.Checklist.CofQMesh) //check if it is the right Mesh
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
	Params[0] = DetectorSize[0]* DetectorSize[1]; //Numer of pixels (size[0]*size[1])
	Params[1] = BigMesh.Shape.dq_per_Voxel; //dq per Voxel
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
	size_t Intsize = sizeof(float) * DetectorSize[0]* DetectorSize[1];
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
	err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(global_size), cl::NullRange, NULL, &cl_event);
	Profiler.Tic();
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

	// Mirrow and Postprocess Data:

}
