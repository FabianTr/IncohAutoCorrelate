#define CL_ENABLE_EXCEPTIONS

#include <cstdio>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <array>
#include <omp.h>
#include <cmath>

#include <CL/cl.hpp>

#include "Settings.h"

//my Classes
#include "Detector.h"

#include "ArrayOperators.h"
#include "ProfileTime.h"
#include <Eigen/Dense>









int main()
{
	int end;
	ProfileTime profiler;
	Settings Options;
	Options.echo = true;

	Options.F_I_Conversion.Step = 0.01f;
	


	Options.Echo("Load Open CL stuff:\n");

	Options.SetUp_OpenCL();

	
	//std::cout << "Program ended\n";
	//std::cin >> end;
	//return 0;



	//TestDetectorSparseList();

	Detector TestDet;

	TestDet.LoadPixelMap("/home/trostfab/LR17Stuff/PixelMap_X123.h5", "PixelMap_X123");

	

	Options.Echo("Load Streamfile");

	Options.MReference << 6.227, 0, 0,
		0, 8.066, 0,
		0, 0, 11.1;



	Options.LoadStreamFile("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/indexing/3fs_JF.stream","entry_1/instrument_1/detector_2/detector_corrected/data",false);
	//Options.AlternateDetectorSourcePath("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/cheetah/hdf5");

	//for (int i = 0; i <10; i++)
	//{
	//	std::cout << Options.HitEvents[i].Filename << " index: " << Options.HitEvents[i].SerialNumber <<" Event: " << Options.HitEvents[i].Event << "\n";
	//}









	
	Options.Echo("Load intensity data for Hit 0");
	TestDet.LoadIntensityData(&Options.HitEvents[0]);

	std::cout << "\n";
	for (int iy = 0; iy <20; iy++)
	{
		for (int ix = 0; ix <20; ix++)
		{
			std::cout << TestDet.Intensity[ix + TestDet.DetectorSize[1] * iy] << "   ";
		}
		std::cout << "\n";
	}

	Options.Echo("Create k-map");
	TestDet.Calc_kMap();

	Options.Echo("Create BigMesh");
	ACMesh BigMesh;
	BigMesh.Options = &Options;
	BigMesh.CreateBigMeshForDetector(TestDet, TestDet.DetectorSize[0] + 1);

	Options.Echo("Create sparese detector");
	TestDet.CreateSparseHitList(3.2f, 6.4f);//
	Detector::AutoCorrFlags ACflags;
	ACflags.InterpolationMode = Settings::Interpolation::Linear;

	Options.Echo("check");
	TestDet.AutoCorrelateSparseList(BigMesh, ACflags);



	//Options.Echo("AutoCorrelate");

	//profiler.Tic();

	//for (int i = 0; i < 1000; i++)
	//{
	//	TestDet.LoadIntensityData(&Options.HitEvents[i]);
	//	TestDet.CreateSparseHitList(3.0);

	//	Detector::AutoCorrFlags flags;
	//	flags.InterpolationMode = Settings::Interpolation::NearestNeighbour;
	//	TestDet.AutoCorrelateSparseList(BigMesh, flags);

	//}
	//
	//profiler.Toc(true);

	Options.Echo("Load and average intensities (1000x)");
	profiler.Tic();
	TestDet.LoadAndAverageIntensity(Options.HitEvents, 3.2f,6.4f, 0, 1000);
	profiler.Toc(true);
	std::cout << "done.\n";


	Options.Echo("Create C(q) - Mesh");

	ACMesh CQMesh;
	CQMesh.CreateBigMesh_CofQ_ForDetector(TestDet, 1025);


	std::vector<Settings::HitEvent> SmallEventList;
	for (unsigned int i = 0; i < 1000; i++)
	{
		SmallEventList.push_back(Options.HitEvents[i]);
	}

	Detector::AutoCorrFlags flags;
	flags.InterpolationMode = Settings::Interpolation::Linear;
	TestDet.AutoCorrelate_CofQ(CQMesh, flags, Options.HitEvents, 0, 1, Options);

	//int ind = 0;
	//for (int i = 0; i < 10; i++)
	//{
	//	for (int j = 0; j < 10; j++)
	//	{
	//		std::cout << Options.HitEvents[ind].PhotonCount << " : " << Options.HitEvents[ind].MeanIntensity << "\t";
	//		ind++;
	//	}
	//	std::cout << "\n";
	//}
	//std::cout << "\n\n\n";

	//for (int iy = 0; iy <10; iy++)
	//{
	//	for (int ix = 0; ix <10; ix++)
	//	{
	//		std::cout << TestDet.Intensity[ix + TestDet.DetectorSize[1] * iy] << "\t";
	//	}
	//	std::cout << "\n";
	//}
	//std::cout << "\n\n\n";

	//double MPI = 0;
	//double MinPi = 99999;
	//double MaxPi = 0;
	//for (int i = 0; i < 5000; i++)
	//{
	//	MPI += Options.HitEvents[i].PhotonCount;
	//	if (Options.HitEvents[i].PhotonCount > MaxPi)
	//		MaxPi = Options.HitEvents[i].PhotonCount;
	//	if (Options.HitEvents[i].PhotonCount < MinPi)
	//		MinPi = Options.HitEvents[i].PhotonCount;
	//}
	//MPI = MPI / 10000;

	//for (int iy = 0; iy <10; iy++)
	//{
	//	for (int ix = 0; ix <10; ix++)
	//	{
	//		std::cout << TestDet.Intensity[ix + TestDet.DetectorSize[1] * iy] * (MPI) << "\t";
	//	}
	//	std::cout << "\n";
	//}

	//std::cout << "Max PI: " << MaxPi << "\t Min PI: " << MinPi << "\t M PI: " << MPI << "\n";


	
	std::cout << "Program ended\n";
	std::cin >> end;
    return 0;
}