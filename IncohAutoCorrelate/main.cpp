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

	TestDet.LoadPixelMap("/gpfs/cfel/cxi/scratch/user/trostfab/PixelMap/PixelMap_J.h5", "data/data");


	Options.Echo("Create k-map");
	TestDet.Calc_kMap();

	std::cout << "Max k = [" << TestDet.Max_k[0] << "; " << TestDet.Max_k[1] << "; " << TestDet.Max_k[2] << "]\n";
	std::cout << "Min k = [" << TestDet.Min_k[0] << "; " << TestDet.Min_k[1] << "; " << TestDet.Min_k[2] << "]\n";
	std::cout << "Max q = [" << TestDet.Max_q[0] << "; " << TestDet.Max_q[1] << "; " << TestDet.Max_q[2] << "]\n";

	Options.Echo("\nLoad pixel-mask");
	TestDet.PixelMask = new int[1024 * 1024]();
	ArrayOperators::LoadArrayFromFile<int>("/gpfs/cfel/cxi/scratch/user/trostfab/PixelMap/PixelMask_thr03.bin", TestDet.PixelMask,1024*1024);

	//for (int i = 0; i < 50; i++)
	//{
	//	for (int j = 0; j < 50; j++)
	//	{
	//		std::cout << TestDet.PixelMask[i + 1024 * j] << "\t ";
	//	}
	//	std::cout << "\n";
	//}


	//Options.Echo("Load HitEventList");

	//Options.LoadHitEventListFromFile("/gpfs/cfel/cxi/scratch/user/trostfab/IACC_TESTSPACE/test.xml");


	Options.Echo("Load Streamfile");
	//Hb reference-unit-cell
	Options.MReference << 6.227, 0, 0,	0, 8.066, 0,	0, 0, 11.1;
	//Options.LoadStreamFile("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/indexing/3fs_JF.stream","entry_1/instrument_1/detector_2/detector_corrected/data",false);

	{ // Get Sparse C(q)  --- TEST
		Options.LoadHitEventListFromFile("/gpfs/cfel/cxi/scratch/user/trostfab/IACC_TESTSPACE/HitEventList_3fs_JF.xml");

		TestDet.Intensity = new float[1024 * 1024];
		
		ArrayOperators::LoadArrayFromFile("/gpfs/cfel/cxi/scratch/user/trostfab/IACC_TESTSPACE/IntensityAv_3fs_JF.bin", TestDet.Intensity, 1024 * 1024);
		//apply Pixelmask
		ArrayOperators::ParMultiplyElementwise(TestDet.Intensity, TestDet.PixelMask, 1024 * 1024);
		//for (int i = 0; i < 50; i++)
		//{
		//	for (int j = 0; j < 50; j++)
		//	{
		//		std::cout << TestDet.Intensity[i + 1024 * j] << " ";
		//	}
		//	std::cout << "\n";
		//}

		ACMesh smallCQMesh;
		smallCQMesh.CreateSmallMesh_CofQ_ForDetector(TestDet, 1001);

		Detector::AutoCorrFlags flags;
		flags.InterpolationMode = Settings::Interpolation::NearestNeighbour;

		std::cout << "Calc C(q) (small)\n";
		TestDet.AutoCorrelate_CofQ_SmallMesh(smallCQMesh, flags, Options);

		//
		//{
		//	ACMesh CQMesh;
		//	CQMesh.CreateBigMesh_CofQ_ForDetector(TestDet, 1025);
		//	TestDet.AutoCorrelate_CofQ(CQMesh, flags, Options.HitEvents, 0, 1, Options);
		//}

		ArrayOperators::SafeArrayToFile("/gpfs/cfel/cxi/scratch/user/trostfab/IACC_TESTSPACE/TEST_Cq_1001_half.bin", smallCQMesh.CQMesh, smallCQMesh.Shape.Size_AB*smallCQMesh.Shape.Size_AB*smallCQMesh.Shape.Size_C, ArrayOperators::FileType::Binary);






		std::cout << "Program ended\n";
		std::cin >> end;
		return 0;
	}






	
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


	Options.Echo("Create SmallMesh");
	ACMesh SmallMesh;
	SmallMesh.CreateSmallMeshForDetector(TestDet, TestDet.DetectorSize[0] + 1);

	Options.Echo("Create BigMesh");
	ACMesh BigMesh;
	
	BigMesh.CreateBigMeshForDetector(TestDet, TestDet.DetectorSize[0] + 1);

	std::cout << "SmallMesh Size: " << SmallMesh.Shape.Size_C <<"\t; dq/dV: " << SmallMesh.Shape.dq_per_Voxel <<  "\n";
	std::cout << "BiglMesh Size: " << BigMesh.Shape.Size_C << "\t; dq/dV: " << BigMesh.Shape.dq_per_Voxel << "\n";

	std::cout << "dq/dV - RATIO: " << SmallMesh.Shape.dq_per_Voxel / BigMesh.Shape.dq_per_Voxel << "\n";

	std::cout << "Program ended\n";
	std::cin >> end;
	return 0;




	BigMesh.Options = &Options;

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

	//TestDet.LoadAndAverageIntensity(Options.HitEvents, 3.2f, 6.4f);

	profiler.Toc(true);
	std::cout << "done.\n";


	/*Options.Echo("Save EventList as XML");
	Options.SafeHitEventListToFile("/gpfs/cfel/cxi/scratch/user/trostfab/IACC_TESTSPACE/HitEventList_3fs_JF_TMP.xml");
	Options.Echo("Save averaged Intensity");
	ArrayOperators::SafeArrayToFile("/gpfs/cfel/cxi/scratch/user/trostfab/IACC_TESTSPACE/IntensityAv_3fs_JF_TMP.bin",TestDet.Intensity,TestDet.DetectorSize[0]* TestDet.DetectorSize[1],ArrayOperators::FileType::Binary);
*/


	Options.Echo("Create C(q) - Mesh");

	ACMesh CQMesh;
	CQMesh.CreateBigMesh_CofQ_ForDetector(TestDet, 1025);


	std::vector<Settings::HitEvent> SmallEventList;
	for (unsigned int i = 0; i < 1000; i++)
	{
		SmallEventList.push_back(Options.HitEvents[i]);
	}

	Detector::AutoCorrFlags flags;
	flags.InterpolationMode = Settings::Interpolation::NearestNeighbour;
	TestDet.AutoCorrelate_CofQ(CQMesh, flags, Options.HitEvents, 0, 3, Options);

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