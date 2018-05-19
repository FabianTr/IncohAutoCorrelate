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

#include "Statistics.h"


void Test_CQ_small(Settings &Options, Detector &Det)
{

	int const MeshSize = 1001;
	float const QZoom = 1.0f;

	ProfileTime profiler;

	 // Get Sparse C(q)  --- TEST
		//Options.LoadHitEventListFromFile("/gpfs/cfel/cxi/scratch/user/trostfab/IACC_TESTSPACE/HitEventList_3fs_JF.xml");

	Det.Intensity = new float[1024 * 1024];

	//load integrated intensity
	ArrayOperators::LoadArrayFromFile("/gpfs/cfel/cxi/scratch/user/trostfab/IACC_TESTSPACE/IntensityAv_3fs_JF.bin", Det.Intensity, 1024 * 1024);
	//apply Pixelmask
	ArrayOperators::ParMultiplyElementwise(Det.Intensity, Det.PixelMask, 1024 * 1024);

	

	float Min=0, Max=0, Mean=0;

	ArrayOperators::Min_Max_Mean_Value(Det.Intensity, Det.DetectorSize[0] * Det.DetectorSize[1], Min, Max, Mean);

	std::cout << "Min: " << Min << "    Max: " << Max << "    Mean: " << Mean<<"\n";

	std::cout << "Size of long: " << sizeof(uint64_t) << "\n";


	long Multiplicator = 1;
	for (; 1 > Mean*Mean * Multiplicator; )
	{
		Multiplicator *= 10;
	}
	Multiplicator = Multiplicator * 100000;
	std::cout << "Multiplicator: " << Multiplicator << "\n";

	
	//return;

	//for (int i = 0; i < 50; i++)
	//{
	//	for (int j = 0; j < 50; j++)
	//	{
	//		std::cout << TestDet.Intensity[i + 1024 * j] << " ";
	//	}
	//	std::cout << "\n";
	//}

	ACMesh smallCQMesh;
	smallCQMesh.CreateSmallMesh_CofQ_ForDetector(Det, MeshSize,QZoom);


	std::cout << "MAX VOXEL OCC: " << 2 * smallCQMesh.Shape.Max_Q / smallCQMesh.Shape.dq_per_Voxel << "\n";

	
	Detector::AutoCorrFlags flags;
	flags.InterpolationMode = Settings::Interpolation::NearestNeighbour;

	


	std::cout << "Calc C(q) (small)\n";
	Det.AutoCorrelate_CofQ_SmallMesh(smallCQMesh, flags, Options);


	std::cout << "q-Max: " << smallCQMesh.Shape.Max_Q << "\n";

	//
	//{
	//	ACMesh CQMesh;
	//	CQMesh.CreateBigMesh_CofQ_ForDetector(TestDet, 1025);
	//	TestDet.AutoCorrelate_CofQ(CQMesh, flags, Options.HitEvents, 0, 1, Options);
	//}

	ArrayOperators::SafeArrayToFile("/gpfs/cfel/cxi/scratch/user/trostfab/IACC_TESTSPACE/Cq_1003_small.bin", smallCQMesh.CQMesh, smallCQMesh.Shape.Size_AB*smallCQMesh.Shape.Size_AB*smallCQMesh.Shape.Size_C, ArrayOperators::FileType::Binary);
	std::cout << "Saved as: /gpfs/cfel/cxi/scratch/user/trostfab/IACC_TESTSPACE/Cq_1003_small.bin \n";


	std::cout << "\n\n\nMerge and weight C(q):\n";
	ACMesh MergedCq;
	MergedCq.CreateBigMesh_CofQ_ForDetector(Det, MeshSize, QZoom);

	//Det.Merge_smallCofQ(MergedCq, smallCQMesh, Options.HitEvents, 0, 1000, Options, flags);
	Det.Merge_smallCofQ(MergedCq, smallCQMesh, Options.HitEvents, Options, flags);

	ArrayOperators::SafeArrayToFile("/gpfs/cfel/cxi/scratch/user/trostfab/IACC_TESTSPACE/Cq_1003_Big.bin", MergedCq.CQMesh, MergedCq.Shape.Size_AB*MergedCq.Shape.Size_AB*MergedCq.Shape.Size_C, ArrayOperators::FileType::Binary);
	std::cout << "Saved as: /gpfs/cfel/cxi/scratch/user/trostfab/IACC_TESTSPACE/Cq_1003_Big.bin \n";
}

void Load_and_average_Intensities(Settings &Options, Detector &Det)
{
	ProfileTime profiler;
	Options.Echo("Load and average intensities (all)");
	profiler.Tic();
	Det.LoadAndAverageIntensity(Options.HitEvents, 3.2f, 6.4f, true);

	Det.LoadAndAverageIntensity(Options.HitEvents, 3.2f, 6.4f);

	profiler.Toc(true);
	std::cout << "done.\n";

	//Options.Echo("Save EventList as XML");
	//Options.SafeHitEventListToFile("/gpfs/cfel/cxi/scratch/user/trostfab/IACC_TESTSPACE/HitEventList_3fs_Jungfr.xml");
	//Options.Echo("Save averaged Intensity");
	//ArrayOperators::SafeArrayToFile("/gpfs/cfel/cxi/scratch/user/trostfab/IACC_TESTSPACE/IntensityAv_3fs_JF_CHECK-FOR-PIXELMASK.bin", Det.Intensity, Det.DetectorSize[0] * Det.DetectorSize[1], ArrayOperators::FileType::Binary);

}

void GetOrientationStatistic(Settings &Options)
{
	Options.Echo("calc orientations");
	float* OrientationVectors;

	Statistics::Get_OrientationSphere(OrientationVectors, Options.HitEvents);
	std::cout << "Size of HitEvents: " << Options.HitEvents.size() << "\n";

	for (int i = 0; i < 10; i++)
	{
		std::cout << sqrtf(OrientationVectors[3 * i + 0] * OrientationVectors[3 * i + 0] + OrientationVectors[3 * i + 1] * OrientationVectors[3 * i + 1] + OrientationVectors[3 * i + 2] * OrientationVectors[3 * i + 2]) << " : " << OrientationVectors[3 * i + 0] << ", " << OrientationVectors[3 * i + 1] << ", " << OrientationVectors[3 * i + 2] << "\n";
	}

	ArrayOperators::SafeArrayToFile("/gpfs/cfel/cxi/scratch/user/trostfab/IACC_TESTSPACE/Orientations.bin", OrientationVectors, 3 * Options.HitEvents.size(), ArrayOperators::FileType::Binary);
}

void AutoCorrelateEvents(Settings &Options, Detector &Det)
{
	ProfileTime profiler;
	Options.Echo("Create BigMesh");
	ACMesh BigMesh;

	BigMesh.CreateBigMeshForDetector(Det, 1001, 1.0f);

	std::cout << "BiglMesh Size: " << BigMesh.Shape.Size_C << "\t; dq/dV: " << BigMesh.Shape.dq_per_Voxel << "\n";


	BigMesh.Options = &Options;

	Options.Echo("AutoCorrelate");
	std::cout << Options.HitEvents.size() << " Events\n";

	profiler.Tic();

	for (int i = 0; i < Options.HitEvents.size(); i++) // Options.HitEvents.size()
	{
		if (i%50 == 0)
			std::cout << i << "/" << Options.HitEvents.size() << std::endl;
		Det.LoadIntensityData(&Options.HitEvents[i]);
		ArrayOperators::ParMultiplyElementwise(Det.Intensity, Det.PixelMask, 1024 * 1024);
		Det.CreateSparseHitList(3.2f, 6.4f);


		//for (int i = 0; i < Det.SparseHitList.size(); i++)
		//{
		//	std::cout << Det.SparseHitList[i][0] << ", " << Det.SparseHitList[i][1]
		//		<< Det.SparseHitList[i][2] <<": " << Det.SparseHitList[i][3] << "\n";
		//}

		//int x;
		//std::cin >> x;


		Detector::AutoCorrFlags flags;
		flags.InterpolationMode = Settings::Interpolation::NearestNeighbour;
		Det.AutoCorrelateSparseList(BigMesh, flags);

	}

	profiler.Toc(true);

	
	double* ACMesh = new double[BigMesh.Shape.Size_AB*BigMesh.Shape.Size_AB*BigMesh.Shape.Size_C]();
	#pragma omp parallel for
	for (int i = 0; i < BigMesh.Shape.Size_AB*BigMesh.Shape.Size_AB*BigMesh.Shape.Size_C; i++)
	{
		ACMesh[i] = Options.IntToFloat(BigMesh.Mesh[i]);
	}



	ArrayOperators::SafeArrayToFile("/gpfs/cfel/cxi/scratch/user/trostfab/IACC_TESTSPACE/AC_UW_1003.bin", ACMesh, BigMesh.Shape.Size_AB*BigMesh.Shape.Size_AB*BigMesh.Shape.Size_C, ArrayOperators::FileType::Binary);
	std::cout << "Saved as: /gpfs/cfel/cxi/scratch/user/trostfab/IACC_TESTSPACE/AC_UW_1003.bin \n";

}


void CombineStuff()
{
	int size = 1003 * 1003 * 1003;

	double * CQ = new double[size]();

	double * AC = new double[size]();

	//double * AC1 = new double[1003 * 1003 * 1003]();
	//double * AC2 = new double[1003 * 1003 * 1003]();
	//double * AC3 = new double[1003 * 1003 * 1003]();
	//double * AC4 = new double[1003 * 1003 * 1003]();
	//double * AC5 = new double[1003 * 1003 * 1003]();

	double * AC_Final = new double[1003 * 1003 * 1003]();

	std::cout << "\n\n\n*************************\n";

	std::cout << "\n Load C(q)\n";
	ArrayOperators::LoadArrayFromFile("/gpfs/cfel/cxi/scratch/user/trostfab/IACC_TESTSPACE/TEST_Cq_1001_Big.bin", CQ, size);

	//std::cout << "\n Load AC1: 0 - 20k\n";
	//ArrayOperators::LoadArrayFromFile("/gpfs/cfel/cxi/scratch/user/trostfab/IACC_TESTSPACE/TEST_AC_UW_0-20k.bin", AC1, 1003 * 1003 * 1003);

	//std::cout << "Load AC2: 20 - 40k\n";
	//ArrayOperators::LoadArrayFromFile("/gpfs/cfel/cxi/scratch/user/trostfab/IACC_TESTSPACE/TEST_AC_UW_20k-40k.bin", AC2, 1003 * 1003 * 1003);

	//std::cout << "Load AC3: 40 - 60k\n";
	//ArrayOperators::LoadArrayFromFile("/gpfs/cfel/cxi/scratch/user/trostfab/IACC_TESTSPACE/TEST_AC_UW_40k-60k.bin", AC3, 1003 * 1003 * 1003);

	//std::cout << "Load AC4: 60 - 80k\n";
	//ArrayOperators::LoadArrayFromFile("/gpfs/cfel/cxi/scratch/user/trostfab/IACC_TESTSPACE/TEST_AC_UW_60k-80k.bin", AC4, 1003 * 1003 * 1003);

	//std::cout << "Load AC2: 80 - 96k\n";
	//ArrayOperators::LoadArrayFromFile("/gpfs/cfel/cxi/scratch/user/trostfab/IACC_TESTSPACE/TEST_AC_UW_80k-96k.bin", AC5, 1003 * 1003 * 1003);



	//std::cout << "\n Combine ACs\n";

	//#pragma omp parallel for
	//for (int i = 0; i < 1003*1003*1003; i++)
	//{
	//	AC[i] = AC1[i] + AC2[i] + AC3[i] + AC4[i] + AC5[i];
	//}


	std::cout << "\n Load AC ...\n";
	ArrayOperators::LoadArrayFromFile("/gpfs/cfel/cxi/scratch/user/trostfab/IACC_TESTSPACE/AC_UW_1003.bin", AC, size);

	std::cout << "\n Apply C(q) ...\n";

	#pragma omp parallel for
	for (int i = 0; i <size; i++)
	{
		if (CQ[i] <= 0)
			CQ[i] = 0;

		AC_Final[i] = AC[i] / CQ[i];
	}


	ArrayOperators::SafeArrayToFile("/gpfs/cfel/cxi/scratch/user/trostfab/IACC_TESTSPACE/TEST_AC_Final_1003.bin", AC_Final, size, ArrayOperators::FileType::Binary);
	std::cout << "Saved as: /gpfs/cfel/cxi/scratch/user/trostfab/IACC_TESTSPACE/TEST_AC_Final_1003.bin \n";

}


int main()
{
	//Run Stiings:

	const bool HitsFromXml = true; //otherwise from stream


	//****************************************************************************************************************************************************
	//Basic Startup (needed for [nearly] everything )
	//
	int end;
	ProfileTime profiler;
	Settings Options;
	Options.echo = true;

	Options.F_I_Conversion.Step = 0.01f;
	
	Options.Echo("Load Open CL stuff:\n");

	Options.SetUp_OpenCL();


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




	
	//Hb reference-unit-cell
	Options.MReference << 6.227, 0, 0, 0, 8.066, 0, 0, 0, 11.1;

	if (HitsFromXml)
	{
		Options.Echo("Load Events from XML");
		Options.LoadHitEventListFromFile("/gpfs/cfel/cxi/scratch/user/trostfab/IACC_TESTSPACE/HitEventList_3fs_Jungfr.xml");
	}
	else
	{
		Options.Echo("Load Streamfile");
		Options.LoadStreamFile("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/indexing/3fs_JF.stream", "entry_1/instrument_1/detector_2/detector_corrected/data", false);
	}


	//****************************************************************************************************************************************************


	


	//Test_CQ_small(Options, TestDet);

	AutoCorrelateEvents(Options, TestDet);


	CombineStuff();

	/*std::cout << "Program ended\n";
	std::cin >> end;*/
	return 0;






	








	//	std::cout << "Program ended\n";
	//	std::cin >> end;
	//	return 0;
	//}






	
	//Options.Echo("Load intensity data for Hit 0");
	//TestDet.LoadIntensityData(&Options.HitEvents[0]);

	//std::cout << "\n";
	//for (int iy = 0; iy <20; iy++)
	//{
	//	for (int ix = 0; ix <20; ix++)
	//	{
	//		std::cout << TestDet.Intensity[ix + TestDet.DetectorSize[1] * iy] << "   ";
	//	}
	//	std::cout << "\n";
	//}



	//{
	//	Options.Echo("Create BigMesh");
	//	ACMesh BigMesh;

	//	BigMesh.CreateBigMeshForDetector(TestDet, TestDet.DetectorSize[0] + 1);

	//	std::cout << "BiglMesh Size: " << BigMesh.Shape.Size_C << "\t; dq/dV: " << BigMesh.Shape.dq_per_Voxel << "\n";


	//	BigMesh.Options = &Options;

	//	Options.Echo("Create sparese detector");
	//	TestDet.CreateSparseHitList(3.2f, 6.4f);//
	//	Detector::AutoCorrFlags ACflags;
	//	ACflags.InterpolationMode = Settings::Interpolation::Linear;

	//	Options.Echo("check");
	//	TestDet.AutoCorrelateSparseList(BigMesh, ACflags);
	//}


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