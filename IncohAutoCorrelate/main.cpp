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
#include <thread>


//my Classes
#include "Detector.h"
#include "Settings.h"
#include "ArrayOperators.h"
#include "ProfileTime.h"
#include <Eigen/Dense>
#include "Statistics.h"
#include "RunIAC.h"
#include "Crystal.h"
#include "Simulator.h"


void Test_CQ_small(Settings &Options, Detector &Det)
{

	int const MeshSize = 501;
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

	ArrayOperators::SafeArrayToFile("/gpfs/cfel/cxi/scratch/user/trostfab/IACC_TESTSPACE/Cq_503_small.bin", smallCQMesh.CQMesh, smallCQMesh.Shape.Size_AB*smallCQMesh.Shape.Size_AB*smallCQMesh.Shape.Size_C, ArrayOperators::FileType::Binary);
	std::cout << "Saved as: /gpfs/cfel/cxi/scratch/user/trostfab/IACC_TESTSPACE/Cq_503_small.bin \n";


	std::cout << "\n\n\nMerge and weight C(q):\n";
	ACMesh MergedCq;
	MergedCq.CreateBigMesh_CofQ_ForDetector(Det, MeshSize, QZoom);

	//Det.Merge_smallCofQ(MergedCq, smallCQMesh, Options.HitEvents, 0, 1000, Options, flags);
	Det.Merge_smallCofQ(MergedCq, smallCQMesh, Options.HitEvents, Options, flags);

	ArrayOperators::SafeArrayToFile("/gpfs/cfel/cxi/scratch/user/trostfab/IACC_TESTSPACE/Cq_503_Big.bin", MergedCq.CQMesh, MergedCq.Shape.Size_AB*MergedCq.Shape.Size_AB*MergedCq.Shape.Size_C, ArrayOperators::FileType::Binary);
	std::cout << "Saved as: /gpfs/cfel/cxi/scratch/user/trostfab/IACC_TESTSPACE/Cq_503_Big.bin \n";
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
	int const MeshSize = 501;
	float const QZoom = 1.0f;


	ProfileTime profiler;
	Options.Echo("Create BigMesh");
	ACMesh BigMesh;

	BigMesh.CreateBigMeshForDetector(Det, MeshSize, QZoom);

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
		Det.AutoCorrelateSparseList(BigMesh, flags,true);

	}

	profiler.Toc(true);

	
	double* ACMesh = new double[BigMesh.Shape.Size_AB*BigMesh.Shape.Size_AB*BigMesh.Shape.Size_C]();
	#pragma omp parallel for
	for (int i = 0; i < BigMesh.Shape.Size_AB*BigMesh.Shape.Size_AB*BigMesh.Shape.Size_C; i++)
	{
		ACMesh[i] = Options.IntToFloat(BigMesh.Mesh[i]);
	}



	ArrayOperators::SafeArrayToFile("/gpfs/cfel/cxi/scratch/user/trostfab/IACC_TESTSPACE/AC_UW_503.bin", ACMesh, BigMesh.Shape.Size_AB*BigMesh.Shape.Size_AB*BigMesh.Shape.Size_C, ArrayOperators::FileType::Binary);
	std::cout << "Saved as: /gpfs/cfel/cxi/scratch/user/trostfab/IACC_TESTSPACE/AC_UW_503.bin \n";

}


void CombineStuff()
{
	int size = 503 * 503 * 503;

	double * CQ = new double[size]();

	double * AC = new double[size]();

	//double * AC1 = new double[1003 * 1003 * 1003]();
	//double * AC2 = new double[1003 * 1003 * 1003]();
	//double * AC3 = new double[1003 * 1003 * 1003]();
	//double * AC4 = new double[1003 * 1003 * 1003]();
	//double * AC5 = new double[1003 * 1003 * 1003]();

	double * AC_Final = new double[size]();

	std::cout << "\n\n\n*************************\n";

	std::cout << "\n Load C(q)\n";
	ArrayOperators::LoadArrayFromFile("/gpfs/cfel/cxi/scratch/user/trostfab/IACC_TESTSPACE/Cq_503_Big.bin", CQ, size);

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
	ArrayOperators::LoadArrayFromFile("/gpfs/cfel/cxi/scratch/user/trostfab/IACC_TESTSPACE/AC_UW_503.bin", AC, size);

	std::cout << "\n Apply C(q) ...\n";

	#pragma omp parallel for
	for (int i = 0; i <size; i++)
	{
		if (CQ[i] <= 0)
			AC_Final[i] = 0;

		AC_Final[i] = AC[i] / CQ[i];
	}


	ArrayOperators::SafeArrayToFile("/gpfs/cfel/cxi/scratch/user/trostfab/IACC_TESTSPACE/TEST_AC_Final_503.bin", AC_Final, size, ArrayOperators::FileType::Binary);
	std::cout << "Saved as: /gpfs/cfel/cxi/scratch/user/trostfab/IACC_TESTSPACE/TEST_AC_Final_503.bin \n";

}


void Simulate(Settings & Options, std::string PixelMap_Path )
{
	//Custom Settings
	//Crystal Size (in Unitcells) 
	unsigned int CrystalSize[3];
	CrystalSize[0] = 50;
	CrystalSize[1] = 50;
	CrystalSize[2] = 50;
	//
	Simulator::SimulationSettings SimSettings;
	
	SimSettings.AutoPixelOrientation = true;
	SimSettings.AutoPixelSize = true;

	SimSettings.NumberOfSimulations = 100;

	SimSettings.Modes = 1;
	SimSettings.AveragePhotonesPerEmitterOnDetector = 1.0f;
	SimSettings.PoissonSample = true;
	SimSettings.SubSampling = 3;

	SimSettings.CrystSettings.FlYield = 1.0f;
	SimSettings.CrystSettings.Incoherent = true;
	SimSettings.CrystSettings.Isotropie = 1.0f;
	SimSettings.CrystSettings.RandOrientation = true;

	

	//End Settings
	//Prepare crystal (here Hb)
	double LatticeVector[3][3];

	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			LatticeVector[i][j] = Options.MReference(i, j) / 1000.0; //convert nanometer to microns (same unit as Pixel-map)
		}
	}
	std::vector<std::array<double, 3>> UnitCell; //Hardcode UC for Hb (1gzx)
	std::array<double, 3> t_pos;
	t_pos = { 15.817 / 10000.0, 16.279 / 10000.0, 14.682 / 10000.0 }; //convert anstr�m to microns
	UnitCell.push_back(t_pos);
	t_pos = { -10.262 / 10000.0, -4.051 / 10000.0, -0.010 / 10000.0 }; //convert anstr�m to microns
	UnitCell.push_back(t_pos);
	t_pos = { 6.437 / 10000.0, -16.819 / 10000.0, 12.649 / 10000.0 }; //convert anstr�m to microns
	UnitCell.push_back(t_pos);
	t_pos = { 2.097 / 10000.0, 11.532 / 10000.0, 34.460 / 10000.0 }; //convert anstr�m to microns
	UnitCell.push_back(t_pos);
	

	Crystal Cryst(LatticeVector, CrystalSize, UnitCell);

	Detector Sim_Det;

	Sim_Det.LoadPixelMap(PixelMap_Path, "data/data");
	Sim_Det.Calc_kMap();

	Simulator Sim;
	Simulator::SimulationOutput Sim_Output;
	
	Sim.Simulate(Cryst, Sim_Det, SimSettings, Sim_Output, Options);


}





int main()
{
	std::cout << "\n\n\n===================\n     IncohAutoCorrelate\n===================\n\n";
	//omp_set_nested(1);
	//omp_set_max_active_levels(2);


	

	ProfileTime profiler;
	Settings Options;
	Options.echo = true;

	Options.Echo("Set up OpenCl Devices");
	Options.SetUp_OpenCL();


	Options.F_I_Conversion.Step = 0.01f;
	//Hb reference-unit-cell in nm
	Options.MReference << 6.227, 0, 0, 0, 8.066, 0, 0, 0, 11.1;


	//1: AutoCorrelate

	int RunMode = 5; 
	switch (RunMode)
	{
	case 1: //Autocorrelate Hb Jungfrau 3fs
		std::cout << "\n******************************\nRun IncohAutoCorrelate in Autocorrelation-mode\n******************************\n";
	{
		const bool HitsFromXml = true; //otherwise from stream

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

		//Further Settings

		RunIAC::CreateCQ_Settings CQ_Settings;
		RunIAC::CreateAC_Settings AC_Settings;
		{
			CQ_Settings.AC_Merge_Flags.InterpolationMode = Settings::Interpolation::NearestNeighbour;
			CQ_Settings.AC_Small_Flags.InterpolationMode = Settings::Interpolation::NearestNeighbour;

			CQ_Settings.AVIntensity_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/IACC_TESTSPACE/IntensityAv_3fs_JF.bin";
			CQ_Settings.PixelMap_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/PixelMap/PixelMap_J.h5";
			CQ_Settings.PixelMap_DataSet = "data/data";
			CQ_Settings.PixelMask_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/PixelMap/PixelMask_thr03.bin";

			CQ_Settings.echo = true;

			CQ_Settings.MeshSize = 501;
			CQ_Settings.QZoom = 8.0f;

			CQ_Settings.SaveSmall_CQ = false;
			CQ_Settings.SaveBig_CQ = true;
			CQ_Settings.BigCQ_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/IACC_TESTSPACE/Cq_503-Z8_Big.bin";

			//ac shared settings
			AC_Settings.AC_FirstMap_Flags = CQ_Settings.AC_Small_Flags;
			AC_Settings.AC_SecondMap_Flags = CQ_Settings.AC_Merge_Flags;
			AC_Settings.MeshSize = CQ_Settings.MeshSize;
			AC_Settings.QZoom = CQ_Settings.QZoom;
			AC_Settings.PixelMap_Path = CQ_Settings.PixelMap_Path;
			AC_Settings.PixelMap_DataSet = CQ_Settings.PixelMap_DataSet;
			AC_Settings.PixelMask_Path = CQ_Settings.PixelMask_Path;

			//ac further settings
			AC_Settings.SaveBig_AC = true;
			AC_Settings.BigAC_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/IACC_TESTSPACE/AC_UW_503-Z8_Big.bin";

			AC_Settings.DoubleMap = true;
			AC_Settings.echo = true;
			AC_Settings.PhotonOffset = 3.2f;
			AC_Settings.PhotonStep = 6.4f;
		}

		//

		ACMesh CQ;
		ACMesh AC;

		profiler.Tic();

		Options.Echo("Launch threads");

		RunIAC::Create_CQ_Mesh(CQ, CQ_Settings, Options);
		RunIAC::Run_AC_UW(AC, AC_Settings, Options);



		Options.Echo("Merge Stuff");

		double * FinalAC = new double[CQ.Shape.Size_AB*CQ.Shape.Size_AB*CQ.Shape.Size_AB];

		RunIAC::Merge_ACandCQ(FinalAC, AC, CQ, Options);

		ArrayOperators::SafeArrayToFile("/gpfs/cfel/cxi/scratch/user/trostfab/IACC_TESTSPACE/AC_Final_503-Z8.bin", FinalAC, CQ.Shape.Size_AB*CQ.Shape.Size_AB*CQ.Shape.Size_AB, ArrayOperators::FileType::Binary);
		std::cout << "Saved as: /gpfs/cfel/cxi/scratch/user/trostfab/IACC_TESTSPACE/AC_Final_503-Z8.bin \n";



		std::cout << "\n\n Finished in ";
		profiler.Toc(true);
		return 0;
	}
		break;
	case 5: //Autocorrelate Single Molecule Jungfrau 120mm
		std::cout << "\n******************************\nRun IncohAutoCorrelate in Autocorrelation-mode for single molecule\n******************************\n";
	{
		RunIAC::CreateSM_Settings SM_Settings;

		//SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_139-xes.h5");
		//SM_Settings.H5Dataset.push_back("/mfxlr1716/run139/Jungfrau1M/calib/data");

		//SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_140-xes.h5");
		//SM_Settings.H5Dataset.push_back("/mfxlr1716/run140/Jungfrau1M/calib/data");

		//SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_141-xes.h5");
		//SM_Settings.H5Dataset.push_back("/mfxlr1716/run141/Jungfrau1M/calib/data");

		//SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_142-xes.h5");
		//SM_Settings.H5Dataset.push_back("/mfxlr1716/run142/Jungfrau1M/calib/data");

		//SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_143-xes.h5");
		//SM_Settings.H5Dataset.push_back("/mfxlr1716/run143/Jungfrau1M/calib/data");

		//SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_146-xes.h5");
		//SM_Settings.H5Dataset.push_back("/mfxlr1716/run146/Jungfrau1M/calib/data");

		//SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_147-xes.h5");
		//SM_Settings.H5Dataset.push_back("/mfxlr1716/run147/Jungfrau1M/calib/data");

		//SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_159-xes.h5");
		//SM_Settings.H5Dataset.push_back("/mfxlr1716/run159/Jungfrau1M/calib/data");

		//SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_160-xes.h5");
		//SM_Settings.H5Dataset.push_back("/mfxlr1716/run160/Jungfrau1M/calib/data");

		//SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_161-xes.h5");
		//SM_Settings.H5Dataset.push_back("/mfxlr1716/run161/Jungfrau1M/calib/data");

		//SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_162-xes.h5");
		//SM_Settings.H5Dataset.push_back("/mfxlr1716/run162/Jungfrau1M/calib/data");

		//SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_163-xes.h5");
		//SM_Settings.H5Dataset.push_back("/mfxlr1716/run163/Jungfrau1M/calib/data");

		//SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_164-xes.h5");
		//SM_Settings.H5Dataset.push_back("/mfxlr1716/run164/Jungfrau1M/calib/data");

		//SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_165-xes.h5");
		//SM_Settings.H5Dataset.push_back("/mfxlr1716/run165/Jungfrau1M/calib/data");

		//SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_166-xes.h5");
		//SM_Settings.H5Dataset.push_back("/mfxlr1716/run166/Jungfrau1M/calib/data");

		//SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_167-xes.h5");
		//SM_Settings.H5Dataset.push_back("/mfxlr1716/run167/Jungfrau1M/calib/data");

		//SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_168-xes.h5");
		//SM_Settings.H5Dataset.push_back("/mfxlr1716/run168/Jungfrau1M/calib/data");

		//SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_169-xes.h5");
		//SM_Settings.H5Dataset.push_back("/mfxlr1716/run169/Jungfrau1M/calib/data");

		SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_203-xes.h5");
		SM_Settings.H5Dataset.push_back("/mfxlr1716/run203/Jungfrau1M/calib/data");

		SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_204-xes.h5");
		SM_Settings.H5Dataset.push_back("/mfxlr1716/run204/Jungfrau1M/calib/data");

		SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_205-xes.h5");
		SM_Settings.H5Dataset.push_back("/mfxlr1716/run205/Jungfrau1M/calib/data");

		SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_206-xes.h5");
		SM_Settings.H5Dataset.push_back("/mfxlr1716/run206/Jungfrau1M/calib/data");

		SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_207-xes.h5");
		SM_Settings.H5Dataset.push_back("/mfxlr1716/run207/Jungfrau1M/calib/data");

		SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_208-xes.h5");
		SM_Settings.H5Dataset.push_back("/mfxlr1716/run208/Jungfrau1M/calib/data");

		SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_209-xes.h5");
		SM_Settings.H5Dataset.push_back("/mfxlr1716/run209/Jungfrau1M/calib/data");

		//SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_253-xes.h5");
		//SM_Settings.H5Dataset.push_back("/mfxlr1716/run253/Jungfrau1M/calib/data");

		//SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_255-xes.h5");
		//SM_Settings.H5Dataset.push_back("/mfxlr1716/run255/Jungfrau1M/calib/data");

		//SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_256-xes.h5");
		//SM_Settings.H5Dataset.push_back("/mfxlr1716/run256/Jungfrau1M/calib/data");

		//SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_257-xes.h5");
		//SM_Settings.H5Dataset.push_back("/mfxlr1716/run257/Jungfrau1M/calib/data");

		//SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_258-xes.h5");
		//SM_Settings.H5Dataset.push_back("/mfxlr1716/run258/Jungfrau1M/calib/data");

		//SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_263-xes.h5");
		//SM_Settings.H5Dataset.push_back("/mfxlr1716/run263/Jungfrau1M/calib/data");

		//SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_264-xes.h5");
		//SM_Settings.H5Dataset.push_back("/mfxlr1716/run264/Jungfrau1M/calib/data");

		//SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_265-xes.h5");
		//SM_Settings.H5Dataset.push_back("/mfxlr1716/run265/Jungfrau1M/calib/data");

		//SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_266-xes.h5");
		//SM_Settings.H5Dataset.push_back("/mfxlr1716/run266/Jungfrau1M/calib/data");

		//SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_267-xes.h5");
		//SM_Settings.H5Dataset.push_back("/mfxlr1716/run267/Jungfrau1M/calib/data");

		//SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_268-xes.h5");
		//SM_Settings.H5Dataset.push_back("/mfxlr1716/run268/Jungfrau1M/calib/data");


//Test (BED-Noise)


		//SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/user/trostfab/SM_Noise_Sim/NoiseStack0.cxi");
		//SM_Settings.H5Dataset.push_back("/data");

		//SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/user/trostfab/SM_Noise_Sim/NoiseStack1.cxi");
		//SM_Settings.H5Dataset.push_back("/data");

		//SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/user/trostfab/SM_Noise_Sim/NoiseStack2.cxi");
		//SM_Settings.H5Dataset.push_back("/data");

//

		SM_Settings.PixelMap_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/PixelMap/PixelMap_J.h5";
		SM_Settings.PixelMap_DataSet = "data/data";
		SM_Settings.PixelMask_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/PixelMap/PixelMask_thr03.bin";

		SM_Settings.Output_AV_Int_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/NanoStar/Block3_HR_avINT.bin";
		SM_Settings.Output_CQ_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/NanoStar/Block3_HR_CQ.bin";
		SM_Settings.Output_ACUW_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/NanoStar/Block3_HR_ACuw.bin";
		SM_Settings.Output_AC_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/NanoStar/Block3_HR_AC.bin";
		SM_Settings.Output_Q_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/NanoStar/Block3_HR_Q.bin";



		
		SM_Settings.ArraySize = 3000;

		AC1D Results;

		RunIAC::Run_AC_SM_Full(Results, SM_Settings, Options);


	}
	break;
	case 6: //Autocorrelate Single Molecule Jungfrau 500mm
	std::cout << "\n******************************\nRun IncohAutoCorrelate in Autocorrelation-mode for single molecule, Jungfr. 500mm \n******************************\n";
	{
		RunIAC::CreateSM_Settings SM_Settings;

		SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_111-xes.h5");
		SM_Settings.H5Dataset.push_back("/mfxlr1716/run111/Jungfrau1M/calib/data");

		SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_119-xes.h5");
		SM_Settings.H5Dataset.push_back("/mfxlr1716/run119/Jungfrau1M/calib/data");

		SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_120-xes.h5");
		SM_Settings.H5Dataset.push_back("/mfxlr1716/run120/Jungfrau1M/calib/data");

		SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_121-xes.h5");
		SM_Settings.H5Dataset.push_back("/mfxlr1716/run121/Jungfrau1M/calib/data");

		SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_122-xes.h5");
		SM_Settings.H5Dataset.push_back("/mfxlr1716/run122/Jungfrau1M/calib/data");

		SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_123-xes.h5");
		SM_Settings.H5Dataset.push_back("/mfxlr1716/run123/Jungfrau1M/calib/data");


		//

		SM_Settings.PixelMap_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/PixelMap/PixelMap_J.h5";
		SM_Settings.PixelMap_DataSet = "data/data";
		SM_Settings.PixelMask_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/PixelMap/PixelMask_thr03.bin";

		SM_Settings.Output_AV_Int_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/NanoStar/J500_avINT.bin";
		SM_Settings.Output_CQ_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/NanoStar/J500_CQ.bin";
		SM_Settings.Output_ACUW_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/NanoStar/J500_ACuw.bin";
		SM_Settings.Output_AC_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/NanoStar/J500_AC.bin";
		SM_Settings.Output_Q_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/NanoStar/J500_Q.bin";




		SM_Settings.ArraySize = 1000;

		AC1D Results;

		RunIAC::Run_AC_SM_Full(Results, SM_Settings, Options);


	}
	break;

	case 7: //Autocorrelate Single Molecule Epix 300mm
		std::cout << "\n******************************\nRun IncohAutoCorrelate in Autocorrelation-mode for single molecule EPIX\n******************************\n";
		{
			RunIAC::CreateSM_Settings SM_Settings;

			//SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_139-xes.h5");
			//SM_Settings.H5Dataset.push_back("/mfxlr1716/run139/Epix100a/calib/data");

			//SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_140-xes.h5");
			//SM_Settings.H5Dataset.push_back("/mfxlr1716/run140/Epix100a/calib/data");

			//SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_141-xes.h5");
			//SM_Settings.H5Dataset.push_back("/mfxlr1716/run141/Epix100a/calib/data");

			//SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_142-xes.h5");
			//SM_Settings.H5Dataset.push_back("/mfxlr1716/run142/Epix100a/calib/data");

			//SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_143-xes.h5");
			//SM_Settings.H5Dataset.push_back("/mfxlr1716/run143/Epix100a/calib/data");

			//SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_146-xes.h5");
			//SM_Settings.H5Dataset.push_back("/mfxlr1716/run146/Epix100a/calib/data");

			//SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_147-xes.h5");
			//SM_Settings.H5Dataset.push_back("/mfxlr1716/run147/Epix100a/calib/data");

			//SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_159-xes.h5");
			//SM_Settings.H5Dataset.push_back("/mfxlr1716/run159/Epix100a/calib/data");

			//SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_160-xes.h5");
			//SM_Settings.H5Dataset.push_back("/mfxlr1716/run160/Epix100a/calib/data");

			//SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_161-xes.h5");
			//SM_Settings.H5Dataset.push_back("/mfxlr1716/run161/Epix100a/calib/data");

			//SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_162-xes.h5");
			//SM_Settings.H5Dataset.push_back("/mfxlr1716/run162/Epix100a/calib/data");

			//SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_163-xes.h5");
			//SM_Settings.H5Dataset.push_back("/mfxlr1716/run163/Epix100a/calib/data");

			//SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_164-xes.h5");
			//SM_Settings.H5Dataset.push_back("/mfxlr1716/run164/Epix100a/calib/data");

			//SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_165-xes.h5");
			//SM_Settings.H5Dataset.push_back("/mfxlr1716/run165/Epix100a/calib/data");

			//SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_166-xes.h5");
			//SM_Settings.H5Dataset.push_back("/mfxlr1716/run166/Epix100a/calib/data");

			//SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_167-xes.h5");
			//SM_Settings.H5Dataset.push_back("/mfxlr1716/run167/Epix100a/calib/data");

			//SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_168-xes.h5");
			//SM_Settings.H5Dataset.push_back("/mfxlr1716/run168/Epix100a/calib/data");

			//SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_169-xes.h5");
			//SM_Settings.H5Dataset.push_back("/mfxlr1716/run169/Epix100a/calib/data");

			SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_203-xes.h5");
			SM_Settings.H5Dataset.push_back("/mfxlr1716/run203/Epix100a/calib/data");

			SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_204-xes.h5");
			SM_Settings.H5Dataset.push_back("/mfxlr1716/run204/Epix100a/calib/data");

			SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_205-xes.h5");
			SM_Settings.H5Dataset.push_back("/mfxlr1716/run205/Epix100a/calib/data");

			SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_206-xes.h5");
			SM_Settings.H5Dataset.push_back("/mfxlr1716/run206/Epix100a/calib/data");

			SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_207-xes.h5");
			SM_Settings.H5Dataset.push_back("/mfxlr1716/run207/Epix100a/calib/data");

			SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_208-xes.h5");
			SM_Settings.H5Dataset.push_back("/mfxlr1716/run208/Epix100a/calib/data");

			SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_209-xes.h5");
			SM_Settings.H5Dataset.push_back("/mfxlr1716/run209/Epix100a/calib/data");

			//SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_253-xes.h5");
			//SM_Settings.H5Dataset.push_back("/mfxlr1716/run253/Epix100a/calib/data");

			//SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_255-xes.h5");
			//SM_Settings.H5Dataset.push_back("/mfxlr1716/run255/Epix100a/calib/data");

			//SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_256-xes.h5");
			//SM_Settings.H5Dataset.push_back("/mfxlr1716/run256/Epix100a/calib/data");

			//SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_257-xes.h5");
			//SM_Settings.H5Dataset.push_back("/mfxlr1716/run257/Epix100a/calib/data");

			//SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_258-xes.h5");
			//SM_Settings.H5Dataset.push_back("/mfxlr1716/run258/Epix100a/calib/data");

			//SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_263-xes.h5");
			//SM_Settings.H5Dataset.push_back("/mfxlr1716/run263/Epix100a/calib/data");

			//SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_264-xes.h5");
			//SM_Settings.H5Dataset.push_back("/mfxlr1716/run264/Epix100a/calib/data");

			//SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_265-xes.h5");
			//SM_Settings.H5Dataset.push_back("/mfxlr1716/run265/Epix100a/calib/data");

			//SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_266-xes.h5");
			//SM_Settings.H5Dataset.push_back("/mfxlr1716/run266/Epix100a/calib/data");

			//SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_267-xes.h5");
			//SM_Settings.H5Dataset.push_back("/mfxlr1716/run267/Epix100a/calib/data");

			//SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_268-xes.h5");
			//SM_Settings.H5Dataset.push_back("/mfxlr1716/run268/Epix100a/calib/data");


			//Test (BED-Noise)


			//SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/user/trostfab/SM_Noise_Sim/NoiseStack0.cxi");
			//SM_Settings.H5Dataset.push_back("/data");

			//SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/user/trostfab/SM_Noise_Sim/NoiseStack1.cxi");
			//SM_Settings.H5Dataset.push_back("/data");

			//SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/user/trostfab/SM_Noise_Sim/NoiseStack2.cxi");
			//SM_Settings.H5Dataset.push_back("/data");

			//

			SM_Settings.PixelMap_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/PixelMap/PixelMap_J.h5";
			SM_Settings.PixelMap_DataSet = "data/data";
		//	SM_Settings.PixelMask_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/PixelMap/PixelMask_thr03.bin";

			SM_Settings.Output_AV_Int_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/NanoStar/EPIX/Block3_HR_avINT.bin";
			SM_Settings.Output_CQ_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/NanoStar/EPIX/Block3_HR_CQ.bin";
			SM_Settings.Output_ACUW_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/NanoStar/EPIX/Block3_HR_ACuw.bin";
			SM_Settings.Output_AC_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/NanoStar/EPIX/Block3_HR_AC.bin";
			SM_Settings.Output_Q_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/NanoStar/EPIX/Block3_HR_Q.bin";




			SM_Settings.ArraySize = 1500;

			AC1D Results;

			SM_Settings.JungfrDet = false; //to not expect two panels
			RunIAC::Run_AC_SM_Full(Results, SM_Settings, Options);


		}
		break;

	case 10: //Simmulate
	{
		std::cout << "\n******************************\nRun IncohAutoCorrelate in Simulation-mode\n******************************\n";
		{
			std::string PixelMap_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/PixelMap/PixelMap_J.h5";
			Simulate(Options, PixelMap_Path);
		}
	}
	break;
	}//end switch


	std::cout << "Ended.";
	return 0;

	////Run Stiings:

	//const bool HitsFromXml = true; //otherwise from stream


	////****************************************************************************************************************************************************
	////Basic Startup (needed for [nearly] everything )
	////
	//int end;
	//ProfileTime profiler;
	//Settings Options;
	//Options.echo = true;

	//Options.F_I_Conversion.Step = 0.01f;
	//
	//Options.Echo("Load Open CL stuff:\n");

	//Options.SetUp_OpenCL();


	//Detector TestDet;

	//TestDet.LoadPixelMap("/gpfs/cfel/cxi/scratch/user/trostfab/PixelMap/PixelMap_J.h5", "data/data");

	//Options.Echo("Create k-map");
	//TestDet.Calc_kMap();

	//std::cout << "Max k = [" << TestDet.Max_k[0] << "; " << TestDet.Max_k[1] << "; " << TestDet.Max_k[2] << "]\n";
	//std::cout << "Min k = [" << TestDet.Min_k[0] << "; " << TestDet.Min_k[1] << "; " << TestDet.Min_k[2] << "]\n";
	//std::cout << "Max q = [" << TestDet.Max_q[0] << "; " << TestDet.Max_q[1] << "; " << TestDet.Max_q[2] << "]\n";

	//Options.Echo("\nLoad pixel-mask");
	//TestDet.PixelMask = new int[1024 * 1024]();
	//ArrayOperators::LoadArrayFromFile<int>("/gpfs/cfel/cxi/scratch/user/trostfab/PixelMap/PixelMask_thr03.bin", TestDet.PixelMask,1024*1024);




	//
	////Hb reference-unit-cell
	//Options.MReference << 6.227, 0, 0, 0, 8.066, 0, 0, 0, 11.1;

	//if (HitsFromXml)
	//{
	//	Options.Echo("Load Events from XML");
	//	Options.LoadHitEventListFromFile("/gpfs/cfel/cxi/scratch/user/trostfab/IACC_TESTSPACE/HitEventList_3fs_Jungfr.xml");
	//}
	//else
	//{
	//	Options.Echo("Load Streamfile");
	//	Options.LoadStreamFile("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/indexing/3fs_JF.stream", "entry_1/instrument_1/detector_2/detector_corrected/data", false);
	//}


	////****************************************************************************************************************************************************


	//


	//Test_CQ_small(Options, TestDet);

	//AutoCorrelateEvents(Options, TestDet);


	//CombineStuff();

	///*std::cout << "Program ended\n";
	//std::cin >> end;*/
	//return 0;






	//








	////	std::cout << "Program ended\n";
	////	std::cin >> end;
	////	return 0;
	////}






	//
	////Options.Echo("Load intensity data for Hit 0");
	////TestDet.LoadIntensityData(&Options.HitEvents[0]);

	////std::cout << "\n";
	////for (int iy = 0; iy <20; iy++)
	////{
	////	for (int ix = 0; ix <20; ix++)
	////	{
	////		std::cout << TestDet.Intensity[ix + TestDet.DetectorSize[1] * iy] << "   ";
	////	}
	////	std::cout << "\n";
	////}



	////{
	////	Options.Echo("Create BigMesh");
	////	ACMesh BigMesh;

	////	BigMesh.CreateBigMeshForDetector(TestDet, TestDet.DetectorSize[0] + 1);

	////	std::cout << "BiglMesh Size: " << BigMesh.Shape.Size_C << "\t; dq/dV: " << BigMesh.Shape.dq_per_Voxel << "\n";


	////	BigMesh.Options = &Options;

	////	Options.Echo("Create sparese detector");
	////	TestDet.CreateSparseHitList(3.2f, 6.4f);//
	////	Detector::AutoCorrFlags ACflags;
	////	ACflags.InterpolationMode = Settings::Interpolation::Linear;

	////	Options.Echo("check");
	////	TestDet.AutoCorrelateSparseList(BigMesh, ACflags);
	////}


	////Options.Echo("AutoCorrelate");

	////profiler.Tic();

	////for (int i = 0; i < 1000; i++)
	////{
	////	TestDet.LoadIntensityData(&Options.HitEvents[i]);
	////	TestDet.CreateSparseHitList(3.0);

	////	Detector::AutoCorrFlags flags;
	////	flags.InterpolationMode = Settings::Interpolation::NearestNeighbour;
	////	TestDet.AutoCorrelateSparseList(BigMesh, flags);

	////}
	////
	////profiler.Toc(true);




	//






	//Options.Echo("Create C(q) - Mesh");

	//ACMesh CQMesh;
	//CQMesh.CreateBigMesh_CofQ_ForDetector(TestDet, 1025);


	//std::vector<Settings::HitEvent> SmallEventList;
	//for (unsigned int i = 0; i < 1000; i++)
	//{
	//	SmallEventList.push_back(Options.HitEvents[i]);
	//}

	//Detector::AutoCorrFlags flags;
	//flags.InterpolationMode = Settings::Interpolation::NearestNeighbour;
	//TestDet.AutoCorrelate_CofQ(CQMesh, flags, Options.HitEvents, 0, 3, Options);

	////int ind = 0;
	////for (int i = 0; i < 10; i++)
	////{
	////	for (int j = 0; j < 10; j++)
	////	{
	////		std::cout << Options.HitEvents[ind].PhotonCount << " : " << Options.HitEvents[ind].MeanIntensity << "\t";
	////		ind++;
	////	}
	////	std::cout << "\n";
	////}
	////std::cout << "\n\n\n";

	////for (int iy = 0; iy <10; iy++)
	////{
	////	for (int ix = 0; ix <10; ix++)
	////	{
	////		std::cout << TestDet.Intensity[ix + TestDet.DetectorSize[1] * iy] << "\t";
	////	}
	////	std::cout << "\n";
	////}
	////std::cout << "\n\n\n";

	////double MPI = 0;
	////double MinPi = 99999;
	////double MaxPi = 0;
	////for (int i = 0; i < 5000; i++)
	////{
	////	MPI += Options.HitEvents[i].PhotonCount;
	////	if (Options.HitEvents[i].PhotonCount > MaxPi)
	////		MaxPi = Options.HitEvents[i].PhotonCount;
	////	if (Options.HitEvents[i].PhotonCount < MinPi)
	////		MinPi = Options.HitEvents[i].PhotonCount;
	////}
	////MPI = MPI / 10000;

	////for (int iy = 0; iy <10; iy++)
	////{
	////	for (int ix = 0; ix <10; ix++)
	////	{
	////		std::cout << TestDet.Intensity[ix + TestDet.DetectorSize[1] * iy] * (MPI) << "\t";
	////	}
	////	std::cout << "\n";
	////}

	////std::cout << "Max PI: " << MaxPi << "\t Min PI: " << MinPi << "\t M PI: " << MPI << "\n";
	//
	//std::cout << "Program ended\n";
	//std::cin >> end;
 //   return 0;
}