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
#include <algorithm>

#include <CL/cl.hpp>
#include <thread>

#include "UnitTest.h"



//QD Test incl
#include <Eigen/SVD>


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
#include "PPP.h"
#include "MainRunModes.h"

#include "ExoticAnalysis.h"




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



	float Min = 0, Max = 0, Mean = 0;

	ArrayOperators::Min_Max_Mean_Value(Det.Intensity, Det.DetectorSize[0] * Det.DetectorSize[1], Min, Max, Mean);

	std::cout << "Min: " << Min << "    Max: " << Max << "    Mean: " << Mean << "\n";

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

	ACMesh smallCQMesh(&Options);
	smallCQMesh.CreateSmallMesh_CofQ_ForDetector(Det, MeshSize, QZoom);


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
	ACMesh MergedCq(&Options);
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
	ACMesh BigMesh(&Options);

	BigMesh.CreateBigMeshForDetector(Det, MeshSize, QZoom);

	std::cout << "BiglMesh Size: " << BigMesh.Shape.Size_C << "\t; dq/dV: " << BigMesh.Shape.dq_per_Voxel << "\n";


	BigMesh.Options = &Options;

	Options.Echo("AutoCorrelate");
	std::cout << Options.HitEvents.size() << " Events\n";

	profiler.Tic();

	for (unsigned int i = 0; i < Options.HitEvents.size(); i++) // Options.HitEvents.size()
	{
		if (i % 50 == 0)
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
		Det.AutoCorrelateSparseList(BigMesh, flags, flags, true, Options);

	}

	profiler.Toc(true);


	double* ACMesh = new double[BigMesh.Shape.Size_AB*BigMesh.Shape.Size_AB*BigMesh.Shape.Size_C]();
#pragma omp parallel for
	for (unsigned int i = 0; i < BigMesh.Shape.Size_AB*BigMesh.Shape.Size_AB*BigMesh.Shape.Size_C; i++)
	{
		ACMesh[i] = Options.IntToFloat(BigMesh.Mesh[i]);
	}


	ArrayOperators::SafeArrayToFile("/gpfs/cfel/cxi/scratch/user/trostfab/IACC_TESTSPACE/AC_UW_503.bin", ACMesh, BigMesh.Shape.Size_AB*BigMesh.Shape.Size_AB*BigMesh.Shape.Size_C, ArrayOperators::FileType::Binary);
	std::cout << "Saved as: /gpfs/cfel/cxi/scratch/user/trostfab/IACC_TESTSPACE/AC_UW_503.bin \n";

}



void CombineStuff(std::string Fr_AC_UW, std::string Fr_CQ, std::string Fw_AC, int size)
{
	//int size = 503 * 503 * 503;

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
	ArrayOperators::LoadArrayFromFile(Fr_CQ, CQ, size);


	std::cout << "\n Load AC ...\n";
	ArrayOperators::LoadArrayFromFile(Fr_AC_UW, AC, size);

	std::cout << "\n Apply C(q) ...\n";

#pragma omp parallel for
	for (int i = 0; i < size; i++)
	{
		if (CQ[i] <= 0)
			AC_Final[i] = 0;

		AC_Final[i] = AC[i] / CQ[i];
	}


	ArrayOperators::SafeArrayToFile(Fw_AC, AC_Final, size, ArrayOperators::FileType::Binary);
	std::cout << "Saved as: " << Fw_AC << "\n";

}


void Simulate(Settings & Options, std::string PixelMap_Path, std::string PixMapDataSet, int CrystSize)
{

	Simulator::SimulationSettings SimSettings;

	SimSettings.UnitCells[0] = CrystSize;
	SimSettings.UnitCells[1] = CrystSize;
	SimSettings.UnitCells[2] = CrystSize;

	SimSettings.AutoPixelOrientation = true;
	SimSettings.AutoPixelSize = true;

	SimSettings.NumberOfSimulations = 1; // 250000

	SimSettings.Modes = 1;
	//Detector coverage factor lib:   (see Wirkungsgrad.nb: SqDetCoverage[l_ (edge), dist_] := ArcTan[l ^ 2 / (2 * dist*Sqrt[4 * dist ^ 2 + 2 * l ^ 2])] / Pi; )
	//
	//     ******** - Jungfrau - ********
	// 0.0275 = 2.75% ~= Jungfr coverage at 120mm
	//     ******** - 350^2 - ********
	// 0.000152243 <= 350x350 50mum at 400mm
	// 0.000270554 <= 350x350 50mum at 300mm
	// 0.00241857  <= 350x350 50mum at 100mm
	// 0.00945996 <= 350x350 50mum at 50mm
	//     ******** - 250^2 - ********
	// 0.000198819 <= 250x250 50mum at 250mm
	// 0.000138095 <= 250x250 50mum at 300mm
	//     ******** - 500^2 - ********
	// 0.0187349 <= 500x500 100mum at 100mm
	//
	float NG = 10000.0; //1.0

	SimSettings.AveragePhotonesPerEmitterOnDetector = NG * 0.0187349; // *0.0275f;//0.0275 = 2.75% ~= Jungfr coverage at 120mm
	SimSettings.PoissonSample = true;
	SimSettings.SubSampling = 1; // 3 => (2*3+1)^2 = 49

	SimSettings.Wavelength = 1.94; //1.94A = 194pm Fe ;; 1.44A Zn

	SimSettings.Value_per_Photon = 1.0f;//6.4

	SimSettings.CrystSettings.FlYield = 1.0f; //0.1
	SimSettings.CrystSettings.Incoherent = true;
	SimSettings.CrystSettings.Isotropie = 1.0f;
	SimSettings.CrystSettings.RandOrientation = false;

	SimSettings.SaveResults = true;

	//SimSettings.Filename_Intensity = "/gpfs/cfel/cxi/scratch/user/trostfab/Simulation/SNR/HB_UC=" + std::to_string(SimSettings.UnitCells[0]) + "x" + std::to_string(SimSettings.UnitCells[1]) + "x" + std::to_string(SimSettings.UnitCells[2]) +  "_NG=" + std::to_string((int)NG) + "_NP=" + std::to_string(SimSettings.NumberOfSimulations) + ".h5";
	//SimSettings.Filename_XML       = "/gpfs/cfel/cxi/scratch/user/trostfab/Simulation/SNR/HB_UC=" + std::to_string(SimSettings.UnitCells[0]) + "x" + std::to_string(SimSettings.UnitCells[1]) + "x" + std::to_string(SimSettings.UnitCells[2]) +  "_NG=" + std::to_string((int)NG) + "_NP=" + std::to_string(SimSettings.NumberOfSimulations) + ".xml";

	
	//SimSettings.Filename_Intensity = "/gpfs/cfel/cxi/scratch/user/trostfab/Simulation/PlacementTest/165_MOF_UC=" + std::to_string(SimSettings.UnitCells[0]) + "x" + std::to_string(SimSettings.UnitCells[1]) + "x" + std::to_string(SimSettings.UnitCells[2]) +  "_NG=" + std::to_string((int)NG) + "_NP=" + std::to_string(SimSettings.NumberOfSimulations) + ".h5";
	//SimSettings.Filename_XML       = "/gpfs/cfel/cxi/scratch/user/trostfab/Simulation/PlacementTest/165_MOF_UC=" + std::to_string(SimSettings.UnitCells[0]) + "x" + std::to_string(SimSettings.UnitCells[1]) + "x" + std::to_string(SimSettings.UnitCells[2]) +  "_NG=" + std::to_string((int)NG) + "_NP=" + std::to_string(SimSettings.NumberOfSimulations) + ".xml";

	//**********************
	//SimSettings.Filename_Intensity = "/gpfs/cfel/cxi/scratch/user/trostfab/Simulation/SNR/100mm/SNR_NG100_" + std::to_string(SimSettings.UnitCells[0]) + "x" + std::to_string(SimSettings.UnitCells[1]) + "x" + std::to_string(SimSettings.UnitCells[2]) + ".h5";
	//      SimSettings.Filename_XML = "/gpfs/cfel/cxi/scratch/user/trostfab/Simulation/SNR/100mm/SNR_NG100_" + std::to_string(SimSettings.UnitCells[0]) + "x" + std::to_string(SimSettings.UnitCells[1]) + "x" + std::to_string(SimSettings.UnitCells[2]) + ".xml";
	//**********************

	SimSettings.Filename_Intensity = "/gpfs/cfel/cxi/scratch/user/trostfab/Simulation/EXAMPLE_Incoherent.h5";
	SimSettings.Filename_XML = "/gpfs/cfel/cxi/scratch/user/trostfab/Simulation/EXAMPLE_Incoherent.xml";


	//SimSettings.Filename_Intensity = "/gpfs/cfel/cxi/scratch/user/trostfab/Simulation/HbTest1/TestPixMa.h5";//Sim20_Fixed_NP_3
	//SimSettings.Filename_XML = "/gpfs/cfel/cxi/scratch/user/trostfab/Simulation/HbTest1/TestPixMa.xml";//Sim20_Fixed_NP_3
	//SimSettings.Filename_Intensity = "/gpfs/cfel/cxi/scratch/user/trostfab/Simulation/Sim_Jf_1000ppe_Y01_rdO_I1_1.h5";
	//SimSettings.Filename_XML = "/gpfs/cfel/cxi/scratch/user/trostfab/Simulation/Sim_Jf_1000ppe_Y01_rdO_I1_1.xml";
	SimSettings.Dataset = "data";


	//End Settings
	//Prepare crystal (here Hb)
	double LatticeVector[3][3];

	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			LatticeVector[i][j] = Options.MReference(i, j) * 10;//nm -> A 

			std::cout << LatticeVector[i][j] << "    ";
			//LatticeVector[i][j] = Options.MReference(i, j) / 1000.0; //convert nanometer to microns (same unit as Pixel-map)
		}
		std::cout << "\n";
	}
	std::vector<std::array<double, 3>> UnitCell; //Hardcode unitcell for Hb (1gzx)
	std::array<double, 3> t_pos;
	
	{
		//Hardcoded Simple
		t_pos = { 0 , 0 , 0 };
		UnitCell.push_back(t_pos);
	}
	//{
	//	//Hardcoded Hb
	//	t_pos = { 15.817 , 16.279 , 14.682 };
	//	UnitCell.push_back(t_pos);
	//	t_pos = { -10.262, -4.051 , -0.010 };
	//	UnitCell.push_back(t_pos);
	//	t_pos = { 6.437 , -16.819 , 12.649 };
	//	UnitCell.push_back(t_pos);
	//	t_pos = { 2.097 , 11.532 , 34.460 };
	//	UnitCell.push_back(t_pos);
	//}
	//{
	//	//Hardcoded MOF dummy
	//	t_pos = { 0 ,0 , 0 };
	//	UnitCell.push_back(t_pos);
	//	t_pos = { 10, 5 , 2 };
	//	UnitCell.push_back(t_pos);
	//	t_pos = { 7 , 4 , 5 };
	//	UnitCell.push_back(t_pos);
	//	t_pos = { 2.097 , 1.532 , 3.460 };
	//	UnitCell.push_back(t_pos);
	//	t_pos = { 5.817 , 6.279 , 4.682 };
	//	UnitCell.push_back(t_pos);
	//	//t_pos = { -1.262, -4.051 , -0.010 };
	//	//UnitCell.push_back(t_pos);
	//	//t_pos = { 6.437 , -6.819 , 2.649 };
	//	//UnitCell.push_back(t_pos);
	//}

	//t_pos = { 15.817 / 10000.0, 16.279 / 10000.0, 14.682 / 10000.0 }; //convert anström to microns
	//UnitCell.push_back(t_pos);
	//t_pos = { -10.262 / 10000.0, -4.051 / 10000.0, -0.010 / 10000.0 }; //convert anström to microns
	//UnitCell.push_back(t_pos);
	//t_pos = { 6.437 / 10000.0, -16.819 / 10000.0, 12.649 / 10000.0 }; //convert anström to microns
	//UnitCell.push_back(t_pos);
	//t_pos = { 2.097 / 10000.0, 11.532 / 10000.0, 34.460 / 10000.0 }; //convert anström to microns
	//UnitCell.push_back(t_pos);

	SimSettings.CrystalSize[0] = SimSettings.UnitCells[0] * sqrt(LatticeVector[0][0] * LatticeVector[0][0] + LatticeVector[1][0] * LatticeVector[1][0] + LatticeVector[2][0] * LatticeVector[2][0]);
	SimSettings.CrystalSize[1] = SimSettings.UnitCells[1] * sqrt(LatticeVector[0][1] * LatticeVector[0][1] + LatticeVector[1][1] * LatticeVector[1][1] + LatticeVector[2][1] * LatticeVector[2][1]);
	SimSettings.CrystalSize[2] = SimSettings.UnitCells[2] * sqrt(LatticeVector[0][2] * LatticeVector[0][2] + LatticeVector[1][2] * LatticeVector[1][2] + LatticeVector[2][2] * LatticeVector[2][2]);

	Crystal Cryst(LatticeVector, SimSettings.UnitCells, UnitCell);

	Detector Sim_Det;

	Sim_Det.LoadPixelMap(PixelMap_Path, PixMapDataSet );
	Sim_Det.Calc_kMap();

	Simulator Sim;
	Simulator::SimulationOutput Sim_Output;

	//Sim.Simulate(Cryst, Sim_Det, SimSettings, Sim_Output, Options);

	Sim.ParSimulate(Cryst, Sim_Det, SimSettings, Sim_Output, Options);

}

void QDSpeckleContr(Settings & Options)
{
	std::cout << "Do Speckle contrast stuff...\n";
	Detector Det;
	Det.LoadPixelMap("/home/trostfab/scratch/LU56/PixelMap_ePix_LU56_V1.h5", "PixelMap");
	Det.LoadPixelMask("/home/trostfab/scratch/LU56/PixelMask_ePix_V1.h5", "data/data");
	Options.LoadHitEventListFromFile("/home/trostfab/scratch/LU56/UnorientetGainCorrectedLAP.xml");


	Statistics::SpeckleContrastStatistics SCS;

	SCS = Statistics::GetSpeckleContrastStatistics(Options, Det, 0, Options.HitEvents.size());

	std::ofstream myfile;
	myfile.open("/home/trostfab/scratch/LU56/eval/SCS.csv");

	for (unsigned int i = 0; i < SCS.SCC_Statistics.size(); i++)
	{
		myfile << SCS.SCC_Statistics[i].MeanPhotonDensity << "; ";
		for (unsigned int j = 0; j < SCS.Nmax; j++)
		{
			if (SCS.SCC_Statistics[i].Probability.size() > j)
			{
				myfile << SCS.SCC_Statistics[i].Probability[j];
			}
			else
			{
				myfile << 0;
			}
			if (j < ( SCS.Nmax - 1) )
			{
				myfile << ", ";
			}
		}
		myfile << "\n";
	}
	myfile.close();
	std::cout << "Done";
}

void QDTests(Settings & Options)
{
	UnitTest UT;
	UT.TestFitting(Options);	
}

void runUnitTest(Settings & Options)
{
	UnitTest UT;

	Options.SetUp_OpenCL();

	//UT.TestKabschImplementation();
	//UT.StreamFileReader(Options);
	UT.TestACandCQmapping(Options, "/gpfs/cfel/cxi/scratch/user/trostfab/LU56/UT_lin_nn_UO.xml", 500, true);

}

int main(int argc, char** argv)
{
	std::cout << "\n\n\n======================================\n     IncohAutoCorrelate (IAC)\n======================================\n     Version "<< Settings::GetVersion() <<"\n" << std::endl;
	Settings Options;

	//Parse Arg for Run-Mode-Fork
	std::string Arg1= "h";
	if (argc > 1)
	{
		Arg1 = argv[1];
		std::transform(Arg1.begin(), Arg1.end(), Arg1.begin(), tolower);
	}
	
	if (Arg1 == "h" || Arg1 == "help" || Arg1 == "-help" || Arg1 == "-h" || Arg1 == "?" || Arg1 == "-?")
	{
		if (argc == 2)
		{
			std::cout << "This is a list of valid arguments. Use \"-h Arg\" to obtain more detailed information about \"Arg\". \n";

			std::cout << "\n---   General stuff   ---\n";
			std::cout << "help [h] \t: Displays a List of valid arguments.\n";

			std::cout << "\n---   Data processing mode   ---\n";


			std::cout << "\n---   Evaluation mode   ---\n";


			std::cout << "\n   --->   Scan Data   ---\n";
			std::cout << "-XMLfromH5 \t: -XMLfromH5 \"H5path1, Dataset1; H5path2, ...\" \"Output.xml\" \n";

			std::cout << "\n---   Simulation mode   ---\n";

		}
		else
		{
			std::string Arg2 = argv[2];
			std::transform(Arg2.begin(), Arg2.end(), Arg2.begin(), tolower);
			if (Arg2 == "h" || Arg2 == "help" || Arg2 == "-help" || Arg2 == "-h" || Arg2 == "?" || Arg2 == "-?")
			{
				std::cout << "Are you f... kidding me?\n";
			}
			else if (Arg2 == "xmlfromh5" || Arg2 == "-xmlfromh5")
			{
				std::cout << "-xmlfromh5 \"h5path1.h5,h5path2.h5,...;dataset1,dataset2,...\" \"XMLOutputPath.xml\" \n";
				std::cout << "Crawls through a list of hdf5-files and creates an xml-event entry for each data-slide found in the according dataset. As no orientation information is stored in a xml-file.\n";
			}

		}
		return 0;
	}
	else if (Arg1 == "xmlfromh5" || Arg1 == "-xmlfromh5")
	{
		if (argc < 4)
		{
			std::cerr << "-XMLfromH5 requires two additional arguments (\"H5path1, H5path2, ... ;Dataset1, Dataset2, ...\" \"Output.xml\")\n";
			return -1;
		}
		else 
		{
			std::string Arg2 = argv[2];
			std::string Arg3 = argv[3];
			return MainRunModes::Create_XMLHitlist_from_H5Stack_script(Arg2, Arg3, Options);
		}
	}
	else if (Arg1 == "evaluate" || Arg1 == "-evaluate" || Arg1 == "e" || Arg1 == "-e")
	{
		std::cout << "Run ICA in evaluation mode - autocorrelation\n";
		if (argc < 3)
		{
			std::cerr << "-Evaluate requires one additional argument (\"Settings.xml\")\n";
			return -1;
		}
		else
		{
			std::string Arg2 = argv[2];
			return MainRunModes::AutoCorrelateData(Arg2, Options);
		}
	}
	else if (Arg1 == "exampleconfig" || Arg1 == "-exampleconfig" || Arg1 == "ec" || Arg1 == "-ec")
	{
		if (argc < 3)
		{
			std::cerr << "-EvaluateConfig requires one additional argument (\"ExampleEvaluationSettings.xml\")\n";
			return -1;
		}
		else
		{
			std::string Arg2 = argv[2];
			return MainRunModes::Create_Example_Config_File(Arg2, Options);
		}
	}
	else if (Arg1 == "exampleevaluationconfig" || Arg1 == "-exampleevaluationconfig" || Arg1 == "eec" || Arg1 == "-eec")
	{
		if (argc < 3)
		{
			std::cerr << "-ExampleEvaluateConfig requires one additional argument (\"ExampleEvaluationSettings.xml\")\n";
			return -1;
		}
		else
		{
			std::string Arg2 = argv[2];
			return MainRunModes::Create_Example_Evaluation_Config_File(Arg2, Options);
		}
	}
	else if (Arg1 == "averageintensity" || Arg1 == "-averageintensity" || Arg1 == "avi" || Arg1 == "-avi")
	{
		if (argc < 3)
		{
			std::cerr << "-AverageIntensity requires one additional argument (\"EvaluationSettings.xml\")\n";
			std::cerr << "Additional Info: -AverageIntensity uses Photon thresholding, ensure you set it in \"EvaluationSettings.xml\" accordingly.\n";
			return -1;
		}
		else
		{
			std::string Arg2 = argv[2];
			return MainRunModes::AverageIntensity(Arg2, Options);
		}
	}
	else if (Arg1 == "pixelwisehistograms" || Arg1 == "-pixelwisehistograms" || Arg1 == "ph" || Arg1 == "-ph")
	{
		if (argc < 3)
		{
			std::cerr << "-PixelwiseHistograms requires one additional argument (\"Settings.xml\")\n";
			std::cerr << "Additional Info: -pixelwisehistograms uses the \"root.StatisticsSettings.PixelHistograms. (...)\" part.\n ";
			return -1;
		}
		else
		{
			std::string Arg2 = argv[2];
			return MainRunModes::CreateAllPixelHistograms(Arg2, Options);
		}
	}
	else if (Arg1 == "gainandlapcorrection" || Arg1 == "-gainandlapcorrection" || Arg1 == "glc" || Arg1 == "-glc")
	{
		std::cout << "run gain correction and LAP-photon counting\n";
		if (argc < 3)
		{
			std::cerr << "-GainAndLAPCorrection requires one additional argument (\"Settings.xml\")\n";
			std::cerr << "Additional Info: -pixelwisehistograms uses the \"root.StatisticsSettings.PixelHistograms. (...)\" part.\n ";
			return -1;
		}
		else
		{
			std::string Arg2 = argv[2];
			return MainRunModes::GainCorrectionAndLAP(Arg2, Options);
		}
	}
	else if (Arg1 == "gaincorrection" || Arg1 == "-gaincorrection" || Arg1 == "gc" || Arg1 == "-gc")
	{
		std::cout << "run gain correction\n";
		if (argc < 3)
		{
			std::cerr << "-GainPCorrection requires one additional argument (\"Settings.xml\")\n";
			//std::cerr << "Additional Info: -pixelwisehistograms uses the \"root.StatisticsSettings.PixelHistograms. (...)\" part.\n ";
			return -1;
		}
		else
		{
			std::string Arg2 = argv[2];
			return MainRunModes::GainCorrection(Arg2, Options);
		}
	}
	else if (Arg1 == "fastgaincorrection" || Arg1 == "-fastgaincorrection" || Arg1 == "fgc" || Arg1 == "-fgc")
	{
	std::cout << "run fast gain correction\n";

	if (argc < 8)
		{
		//Arg1: XML List - in
		//Arg2: Path GainMap.h5
		//Arg3: Dataset 'offset'
		//Arg4: Dataset 'gain'
		//Arg5: H5 Out
		//Arg6: H5 Dataset Out
		//Arg7: XML List - out


			std::cerr << "-FastGainCorrection requires the following arguments:\n";
			std::cerr << "* XML EventList input\n";
			std::cerr << "* GainMap path (.h5)\n";
			std::cerr << "* GainMap offset Dataset\n";
			std::cerr << "* GainMap gain Dataset\n";
			std::cerr << "* New H5File path\n";
			std::cerr << "* New H5File Dataset\n";
			std::cerr << "* New XML EventList path" << std::endl;
			return -1;
		}
	else
	{
		std::string Arg2 = argv[2];
		std::string Arg3 = argv[3];
		std::string Arg4 = argv[4];
		std::string Arg5 = argv[5];
		std::string Arg6 = argv[6];
		std::string Arg7 = argv[7];
		std::string Arg8 = argv[8];
		return MainRunModes::FastGainCorrection(Arg2, Arg3, Arg4, Arg5, Arg6, Arg7, Arg8, Options);
	}
	}
	else if (Arg1 == "darkcalibration" || Arg1 == "-darkcalibration" || Arg1 == "dc" || Arg1 == "-dc")
	{
		std::cout << "run dark calibration\n";
		if (argc < 3)
		{
			std::cerr << "-DarkCalibration requires one additional argument (\"Settings.xml\")\n";
			std::cerr << "Additional Info: -DarkCalibration uses the \"root.PatternPreProcessing.DarkFieldCorrection. (...)\" part.\n ";
			return -1;
		}
		else
		{
			std::string Arg2 = argv[2];
			return MainRunModes::DarkCalibration(Arg2, Options);
		}
	}
	else if (Arg1 == "fastdarkcalibration" || Arg1 == "-fastdarkcalibration" || Arg1 == "fdc" || Arg1 == "-fdc")
	{
		std::cout << "run (fast) dark calibration\n";
		if (argc < 7)
		{
			//required Arguments (5 -> 7 insg.):
			// XML List - in
			// Path Dark.h5
			// Dataset 'data'
			// H5 Out
			// XML List - out
			

			std::cerr << "-FastDarkCalibration requires the following arguments:\n";
			std::cerr << "* XML EventList input\n";
			std::cerr << "* Dark file path (.h5)\n";
			std::cerr << "* Dark file Dataset\n";
			std::cerr << "* New H5File path\n";
			std::cerr << "* New H5File Dataset\n";
			std::cerr << "* New XML EventList path" << std::endl;
			return -1;
		}
		else
		{
			std::string Arg2 = argv[2];
			std::string Arg3 = argv[3];
			std::string Arg4 = argv[4];
			std::string Arg5 = argv[5];
			std::string Arg6 = argv[6];
			std::string Arg7 = argv[7];

			return MainRunModes::FastDarkCalibration(Arg2, Arg3, Arg4, Arg5, Arg6, Arg7, Options);
		}
	}
	else if (Arg1 == "photongaussfit" || Arg1 == "-photongaussfit" || Arg1 == "pgf" || Arg1 == "-pgf")
	{
	std::cout << "run photon-gauss-fit algo\n";
	if (argc < 3)
	{
		std::cerr << "-PhotonGaussFit requires one additional argument (\"Settings.info\")\n";
		std::cerr << "Additional Info: -PhotonGaussFit uses the \"root.PatternPreProcessing.PhotonGaussFit. (...)\" part.\n ";
		return -1;
	}
	else
	{
		std::string Arg2 = argv[2];
		return MainRunModes::PhotonGaussFit(Arg2, Options);
	}
	}
	else if (Arg1 == "sortbyintensity" || Arg1 == "-sortbyintensity" || Arg1 == "si" || Arg1 == "-si")
	{
		if (argc < 4)
		{
			std::cerr << "-SortByIntensity requires two additional arguments (\"EventList.xml\" \"Output.csv\")\n";
			return -1;
		}
		else
		{
			std::string Arg2 = argv[2];
			std::string Arg3 = argv[3];
			return MainRunModes::SortHitsByMeanIntensity(Arg2, Arg3, Options);
		}
	}
	else if (Arg1 == "sortxmlbyintensity" || Arg1 == "-sortxmlbyintensity" || Arg1 == "six" || Arg1 == "-six")
	{
		if (argc < 4)
		{
			std::cerr << "-SortXmlByIntensity requires two additional arguments (\"EventList.xml\" \"SortedEventList.xmli\")\n";
			return -1;
		}
		else
		{
			std::string Arg2 = argv[2];
			std::string Arg3 = argv[3];
			return MainRunModes::SortXMLHitsByMeanIntensity(Arg2, Arg3, Options);
		}
	}
	else if (Arg1 == "xmlfromcsv" || Arg1 == "-xmlfromcsv" || Arg1 == "xfc" || Arg1 == "-xfc")
	{
		if (argc < 4)
		{
			std::cerr << "-XmlFromCsv requires two additional arguments (\"EventList_IN.CSV\" \"EventList_Out.xml\")\n";
			return -1;
		}
		else
		{
			std::string Arg2 = argv[2];
			std::string Arg3 = argv[3];
			return MainRunModes::GetHitListFromCSVFile (Arg2, Arg3, Options);
		}
	}
	else if (Arg1 == "xmlfromstreamfile" || Arg1 == "-xmlfromstreamfile" || Arg1 == "xfs" || Arg1 == "-xfs")
	{ //implement a reference matrix input for oritentation calculation!!!
		if (argc < 4)
		{
			std::cerr << "-XmlFromStreamFile requires three additional arguments (\"EventList_IN.stream\" \"H5-Path\" \"EventList_Out.xml\")\n";
			return -1;
		}
		else
		{
			//Options.MReference << 1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0;
			Options.MReference << 6.227, 0, 0, 0, 8.066, 0, 0, 0, 11.1; //LR17-Hb


			std::cout << "Use reference unit cell:\n";
			std::cout << Options.MReference(0, 0) << ", " << Options.MReference(0, 1) << ", " << Options.MReference(0, 2) << "\n";
			std::cout << Options.MReference(1, 0) << ", " << Options.MReference(1, 1) << ", " << Options.MReference(1, 2) << "\n";
			std::cout << Options.MReference(2, 0) << ", " << Options.MReference(2, 1) << ", " << Options.MReference(2, 2) << std::endl;

			std::string Arg2 = argv[2];
			std::string Arg3 = argv[3];
			std::string Arg4 = argv[4];
			return MainRunModes::GetHitListFromStreamFile(Arg2, Arg3, Arg4, Options);
		}
	}
	else if (Arg1 == "mergexmllists" || Arg1 == "-mergexmllists" || Arg1 == "mxl" || Arg1 == "-mxl")
	{
	std::cout << "Merge XML-Hit lits - mode " << std::endl;
		std::string Arg2;
		std::string Arg3;
		std::string Arg4; 
		if (argc < 4)
		{
			std::cerr << "-MergeXmlLists requires two additional arguments (\"HitListOut.xml\" \"List1.xml, List2.xml, ...\" )\n";
			std::cerr << "-MergeXmlLists can also have three additional arguments (\"HitListOut.xml\" \"List1.xml, List2.xml, ...\" \"SuplInfo1, SuplInfo2, ...\" )\n";
			return -1;
		}
		if (argc == 4)
		{
			Arg2 = argv[2];
			Arg3 = argv[3];
			Arg4 = "";
		}
		else
		{
			Arg2 = argv[2];
			Arg3 = argv[3];
			Arg4 = argv[4];
		}
		return MainRunModes::MergeXMLHitLits(Arg2, Arg3, Arg4, Options);
	}
	else if (Arg1 == "simulate" || Arg1 == "-simulate" || Arg1 == "s" || Arg1 == "-s")
	{
		std::cout << "Run ICA in simulation mode - autocorrelation" << std::endl;;
		if (argc < 3)
		{
			std::cerr << "-Simulate requires one additional argument (\"Settings.xml\")\n";
			return -1;
		}
		else
		{
			std::string Arg2 = argv[2];
			return MainRunModes::Simulate(Arg2, Options);
		}
	}
	else if (Arg1 == "generatepixelmap" || Arg1 == "-generatepixelmap" || Arg1 == "gpm" || Arg1 == "-gpm")
	{
	std::cout << "Run ICA in simulation mode - Generate PixelMap" << std::endl;;
	if (argc < 3)
	{
		std::cerr << "-GeneratePixelMap requires one additional argument (\"Settings.xml\")\n";
		return -1;
	}
	else
	{
		std::string Arg2 = argv[2];
		return MainRunModes::GeneratePixelMap(Arg2, Options);
	}
	}
	else if (Arg1 == "specklecontraststatistic" || Arg1 == "-specklecontraststatistic" || Arg1 == "scs" || Arg1 == "-scs")
	{
	std::cout << "Run ICA to generate \"speckle contrast statistics\"" << std::endl;
	if (argc < 3)
	{
		std::cerr << "-SpeckleContrastStatistic requires one additional argument (\"Settings.xml\")\n";
		return -1;
	}
	else
	{
		std::string Arg2 = argv[2];
		return MainRunModes::GenerateSpeckleContrastStatistics(Arg2, Options);
	}
	}
	else if (Arg1 == "chargesharingfit" || Arg1 == "-chargesharingfit" || Arg1 == "csf" || Arg1 == "-csf")
	{
	std::cout << "Run ICA to fit charge sharing radius of isolated photons" << std::endl;
	if (argc < 3)
	{
		std::cerr << "-ChargeSharingFit requires one additional argument (\"Settings.xml\")\n";
		return -1;
	}
	else
	{
		std::string Arg2 = argv[2];
		return MainRunModes::IsolatedPhotonChargeSharingFit(Arg2, Options);
	}
	}
	else if (Arg1 == "qdtest" || Arg1 == "-qdtest" )
	{
		QDTests(Options);
		return 0;
	}
	else if (Arg1 == "unittest" || Arg1 == "-unittest" || Arg1 == "ut" || Arg1 == "-ut")
	{
		runUnitTest(Options);
		return 0;
	}

	else if (Arg1 == "filterfor2photonevents" || Arg1 == "-filterfor2photonevents" )
	{
	std::cout << "Run ICA to filter Event-List for two photon events" << std::endl;
	if (argc < 4)
	{
		std::cerr << "-FilterFor2PhotonEvents requires two additional arguments (\"EventlistInput.info\", \"EventlistOutput.info\")\n";
		return -1;
	}

	Statistics::HitListFilterCondition Condition;
	Condition.TwoPhotonEvent = true;
	Statistics::FilterHitList(argv[2], argv[3], Condition);

	return 0;
	}


	else
	{
		std::cout << "Invalid argument: \"" << Arg1 << "\"\n";
		return 0;
	}


	return 0;
}