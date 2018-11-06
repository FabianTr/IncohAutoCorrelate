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

	for (int i = 0; i < Options.HitEvents.size(); i++) // Options.HitEvents.size()
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
		Det.AutoCorrelateSparseList(BigMesh, flags, true, Options);

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

	for (int i = 0; i < SCS.SCC_Statistics.size(); i++)
	{
		myfile << SCS.SCC_Statistics[i].MeanPhotonDensity << "; ";
		for (int j = 0; j < SCS.Nmax; j++)
		{
			if (SCS.SCC_Statistics[i].Probability.size() > j)
			{
				myfile << SCS.SCC_Statistics[i].Probability[j];
			}
			else
			{
				myfile << 0;
			}
			if (j < SCS.Nmax - 1)
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

	std::cout << "Running Cluster Stuff\n\n";

	Options.SetUp_OpenCL();

	MainRunModes::AllSettings InputSettings = MainRunModes::LoadSettings("/home/trostfab/scratch/LU56/ClusterTest/Settings_Clusters.xml",Options);

	std::string OutPath = "/home/trostfab/scratch/LU56/ClusterTest/Clusters.csv";

	std::vector<std::vector<unsigned int>> Clusters;
	std::vector<std::string> ClHeader;

	//Ghost Cluster
	std::vector<unsigned int> GhostCluster;
	GhostCluster.push_back(276 + 211 * 553 + 275 * 553 * 553);
	GhostCluster.push_back(276 + 210 * 553 + 275 * 553 * 553);
	GhostCluster.push_back(276 + 209 * 553 + 275 * 553 * 553);
	GhostCluster.push_back(276 + 208 * 553 + 275 * 553 * 553);
	GhostCluster.push_back(277 + 208 * 553 + 275 * 553 * 553);
	GhostCluster.push_back(277 + 209 * 553 + 275 * 553 * 553);

	Clusters.push_back(GhostCluster);
	ClHeader.push_back("Ghost");

	//Cental Peak (CP)
	Clusters.push_back({ 276 + 276 * 553 + 276 * 553 * 553 });
	ClHeader.push_back("CP (276,276,276)");
	//Clusters around CP
	for (unsigned int ms = 273; ms <= 276; ms++)
	{
		for (unsigned int ss = 273; ss <= 279; ss++)
		{
			if (ms == ss && ss == 276) //exclude CP here
				continue;

			Clusters.push_back({ 276 + ms * 553 + ss * 553 * 553 });
			ClHeader.push_back("(276," + std::to_string(ms) + "," + std::to_string(ms)+ ")");
		}
	}

	ExoticAnalysis::CompareClustersInAC_PatternByPattern(Options, InputSettings.EvaluationSettings, Clusters, OutPath, ClHeader, true);

	
}



int main(int argc, char** argv)
{
	std::cout << "\n\n\n===================\n     IncohAutoCorrelate (IAC)\n===================\n\n";
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
		if (argc = 2)
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
			std::cerr << "-XMLfromH5 requires two additional arguments (\"H5path1, Dataset1; H5path2, ...\" \"Output.xml\")\n";
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
			std::cerr << "Additional Info: -pixelwisehistograms uses the \"root.StatisticsSettings.PixelHistograms. (...)\" part.\n ";
			return -1;
		}
		else
		{
			std::string Arg2 = argv[2];
			return MainRunModes::GainCorrection(Arg2, Options);
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
	else if (Arg1 == "qdtest" || Arg1 == "-qdtest" )
	{
		QDTests(Options);
		return 0;
	}
	else
	{
		std::cout << "Invalid argument: \"" << Arg1 << "\"\n";
		return 0;
	}




	 

	//Old Stuff -> TODO: Integrate into Mode-Fork

	//QDTests();
	return 0;
	//omp_set_nested(1);
	//omp_set_max_active_levels(2);

	int CrystSize = 100;
	if (argc > 1)
	{
		CrystSize = std::stoi(argv[1]);
	}
	std::cout << "Unitcells: " << CrystSize << "^3\n";



	ProfileTime profiler;
	Options.echo = true;

	Options.Echo("Set up OpenCl Devices");
	Options.SetUp_OpenCL();


	Options.F_I_Conversion.Step = 0.01f;
	Options.F_I_Conversion.Offset = 0.0f;
	////Hb reference-unit-cell in nm
	//Options.MReference << 6.227, 0, 0, 0, 8.066, 0, 0, 0, 11.1;

	//Simple
	Options.MReference << 1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0;
	////MOF
	//Options.MReference << 2.6, 0, 0, 0, 2.6, 0, 0, 0, 0.7;

	bool Panelwise = false;
	int RunMode = 10;

	int N_autorun = 1;
	if (Panelwise)
		N_autorun = 16;

	for (int i_autorun = 1; i_autorun <= N_autorun; i_autorun++)
	{

		std::cout << "\n\n\n***************************\nRun for Panel: " << i_autorun << "\n---------------------------\n\n";

		//1: AutoCorrelate


		switch (RunMode)
		{
		case -10: //Generate PixelMap for Simulation
		{
			std::cout << "Generate Pixelmap for Simulator\n";
			Simulator Sim;
			Sim.GeneratePixelMap("/gpfs/cfel/cxi/scratch/user/trostfab/PixelMap/PixelMap_300x300_50mu_500mm_SIM.h5", "PixelMap", 300, 300, 50, {-5.0e5, 0, 0}, {1, 0, 0}, {0, 1, 0});
		}
		break;
		case 0: //Combine ACuw and C(q)
		{
			std::string Cq_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/IACC_TESTSPACE/Cq_503-Z4_Big_Seg" + std::to_string(i_autorun) + ".bin";
			std::string ACuw_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/IACC_TESTSPACE/AC_UW_503-Z4_Big_Seg" + std::to_string(i_autorun) + ".bin";
			std::string AC_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/IACC_TESTSPACE/AC_503-Z4_Big_Seg" + std::to_string(i_autorun) + ".bin";
			CombineStuff(ACuw_Path, Cq_Path, AC_Path, 503 * 503 * 503);
		}
		break;
		case 1: //Autocorrelate Hb Jungfrau 3fs
			std::cout << "\n******************************\nRun IncohAutoCorrelate in Autocorrelation-mode for Jungfrau\n******************************\n";
			{

				std::string Prefix = "3fs_invRot";

				RunIAC::CreateDataEval_Settings EvalSettings;
				EvalSettings.XML_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/IACC_TESTSPACE/HitEventList_3fs_Jungfr.xml";

				EvalSettings.PixelMap_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/PixelMap/PixelMap_J.h5";
				EvalSettings.PixelMap_DataSet = "data/data";
				EvalSettings.PixelMask_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/PixelMap/PixelMask_thr03.bin";
				//
				EvalSettings.EchoLevel = 3;
				//
				EvalSettings.InvertRotMatrix = true;
				//
				EvalSettings.AngularAveraged = false;
				EvalSettings.DoubleMap = true;
				EvalSettings.FractionalCq = false;
				EvalSettings.RestrictStackToBoundaries = false;
				EvalSettings.MeshSize = 501;
				EvalSettings.QZoom = 2.0f;

				EvalSettings.PhotonOffset = 3.2f;
				EvalSettings.PhotonStep = 6.4;


				//EvalSettings.LowerBoundary = 0;
				//EvalSettings.UpperBoundary = 100;
				//EvalSettings.RestrictStackToBoundaries = true;


				EvalSettings.Out_AvIntensity_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/IACC_TESTSPACE/" + Prefix + "_avIntensity_.bin";
				EvalSettings.Out_ACuw_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/IACC_TESTSPACE/" + Prefix + "_uwAC.bin";
				EvalSettings.Out_Cq_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/IACC_TESTSPACE/" + Prefix + "_CQ.bin";
				EvalSettings.Out_Cq_small_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/IACC_TESTSPACE/" + Prefix + "_CQsmall.bin";
				EvalSettings.Out_Final_AC_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/IACC_TESTSPACE/" + Prefix + "_AC.bin";
				EvalSettings.Out_Q_Vector = "/gpfs/cfel/cxi/scratch/user/trostfab/IACC_TESTSPACE/" + Prefix + "_Q.bin";

				RunIAC::Run_AutoCorr_DataEval(Options, EvalSettings);

			}
			break;
		case 5: //Autocorrelate Single Molecule Jungfrau 120mm
			std::cout << "\n******************************\nRun IncohAutoCorrelate in Autocorrelation-mode for single molecule (Jungfr)\n******************************\n";
			{
				RunIAC::CreateSM_Settings SM_Settings;

				bool Block1 = false;
				bool Block2 = false;
				bool Block3 = true;
				bool BlockNoise = false;

				if (Block1)
				{
					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_139-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run139/Jungfrau1M/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_140-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run140/Jungfrau1M/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_141-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run141/Jungfrau1M/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_142-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run142/Jungfrau1M/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_143-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run143/Jungfrau1M/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_146-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run146/Jungfrau1M/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_147-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run147/Jungfrau1M/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_159-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run159/Jungfrau1M/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_160-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run160/Jungfrau1M/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_161-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run161/Jungfrau1M/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_162-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run162/Jungfrau1M/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_163-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run163/Jungfrau1M/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_164-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run164/Jungfrau1M/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_165-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run165/Jungfrau1M/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_166-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run166/Jungfrau1M/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_167-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run167/Jungfrau1M/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_168-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run168/Jungfrau1M/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_169-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run169/Jungfrau1M/calib/data");
				}

				if (Block2)
				{
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
				}

				if (Block3)
				{
					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_253-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run253/Jungfrau1M/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_255-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run255/Jungfrau1M/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_256-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run256/Jungfrau1M/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_257-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run257/Jungfrau1M/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_258-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run258/Jungfrau1M/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_263-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run263/Jungfrau1M/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_264-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run264/Jungfrau1M/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_265-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run265/Jungfrau1M/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_266-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run266/Jungfrau1M/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_267-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run267/Jungfrau1M/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_268-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run268/Jungfrau1M/calib/data");
				}

				//Test (BED-Noise)

				if (BlockNoise)
				{
					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/user/trostfab/SM_Noise_Sim/NoiseStack0.cxi");
					SM_Settings.H5Dataset.push_back("/data");

					//SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/user/trostfab/SM_Noise_Sim/NoiseStack1.cxi");
					//SM_Settings.H5Dataset.push_back("/data");

					//SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/user/trostfab/SM_Noise_Sim/NoiseStack2.cxi");
					//SM_Settings.H5Dataset.push_back("/data");
				}
				//

				SM_Settings.PixelMap_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/PixelMap/PixelMap_J.h5";


				SM_Settings.PixelMap_DataSet = "data/data";

				//SM_Settings.PixelMask_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/PixelMap/PixelMask_thr03.bin";

				std::string Praefix = "Block3_Thr48";

				if (Panelwise)
				{//auto adapt Panel
					//std::string MaskPath = "/gpfs/cfel/cxi/scratch/user/trostfab/PixelMap/PixelMask_Jungfr_Seg";
					SM_Settings.PixelMask_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/PixelMap/PixelMask_Jungfr_Seg" + std::to_string(i_autorun) + ".bin";

					SM_Settings.Output_AV_Int_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/NanoStar/" + Praefix + "_Pan" + std::to_string(i_autorun) + "_avINT.bin";
					SM_Settings.Output_CQ_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/NanoStar/" + Praefix + "_Pan" + std::to_string(i_autorun) + "_CQ.bin";
					SM_Settings.Output_ACUW_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/NanoStar/" + Praefix + "_Pan" + std::to_string(i_autorun) + "_ACuw.bin";
					SM_Settings.Output_AC_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/NanoStar/" + Praefix + "_Pan" + std::to_string(i_autorun) + "_AC.bin";
				}//auto adapt Panel
				if (!Panelwise)
				{
					SM_Settings.PixelMask_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/PixelMap/PixelMask_thr03.bin";
					SM_Settings.Output_AV_Int_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/NanoStar/" + Praefix + "_avINT.bin";
					SM_Settings.Output_CQ_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/NanoStar/" + Praefix + "_CQ.bin";
					SM_Settings.Output_ACUW_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/NanoStar/" + Praefix + "_ACuw.bin";
					SM_Settings.Output_AC_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/NanoStar/" + Praefix + "_AC.bin";
				}//auto adapt Panel


				//SM_Settings.PixelMask_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/PixelMap/PixelMask_Jungfr_Seg1.bin";
				//SM_Settings.Output_AV_Int_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/NanoStar/Block3_Pan1_avINT.bin";
				//SM_Settings.Output_CQ_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/NanoStar/Block3_Pan1_CQ.bin";
				//SM_Settings.Output_ACUW_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/NanoStar/Block3_Pan1_ACuw.bin";
				//SM_Settings.Output_AC_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/NanoStar/Block3_Pan1_AC.bin";
				SM_Settings.Output_Q_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/NanoStar/Q_Jungfr120.bin";

				SM_Settings.PhotonOffset = 4.8;//3.2
				SM_Settings.PhotonStep = 6.4;//6.4


				SM_Settings.ArraySize = 1200;

				AC1D Results;


				RunIAC::Run_AC_SM_Full(Results, SM_Settings, Options);


			}
			break;
		case 501: //Autocorrelate Single Molecule Jungfrau 120mm FRACTIONAL
			std::cout << "\n******************************\nRun IncohAutoCorrelate in Fractional Autocorrelation-mode for single molecule (Jungfr)\n******************************\n";
			{
				RunIAC::CreateSM_Settings SM_Settings;

				unsigned int FracSize = 100;
				std::cout << "Fractionsize = " << FracSize << "\n";

				Options.HitEvents.clear();
				Options.LoadHitEventListFromFile("/gpfs/cfel/cxi/scratch/user/trostfab/PPPData/HitEventList_SM_Jungfr_Block3_LAP.xml");


				//

				SM_Settings.PixelMap_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/PixelMap/PixelMap_J.h5";
				SM_Settings.PixelMap_DataSet = "data/data";

				//SM_Settings.PixelMask_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/PixelMap/PixelMask_thr03.bin";

				std::string Praefix = "Block3_LAP";

				if (Panelwise)
				{//auto adapt Panel
				 //std::string MaskPath = "/gpfs/cfel/cxi/scratch/user/trostfab/PixelMap/PixelMask_Jungfr_Seg";
					SM_Settings.PixelMask_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/PixelMap/PixelMask_Jungfr_Seg" + std::to_string(i_autorun) + ".bin";

					SM_Settings.Output_AV_Int_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/NanoStar/Fractional LAP/" + Praefix + "_Pan" + std::to_string(i_autorun) + "_Frac" + std::to_string(FracSize) + "_avINT.bin";
					SM_Settings.Output_CQ_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/NanoStar/Fractional LAP/" + Praefix + "_Pan" + std::to_string(i_autorun) + "_Frac" + std::to_string(FracSize) + "_CQ.bin";
					SM_Settings.Output_ACUW_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/NanoStar/Fractional LAP/" + Praefix + "_Pan" + std::to_string(i_autorun) + "_Frac" + std::to_string(FracSize) + "_ACuw.bin";
					SM_Settings.Output_AC_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/NanoStar/Fractional LAP/" + Praefix + "_Pan" + std::to_string(i_autorun) + "_Frac" + std::to_string(FracSize) + "_AC.bin";
				}//auto adapt Panel
				if (!Panelwise)
				{
					SM_Settings.PixelMask_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/PixelMap/PixelMask_thr03.bin";
					SM_Settings.Output_AV_Int_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/NanoStar/Fractional LAP/TMPavINT.bin";
					SM_Settings.Output_CQ_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/NanoStar/Fractional LAP/TMPcq_CQ.bin";
					SM_Settings.Output_ACUW_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/NanoStar/Fractional LAP/TMPacuw_ACuw.bin";
					SM_Settings.Output_AC_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/NanoStar/Fractional LAP/" + Praefix + "_Frac" + std::to_string(FracSize) + "_AC.bin";
				}//auto adapt Panel


				 //SM_Settings.PixelMask_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/PixelMap/PixelMask_Jungfr_Seg1.bin";
				 //SM_Settings.Output_AV_Int_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/NanoStar/Block3_Pan1_avINT.bin";
				 //SM_Settings.Output_CQ_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/NanoStar/Block3_Pan1_CQ.bin";
				 //SM_Settings.Output_ACUW_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/NanoStar/Block3_Pan1_ACuw.bin";
				 //SM_Settings.Output_AC_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/NanoStar/Block3_Pan1_AC.bin";
				SM_Settings.Output_Q_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/NanoStar/Fractional LAP/Q_Jungfr120_FRAC.bin";

				SM_Settings.PhotonOffset = 0.99;//3.2
				SM_Settings.PhotonStep = 0.99;//6.4


				SM_Settings.ArraySize = 1200;

				AC1D Results;


				RunIAC::Run_AC_SM_fractionalCQ(Results, SM_Settings, Options, FracSize, true);


			}
			break;
		case 5001:
			std::cout << "\n******************************\nRun Statistics for single molecule (Jungfrau)\n******************************\n";
			{
				RunIAC::CreateSM_Settings SM_Settings;

				bool Block1 = false;
				bool Block2 = false;
				bool Block3 = true;
				bool BlockNoise = false;

				if (Block1)
				{
					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_139-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run139/Jungfrau1M/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_140-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run140/Jungfrau1M/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_141-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run141/Jungfrau1M/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_142-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run142/Jungfrau1M/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_143-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run143/Jungfrau1M/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_146-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run146/Jungfrau1M/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_147-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run147/Jungfrau1M/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_159-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run159/Jungfrau1M/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_160-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run160/Jungfrau1M/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_161-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run161/Jungfrau1M/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_162-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run162/Jungfrau1M/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_163-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run163/Jungfrau1M/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_164-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run164/Jungfrau1M/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_165-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run165/Jungfrau1M/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_166-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run166/Jungfrau1M/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_167-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run167/Jungfrau1M/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_168-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run168/Jungfrau1M/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_169-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run169/Jungfrau1M/calib/data");
				}

				if (Block2)
				{
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
				}

				if (Block3)
				{
					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_253-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run253/Jungfrau1M/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_255-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run255/Jungfrau1M/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_256-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run256/Jungfrau1M/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_257-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run257/Jungfrau1M/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_258-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run258/Jungfrau1M/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_263-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run263/Jungfrau1M/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_264-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run264/Jungfrau1M/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_265-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run265/Jungfrau1M/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_266-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run266/Jungfrau1M/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_267-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run267/Jungfrau1M/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_268-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run268/Jungfrau1M/calib/data");
				}

				//Test (BED-Noise)

				if (BlockNoise)
				{
					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/user/trostfab/SM_Noise_Sim/NoiseStack0.cxi");
					SM_Settings.H5Dataset.push_back("/data");

					//SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/user/trostfab/SM_Noise_Sim/NoiseStack1.cxi");
					//SM_Settings.H5Dataset.push_back("/data");

					//SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/user/trostfab/SM_Noise_Sim/NoiseStack2.cxi");
					//SM_Settings.H5Dataset.push_back("/data");
				}
				//

				SM_Settings.PixelMap_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/PixelMap/PixelMap_J.h5";


				SM_Settings.PixelMap_DataSet = "data/data";

				//SM_Settings.PixelMask_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/PixelMap/PixelMask_thr03.bin";

				std::string Praefix = "Block3_Thr48";

				if (Panelwise)
				{//auto adapt Panel
				 //std::string MaskPath = "/gpfs/cfel/cxi/scratch/user/trostfab/PixelMap/PixelMask_Jungfr_Seg";
					SM_Settings.PixelMask_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/PixelMap/PixelMask_Jungfr_Seg" + std::to_string(i_autorun) + ".bin";

					SM_Settings.Output_AV_Int_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/NanoStar/" + Praefix + "_Pan" + std::to_string(i_autorun) + "_avINT.bin";
					SM_Settings.Output_CQ_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/NanoStar/" + Praefix + "_Pan" + std::to_string(i_autorun) + "_CQ.bin";
					SM_Settings.Output_ACUW_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/NanoStar/" + Praefix + "_Pan" + std::to_string(i_autorun) + "_ACuw.bin";
					SM_Settings.Output_AC_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/NanoStar/" + Praefix + "_Pan" + std::to_string(i_autorun) + "_AC.bin";
				}//auto adapt Panel
				if (!Panelwise)
				{
					SM_Settings.PixelMask_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/PixelMap/PixelMask_thr03.bin";
					SM_Settings.Output_AV_Int_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/NanoStar/" + Praefix + "_avINT.bin";
					SM_Settings.Output_CQ_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/NanoStar/" + Praefix + "_CQ.bin";
					SM_Settings.Output_ACUW_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/NanoStar/" + Praefix + "_ACuw.bin";
					SM_Settings.Output_AC_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/NanoStar/" + Praefix + "_AC.bin";
				}//auto adapt Panel


				 //SM_Settings.PixelMask_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/PixelMap/PixelMask_Jungfr_Seg1.bin";
				 //SM_Settings.Output_AV_Int_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/NanoStar/Block3_Pan1_avINT.bin";
				 //SM_Settings.Output_CQ_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/NanoStar/Block3_Pan1_CQ.bin";
				 //SM_Settings.Output_ACUW_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/NanoStar/Block3_Pan1_ACuw.bin";
				 //SM_Settings.Output_AC_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/NanoStar/Block3_Pan1_AC.bin";
				SM_Settings.Output_Q_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/NanoStar/Q_Jungfr120.bin";

				SM_Settings.PhotonOffset = 4.8;//3.2
				SM_Settings.PhotonStep = 6.4;//6.4


				SM_Settings.ArraySize = 1200;



				RunIAC::Print_Statistics_SM(SM_Settings, Options);
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

				SM_Settings.PixelMap_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/PixelMap/PixelMap_J500.h5";
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
			std::cout << "\n******************************\nRun IncohAutoCorrelate in Autocorrelation-mode for single molecule (ePix)\n******************************\n";
			{
				RunIAC::CreateSM_Settings SM_Settings;

				bool Block_1 = false; // Adapt file names !!!
				bool Block_2 = false;
				bool Block_3 = true;


				if (Block_1)//Block 1
				{
					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_139-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run139/Epix100a/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_140-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run140/Epix100a/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_141-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run141/Epix100a/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_142-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run142/Epix100a/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_143-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run143/Epix100a/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_146-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run146/Epix100a/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_147-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run147/Epix100a/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_159-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run159/Epix100a/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_160-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run160/Epix100a/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_161-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run161/Epix100a/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_162-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run162/Epix100a/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_163-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run163/Epix100a/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_164-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run164/Epix100a/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_165-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run165/Epix100a/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_166-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run166/Epix100a/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_167-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run167/Epix100a/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_168-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run168/Epix100a/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_169-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run169/Epix100a/calib/data");
				}

				if (Block_2)//Block2
				{
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
				}

				if (Block_3)
				{
					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_253-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run253/Epix100a/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_255-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run255/Epix100a/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_256-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run256/Epix100a/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_257-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run257/Epix100a/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_258-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run258/Epix100a/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_263-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run263/Epix100a/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_264-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run264/Epix100a/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_265-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run265/Epix100a/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_266-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run266/Epix100a/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_267-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run267/Epix100a/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_268-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run268/Epix100a/calib/data");
				}


				SM_Settings.PixelMap_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/PixelMap/Epix_rough.h5";
				SM_Settings.PixelMap_DataSet = "geometry";
				SM_Settings.PixelMask_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/PixelMap/ePix_mask1.bin";

				std::string Praefix = "Block3_Thr75";

				SM_Settings.Output_AV_Int_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/NanoStar/EPIX/" + Praefix + "_ePix_avINT.bin";
				SM_Settings.Output_CQ_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/NanoStar/EPIX/" + Praefix + "_ePix_CQ.bin";
				SM_Settings.Output_ACUW_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/NanoStar/EPIX/" + Praefix + "_ePix_ACuw.bin";
				SM_Settings.Output_AC_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/NanoStar/EPIX/" + Praefix + "_ePix_AC.bin";
				SM_Settings.Output_Q_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/NanoStar/EPIX/" + Praefix + "_ePix_Q.bin";

				//Epix ADU differ from Jungfrau!! (needs maybe refinement)
				SM_Settings.PhotonOffset = 75;//50
				SM_Settings.PhotonStep = 100;
				//

				SM_Settings.ArraySize = 750;


				AC1D Results;

				SM_Settings.JungfrDet = false; //to not expect two panels
				RunIAC::Run_AC_SM_Full(Results, SM_Settings, Options);


			}
			break;

		case 701:
			std::cout << "\n******************************\nRun Statistics for single molecule (ePix)\n******************************\n";
			{
				RunIAC::CreateSM_Settings SM_Settings;

				bool Block_1 = false; // Adapt file names !!!
				bool Block_2 = false;
				bool Block_3 = true;


				if (Block_1)//Block 1
				{
					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_139-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run139/Epix100a/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_140-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run140/Epix100a/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_141-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run141/Epix100a/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_142-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run142/Epix100a/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_143-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run143/Epix100a/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_146-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run146/Epix100a/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_147-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run147/Epix100a/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_159-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run159/Epix100a/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_160-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run160/Epix100a/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_161-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run161/Epix100a/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_162-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run162/Epix100a/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_163-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run163/Epix100a/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_164-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run164/Epix100a/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_165-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run165/Epix100a/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_166-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run166/Epix100a/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_167-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run167/Epix100a/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_168-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run168/Epix100a/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_169-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run169/Epix100a/calib/data");
				}

				if (Block_2)//Block2
				{
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
				}

				if (Block_3)
				{
					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_253-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run253/Epix100a/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_255-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run255/Epix100a/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_256-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run256/Epix100a/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_257-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run257/Epix100a/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_258-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run258/Epix100a/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_263-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run263/Epix100a/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_264-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run264/Epix100a/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_265-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run265/Epix100a/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_266-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run266/Epix100a/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_267-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run267/Epix100a/calib/data");

					SM_Settings.Files.push_back("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/SegmentedSM/run_268-xes.h5");
					SM_Settings.H5Dataset.push_back("/mfxlr1716/run268/Epix100a/calib/data");
				}


				SM_Settings.PixelMap_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/PixelMap/Epix_rough.h5";
				SM_Settings.PixelMap_DataSet = "geometry";
				SM_Settings.PixelMask_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/PixelMap/ePix_mask1.bin";

				std::string Praefix = "Block3_Thr75";

				SM_Settings.Output_AV_Int_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/NanoStar/EPIX/" + Praefix + "_ePix_avINT.bin";
				SM_Settings.Output_CQ_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/NanoStar/EPIX/" + Praefix + "_ePix_CQ.bin";
				SM_Settings.Output_ACUW_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/NanoStar/EPIX/" + Praefix + "_ePix_ACuw.bin";
				SM_Settings.Output_AC_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/NanoStar/EPIX/" + Praefix + "_ePix_AC.bin";
				SM_Settings.Output_Q_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/NanoStar/EPIX/" + Praefix + "_ePix_Q.bin";

				//Epix ADU differ from Jungfrau!! (needs maybe refinement)
				SM_Settings.PhotonOffset = 75;//50
				SM_Settings.PhotonStep = 100;
				//

				SM_Settings.ArraySize = 750;




				SM_Settings.JungfrDet = false; //to not expect two panels
				RunIAC::Print_Statistics_SM(SM_Settings, Options);
			}
			break;

		case 10: //Simulate
		{
			std::cout << "\n******************************\nRun IncohAutoCorrelate in Simulation-mode\n******************************\n";
			{
				bool NewPixelMap = true;

				if (NewPixelMap)
				{
					Simulator Sim;

					std::string PixelMap_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/PixelMap/PixelMap_1000x1000_100mu_100mm_SIM.h5";


					Sim.GeneratePixelMap(PixelMap_Path, "PixelMap", 1000, 1000, 100, { -1.0e5, 0, 0 }, {0 , 0, 1.0f }, { 0, 1.0f, 0 });

					Simulate(Options, PixelMap_Path, "PixelMap", CrystSize);
				}
				else
				{
					std::string PixelMap_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/PixelMap/PixelMap_J.h5";
					Simulate(Options, PixelMap_Path, "data/data", CrystSize);
				}
			}
		}
		break;

		case 20: //Evaluate simulated data Block (Hb)
		{
			std::cout << "\n******************************\nRun IncohAutoCorrelate in Autocorrelation-mode for simulated Jungfrau data\n******************************\n";

			// / NEAR100_HB_UC =

			//std::string Prefix = "165_MOFro_UC=30_NP=1000_NG=1e4";

		//	std::string Prefix = "SNR_24x24x24";//
			std::string Prefix = "SNR_NG100_"+ std::to_string(CrystSize) +"x" + std::to_string(CrystSize) + "x" + std::to_string(CrystSize) + "";//

			RunIAC::CreateDataEval_Settings EvalSettings;
			//EvalSettings.XML_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/Simulation/HbTest1/TestPixMa.xml";

			//EvalSettings.XML_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/Simulation/PlacementTest/165_MOF_UC=30x30x30_NG=10000_NP=1000.xml";
			EvalSettings.XML_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/Simulation/SNR/100mm/SNR_NG100_" + std::to_string(CrystSize) + "x" + std::to_string(CrystSize) + "x" + std::to_string(CrystSize) + ".xml";

			//EvalSettings.XML_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/Simulation/realistic/Stack100k_1.xml";

			//EvalSettings.XML_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/Simulation/SNR/HB_UC=15x15x15_NG=10_NP=250000.xml";


			EvalSettings.DetDisturb = false;
			EvalSettings.DetDisturb_Shift = 2000;
			EvalSettings.DetDisturb_Rot = 1;


			//"/gpfs/cfel/cxi/scratch/user/trostfab/Simulation/SNR/HB_UC=" + std::to_string(SimSettings.UnitCells[0]) + "x" + std::to_string(SimSettings.UnitCells[1]) + "x" + std::to_string(SimSettings.UnitCells[2]) + "x" + "_NG=" + std::to_string(NG) + "1e1_NP=" + std::to_string(SimSettings.NumberOfSimulations) + ".xml";

			//EvalSettings.PixelMap_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/PixelMap/PixelMap_J.h5";
			//EvalSettings.PixelMap_DataSet = "data/data";

			//EvalSettings.PixelMap_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/PixelMap/PixelMap_250x250_50mu_300mm_SIM.h5";

			EvalSettings.PixelMap_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/PixelMap/PixelMap_500x500_100mu_100mm_SIM.h5";
			EvalSettings.PixelMap_DataSet = "PixelMap";
			EvalSettings.PixelMask_Path = "";
			//
			EvalSettings.EchoLevel = 3;
			//
			EvalSettings.AngularAveraged = false;
			EvalSettings.DoubleMap = true;
			EvalSettings.FractionalCq = false;
			EvalSettings.RestrictStackToBoundaries = false;
			EvalSettings.MeshSize = 501;
			EvalSettings.QZoom = 1.0f;

			EvalSettings.PhotonOffset = 0.000000001f;
			EvalSettings.PhotonStep =   1.0f;

			

				
			//EvalSettings.Out_AvIntensity_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/Simulation/Eval/" + Prefix + "_avIntensity_.bin";
			//EvalSettings.Out_ACuw_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/Simulation/Eval/" + Prefix + "_uwAC_"+ std::to_string(EvalSettings.MeshSize) +"-Z"+ std::to_string(EvalSettings.QZoom) +".bin";
			//EvalSettings.Out_Cq_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/Simulation/Eval/" + Prefix + "_CQ_" + std::to_string(EvalSettings.MeshSize) + "-Z" + std::to_string(EvalSettings.QZoom) + ".bin";
			//EvalSettings.Out_Cq_small_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/Simulation/Eval/" + Prefix + "_CQsmall_" + std::to_string(EvalSettings.MeshSize) + "-Z" + std::to_string(EvalSettings.QZoom) + ".bin";
			//EvalSettings.Out_Final_AC_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/Simulation/Eval/" + Prefix + "_AC_" + std::to_string(EvalSettings.MeshSize) + "-Z" + std::to_string(EvalSettings.QZoom) + ".bin";
			//EvalSettings.Out_Q_Vector = "/gpfs/cfel/cxi/scratch/user/trostfab/Simulation/Eval/" + Prefix + "_Q_" + std::to_string(EvalSettings.MeshSize) + "-Z" + std::to_string(EvalSettings.QZoom) + ".bin";

			//EvalSettings.Out_AvIntensity_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/Simulation/SNR/" + Prefix + "_avIntensity_.bin";
			//EvalSettings.Out_ACuw_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/Simulation/SNR/" + Prefix + "_uwAC_" + std::to_string(EvalSettings.MeshSize) + "-Z" + std::to_string(EvalSettings.QZoom) + ".bin";
			//EvalSettings.Out_Cq_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/Simulation/SNR/" + Prefix + "_CQ_" + std::to_string(EvalSettings.MeshSize) + "-Z" + std::to_string(EvalSettings.QZoom) + ".bin";
			//EvalSettings.Out_Cq_small_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/Simulation/SNR/" + Prefix + "_CQsmall_" + std::to_string(EvalSettings.MeshSize) + "-Z" + std::to_string(EvalSettings.QZoom) + ".bin";
			//EvalSettings.Out_Final_AC_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/Simulation/SNR/" + Prefix + "_AC_" + std::to_string(EvalSettings.MeshSize) + "-Z" + std::to_string(EvalSettings.QZoom) + ".bin";
			//EvalSettings.Out_Q_Vector = "/gpfs/cfel/cxi/scratch/user/trostfab/Simulation/SNR/" + Prefix + "_Q_" + std::to_string(EvalSettings.MeshSize) + "-Z" + std::to_string(EvalSettings.QZoom) + ".bin";

			//EvalSettings.Out_AvIntensity_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/Simulation/SNR/" + Prefix + "_avIntensity_.bin";
			//EvalSettings.Out_ACuw_Path = "";// "/gpfs/cfel/cxi/scratch/user/trostfab/Simulation/SNR/" + Prefix + "_uwAC_" + std::to_string(EvalSettings.MeshSize) + "-Z" + std::to_string(EvalSettings.QZoom) + ".bin";
			//EvalSettings.Out_Cq_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/Simulation/SNR/" + Prefix + "_CQ_" + std::to_string(EvalSettings.MeshSize) + "-Z" + std::to_string(EvalSettings.QZoom) + ".bin";
			//EvalSettings.Out_Cq_small_Path = "";// v"/gpfs/cfel/cxi/scratch/user/trostfab/Simulation/SNR/" + Prefix + "_CQsmall_" + std::to_string(EvalSettings.MeshSize) + "-Z" + std::to_string(EvalSettings.QZoom) + ".bin";
			//EvalSettings.Out_Final_AC_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/Simulation/SNR/" + Prefix + "_AC_" + std::to_string(EvalSettings.MeshSize) + "-Z" + std::to_string(EvalSettings.QZoom) + ".bin";
			//EvalSettings.Out_Q_Vector = "/gpfs/cfel/cxi/scratch/user/trostfab/Simulation/SNR/" + Prefix + "_Q_" + std::to_string(EvalSettings.MeshSize) + "-Z" + std::to_string(EvalSettings.QZoom) + ".bin";

			EvalSettings.Out_AvIntensity_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/Simulation/SNR/100mm/" + Prefix + "_avIntensity_.bin";
			EvalSettings.Out_ACuw_Path = "";
			EvalSettings.Out_Cq_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/Simulation/SNR/100mm/" + Prefix + "_CQ.bin";
			EvalSettings.Out_Cq_small_Path = "";
			EvalSettings.Out_Final_AC_Path = "/gpfs/cfel/cxi/scratch/user/trostfab/Simulation/SNR/100mm/" + Prefix + "_AC.bin";
			EvalSettings.Out_Q_Vector = "";



			RunIAC::Run_AutoCorr_DataEval(Options, EvalSettings);

			return 0;
		}

		//break;

		case 50: //Photon Hitfinding
		{
			Detector Det;
			//needed only for DetClass to work propaly (ToDo: Check if it is really needed)
			Det.LoadPixelMap("/gpfs/cfel/cxi/scratch/user/trostfab/PixelMap/PixelMap_J.h5", "data/data");
			Det.LoadPixelMask("/gpfs/cfel/cxi/scratch/user/trostfab/PixelMap/PixelMask_thr03.bin");

			std::string XML_IN = "/gpfs/cfel/cxi/scratch/user/trostfab/IACC_TESTSPACE/HitEventList_3fs_Jungfr.xml";
			std::string XML_OUT = "/gpfs/cfel/cxi/scratch/user/trostfab/PPPData/HitEventList_3fs_Jungfr_LAP.xml";
			std::string H5_OUT = "/gpfs/cfel/cxi/scratch/user/trostfab/PPPData/IntensityData_3fs_Jungfr_LAP.h5";
			std::string H5_Dataset_OUT = "LAP";

			std::vector<PPP::DetectorPanel> DetPanels;
			//Jungfrau Panels:
			{
				PPP::DetectorPanel DetPan;
				//first Jungfrau Panel
				DetPan.FirstInd = 0;
				DetPan.Scans[0] = 1024;
				DetPan.Scans[1] = 512;

				DetPanels.push_back(DetPan);
				//second Jungfrau Panel
				DetPan.FirstInd = 1024 * 512;
				DetPan.Scans[0] = 1024;
				DetPan.Scans[1] = 512;

				DetPanels.push_back(DetPan);
			}

			PPP::ProcessData_PF_LAP(XML_IN, XML_OUT, H5_OUT, H5_Dataset_OUT, DetPanels, Det, Det.DetectorSize[0] * Det.DetectorSize[1], 6.4f, 0.5f, 0.9f);

		}
		break;

		case 500: //Photon Hitfinding UnitTest
		{
			Detector Det;
			//needed only for DetClass to work propaly (ToDo: Check if it is really needed)
			Det.LoadPixelMap("/gpfs/cfel/cxi/scratch/user/trostfab/PixelMap/PixelMap_J.h5", "data/data");

			Det.LoadPixelMask("/gpfs/cfel/cxi/scratch/user/trostfab/PixelMap/PixelMask_thr03.bin");


			//load Hit events from Jungfrau Data:
			Options.Echo("Load Events from XML");
			Options.LoadHitEventListFromFile("/gpfs/cfel/cxi/scratch/user/trostfab/IACC_TESTSPACE/HitEventList_3fs_Jungfr.xml");

			Det.LoadIntensityData(&Options.HitEvents[0]);
			Det.ApplyPixelMask();

			//save Intensity before Photon finding
			ArrayOperators::SafeArrayToFile("/gpfs/cfel/cxi/scratch/user/trostfab/Tests/IntBeforePhotonFinding.bin", Det.Intensity, Det.DetectorSize[0] * Det.DetectorSize[1], ArrayOperators::Binary);

			std::vector<PPP::DetectorPanel> DetPanels;

			PPP::DetectorPanel DetPan;
			//first Jungfrau Panel
			DetPan.FirstInd = 0;
			DetPan.Scans[0] = 1024;
			DetPan.Scans[1] = 512;

			DetPanels.push_back(DetPan);
			//second Jungfrau Panel
			DetPan.FirstInd = 1024 * 512;
			DetPan.Scans[0] = 1024;
			DetPan.Scans[1] = 512;

			DetPanels.push_back(DetPan);

			//Run Photon Hitfinder
			PPP::PhotonFinder_LargestAdjacentPixel(Det.Intensity, DetPanels, 1024 * 1024, 6.4f, 0.5f, 0.9f);

			//save Intensity after Photon finding
			ArrayOperators::SafeArrayToFile("/gpfs/cfel/cxi/scratch/user/trostfab/Tests/IntAfterPhotonFinding.bin", Det.Intensity, Det.DetectorSize[0] * Det.DetectorSize[1], ArrayOperators::Binary);


		}
		break;


		case -1://Histogram Jungfr
		{
			std::cout << "\n******************************\nRun Statistics Histogram for Jungfrau\n******************************\n";
			Detector RefDet;
			RefDet.LoadPixelMap("/gpfs/cfel/cxi/scratch/user/trostfab/PixelMap/PixelMap_J.h5", "data/data");
			RefDet.LoadPixelMask("/gpfs/cfel/cxi/scratch/user/trostfab/PixelMap/PixelMask_thr03.bin");
			
			//Options.LoadHitEventListFromFile("/gpfs/cfel/cxi/scratch/user/trostfab/IACC_TESTSPACE/HitEventList_3fs_Jungfr.xml");
			Options.LoadStreamFile("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/indexing/10fs_JF.stream", "entry_1/instrument_1/detector_2/detector_corrected/data", false);
			


			Options.SafeHitEventListToFile("/gpfs/cfel/cxi/scratch/user/trostfab/IACC_TESTSPACE/HitEventList_10fs_Jungfr.xml");

			Statistics::Histogram Hist = Statistics::Make_AllPixel_Histogram(Options, RefDet, 100, 0, 50);
			Hist.SafeToFile("/gpfs/cfel/cxi/scratch/user/trostfab/Statistics/Hist_Jungfr_Hb10fs.bin");


		}
		break;
		case -2://Histogram Jungfr Pixelwise SM Block3
		{
			std::cout << "\n******************************\nRun Statistics Histogram for Jungfrau SM Block 3\n******************************\n";
			Detector RefDet;
			RefDet.LoadPixelMap("/gpfs/cfel/cxi/scratch/user/trostfab/PixelMap/PixelMap_J.h5", "data/data");
			RefDet.LoadPixelMask("/gpfs/cfel/cxi/scratch/user/trostfab/PixelMap/PixelMask_thr03.bin");
			Options.LoadHitEventListFromFile("/gpfs/cfel/cxi/scratch/user/trostfab/SM_RAWs/HitEventList_SM_Jungfr_PasanaCalib_Block3.xml");


			int HistSize = 55;
			std::vector<Statistics::Histogram> HistStack = Statistics::MakePixelHistogramStack(Options, RefDet, HistSize, -1.0, 21.0);

			double * NormalizedStack = new double[HistSize * RefDet.DetectorSize[0] * RefDet.DetectorSize[1]]();


			#pragma omp parallel for
			for (unsigned int i = 0; i < RefDet.DetectorSize[0] * RefDet.DetectorSize[1]; i++)
			{
				unsigned long AllBinCount = 0;
				for (unsigned int j = 0; j < HistSize; j++) //sum up all Bins (without over and underflow)
				{
					AllBinCount += HistStack[i].HistogramContent[j];
				}
				if (AllBinCount > 0)
				{
					for (unsigned int j = 0; j < HistSize; j++) //normalize
					{
						NormalizedStack[HistSize*i + j] = ((double)HistStack[i].HistogramContent[j] / ((double)AllBinCount));
					}
				}
			}

			ArrayOperators::SafeArrayToFile("/gpfs/cfel/cxi/scratch/user/trostfab/Statistics/PixelwiseHist_Jungfr_SM_Block3_55_-1_21.bin", NormalizedStack, HistSize * RefDet.DetectorSize[0] * RefDet.DetectorSize[1], ArrayOperators::Binary);

			delete[] NormalizedStack;
		}
		break;

		case -5000: //Create flat 1
		{
			float * FlatOne = new float[1024 * 1024];
			for (int i = 0; i < 1024*1024; i++)
			{
				FlatOne[i] = 1.0f;
			}
			ArrayOperators::SafeArrayToFile("/gpfs/cfel/cxi/scratch/user/trostfab/Simulation/Eval/JungfrFlatOne.bin", FlatOne, 1024*1024, ArrayOperators::FileType::Binary);
			delete[] FlatOne;
		}
		break;

		}//end switch

	}

	std::cout << "Ended.";
	return 0;


}