#include "UnitTest.h"
#include <iostream>
#include <unordered_set>
#include <set>

#include "ArrayOperators.h"
#include "MainRunModes.h"
#include "Detector.h"
#include "RunIAC.h"


UnitTest::UnitTest() :mt(std::random_device{}())
{
}


UnitTest::~UnitTest()
{
}

inline double UnitTest::Drand()
{
	return rnd(mt);
}

bool UnitTest::TestKabschImplementation()
{
	std::array<float, 9> RefMat;
	RefMat[0] = 1.0;
	RefMat[1] = 1.0;
	RefMat[2] = 1.0;
	RefMat[3] = 0;
	RefMat[4] = 1.0;
	RefMat[5] = 0;
	RefMat[6] = 0;
	RefMat[7] = 1.0;
	RefMat[8] = 1.0;

	std::array<float, 9> RotRefMat;
	RotRefMat = RefMat;

	std::cout << "Test Matrix is:\n"
		<< RefMat[0] << " \t" << RefMat[1] << " \t" << RefMat[2] << "\n"
		<< RefMat[3] << " \t" << RefMat[4] << " \t" << RefMat[5] << "\n"
		<< RefMat[6] << " \t" << RefMat[7] << " \t" << RefMat[8] << "\n" <<  std::endl;


	std::array<float, 9> RotMat;

	// <Create Rotation Matrix>
	std::vector<std::array<double, 3>> Vectors; //empty, just to execute "RotateVectors"
	std::array<double, 3> t;
	t[0] = 1;
	t[1] = 1;
	t[2] = 2;
	Vectors.push_back(t);
	//
	std::array<double, 3> V;
	V[0] = Drand() * 2. - 1.;
	V[1] = (Drand() * 2. - 1.);
	V[2] = (Drand() * 2. - 1.);

	while (V[0] * V[0] + V[1] * V[1] + V[2] * V[2] > 1)
	{
		V[0] = Drand() * 2. - 1.;
		V[1] = (Drand() * 2. - 1.);
		V[2] = (Drand() * 2. - 1.);
	}
	double alpha = Drand() * 2 * M_PIl;

	RotMat = Crystal::RotateVectors(Vectors, alpha, V);

	std::cout << "Rotation Matrix is:\n"
		<< RotMat[0] << " \t" << RotMat[1] << " \t" << RotMat[2] << "\n"
		<< RotMat[3] << " \t" << RotMat[4] << " \t" << RotMat[5] << "\n"
		<< RotMat[6] << " \t" << RotMat[7] << " \t" << RotMat[8] << "\n" << std::endl;
	//<\Create Rotation Matrix>

	//<Rotate Matrix>
	ArrayOperators::RotateMatrix(RotRefMat.data(), RotMat.data());

	std::cout << "Rotated Test Matrix is:\n"
		<< RotRefMat[0] << " \t" << RotRefMat[1] << " \t" << RotRefMat[2] << "\n"
		<< RotRefMat[3] << " \t" << RotRefMat[4] << " \t" << RotRefMat[5] << "\n"
		<< RotRefMat[6] << " \t" << RotRefMat[7] << " \t" << RotRefMat[8] << "\n" << std::endl;

	//<\Rotate Matrix>

	//<Test Kansch Algo>


	float* KRot = new float[9]();
	ArrayOperators::KabschRotationMatrixRetrieval3x3(RotRefMat.data(), RefMat.data(), KRot);

	RefMat[0] = 2.0;
	RefMat[1] = 7.0;
	RefMat[2] = 1.0;
	RefMat[3] = 0;
	RefMat[4] = 2.0;
	RefMat[5] = 0;
	RefMat[6] = 0;
	RefMat[7] = 7.0;
	RefMat[8] = 1.0;

	std::cout << "Retrieved Rotation Matrix is:\n"
		<< KRot[0] << " \t" << KRot[1] << " \t" << KRot[2] << "\n"
		<< KRot[3] << " \t" << KRot[4] << " \t" << KRot[5] << "\n"
		<< KRot[6] << " \t" << KRot[7] << " \t" << KRot[8] << "\n" << std::endl;

	//<\Test Kansch Algo>

	return true;
}

bool UnitTest::StreamFileReader(Settings & Options)
{
	Options.MReference << 6.227, 0, 0, 0, 8.066, 0, 0, 0, 11.1;
	Options.LoadStreamFile("/gpfs/cfel/cxi/scratch/user/trostfab/LR17/data/stream files/3fs_JF.stream", "entry_1/instrument_1/detector_2", false);


	return true;
}

bool UnitTest::TestACandCQmapping(Settings & Options, std::string SettingsPath, int EventNum, bool Rotation)
{
	std::cout << "Options Int to Float ConvOffset = " << Options.F_I_Conversion.Offset << "; ConvStep = " << Options.F_I_Conversion.Step << std::endl;
	ACMesh Mesh_CQ(&Options);
//Load Settings and Event XML
	std::cout << "Load XML Settings" << std::endl;
	MainRunModes::AllSettings RunSettings = MainRunModes::LoadSettings(SettingsPath, Options);
	RunIAC::CreateDataEval_Settings EvalSettings = RunSettings.EvaluationSettings;

	std::cout << "Load XML HitList" << std::endl;
	Options.LoadHitEventListFromFile(RunSettings.EvaluationSettings.XML_Path);
	
	if (!Rotation)
	{
		Options.HitEvents[EventNum].RotMatrix[0] = 1.0f;
		Options.HitEvents[EventNum].RotMatrix[1] = 0.0f;
		Options.HitEvents[EventNum].RotMatrix[2] = 0.0f;
		Options.HitEvents[EventNum].RotMatrix[3] = 0.0f;
		Options.HitEvents[EventNum].RotMatrix[4] = 1.0f;
		Options.HitEvents[EventNum].RotMatrix[5] = 0.0f;
		Options.HitEvents[EventNum].RotMatrix[6] = 0.0f;
		Options.HitEvents[EventNum].RotMatrix[7] = 0.0f;
		Options.HitEvents[EventNum].RotMatrix[8] = 1.0f;
		std::cout << "Rotation Matrix set to unity matrix" << std::endl;
	}
	else
	{
		std::cout << "Rotation Matrix of this event:\n"
			<< Options.HitEvents[EventNum].RotMatrix[0] << " \t" << Options.HitEvents[EventNum].RotMatrix[1] << " \t" << Options.HitEvents[EventNum].RotMatrix[2] << "\n"
			<< Options.HitEvents[EventNum].RotMatrix[3] << " \t" << Options.HitEvents[EventNum].RotMatrix[4] << " \t" << Options.HitEvents[EventNum].RotMatrix[5] << "\n"
			<< Options.HitEvents[EventNum].RotMatrix[6] << " \t" << Options.HitEvents[EventNum].RotMatrix[7] << " \t" << Options.HitEvents[EventNum].RotMatrix[8] << std::endl;
	}



//Load and "average (1)" intensity
	Detector DetInt;
	Detector DetInt2;
	std::cout << "Load pixelmap and pixelmask" << std::endl;
	DetInt.LoadPixelMap(EvalSettings.PixelMap_Path, EvalSettings.PixelMap_DataSet);
	DetInt2.LoadPixelMap(EvalSettings.PixelMap_Path, EvalSettings.PixelMap_DataSet);

	DetInt.Calc_kMap();
	DetInt2.Calc_kMap();

	DetInt.LoadPixelMask(EvalSettings.PixelMask_Path, EvalSettings.PixelMask_Dataset);
	DetInt2.LoadPixelMask(EvalSettings.PixelMask_Path, EvalSettings.PixelMask_Dataset);

	std::cout << "Load one Intensity with Detector::LoadAndAverageIntensity() method, with pixelmask" << std::endl;
	DetInt.LoadAndAverageIntensity(Options.HitEvents, EvalSettings.PhotonOffset, EvalSettings.PhotonStep, EventNum, EventNum + 1, true);

	std::cout << "Check \"Merged\" Intensity \"the stupid way\"" << std::endl;
	{
		std::set<double> val;
		for (unsigned long i = 0; i < DetInt.DetectorSize[0] * DetInt.DetectorSize[1]; i++)
		{
			if (DetInt.Intensity[i] != 0)
				int ascasc = 0;

			double t = DetInt.Intensity[i];

			val.insert(t);
		}
		for (double v : val)
		{
			std::cout << v << "; ";
		}
		std::cout << std::endl;
		//std::cout << "\nPress key\n";
		//int wait;
		//std::cin >> wait;
	}

//C(q) small
	std::cout << "Create small C(q)" << std::endl;
	ACMesh smallMesh(&Options);
	smallMesh.CreateSmallMesh_CofQ_ForDetector(DetInt, EvalSettings.MeshSize, EvalSettings.QZoom);

	Detector::AutoCorrFlags Flags;
	Flags.InterpolationMode = Settings::NearestNeighbour; // First level is always NN by now
	DetInt.AutoCorrelate_CofQ_SmallMesh(smallMesh, Flags, Options);

	{
		std::cout << "Check small C(q) to be integer" << std::endl;
		for (unsigned int i = 0; i < smallMesh.Shape.Size_AB*smallMesh.Shape.Size_AB*smallMesh.Shape.Size_C; i++)
		{
			if (ceil(smallMesh.CQMesh[i]) != floor(smallMesh.CQMesh[i]))
			{
				std::cout << "i = " << i << "; C(q) = " << smallMesh.CQMesh[i] << std::endl;
			}
			//if (smallMesh.CQMesh[i] != 0)
			//	std::cout << "i = " << i << "; C(q) = " << smallMesh.CQMesh[i] << std::endl;
		}
		//std::cout << "Press key\n";
		//int wait;
		//std::cin >> wait;
	}


//Merge and weight
	std::cout << "Weight and Rotate (merge) C(q)" << std::endl;

	Mesh_CQ.CreateBigMesh_CofQ_ForDetector(DetInt, EvalSettings.MeshSize, EvalSettings.QZoom);

	DetInt.Merge_smallCofQ(Mesh_CQ, smallMesh, Options.HitEvents, EventNum, EventNum + 1, Options, EvalSettings.AC_SecondMap_Flags);

	//{
	//	std::cout << "Check Big C(q) to be integer" << std::endl;
	//	double m = 100000000000;
	//	for (unsigned int i = 0; i < Mesh_CQ.Shape.Size_AB*Mesh_CQ.Shape.Size_AB*Mesh_CQ.Shape.Size_C; i++)
	//	{
	//		if (ceil(Mesh_CQ.CQMesh[i]*m) != floor(Mesh_CQ.CQMesh[i]*m))
	//		{
	//			std::cout << "i = " << i << "; C(q) = " << Mesh_CQ.CQMesh[i]*m << std::endl;
	//		}
	//		if (smallMesh.CQMesh[i] != 0)
	//			std::cout << "i = " << i << "; C(q) = " << smallMesh.CQMesh[i]*m << std::endl;
	//	}
	//	std::cout << "Press key\n";
	//	int wait;
	//	std::cin >> wait;
	//}



//AC uw
	std::cout << "Start AC uw Setup" << std::endl;
	ACMesh AC_uw(&Options);
	AC_uw.CreateBigMeshForDetector(DetInt2, EvalSettings.MeshSize, EvalSettings.QZoom);

	std::cout << "Load intensity apply pixelmask and compare to av int" << std::endl;
	DetInt2.LoadIntensityData(&Options.HitEvents[EventNum]);
	ArrayOperators::ParMultiplyElementwise(DetInt2.Intensity, DetInt2.PixelMask, DetInt2.DetectorSize[0] * DetInt2.DetectorSize[1]);

	bool equal = true;
	ArrayOperators::DiscretizeToPhotons(DetInt2.Intensity, EvalSettings.PhotonOffset, EvalSettings.PhotonStep, DetInt2.DetectorSize[0] * DetInt2.DetectorSize[1]);
	for (unsigned int i = 0; i < DetInt2.DetectorSize[0]* DetInt2.DetectorSize[1]; i++)
	{
		if (DetInt.Intensity[i] != DetInt2.Intensity[i])
		{
			std::cout << "intensity mismatch at i = " << i << "; Det1: " << DetInt.Intensity[i] << "; " << "; Det2: " << DetInt2.Intensity[i] << std::endl;
			equal = false;
		}
	}
	if(equal)
		std::cout << "Intensity does match!" << std::endl;

	std::cout << "Sparsify intensity" << std::endl;
	DetInt2.CreateSparseHitList(EvalSettings.PhotonOffset, EvalSettings.PhotonStep); //Sparsificate

	std::cout << "Autocorrelate" << std::endl;
	DetInt2.AutoCorrelateSparseList(AC_uw, EvalSettings.AC_FirstMap_Flags, EvalSettings.AC_SecondMap_Flags, EvalSettings.DoubleMap, Options,1);


	std::cout << "Apply C(q) to unweighted AC\n";

	std::cout << "Options Int to Float ConvOffset = " << Options.F_I_Conversion.Offset << "; ConvStep = " << Options.F_I_Conversion.Step << std::endl;

	double * FinalAC = nullptr;
	RunIAC::Merge_ACandCQ(FinalAC, AC_uw, Mesh_CQ, Options);
	
	if(false)
	{
		std::cout << "Check Output \"the stupid way\"" << std::endl;
		std::set<double> ACVals0;
		for (unsigned long i = 0; i < AC_uw.Shape.Size_AB*AC_uw.Shape.Size_AB*AC_uw.Shape.Size_C; i++)
		{
			if (AC_uw.Mesh[i] != 0 || Mesh_CQ.CQMesh[i] != 0)
				int qweert = 1;
			double t_AC = AC_uw.Mesh[i];
			double t_CQ = Mesh_CQ.CQMesh[i];
			double t = t_AC / t_CQ;
			if (std::isnan(t))
				t = 0.0;
			ACVals0.insert(t);
		}
		for (double v : ACVals0)
		{
			std::cout << v << "; ";
		}
		int wait;
		std::cin >> wait;
	}



	//{
	//	std::cout << "Check Output" << std::endl;
	//	std::set<double> ACVals;
	//	for (unsigned long i = 0; i < AC_uw.Shape.Size_AB*AC_uw.Shape.Size_AB*AC_uw.Shape.Size_C; i++)
	//	{
	//		ACVals.insert(FinalAC[i]);
	//	}

	//	for (double v : ACVals)
	//	{
	//		std::cout << v << "; ";
	//	}
	//	std::cout << std::endl;
	//}




//Save result
	ArrayOperators::SafeArrayToFile(EvalSettings.Out_Final_AC_Path, FinalAC, AC_uw.Shape.Size_AB*AC_uw.Shape.Size_AB*AC_uw.Shape.Size_AB, ArrayOperators::FileType::Binary);
	std::cout << "-> Saved AC (Final form, formated as double) as \"" << EvalSettings.Out_Final_AC_Path << "\"\n";
	std::cout << "--> Mesh shape: " << AC_uw.Shape.Size_AB << " x " << AC_uw.Shape.Size_AB << " x " << AC_uw.Shape.Size_C << std::endl;

	std::cout << "\n\ndone." << std::endl;
	return false;
}
