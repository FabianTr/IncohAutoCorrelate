#include "UnitTest.h"
#include <iostream>
#include <unordered_set>
#include <set>
#include <vector>
#include <math.h>
#include <Eigen/Core>

#include "ArrayOperators.h"
#include "MainRunModes.h"
#include "Detector.h"
#include "RunIAC.h"
#include "Statistics.h"

#define PI 3.14159265

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
	int CPUGPU = 0;

	std::cout << "Options Int to Float ConvOffset = " << Options.F_I_Conversion.Offset << "; ConvStep = " << Options.F_I_Conversion.Step << std::endl;
	ACMesh Mesh_CQ(&Options);
//Load Settings and Event XML
	std::cout << "Load XML Settings" << std::endl;
	MainRunModes::AllSettings RunSettings = MainRunModes::LoadSettings(SettingsPath, Options);
	RunIAC::CreateDataEval_Settings EvalSettings = RunSettings.EvaluationSettings;

	std::cout << "\nINTERPOLATIONS:" << std::endl;
	std::cout << "FirstLevelInterpo = " << EvalSettings.AC_FirstMap_Flags.InterpolationMode << std::endl;
	std::cout << "SecondLevelInterpo = " << EvalSettings.AC_SecondMap_Flags.InterpolationMode << "\n"<< std::endl;


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
			//if (DetInt.Intensity[i] != 0)
			//	int ascasc = 0;

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
	Flags.InterpolationMode = EvalSettings.AC_FirstMap_Flags.InterpolationMode; // First level is always NN by now
	DetInt.AutoCorrelate_CofQ_SmallMesh(smallMesh, Flags, Options);

	if(EvalSettings.AC_FirstMap_Flags.InterpolationMode == 0)
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
	DetInt2.AutoCorrelateSparseList(AC_uw, EvalSettings.AC_FirstMap_Flags, EvalSettings.AC_SecondMap_Flags, EvalSettings.DoubleMap, Options,CPUGPU);


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
			//if (AC_uw.Mesh[i] != 0 || Mesh_CQ.CQMesh[i] != 0)
			//	1+1;
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

bool UnitTest::TestFitting(Settings & Options)
{
	std::cout << "Test Fit functions" << std::endl;

	class Gauss3x3FitTarget
	{
	public:
		bool UseConstraints = false;
		double MaxAbsXY = 0.5;
		double MaxSigma = 0.25;

		double operator()(const std::pair<int, int> XY, const std::vector<double> Parameter, std::vector<double> &ParGrad) const
		{
			//Parameter(3): [x0, y0, sigma]
			const double x0 = Parameter[0];
			const double y0 = Parameter[1];
			const double s = Parameter[2];

			//check sigma = 0:
			if (s == 0)
			{
				ParGrad[0] = 0.0;
				ParGrad[1] = 0.0;
				ParGrad[2] = 0.0;
				return 0.0;
			}

			// XY Integration limits
			const double Xbot = (double)XY.first - 1.5;
			const double Xtop = (double)XY.first - 0.5;
			const double Ybot = (double)XY.second - 1.5;
			const double Ytop = (double)XY.second - 0.5;

			//construction kit
			const double ErfX = std::erf((Xbot - x0) / (std::sqrt(2)*s)) - std::erf((Xtop - x0) / (std::sqrt(2)*s));
			const double ErfY = std::erf((Ybot - y0) / (std::sqrt(2)*s)) - std::erf((Ytop - y0) / (std::sqrt(2)*s));

			const double LogExpXb = -std::pow((x0 - Xbot), 2) / (2.0 * s*s);
			const double ExpXb = LogExpXb<-700?0:std::exp(LogExpXb);

			const double LogExpXt = -std::pow((x0 - Xtop), 2) / (2.0 * s*s);
			const double ExpXt = LogExpXt < -700 ? 0 : std::exp(LogExpXt);

			const double LogExpYb = -std::pow((y0 - Ybot), 2) / (2.0 * s*s);
			double ExpYb = LogExpYb < -700 ? 0 : std::exp(LogExpYb);

			const double LogExpYt = -std::pow((y0 - Ytop), 2) / (2.0 * s*s);
			double ExpYt = LogExpYt < -700 ? 0 : std::exp(LogExpYt);


			//function value
			const double ret = (1.0 / 4.0) * (ErfX * ErfY);


			// d/dx0
			ParGrad[0] = (1.0 / (2.0 * s * std::sqrt(2.0 * M_PI))) * (ExpXt - ExpXb)*ErfY;
			// d/dy0
			ParGrad[1] = (1.0 / (2.0 * s * std::sqrt(2.0 * M_PI))) * (ExpYt - ExpYb)*ErfX;
			// d/ds
			ParGrad[2] = (1.0 / (2.0 * s * s * std::sqrt(2.0 * M_PI))) *   (
				(ExpYb * (y0 - Ybot) + ExpYt * (Ytop - y0)) * ErfX +
				(ExpXb * (x0 - Xbot) + ExpXt * (Xtop - x0)) * ErfY);

			//if (isnan(ret) || isnan(ParGrad[0]) || isnan(ParGrad[1]) || isnan(ParGrad[2]))
			//{
			//	std::cout << "x0 = " << x0 << ";\t y0 = " << y0 << "\t s = " << s << std::endl;

			//	int wait;
			//	std::cin >> wait;
			//}
			return ret;
		}
	};

	class Gauss3x3FitTarget f;

	std::vector<double> ChiSquareX0;
	std::vector<double> ChiSquareY0;
	std::vector<double> ChiSquareS;
	std::vector<double> RetSigma;


	double Noise = 0.1;
	for (size_t idx = 0; idx < 100; idx++)
	{
		double x0 = Drand() - 0.5;
		double y0 = Drand() - 0.5;
		double sigma0 = Drand()*0.25 + 0.000001;

		//x0 = - 0.426578;
		//y0 = - 0.40581;
		sigma0 = 0.25;

		std::vector<std::pair<std::pair<int, int>, double>> data(9);
		std::vector<double> ParGrad(3);
		int j = 0;
		for (int X = 0; X < 3; X++)
		{
			for (int Y = 0; Y < 3; Y++)
			{
				data[j] = { {X,Y}, f({X,Y},{x0,y0, sigma0}, ParGrad) * (1 + Noise * (Drand()-0.5) )};
				std::cout << data[j].second << " \t";
				j++;
			}
			std::cout << std::endl;
		}
		//std::cout << std::endl;

		std::vector<double> StartParams(3);
		StartParams[0] = 0;
		StartParams[1] = 0;
		StartParams[2] = 0.2;


		std::vector<double> FittedParams(3);	
		Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Cov;
		FittedParams = Statistics::detail::GaussNewton(f, data, StartParams,Cov);

		//std::cout << "Cov:\n" << Cov << std::endl;

		ChiSquareX0.push_back(std::pow(FittedParams[0] - x0, 2));
		ChiSquareY0.push_back(std::pow(FittedParams[1] - y0, 2));
		ChiSquareS.push_back(std::pow(FittedParams[2] - sigma0, 2));

		RetSigma.push_back(FittedParams[2]);

		std::cout << "===============\n";
		std::cout << "x0   : " << x0 << " -> " << FittedParams[0] << std::endl;
		std::cout << "y0   : " << y0 << " -> " << FittedParams[1] << std::endl;
		std::cout << "sigma: " << sigma0 << " -> " << FittedParams[2] << std::endl;
		std::cout << "===============\n\n";
	}

	double StAbw_X0 = 0;
	double StAbw_Y0 = 0;
	double StAbw_S = 0;

	double meanRetSigma = 0;

	for (size_t i = 0; i < ChiSquareX0.size(); i++)
	{
		StAbw_X0 += ChiSquareX0[i] / ChiSquareX0.size();
		StAbw_Y0 += ChiSquareY0[i] / ChiSquareY0.size();
		StAbw_S += ChiSquareS[i] / ChiSquareS.size();
		meanRetSigma += RetSigma[i] / ChiSquareS.size();
	}

	double StAbwSigma = 0.0;
	for (size_t i = 0; i < ChiSquareX0.size(); i++)
	{
		StAbwSigma += std::pow(RetSigma[i]- meanRetSigma,2) / ChiSquareS.size();
	}
	StAbwSigma = std::sqrt(StAbwSigma);

	std::cout << "**********************\n";

	std::cout << std::sqrt(StAbw_X0) << "  \t" << std::sqrt(StAbw_Y0) << "  \t" << std::sqrt(StAbw_S) << std::endl;
	std::wcout << "\nMeanSigma = " << meanRetSigma << " +/- " << StAbwSigma << std::endl;




	//
	return true;

}


