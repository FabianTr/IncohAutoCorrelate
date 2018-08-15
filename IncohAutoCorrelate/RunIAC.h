#pragma once
#include <string>
#include "ACMesh.h"
#include "AC1D.h"
#include "Settings.h"
#include <vector>

namespace RunIAC
{

	struct CreateCQ_Settings
	{
	private:

	public:
		int MeshSize = 501; // Size without ZeroPadding 
		float QZoom = 1.0f;

		std::string PixelMap_Path = "";//H5
		std::string PixelMap_DataSet = "";//H5

		std::string PixelMask_Path = "";//Binary

		std::string AVIntensity_Path = "";

		bool SaveSmall_CQ = false;
		std::string SmallCQ_Path = "";

		bool SaveBig_CQ = false;
		std::string BigCQ_Path = "";

		bool echo = false;
		std::string ThreadName = "CQ-Thread";

		Detector::AutoCorrFlags AC_Small_Flags;
		Detector::AutoCorrFlags AC_Merge_Flags;
	};

	void Create_CQ_Mesh(ACMesh& CQ, CreateCQ_Settings CQ_Settings, Settings& PrgSettings);
	void Create_CQ_Mesh(ACMesh& CQ, CreateCQ_Settings CQ_Settings, Settings& PrgSettings, unsigned int LowerBound, unsigned int UpperBound);


	struct CreateAC_Settings
	{
	public:
		int MeshSize = 501; // Size without ZeroPadding 
		float QZoom = 1.0f;

		std::string PixelMap_Path = "";//H5
		std::string PixelMap_DataSet = "";//H5

		std::string PixelMask_Path = "";//Binary

		bool SaveBig_AC = false;
		std::string BigAC_Path = "";

		float PhotonOffset = 3.2f;
		float PhotonStep = 6.4f;

		bool echo = false;
		int StatusEcho = 100;
		std::string ThreadName = "AC-Thread";

		bool DoubleMap = true;

		Detector::AutoCorrFlags AC_SecondMap_Flags; //to Implement
		Detector::AutoCorrFlags AC_FirstMap_Flags;
	};

	void Run_AC_UW(ACMesh& AC, CreateAC_Settings AC_Settings, Settings& PrgSettings);
	void Run_AC_UW(ACMesh& AC, CreateAC_Settings AC_Settings, Settings& PrgSettings, unsigned int LowerBound, unsigned int UpperBound);


	void Merge_ACandCQ(double *& Output, ACMesh & AC, ACMesh & CQ, Settings & PrgSettings);
	void Merge_ACandCQ(double *& Output, double * AC, ACMesh & CQ, Settings & PrgSettings);

	//Combined EVAL
	class CreateDataEval_Settings
	{
	public:
		int MeshSize = 501; // Size without ZeroPadding 
		float QZoom = 1.0f;

		bool AngularAveraged = false; //1D array instead of 3D mesh

		bool RestrictStackToBoundaries = false; //Evaluate only certain stack of events, if true, need to set boundaries
		unsigned int LowerBoundary = 0;
		unsigned int UpperBoundary = 1;

		bool FractionalCq = false; //Generate new CQ for each stack of X events (=> set SizeOfCqFraction accordingly)
		unsigned int SizeOfCqFraction = 100;

		bool InvertRotMatrix = false;

		//Input Paths
		std::string XML_Path = "";
		std::string PixelMap_Path = "";//H5
		std::string PixelMap_DataSet = "";//H5
		std::string PixelMask_Path = "";//Binary int

		//Output Paths
		std::string Out_AvIntensity_Path = "";
		std::string Out_ACuw_Path = "";
		std::string Out_Cq_Path = "";
		std::string Out_Cq_small_Path = "";//only 3D
		std::string Out_Final_AC_Path = "";
		std::string Out_Q_Vector = "";//only 1D AAv

		float PhotonOffset = 0.0f;
		float PhotonStep = 1.0f;

		int EchoLevel = 1;

		bool DoubleMap = true;

		Detector::AutoCorrFlags AC_SecondMap_Flags; //to Implement
		Detector::AutoCorrFlags AC_FirstMap_Flags;
		
		// For simulative data, detector disturbtion
		bool DetDisturb = false;
		double DetDisturb_Shift = 0.0;
		double DetDisturb_Rot = 0.0; //in degree

	};
	void Run_AutoCorr_DataEval(Settings & PrgSettings, CreateDataEval_Settings EvalSettings);


	//Single Molecule

	struct CreateSM_Settings
	{
		std::vector<std::string> Files;
		std::vector<std::string> H5Dataset;

		std::string PixelMap_Path = "";//H5
		std::string PixelMap_DataSet = "";//H5

		std::string PixelMask_Path = "";//Binary

		std::string Output_AV_Int_Path = "";//Binary
		std::string Output_CQ_Path = "";//Binary
		std::string Output_ACUW_Path = "";//Binary
		std::string Output_AC_Path = "";//Binary
		std::string Output_Q_Path = "";//Binary

		float PhotonOffset = 3.2f;
		float PhotonStep = 6.4f;
		
		unsigned int ArraySize = 1000;

		bool JungfrDet = true;
	};


	void Run_AC_SM_Full(AC1D & Output, CreateSM_Settings SM_Settings, Settings& PrgSettings);
	void Run_AC_SM_fractionalCQ(AC1D & AC, CreateSM_Settings SM_Settings, Settings & PrgSettings, unsigned int FractionSize, bool PPPdata);

	//General
	void Load_and_average_Intensities(Settings &Options, Detector &Det, float PhotonThreshold, float PhotonStep, std::string XML_Filename, std::string AvInt_Filename);


	//Helpers
	unsigned int GetH5StackSize(std::string Filename, std::string Dataset);
	

	//Statistics
	void Print_Statistics_SM(CreateSM_Settings SM_Settings, Settings & PrgSettings);

	
	//Pixelmap
	void GeneratePixelMapForSimulator(std::string Filename, std::string Dataset, double Distance, double PixelSize, int SizeFS, int SizeSS, std::array<float,3> Orientation);


}