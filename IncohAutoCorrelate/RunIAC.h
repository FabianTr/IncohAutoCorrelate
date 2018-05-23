#pragma once
#include <string>
#include "ACMesh.h"
#include "Settings.h"

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


	void Merge_ACandCQ(double *& Output, ACMesh AC, ACMesh CQ, Settings & PrgSettings);
	void Merge_ACandCQ(double *& Output, double * AC, ACMesh CQ, Settings & PrgSettings);
}