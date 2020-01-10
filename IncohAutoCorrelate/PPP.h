#pragma once

#include <vector>
#include <string>

#include "Detector.h"
#include "Settings.h"
//PPP Pattern PreProcessing
//This namespace contains methods to preprocess measured data (e.g. dropletisation)

namespace PPP
{
	struct DetectorPanel
	{
		unsigned int FirstInd = 0;
		unsigned int Scans[2] = { 2,2 }; //fs, ss
	};

	struct Create_LAPSettings
	{
		unsigned int DetPanels_Num = 1;
		std::vector<DetectorPanel> DetectorPanels;


		float ADU_perPhoton = 1.0f;
		float SeedThershold = 0.5f;
		float CombinedThershold = 0.9f;


		std::string Output_Path = "";
		std::string Output_Dataset = "";

		std::string Output_NewXML = "";

		std::string GainMapPath = "";
		std::string DatasetOffset = "offset";
		std::string DatasetGain = "gain";
	};

	struct Create_GaussPhotonizeSettings 
	{
		unsigned int DetPanels_Num = 1;
		std::vector<DetectorPanel> DetectorPanels;


		float ADU_perPhoton = 1.0f;
		float ChargeSharingSigma = 0.05;

		std::string Output_Path = "";
		std::string Output_Dataset = "";
		std::string Output_NewXML = "";

		//from eval Settings
		bool RestrictToLimits = false;
		size_t LowLimit=0;
		size_t UpperLimit=0;
	};

	class Create_DarcCalSettings
	{
	public:
		struct DarcCalSegment {
			std::string DarcCalFile = "";
			std::string H5Path = "";

			std::string ForDataContainer = "";
		};

		std::vector<DarcCalSegment> DarcCals;
		unsigned int DefaultIndex;

	};

	struct CreateDarkSettings {

		std::string Output_Path = "";
		std::string Output_Dataset = "";

		std::string Dark_Path = "";
		std::string Dark_Dataset = "";

		std::string Output_NewXML = "";

		bool RestrictToDataSource = false;
		std::string DataSource_Path = "";
	};

	
	//Find Photons by considering the adjacent Pixels.
	void PhotonFinder_LargestAdjacentPixel(float * Intensity, std::vector<DetectorPanel> DetectorPanels, int FullDetSize, float ADU_perPhoton = 1.0f, float SeedThershold = 0.5f, float CombinedThershold = 0.9f);
	void PhotonFinder_GaussFit(float* Intensity, const unsigned int FullDetSize, const Create_GaussPhotonizeSettings GaussPhotonizeSettings);


	void ProcessData_PF_LAP(std::string XML_In, std::string XML_Out, std::string H5_Out, std::string Dataset, std::vector<DetectorPanel> DetectorPanels, Detector &Det, int FullDetSize, float ADU_perPhoton = 1.0f, float SeedThershold = 0.5f, float CombinedThershold = 0.9f);
	void ProcessData_PF_LAP(Detector &Det, Create_LAPSettings LAPSettings, std::string XML_In, bool GainOnly = false);

	void ProcessData_GaussFit(Detector& Det, Create_GaussPhotonizeSettings GaussPhotonizeSettings, std::string XML_In);


	void GainCorrection(Detector & Det, std::string GainCorr_Path, std::string Dataset_Offset, std::string Dataset_Gain, Settings & Options, bool AllowNegativeValues = false);
	
	void ProcessData_DarkFieldCorrection(Detector & Det, CreateDarkSettings DarkSettings, std::string XML_In, Settings & Options);

}