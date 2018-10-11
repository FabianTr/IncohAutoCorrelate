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

	
	//Find Photons by considering the adjacent Pixels.
	void PhotonFinder_LargestAdjacentPixel(float * Intensity, std::vector<DetectorPanel> DetectorPanels, int FullDetSize, float ADU_perPhoton = 1.0f, float SeedThershold = 0.5f, float CombinedThershold = 0.9f);

	void ProcessData_PF_LAP(std::string XML_In, std::string XML_Out, std::string H5_Out, std::string Dataset, std::vector<DetectorPanel> DetectorPanels, Detector &Det, int FullDetSize, float ADU_perPhoton = 1.0f, float SeedThershold = 0.5f, float CombinedThershold = 0.9f);
	void ProcessData_PF_LAP(Detector &Det, Create_LAPSettings LAPSettings, std::string XML_In);

	void GainCorrection(Detector & Det, std::string GainCorr_Path, std::string Dataset_Offset, std::string Dataset_Gain, Settings & Options);
	
}