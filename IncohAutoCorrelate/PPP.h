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
		int FirstInd = 0;
		int Size = 2;
		int Scans[2] = { 1,1 }; //fs, ss
	};


	
	void PhotonFinder_LargestAdjacentPixel(float * Intensity, std::vector<DetectorPanel> DetectorPanels, int FullDetSize, float ADU_perPhoton = 1.0f, float SeedThershold = 0.5f, float CombinedThershold = 0.9f);
	//Find Photons by considering the adjacent Pixels.

	void ProcessData_PF_LAP(std::string XML_In, std::string XML_Out, std::string H5_Out, std::string Dataset, std::vector<DetectorPanel> DetectorPanels, Detector &Det, int FullDetSize, float ADU_perPhoton = 1.0f, float SeedThershold = 0.5f, float CombinedThershold = 0.9f);

	void ProcessData_PF_LAP_SM(std::string XML_In, std::string XML_Out, std::string H5_Out, std::string Dataset, std::vector<DetectorPanel> DetectorPanels, Detector & Det, int FullDetSize, float ADU_perPhoton, float SeedThershold, float CombinedThershold, bool Jungfrau);

	void ProcessData_ConvertSM(std::string XML_In, std::string XML_Out, std::string H5_Out, std::string Dataset, std::vector<DetectorPanel> DetectorPanels, Detector & Det, int FullDetSize, float ADU_perPhoton, float SeedThershold, float CombinedThershold, bool Jungfrau);


}