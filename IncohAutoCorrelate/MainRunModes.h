#pragma once
#include <vector>
#include <string>
#include <array>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/property_tree/info_parser.hpp>

#include "Settings.h"
#include "RunIAC.h"
#include "Statistics.h"
#include "Simulator.h"
#include "PPP.h"

namespace MainRunModes
{
	//Helper
	std::vector<std::string> CSV_Splitter(std::string Input, std::string delimiter);

	struct Create_MiscSettings
	{
		std::string ReportPath = "";
	};

	struct AllSimSettings
	{
		Simulator::SimulationSettings SimSettings;
		Simulator::GeneratePMSettings GPMSettings;

		std::vector<std::array<double, 3>> UnitCell;

		double LatticeVector[3][3];

	};


	struct AllSettings
	{
		//Evaluator
		RunIAC::CreateDataEval_Settings EvaluationSettings;
		//Simulator
		AllSimSettings AllSimulationSettings;
		
		//Statistics
		Statistics::StatisticsSettings StatisticsSettings;
		//PPP.LAP
		PPP::Create_LAPSettings PPPLAPSettings;
		//PPP.Dark
		PPP::CreateDarkSettings PPPDarkSettings;
		//PPP.GaussFit
		PPP::Create_GaussPhotonizeSettings GaussPhotonizeSettings;

		std::string XMLSetting_Path = "";
	};


	//Functions
	//-> Data handling
	int Create_XMLHitlist_from_H5Stack_script(std::string Arg1, std::string Arg2, Settings &Options, std::string SupplementInfo = "");
	int Create_XMLHitlist_from_H5Stack(std::vector<std::string> H5_Paths, std::vector<std::string> H5_Datasets, std::string XML_Output_Path, Settings &Options, std::vector<std::string> SupplementInfo = std::vector<std::string>());


	//->Settings-Files Handle

	//single mode
	RunIAC::CreateDataEval_Settings LoadEvaluationSettings(std::string Filename, Settings &Options);
	Statistics::StatisticsSettings LoadStatisticSettings(std::string Filename, Settings &Options);
	PPP::Create_LAPSettings LoadPPPLAPSettings(std::string Filename, Settings &Options);
	PPP::CreateDarkSettings LoadPPPDarkSettings(std::string Filename, Settings &Options);
	PPP::Create_GaussPhotonizeSettings LoadGaussPhotonizeSettings(std::string Filename, Settings& Options);
	
	AllSimSettings LoadSimulationSettings(std::string Filename, Settings &Options);
	//combined
	MainRunModes::AllSettings LoadSettings(std::string Filename, Settings &Options);

	int Create_Example_Evaluation_Config_File(std::string Filename, Settings &Options);
	int Create_Example_Config_File(std::string Filename, Settings &Options);

	//partial Example Settings
	boost::property_tree::ptree Example_Evaluation_Config_PT(boost::property_tree::ptree pt, Settings &Options);
	boost::property_tree::ptree Example_Statistics_Config_PT(boost::property_tree::ptree pt, Settings &Options); //requires Evaluation Config!
	boost::property_tree::ptree Example_PatternPreProcessing_LAP(boost::property_tree::ptree pt, Settings &Options); //requires Evaluation Config!
	boost::property_tree::ptree Example_Simulation_Config_PT(boost::property_tree::ptree pt, Settings &Options); //requires Evaluation Config!
	boost::property_tree::ptree Example_PatternPreProcessing_Dark(boost::property_tree::ptree pt, Settings &Options); //requires Evaluation Config!
	boost::property_tree::ptree Example_PatternPreProcessing_GaussFit(boost::property_tree::ptree pt, Settings& Options); //requires Evaluation Config!

	//->Data Evaluation
	int AverageIntensity(std::string EvaluationConfigFile, Settings &Options, bool UpdateEventXML = false);

	//DataProcessing PPP
	int GainCorrectionAndLAP(std::string Arg1, Settings &Options);

	int GainCorrection(std::string Arg1, Settings &Options);
	int FastGainCorrection(std::string Arg1, std::string Arg2, std::string Arg3, std::string Arg4, std::string Arg5, std::string Arg6, std::string Arg7, Settings & Options);

	int DarkCalibration(std::string Arg1, Settings &Options);
	int FastDarkCalibration(std::string Arg1, std::string Arg2, std::string Arg3, std::string Arg4, std::string Arg5, std::string Arg6, Settings &Options);

	int PhotonGaussFit(std::string Arg1, Settings& Options);
	//->Statistics
	int CreateAllPixelHistograms(std::string ConfigFile, Settings &Options);
	int GenerateSpeckleContrastStatistics(std::string ConfigFile, Settings &Options);
	int IsolatedPhotonChargeSharingFit(std::string ConfigFile, Settings &Options); //fits the charge sharing radius for isolated photon hits


	//EVALUATE - AUTOCORRELATION mode
	int AutoCorrelateData(std::string ConfigFile, Settings &Options);


	//HitList Handling
		//Sort Events
	int SortHitsByMeanIntensity(std::string Arg1, std::string Arg2, Settings &Options);//xml_in, csv_out

	int SortXMLHitsByMeanIntensity(std::string Arg1, std::string Arg2, Settings &Options);//xml_in, xml_out

	int MergeXMLHitLits(std::string Arg1, std::string Arg2, std::string Arg3, Settings &Options); //xml_out, xml_in[], SupplInfo[] (optional)

	//Events from CSV

	int GetHitListFromCSVFile(std::string Arg1, std::string Arg2, Settings &Options); //CSV, XML

	int GetHitListFromStreamFile(std::string Arg1, std::string Arg2, std::string Arg3, Settings &Options); //CSV, H5, XML

	//Simulation
	int Simulate(std::string ConfigFile, Settings &Options);

	int GeneratePixelMap(std::string Arg1, Settings &Options);
}
