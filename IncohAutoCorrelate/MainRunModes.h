#pragma once
#include <vector>
#include <string>
#include <array>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>

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


		std::string XMLSetting_Path = "";
	};

	//Functions
	//-> Data handling
	int Create_XMLHitlist_from_H5Stack_script(std::string Arg1, std::string Arg2, Settings &Options);
	int Create_XMLHitlist_from_H5Stack(std::vector<std::string> H5_Paths, std::vector<std::string> H5_Datasets, std::string XML_Output_Path, Settings &Options);

	
	//->Settings-Files Handle

	//single mode
	RunIAC::CreateDataEval_Settings LoadEvaluationSettings(std::string Filename, Settings &Options);
	Statistics::StatisticsSettings LoadStatisticSettings(std::string Filename, Settings &Options);
	PPP::Create_LAPSettings LoadPPPLAPSettings(std::string Filename, Settings &Options);
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

	//->Data Evaluation
	int AverageIntensity(std::string EvaluationConfigFile, Settings &Options, bool UpdateEventXML = false);

	//DataProcessing
	int GainCorrectionAndLAP(std::string Arg1, Settings &Options);

	int GainCorrection(std::string Arg1, Settings &Options);
	//->Statistics
	int CreateAllPixelHistograms(std::string ConfigFile, Settings &Options);


	//EVALUATE - AUTOCORRELATION mode
	int AutoCorrelateData(std::string ConfigFile, Settings &Options);



	//Sort Events
	int SortHitsByMeanIntensity(std::string Arg1, std::string Arg2, Settings &Options);//xml_in, csv_out

	int SortXMLHitsByMeanIntensity(std::string Arg1, std::string Arg2, Settings &Options);//xml_in, xml_out

	//Simulation
	int Simulate(std::string ConfigFile, Settings &Options);
}
