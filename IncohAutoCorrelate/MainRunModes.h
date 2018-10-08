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

namespace MainRunModes
{
	//Helper
	std::vector<std::string> CSV_Splitter(std::string Input, std::string delimiter);

	struct AllSettings
	{
		//Evaluator
		RunIAC::CreateDataEval_Settings EvaluationSettings;
		//Simulator
		Simulator::SimulationSettings SimulationSettings;
		//Statistics
		Statistics::StatisticsSettings StatisticsSettings;

	};

	//Functions
	//-> Data handling
	int Create_XMLHitlist_from_H5Stack_script(std::string Arg1, std::string Arg2, Settings &Options);
	int Create_XMLHitlist_from_H5Stack(std::vector<std::string> H5_Paths, std::vector<std::string> H5_Datasets, std::string XML_Output_Path, Settings &Options);

	
	//->Settings-Files Handle
	//single mode
	RunIAC::CreateDataEval_Settings LoadEvaluationSettings(std::string Filename, Settings &Options);
	Statistics::StatisticsSettings LoadStatisticSettings(std::string Filename, Settings &Options);
	//combined
	MainRunModes::AllSettings LoadSettings(std::string Filename, Settings &Options);

	int Create_Example_Evaluation_Config_File(std::string Filename, Settings &Options);
	int Create_Example_Config_File(std::string Filename, Settings &Options);

	//partial Settings
	boost::property_tree::ptree Example_Evaluation_Config_PT(boost::property_tree::ptree pt, Settings &Options);
	boost::property_tree::ptree Example_Statistics_Config_PT(boost::property_tree::ptree pt, Settings &Options); //requires Evaluation Config!
	

	//->Data Evaluation
	int AverageIntensity(std::string EvaluationConfigFile, Settings &Options, bool UpdateEventXML = false);

	//->Statistics
	int CreateAllPixelHistograms(std::string ConfigFile, Settings &Options);

}
