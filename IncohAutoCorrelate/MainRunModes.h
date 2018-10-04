#pragma once
#include <iostream>
#include <vector>
#include <string>


#include "ProfileTime.h"
#include "Settings.h"

namespace MainRunModes
{
	int Create_XMLHitlist_from_H5Stack_script(std::string Arg1, std::string Arg2, Settings Options);
	int Create_XMLHitlist_from_H5Stack(std::vector<std::string> H5_Paths, std::vector<std::string> H5_Datasets, std::string XML_Output_Path, Settings Options);



	std::vector<std::string> CSV_Splitter(std::string Input, std::string delimiter);
}
