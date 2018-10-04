#include "MainRunModes.h"

int MainRunModes::Create_XMLHitlist_from_H5Stack_script(std::string Arg1, std::string Arg2, Settings Options)
{
	std::vector<std::string> Dublett = CSV_Splitter(Arg1, ";");
	if (Dublett.size() != 2)
	{
		std::cerr << "ERROR: while parsing argument for XML-Event list from H5-files.\n";
		std::cerr << "Input need to be in the shape \"Input1.h5, Input2.h5, ...;Dataset1, Dataset2, ...\". \n";
		return -1;
	}
	std::vector<std::string> H5_Paths = CSV_Splitter(Dublett[0], ",");
	std::vector<std::string> H5_Datasets = CSV_Splitter(Dublett[1], ",");

	if (H5_Paths.size() != H5_Datasets.size())
	{
		std::cerr << "ERROR: while parsing argument for XML-Event list from H5-files.\n";
		std::cerr << "Unequal amont of hdf5-paths and datasets. \n";
		return -1;
	}
	return Create_XMLHitlist_from_H5Stack(H5_Paths, H5_Datasets, Arg2, Options);
}

int MainRunModes::Create_XMLHitlist_from_H5Stack(std::vector<std::string> H5_Paths, std::vector<std::string> H5_Datasets, std::string XML_Output_Path, Settings Options)
{
	std::array<unsigned int, 2 > DetSize;
	DetSize = Options.ScanH5Files(H5_Paths, H5_Datasets);
	Options.SafeHitEventListToFile(XML_Output_Path);


	std::cout << "\n\n\n****************************\n     Report:\n\n";
	std::cout << H5_Paths.size() << " HDF5-files scanned and a total of " << Options.HitEvents.size() << " entries found.\n";
	std::cout << "Size of the 2D array (single entrie ^= detector size) is " << DetSize[0] << " x " << DetSize[1] << " pixel\n";
	std::cout << "Because of no orientation information, rotation matrix was set to identity matrix for each entry.\n";
	std::cout << "XML-Hit-List has been saved as \"" << XML_Output_Path << "\".\n";

	return 0;
}

std::vector<std::string> MainRunModes::CSV_Splitter(std::string Input, std::string delimiter)
{
	//Splits "delimiter"-seperated Values within string to vector
	std::vector<std::string> Output;
	size_t pos = 0;
	std::string token;
	bool ParsingAlive = true;
	while ((pos = Input.find(delimiter)) != std::string::npos)
	{
		token = Input.substr(0, pos);
		Output.push_back(token);
		Input.erase(0, pos + delimiter.length());
	}
	Output.push_back(Input);
	return Output;
}

