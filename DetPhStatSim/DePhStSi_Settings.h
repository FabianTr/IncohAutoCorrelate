#pragma once

#include <string>


class DePhStSi_Settings
{
public:
	DePhStSi_Settings();
	~DePhStSi_Settings();


	//Det Parameter
	int DetSize = 1024;
	int SuSa = 12;
	float ChargeSharingSigma = 0.22f;
	float DarkNoise = 0.0f;

	//file stuff
	std::string OutputPath = "SimulationOutput.h5";
	std::string OutputDataset = "data";

	//statistics
	float MeanIntensity = 1.0f;
	float Modes = 1.0f;

	//run parameter
	bool Dark = false;

	unsigned int Pattern = 1;

	void LoadDetectorDePhStSi_Settings(std::string File, std::string ArgStream = "");

	void SafeExampleDePhStSi_Settings(std::string File);



};

