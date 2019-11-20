#pragma once

#include <string>


class Settings
{
public:
	Settings();
	~Settings();


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

	unsigned int Pattern = 1;

	void LoadDetectorSettings(std::string File);

	void SafeExampleSettings(std::string File);



};

