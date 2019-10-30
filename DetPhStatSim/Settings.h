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
	float ChargeSharingSigma = 0.22;
	float DarkNoise = 0.0;

	//file stuff
	std::string OutputPath = "SimulationOutpuit.h5";
	std::string OutputDataset = "data";

	//statistics
	float MeanIntensity = 1.0;
	float Modes = 1.0;

	long Pattern = 1;

	void LoadDetectorSettings(std::string File);





};

