#pragma once
#include "DePhStSi_Settings.h"
#include <string>
#include <vector>
#include <atomic>

class PhStatSimulator
{
private:
	static void SimulatePart(std::vector<std::vector<double>> & DetImage, std::vector<std::vector<double>>& GroundTruth, DePhStSi_Settings& Options,unsigned int Loops, int ThreadNum, std::atomic<int> & counter);

public:
	DePhStSi_Settings Options;
	PhStatSimulator();
	PhStatSimulator(std::string DePhStSi_SettingsPath);

	
	void Simulate();

};

