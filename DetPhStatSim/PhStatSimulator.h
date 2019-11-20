#pragma once
#include "Settings.h"
#include <string>
#include <vector>
#include <atomic>

class PhStatSimulator
{
private:
	static void SimulatePart(std::vector<std::vector<float>> & DetImage, Settings& Options,unsigned int Loops, int ThreadNum, std::atomic<int> & counter);

public:
	Settings Options;
	PhStatSimulator();
	PhStatSimulator(std::string SettingsPath);

	
	void Simulate();

};

