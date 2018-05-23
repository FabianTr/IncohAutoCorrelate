#pragma once

#include <string>

#include "Detector.h"

class Simulator
{
private:

	void WriteIntensityToH5(Detector & Det, std::string Filename, std::string Dataset);

public:
	Simulator();
	~Simulator();



};

