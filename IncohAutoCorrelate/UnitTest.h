#pragma once

#include <string>

#include "Crystal.h"
#include "Settings.h"


class UnitTest
{
private:
	//Random stuff
	inline double Drand();
	std::uniform_real_distribution<double> rnd;
	std::mt19937_64 mt;
public:
	UnitTest();
	~UnitTest();



	bool TestKabschImplementation();

	bool StreamFileReader(Settings & Options);


	bool TestACandCQmapping(Settings & Options, std::string SettingsPath, int EventNum, bool Rotation = true);


	bool TestFitting(Settings & Options);
};

