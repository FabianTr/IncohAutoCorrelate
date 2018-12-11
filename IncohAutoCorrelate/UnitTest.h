#pragma once

#include "Crystal.h"

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
};

