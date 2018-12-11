#include "UnitTest.h"
#include <iostream>
#include "ArrayOperators.h"


UnitTest::UnitTest() :mt(std::random_device{}())
{
}


UnitTest::~UnitTest()
{
}

inline double Crystal::Drand()
{
	return rnd(mt);
}

bool UnitTest::TestKabschImplementation()
{
	std::array<float, 9> RefMat;
	RefMat[0] = 1.0;
	RefMat[1] = 0;
	RefMat[2] = 0;
	RefMat[3] = 0;
	RefMat[4] = 1.0;
	RefMat[5] = 0;
	RefMat[6] = 0;
	RefMat[7] = 0;
	RefMat[8] = 1.0;

	std::array<float, 9> RotMatrix;

	//
	std::vector<std::array<double, 3>> Vectors; //empty, just to execute "RotateVectors"
	std::array<double, 3> t;
	t[0] = 1;
	t[1] = 1;
	t[2] = 2;
	Vectors.push_back(t);
	//


	//Crystal::RotateVectors();

	return true;
}
