#include "UnitTest.h"
#include <iostream>
#include "ArrayOperators.h"


UnitTest::UnitTest() :mt(std::random_device{}())
{
}


UnitTest::~UnitTest()
{
}

inline double UnitTest::Drand()
{
	return rnd(mt);
}

bool UnitTest::TestKabschImplementation()
{
	std::array<float, 9> RefMat;
	RefMat[0] = 1.0;
	RefMat[1] = 1.0;
	RefMat[2] = 1.0;
	RefMat[3] = 0;
	RefMat[4] = 1.0;
	RefMat[5] = 0;
	RefMat[6] = 0;
	RefMat[7] = 1.0;
	RefMat[8] = 1.0;

	std::array<float, 9> RotRefMat;
	RotRefMat = RefMat;

	std::cout << "Test Matrix is:\n"
		<< RefMat[0] << " \t" << RefMat[1] << " \t" << RefMat[2] << "\n"
		<< RefMat[3] << " \t" << RefMat[4] << " \t" << RefMat[5] << "\n"
		<< RefMat[6] << " \t" << RefMat[7] << " \t" << RefMat[8] << "\n" <<  std::endl;


	std::array<float, 9> RotMat;

	// <Create Rotation Matrix>
	std::vector<std::array<double, 3>> Vectors; //empty, just to execute "RotateVectors"
	std::array<double, 3> t;
	t[0] = 1;
	t[1] = 1;
	t[2] = 2;
	Vectors.push_back(t);
	//
	std::array<double, 3> V;
	V[0] = Drand() * 2. - 1.;
	V[1] = (Drand() * 2. - 1.);
	V[2] = (Drand() * 2. - 1.);

	while (V[0] * V[0] + V[1] * V[1] + V[2] * V[2] > 1)
	{
		V[0] = Drand() * 2. - 1.;
		V[1] = (Drand() * 2. - 1.);
		V[2] = (Drand() * 2. - 1.);
	}
	double alpha = Drand() * 2 * M_PIl;

	RotMat = Crystal::RotateVectors(Vectors, alpha, V);

	std::cout << "Rotation Matrix is:\n"
		<< RotMat[0] << " \t" << RotMat[1] << " \t" << RotMat[2] << "\n"
		<< RotMat[3] << " \t" << RotMat[4] << " \t" << RotMat[5] << "\n"
		<< RotMat[6] << " \t" << RotMat[7] << " \t" << RotMat[8] << "\n" << std::endl;
	//<\Create Rotation Matrix>

	//<Rotate Matrix>
	ArrayOperators::RotateMatrix(RotRefMat.data(), RotMat.data());

	std::cout << "Rotated Test Matrix is:\n"
		<< RotRefMat[0] << " \t" << RotRefMat[1] << " \t" << RotRefMat[2] << "\n"
		<< RotRefMat[3] << " \t" << RotRefMat[4] << " \t" << RotRefMat[5] << "\n"
		<< RotRefMat[6] << " \t" << RotRefMat[7] << " \t" << RotRefMat[8] << "\n" << std::endl;

	//<\Rotate Matrix>

	//<Test Kansch Algo>


	float* KRot = new float[9]();
	ArrayOperators::KabschRotationMatrixRetrieval3x3(RotRefMat.data(), RefMat.data(), KRot);

	RefMat[0] = 2.0;
	RefMat[1] = 7.0;
	RefMat[2] = 1.0;
	RefMat[3] = 0;
	RefMat[4] = 2.0;
	RefMat[5] = 0;
	RefMat[6] = 0;
	RefMat[7] = 7.0;
	RefMat[8] = 1.0;

	std::cout << "Retrieved Rotation Matrix is:\n"
		<< KRot[0] << " \t" << KRot[1] << " \t" << KRot[2] << "\n"
		<< KRot[3] << " \t" << KRot[4] << " \t" << KRot[5] << "\n"
		<< KRot[6] << " \t" << KRot[7] << " \t" << KRot[8] << "\n" << std::endl;

	//<\Test Kansch Algo>

	return true;
}

bool UnitTest::StreamFileReader(Settings & Options)
{
	Options.LoadStreamFile("/gpfs/cfel/cxi/scratch/user/trostfab/LR17/data/stream files/3fs_JF.stream", "data/data", false);


	return true;
}
