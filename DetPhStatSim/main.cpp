#include <cstdio>
#include <iostream>
#include <fstream>
#include <algorithm>

#include "ArrayMaths.h"

#include "hdf5Handle.h"
#include "ProfileTime.h"


void Test()
{
	ProfileTime profiler;

	profiler.Tic();

	float * M = new float[1024*1024*16*16]();
	profiler.Toc(true);
	
	ArrayMaths::GetNegativeBinomialArray(M, 1024 * 1024 * 16 * 16, (0.01 / (16.0*16.0)), (1.0 / (16.0*16.0)));

	profiler.Toc(true);

	float * Kernel = new float[31 * 31];

	ArrayMaths::CreateGaussKernel(Kernel, 31 , 0.22 * 16, true);

	ArrayMaths::Convolve2D(M, { 1024 * 1024 * 16 * 16 }, Kernel, { 31,31 });

	profiler.Toc(true);

	float * PixM = new float[1024 * 1024]();
	ArrayMaths::Pixelize2DArray(M, { 1024 * 16, 1024 * 16 }, PixM, { 16,16 });

	profiler.Toc(true);

	//std::vector<hsize_t> Shape;
	//Shape.push_back(9);
	//Shape.push_back(9);


	//for (size_t i = 0; i < 25; i++)
	//{
	//	for (size_t j = 0; j < 25; j++)
	//	{
	//		std::cout << PixM[j + 25 * i] << "\t";
	//	}
	//	std::cout << std::endl;
	//}

	hdf5Handle::H5Quicksave(Kernel, { 31,31 }, "/home/trostfab/kernel.h5", "data");
	hdf5Handle::H5Quicksave(M, {1024*16,1024*16}, "/home/trostfab/M.h5", "data");
	hdf5Handle::H5Quicksave(PixM, { 1024,1024}, "/home/trostfab/PixM.h5", "data");

	return;
	while (true)
	{

		float * M = new float[100];


		float Modes = 0;

		std::cin >> Modes;

		ArrayMaths::GetNegativeBinomialArray(M, 100, 1, Modes);


		for (int i = 0; i < 10; i++)
		{
			for (int j = 0; j < 10; j++)
			{
				std::cout << M[j + 10 * i] << "\t";
			}
			std::cout << std::endl;
		}
		std::cout << "\n"<< std::endl;

	}
}


int main(int argc, char** argv)
{
	std::cout << "Detector Photon Statistics Simulation\n";
	std::cout << "*************************************\n" << std::endl;


	Test();


	std::cout << "the end." << std::endl;
    return 0;
}