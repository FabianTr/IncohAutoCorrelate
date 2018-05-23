#include <hdf5.h>
#include <iostream>

#include "Simulator.h"




void Simulator::WriteIntensityToH5(Detector & Det, std::string Filename, std::string Dataset)
{
	#pragma omp critical
	{
		H5::H5File file(Filename, H5F_ACC_RDWR);
		//H5::DataSet dataset = file.openDataSet(DataSet);

		std::cerr << "To be implemented\n";
		file.close();
	}
}

Simulator::Simulator()
{
}


Simulator::~Simulator()
{
}
