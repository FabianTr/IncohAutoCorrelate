#include <hdf5.h>


#include "Simulator.h"




void Simulator::WriteIntensityToH5(Detector & Det, std::string Filename, std::string Dataset)
{
	H5::H5File file(Filename, H5F_ACC_RDWR);
	//H5::DataSet dataset = file.openDataSet(DataSet);
}

Simulator::Simulator()
{
}


Simulator::~Simulator()
{
}
