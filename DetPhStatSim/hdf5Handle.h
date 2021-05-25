#pragma once

#include <fstream>
#include <iostream>
#include "H5Cpp.h"
#include <vector>
#include <string>


namespace hdf5Handle
{
	std::vector<int> GetH5Dimensions(std::string Filename, std::string Dataset);


	void H5Quicksave(float * Array, std::vector<hsize_t> ArrayShape, std::string Filename, std::string Dataset, int compression = 0);
	void H5Quicksave(double * Array, std::vector<hsize_t> ArrayShape, std::string Filename, std::string Dataset, int compression = 0);
	void H5Quicksave(int32_t * Array, std::vector<hsize_t> ArrayShape, std::string Filename, std::string Dataset, int compression = 0);

}

