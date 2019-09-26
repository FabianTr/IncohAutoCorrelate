#include "ArrayOperators.h"
#include <fstream>
#include <iostream>
#include "H5Cpp.h"


#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/foreach.hpp>
#include <set>
#include <Eigen/SVD>
#include <Eigen/Dense>

namespace ArrayOperators
{
	



	H5Infos GetH5FileInformation(std::string Filename, std::string Dataset)
	{
		H5Infos ret;

		H5::H5File file(Filename, H5F_ACC_RDONLY);
		H5::DataSet dataset = file.openDataSet(Dataset);

		//ret.DataType = dataset.getTypeClass();

		H5::DataSpace DS = dataset.getSpace();

		hsize_t *dims = new hsize_t[DS.getSimpleExtentNdims()];
		DS.getSimpleExtentDims(dims);

		for (size_t i = 0; i < DS.getSimpleExtentNdims(); i++)
		{
			ret.Dimensions.push_back((int)dims[i]);
		}

		delete[] dims;
		return ret;
	}

}