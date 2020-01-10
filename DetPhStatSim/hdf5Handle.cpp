#include "hdf5Handle.h"
#include <fstream>


namespace hdf5Handle
{

	std::vector<int> GetH5Dimensions(std::string Filename, std::string Dataset)
	{
		std::vector<int> ret;

		H5::H5File file(Filename, H5F_ACC_RDONLY);
		H5::DataSet dataset = file.openDataSet(Dataset);

		//ret.DataType = dataset.getTypeClass();

		H5::DataSpace DS = dataset.getSpace();

		hsize_t *dims = new hsize_t[DS.getSimpleExtentNdims()];
		DS.getSimpleExtentDims(dims);

		for (int i = 0; i < DS.getSimpleExtentNdims(); i++)
		{
			ret.push_back((int)dims[i]);
		}

		return ret;
	}

	void H5Quicksave(float * Array, std::vector<hsize_t> ArrayShape, std::string Filename, std::string Dataset)
	{
		H5::H5File file;
		if(std::ifstream(Filename).is_open())
		{ 
			file = H5::H5File(Filename, H5F_ACC_RDWR);
		}
		else
		{
			file = H5::H5File(Filename, H5F_ACC_TRUNC);
		}
		
		H5::DataSpace dataspace((int)ArrayShape.size(), ArrayShape.data());

		H5::DataSet dataset = file.createDataSet(Dataset, H5::PredType::NATIVE_FLOAT, dataspace); // todo: think how to solve that with datatype
		dataset.write(Array, H5::PredType::NATIVE_FLOAT);
		file.close();
	}
	void H5Quicksave(double * Array, std::vector<hsize_t> ArrayShape, std::string Filename, std::string Dataset)
	{
		H5::H5File file;
		if (std::ifstream(Filename).is_open())
		{
			file = H5::H5File(Filename, H5F_ACC_RDWR);
		}
		else
		{
			file = H5::H5File(Filename, H5F_ACC_TRUNC);
		}

		H5::DataSpace dataspace((int)ArrayShape.size(), ArrayShape.data());
		H5::DataSet dataset = file.createDataSet(Dataset, H5::PredType::NATIVE_DOUBLE, dataspace); // todo: think how to solve that with datatype
		dataset.write(Array, H5::PredType::NATIVE_DOUBLE);
		file.close();
	}
	void H5Quicksave(int32_t * Array, std::vector<hsize_t> ArrayShape, std::string Filename, std::string Dataset)
	{
		H5::H5File file;
		if (std::ifstream(Filename).is_open())
		{
			file = H5::H5File(Filename, H5F_ACC_RDWR);
		}
		else
		{
			file = H5::H5File(Filename, H5F_ACC_TRUNC);
		}

		H5::DataSpace dataspace((int)ArrayShape.size(), ArrayShape.data());
		H5::DataSet dataset = file.createDataSet(Dataset, H5::PredType::NATIVE_INT32, dataspace); // todo: think how to solve that with datatype
		dataset.write(Array, H5::PredType::NATIVE_INT32);
		file.close();
	}




}