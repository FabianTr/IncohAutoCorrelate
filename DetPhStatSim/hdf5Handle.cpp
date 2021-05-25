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

	void H5Quicksave(float * Array, std::vector<hsize_t> ArrayShape, std::string Filename, std::string Dataset, int compression)
	{
		if (ArrayShape.size() == 0)
		{
			std::cerr << "ArrayShape is not defined!" << std::endl;
			throw;
		}

		H5::H5File file;
		if(std::ifstream(Filename).is_open())
		{ 
			file = H5::H5File(Filename, H5F_ACC_RDWR);
		}
		else
		{
			file = H5::H5File(Filename, H5F_ACC_TRUNC);
		}

		hsize_t* block = new hsize_t(ArrayShape.size());
		block[0] = 1;
		for (size_t i = 1; i < ArrayShape.size(); i++)
		{
			block[i] = ArrayShape[i];
		}
		H5::DSetCreatPropList plist = H5::DSetCreatPropList();
		plist.setChunk((int)ArrayShape.size(), block);
		plist.setDeflate(compression); //compression

		H5::DataSpace dataspace((int)ArrayShape.size(), ArrayShape.data());

		H5::DataSet dataset = file.createDataSet(Dataset, H5::PredType::NATIVE_FLOAT, dataspace,plist); // todo: think how to solve that with datatype
		dataset.write(Array, H5::PredType::NATIVE_FLOAT);
		file.close();

		delete[] block;
	}
	void H5Quicksave(double * Array, std::vector<hsize_t> ArrayShape, std::string Filename, std::string Dataset, int compression)
	{
		if (ArrayShape.size() == 0)
		{
			std::cerr << "ArrayShape is not defined!" << std::endl;
			throw;
		}

		H5::H5File file;
		if (std::ifstream(Filename).is_open())
		{
			file = H5::H5File(Filename, H5F_ACC_RDWR);
		}
		else
		{
			file = H5::H5File(Filename, H5F_ACC_TRUNC);
		}

		hsize_t* block = new hsize_t(ArrayShape.size());
		block[0] = 1;
		for (size_t i = 1; i < ArrayShape.size(); i++)
		{
			block[i] = ArrayShape[i];
		}
		H5::DSetCreatPropList plist = H5::DSetCreatPropList();
		plist.setChunk((int)ArrayShape.size(), block);
		plist.setDeflate(compression); //compression

		H5::DataSpace dataspace((int)ArrayShape.size(), ArrayShape.data());
		H5::DataSet dataset = file.createDataSet(Dataset, H5::PredType::NATIVE_DOUBLE, dataspace,plist); // todo: think how to solve that with datatype
		dataset.write(Array, H5::PredType::NATIVE_DOUBLE);
		file.close();
	}
	void H5Quicksave(int32_t * Array, std::vector<hsize_t> ArrayShape, std::string Filename, std::string Dataset, int compression)
	{
		if (ArrayShape.size() == 0)
		{
			std::cerr << "ArrayShape is not defined!" << std::endl;
			throw;
		}

		H5::H5File file;
		if (std::ifstream(Filename).is_open())
		{
			file = H5::H5File(Filename, H5F_ACC_RDWR);
		}
		else
		{
			file = H5::H5File(Filename, H5F_ACC_TRUNC);
		}

		hsize_t* block = new hsize_t(ArrayShape.size());
		block[0] = 1;
		for (size_t i = 1; i < ArrayShape.size(); i++)
		{
			block[i] = ArrayShape[i];
		}
		H5::DSetCreatPropList plist = H5::DSetCreatPropList();
		plist.setChunk((int)ArrayShape.size(), block);
		plist.setDeflate(compression); //compression

		H5::DataSpace dataspace((int)ArrayShape.size(), ArrayShape.data());
		H5::DataSet dataset = file.createDataSet(Dataset, H5::PredType::NATIVE_INT32, dataspace,plist); // todo: think how to solve that with datatype
		dataset.write(Array, H5::PredType::NATIVE_INT32);
		file.close();
	}




}