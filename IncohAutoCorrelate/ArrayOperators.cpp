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
	void KabschRotationMatrixRetrieval3x3(float * Input, float * Reference, float * RotationMatrix) //Test this!!! (if its wrong correct and redo LR17)
	{
		//Kabsch algorithm, see https://en.wikipedia.org/wiki/Kabsch_algorithm 

		Eigen::Matrix<float, 3, 3> M_in;
		M_in << Input[0], Input[1], Input[2], Input[3], Input[4], Input[5], Input[6], Input[7], Input[8];

		Eigen::Matrix<float, 3, 3> MReference;
		MReference << Reference[0], Reference[1], Reference[2], Reference[3], Reference[4], Reference[5], Reference[6], Reference[7], Reference[8];


		Eigen::JacobiSVD<Eigen::Matrix<float, 3, 3>> svd(M_in.transpose() * MReference.inverse(), Eigen::ComputeFullU | Eigen::ComputeFullV);

		float det;
		det = (svd.matrixV()*svd.matrixU().transpose()).determinant();

		Eigen::Matrix<float, 3, 3> Diag;
		Diag << 1, 0, 0, 0, 1, 0, 0, 0, det;

		Eigen::Matrix<float, 3, 3> Rot;
		Rot = svd.matrixV()*Diag*svd.matrixU().transpose();

		for (int k = 0; k < 3; k++)
		{
			for (int l = 0; l < 3; l++)
				RotationMatrix[l + 3 * k] = Rot(k, l);
		}

		
	}//

	void SafeArrayToFile(char * Filename, double* Array,unsigned int Size, FileType Type)
	{
		switch (Type)
		{
		case Binary:
		{
			std::ofstream FILE(Filename, std::ios::out | std::ofstream::binary);
			FILE.write(reinterpret_cast<char*>(Array), Size * sizeof(double));
			FILE.flush();
			FILE.close();
			if (!FILE) {
				std::cerr << "ERROR: can't write to file: \"" << Filename << "\"\n";
				std::cerr << "    -> in ArrayOperators::SafeArrayToFile()\n";
				throw;
			}
		}
			break;
		case HDF5:
		{
			std::cerr << "ERROR: HDF5 storage is not implemented\n";
			std::cerr << "    -> in ArrayOperators::SafeArrayToFile()\n";
			throw;




		}
			break;
		}
	}
	void SafeArrayToFile(char * Filename, float * Array, unsigned int Size, FileType Type)
	{
		switch (Type)
		{
		case Binary:
		{
			std::ofstream FILE(Filename, std::ios::out | std::ofstream::binary);
			FILE.write(reinterpret_cast<char*>(Array), Size * sizeof(float));
			FILE.flush();
			FILE.close();
			if (!FILE) {
				std::cerr << "ERROR: can't write to file: \"" << Filename << "\"\n";
				std::cerr << "    -> in ArrayOperators::SafeArrayToFile()\n";
				throw;
			}
		}
		break;
		case HDF5:
		{
			std::cerr << "ERROR: HDF5 storage is not implemented\n";
			std::cerr << "    -> in ArrayOperators::SafeArrayToFile()\n";
			throw;
		}
		break;
		}
	}
	void SafeArrayToFile(char * Filename, unsigned int * Array, unsigned int Size, FileType Type)
	{
		switch (Type)
		{
		case Binary:
		{
			std::ofstream FILE(Filename, std::ios::out | std::ofstream::binary);
			FILE.write(reinterpret_cast<char*>(Array), Size * sizeof(unsigned int));
			FILE.flush();
			FILE.close();
			if (!FILE) {
				std::cerr << "ERROR: can't write to file: \"" << Filename << "\"\n";
				std::cerr << "    -> in ArrayOperators::SafeArrayToFile()\n";
				throw;
			}
		}
		break;
		case HDF5:
		{
			std::cerr << "ERROR: HDF5 storage is not implemented\n";
			std::cerr << "    -> in ArrayOperators::SafeArrayToFile()\n";
			throw;
		}
		break;
		}
	}
	
	void SafeArrayToFile(std::string Filename, double* Array, unsigned int Size, FileType Type)
	{
		switch (Type)
		{
		case Binary:
		{
			std::ofstream FILE(Filename, std::ios::out | std::ofstream::binary);
			FILE.write(reinterpret_cast<char*>(Array), Size * sizeof(double));
			FILE.flush();
			FILE.close();
			if (!FILE) {
				std::cerr << "ERROR: can't write to file: \"" << Filename << "\"\n";
				std::cerr << "    -> in ArrayOperators::SafeArrayToFile()\n";
				throw;
			}
		}
		break;
		case HDF5:
		{
			std::cerr << "ERROR: HDF5 storage is not implemented\n";
			std::cerr << "    -> in ArrayOperators::SafeArrayToFile()\n";
			throw;
		}
		break;
		}
	}
	void SafeArrayToFile(std::string Filename, float * Array, unsigned int Size, FileType Type)
	{
		switch (Type)
		{
		case Binary:
		{
			std::ofstream FILE(Filename, std::ios::out | std::ofstream::binary);
			FILE.write(reinterpret_cast<char*>(Array), Size * sizeof(float));
			FILE.flush();
			FILE.close();
			if (!FILE) {
				std::cerr << "ERROR: can't write to file: \"" << Filename << "\"\n";
				std::cerr << "    -> in ArrayOperators::SafeArrayToFile()\n";
				throw;
			}
		}
		break;
		case HDF5:
		{
			std::cerr << "ERROR: HDF5 storage is not implemented\n";
			std::cerr << "    -> in ArrayOperators::SafeArrayToFile()\n";
			throw;
		}
		break;
		}
	}
	void SafeArrayToFile(std::string Filename, unsigned int * Array, unsigned int Size, FileType Type)
	{
		switch (Type)
		{
		case Binary:
		{
			std::ofstream FILE(Filename, std::ios::out | std::ofstream::binary);
			FILE.write(reinterpret_cast<char*>(Array), Size * sizeof(unsigned int));
			FILE.flush();
			FILE.close();
			if (!FILE) {
				std::cerr << "ERROR: can't write to file: \""<< Filename << "\"\n";
				std::cerr << "    -> in ArrayOperators::SafeArrayToFile()\n";
				throw;
			}
		}
		break;
		case HDF5:
		{
			std::cerr << "ERROR: HDF5 storage is not implemented\n";
			std::cerr << "    -> in ArrayOperators::SafeArrayToFile()\n";
			throw;
		}
		break;
		}
	}
	void SafeArrayToFile(std::string Filename, unsigned long * Array, unsigned int Size, FileType Type)
	{
		switch (Type)
		{
		case Binary:
		{
			std::ofstream FILE(Filename, std::ios::out | std::ofstream::binary);
			FILE.write(reinterpret_cast<char*>(Array), Size * sizeof(unsigned long));
			FILE.flush();
			FILE.close();
			if (!FILE) {
				std::cerr << "ERROR: can't write to file: \"" << Filename << "\"\n";
				std::cerr << "    -> in ArrayOperators::SafeArrayToFile()\n";
				throw;
			}
		}
		break;
		case HDF5:
		{
			std::cerr << "ERROR: HDF5 storage is not implemented\n";
			std::cerr << "    -> in ArrayOperators::SafeArrayToFile()\n";
			throw;
		}
		break;
		}
	}


}