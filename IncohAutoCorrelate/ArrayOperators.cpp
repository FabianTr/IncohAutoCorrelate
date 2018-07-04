#include "ArrayOperators.h"
#include <fstream>
#include <iostream>
#include "H5Cpp.h"


#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/foreach.hpp>
#include <set>

namespace ArrayOperators
{
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



}