#pragma once
#include <vector>
#include <array>

#include <math.h>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <array>
#include <unordered_map>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/info_parser.hpp>

#include <CL/cl.hpp>

#include <omp.h>
#include <Eigen/Dense>

// Conventions:
// ------------
// Matrices stored in 1D arrays (e.g. the rotation matrix) are always row major! (be aware, that for example Matlab and PyQtGraph are Column major but Pylab again row major)
//
// ------------


class Settings
{
	//Class containing all parameters the program has and its helpers functions
private:
	
public:
	const static int XML_HITLIST_VERSION = 8; //Mmm (1.00 -> 100; 1.01 -> 101; 0.1 -> 10)
	const static int INTERNAL_VERSION = 10; //Mmm (1.00 -> 100; 1.01 -> 101; 0.1 -> 10)
	const static int INTERNAL_VERSION_Revision = 6; //rr

	static void SavePropertyTree(boost::property_tree::ptree &pt, std::string Filename)
	{
		std::vector<std::string> filename = Settings::SplitString(Filename, ".");
		std::string filetype = filename[filename.size() - 1];
		std::transform(filetype.begin(), filetype.end(), filetype.begin(), tolower);

		if (filetype == "json")
		{
			boost::property_tree::write_json(Filename, pt);
		}
		else if (filetype == "info")
		{
			boost::property_tree::write_info(Filename, pt);
		}
		else
		{
			if (filetype != "xml")
			{
				std::cout << "WARNING: \"" << filetype << "\" is not supportet.\n  -> Using \"xml\" instead." << std::endl;
			}
			boost::property_tree::write_xml(Filename, pt);
		}
	}

	static void LoadPropertyTree(boost::property_tree::ptree& pt, std::string Filename)
	{
		std::vector<std::string> filename = Settings::SplitString(Filename, ".");
		std::string filetype = filename[filename.size() - 1];
		std::transform(filetype.begin(), filetype.end(), filetype.begin(), tolower);

		if (filetype == "json")
		{
			boost::property_tree::read_json(Filename, pt);
		}
		else if (filetype == "info")
		{
			boost::property_tree::read_info(Filename, pt);
		}
		else
		{
			boost::property_tree::read_xml(Filename, pt);
		}
	}

	static std::string GetVersion()
	{
		std::string ret = std::to_string(INTERNAL_VERSION/100);
		ret += ".";
		if (INTERNAL_VERSION % 100 < 10)
			ret += "0";
		ret += std::to_string(INTERNAL_VERSION % 100) + ".";
		if (INTERNAL_VERSION_Revision / 10 == 0)
			ret += "0";
		ret += std::to_string(INTERNAL_VERSION_Revision);

		return ret;
	}
	
	Settings();
	~Settings();

	struct HitEvent
	{
		float RotMatrix[9]; //ROW MAJOR!!! // in rez space [x1 y1 z1 x2 y2 z2 x3 y3 z3] so it is multiplied as: q = M * q_local = {{q_local_1 * x1 + q_local_2 * y1 + q_local_3 * z1},{q_local_1 * x2 + ...},  {...}}
		std::string Filename;
		std::string Dataset;
		int Event;
		int SerialNumber;

		float MeanIntensity = -1.0f;
		int PhotonCount = -1;

		float HitsPixelRatio = -1.0f; // Number of Pixels with hits / Number of Pixel

		std::string SupplementInfo = "";
	};

	std::vector<HitEvent> HitEvents;

	Eigen::Matrix<float, 3, 3> MReference; //Rows, Colums
	bool echo = true;

	void LoadStreamFile(std::string Filename, std::string DatasetFIntensity, bool InclMultiHits); //Filename for StreamFile, DatasetFIntensity for the Intensity in CXIFile, include Hits with multiple segmented crystals
	std::array<unsigned int,2> ScanH5Files(std::vector<std::string> Filenames, std::vector<std::string> Datasets, bool ResumeOnError = false); //Loads all entrys out of H5 files, returns the 2x2 DetectorSize
	void Echo(std::string output);

	//OpenCL Stuff
	void checkErr(cl_int err, const char * name);
	std::vector<cl::Platform> CL_platforms;
	std::vector<cl::Device> CL_devices;
	cl::Context CL_context;
	cl::Program CL_Program;

	void SetUp_OpenCL();
	//GPU DEVICE POOL
	std::vector<bool> OCL_Available;
	int OCL_ReserveDevice();
	void OCL_FreeDevice(int DeviceIndex);
	unsigned int ThreadSleepForOCLDev = 60000; // Time to wait (per loop) for free device if all devices are occupied in microseconds
	//END DEVICE POOL



	void SafeHitEventListToFile(std::string Filename);

	void SafeHitEventListToFile(std::string Filename, std::vector<Settings::HitEvent> &HitEventList, bool AdditionalInformations = false, std::unordered_map<std::string, std::string> AdditioInfoMap = std::unordered_map<std::string, std::string>{});
	void LoadHitEventListFromFile(std::string Filename);

	void InvertRotationMatrices();



	// AC Settings
	int halfACMeshEdgeLength = 100; //half EdgeLength of (cube) 3D Volume for auto-correlation without padding (intern +1 zeropadding)
	enum Interpolation { NearestNeighbour = 0, Linear = 1 };


	//conversion for ACMesh and Rounding
	struct FloatIntConversion
	{
		float Offset = 0.0f; //shift before rounding to int
		float Step = 0.001f; //steps for discretization
	};
	FloatIntConversion F_I_Conversion;
	inline long FloatToInt(float f)
	{
		return (long)floorf(((f + F_I_Conversion.Offset) / F_I_Conversion.Step) + 0.5f);
	}
	inline long FloatToInt(double f)
	{
		return ( long)floor(((f + F_I_Conversion.Offset) / F_I_Conversion.Step) + 0.5);
	}
	inline float IntToFloat(int i)
	{
		return ((float)i * F_I_Conversion.Step) - (float)(F_I_Conversion.Offset);
	}
	inline float IntToFloat(long i)
	{
		return ((float)i * F_I_Conversion.Step) - (float)(F_I_Conversion.Offset);
	}
	inline float IntToFloat(unsigned long i)
	{
		return ((float)i * F_I_Conversion.Step) - (float)(F_I_Conversion.Offset);
	}

	struct SettingsChecklist {
		bool OpenCL = false;
	};

	SettingsChecklist  Checklist;

	//


	//Sort 
	std::vector<unsigned int> SortHitEventsByIntensity();


	template <typename T>
	static typename std::enable_if<std::is_arithmetic<T>::value>::type SplitString(std::string Input, T * Output, unsigned int ReqSize, std::string delimiter = ";")
	{
		size_t pos = 0;
		std::string token;
		int count = 0;
		while ((pos = Input.find(delimiter)) != std::string::npos)
		{

			if (count >= ReqSize -1)
			{
				throw;
			}

			token = Input.substr(0, pos);

			Output[count] = atof(token.data());

			//std::cout << count << " :" << Val << std::endl;
			Input.erase(0, pos + delimiter.length());
			count++;
		}

		Output[count] = atof(Input.data());

		if (count != ReqSize - 1)
			throw;
	}


	static std::vector<std::string> SplitString(std::string Input,  std::string delimiter = ";")
	{
		std::vector<std::string> Output;
		size_t pos = 0;
		std::string token;
		int count = 0;
		while ((pos = Input.find(delimiter)) != std::string::npos)
		{
			token = Input.substr(0, pos);
			Output.push_back(token);
			Input.erase(0, pos + delimiter.length());
			count++;
		}
		Output.push_back(Input);
		return Output;
	}

};

