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
	const int XML_HITLIST_VERSION = 2; //Mmm (1.00 -> 100; 1.01 -> 101; 0.1 -> 10)
	const int INTERNAL_VERSION = 2; //Mmm (1.00 -> 100; 1.01 -> 101; 0.1 -> 10)

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






	void SafeHitEventListToFile(char* Filename);
	void SafeHitEventListToFile(std::string Filename);
	void SafeHitEventListToFile(std::string Filename, std::vector<Settings::HitEvent> &HitEventList);
	void SafeHitEventListToFile(std::string Filename, std::vector<Settings::HitEvent> &HitEventList, bool AdditionalInformations, std::unordered_map<std::string, std::string> AdditioInfoMap = std::unordered_map<std::string, std::string>{});
	void LoadHitEventListFromFile(char* Filename);
	void LoadHitEventListFromFile(std::string Filename);

	void InvertRotationMatrices();



	// AC Settings
	int halfACMeshEdgeLength = 100; //half EdgeLength of (cube) 3D Volume for auto-correlation without padding (intern +1 zeropadding)
	enum Interpolation { NearestNeighbour = 0, Linear = 1 };


	//conversion for ACMesh and Rounding
	struct FloatIntConversion
	{
		float Offset = 0.0f; //shift before rounding to int
		float Step = 1000.0f; //steps for discretization
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

};

