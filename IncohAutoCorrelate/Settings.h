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


class Settings
{
	//Class containing all parameters the program has and its helpers functions
public:
	Settings();
	~Settings();

	void LoadStreamFile(char * Filename, char* DatasetFIntensity, bool InclMultiHits); //Filename for StreamFile, DatasetFIntensity for the Intensity in CXIFile, include Hits with multiple segmented crystals
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



	bool echo = false;

	struct HitEvent
	{
		float RotMatrix[9]; // in rez space [x1 y1 z1 x2 y2 z2 x3 y3 z3] so it is multiplied as: q = M * q_local = {{q_local_1 * x1 + q_local_2 * y1 + q_local_3 * z1},{q_local_1 * x2 + ...},  {...}}
		std::string Filename;
		std::string Dataset;
		int Event;
		int SerialNumber;

		float MeanIntensity = -1;
		int PhotonCount = -1;
	};

	std::vector<HitEvent> HitEvents;
	void SafeHitEventListToFile(char* Filename);
	void SafeHitEventListToFile(std::string Filename);
	void SafeHitEventListToFile(std::string Filename, std::vector<Settings::HitEvent> &HitEventList);
	void SafeHitEventListToFile(std::string Filename, std::vector<Settings::HitEvent> &HitEventList, bool AdditionalInformations, std::unordered_map<std::string, std::string> AdditioInfoMap = std::unordered_map<std::string, std::string>{});
	void LoadHitEventListFromFile(char* Filename);
	void LoadHitEventListFromFile(std::string Filename);


	// AC Settings
	int halfACMeshEdgeLength = 100; //half EdgeLength of (cube) 3D Volume for auto-correlation without padding (intern +1 zeropadding)
	enum Interpolation { NearestNeighbour = 0, Linear = 1 };


	//conversion for ACMesh and Rounding
	struct FloatIntConversion
	{
		float Offset = 0; //shift before rounding to int
		float Step = 1; //steps for discretization
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


	Eigen::Matrix<float, 3, 3> MReference;
};

