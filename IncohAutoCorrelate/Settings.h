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
#include <CL/cl.hpp>

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
	std::vector<cl::Platform> platforms;
	std::vector<cl::Device> devices;


	void SetUp_OpenCL();

	//


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

	// AC Settings
	int halfACMeshEdgeLength = 100; //half EdgeLength of (cube) 3D Volume for auto-correlation without padding (intern +1 zeropadding)
	enum Interpolation {NearestNeighbour = 0, Linear = 1};


	//conversion for ACMesh and Rounding
	struct FloatIntConversion
	{
		float Offset=0; //shift before rounding to int
		float Step=1; //steps for discretization
	};
	FloatIntConversion F_I_Conversion;
	inline int FloatToInt(float f)
	{
		return (int)floorf(((f + F_I_Conversion.Offset ) / F_I_Conversion.Step) + 0.5f);
	}
	inline float IntToFloat( int i)
	{
		return (float)(i * F_I_Conversion.Step) - (float)(F_I_Conversion.Offset );
	}


};

