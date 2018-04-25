#pragma once
#include <vector>
#include <array>

#include <cstdio>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <array>

class Settings
{
	//Class containing all parameters the program has and its helpers functions
public:
	Settings();
	~Settings();

	void LoadStreamFile(char * Filename, char* DatasetFIntensity, bool InclMultiHits); //Filename for StreamFile, DatasetFIntensity for the Intensity in CXIFile, include Hits with multiple segmented crystals
	void Echo(std::string output);


	bool echo = false;

	struct HitEvent
	{
		std::array<double, 9> RotMatrix; // in rez space [x1 y1 z1 x2 y2 z2 x3 y3 z3]
		std::string Filename;
		std::string Dataset;
		int Event;
		int SerialNumber;

		float MeanIntensity = 0;
	};

	std::vector<HitEvent> HitEvents;

	// AC Settings
	int halfACMeshEdgeLength = 100; //half EdgeLength of (cube) 3D Volume for auto-correlation without padding (intern +1 zeropadding)
	enum Interpolation {NearestNeighbour = 0, Linear = 1};

	

};

