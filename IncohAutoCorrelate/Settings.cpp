#define CL_ENABLE_EXCEPTIONS

#include "Settings.h"
#include <string>
#include <math.h>




Settings::Settings()
{
}


Settings::~Settings()
{
}




void Settings::LoadStreamFile(char* Filename,char* DatasetFIntensity, bool InclMultiHits)
{

	Echo("Start reading stream-file.");

	using std::getline;
	using std::stringstream;

	std::vector<HitEvent> tmpHitEvents;

	std::ifstream File(Filename);
	HitEvent currEvent;
	for (std::string line; getline(File, line);)
	{


		if (line.compare(0, 15, "Image filename:") == 0)
		{
			currEvent = HitEvent();
			stringstream ss(line);
			ss.seekg(16);
			ss >> currEvent.Filename;
			continue;
		}
		if (line.compare(0, 20, "Image serial number:") == 0)
		{
			stringstream ss(line);
			ss.seekg(21);
			ss >> currEvent.SerialNumber;
			continue;
		}
		if (line.compare(0, 6, "Event:") == 0)
		{
			stringstream ss(line);
			ss.seekg(9);
			ss >> currEvent.Event;
			continue;
		}
		if (line.compare("--- Begin crystal") == 0)
		{
			//shots.push_back(CRYSTAL());

			while (getline(File, line))
			{
				if (line.find("Cell parameters") == 0) break;
			}
			for (size_t i = 0; getline(File, line) && i<3; ++i) {
				float x, y, z;
				stringstream ss(line);
				ss.seekg(7);
				ss >> x >> y >> z;
				//         cout << line << endl;
				currEvent.RotMatrix[i * 3 + 0] = x;
				currEvent.RotMatrix[i * 3 + 1] = y;
				currEvent.RotMatrix[i * 3 + 2] = z;
			}
			currEvent.Dataset = DatasetFIntensity;
			tmpHitEvents.push_back(currEvent);

		}
	}

	if (tmpHitEvents.size() == 0)
	{
		std::cerr << "ERROR: empty stream file.\n";
		std::cerr << "   ->: in Settings::LoadStreamFile()\n";
		throw;
	}

	if (!InclMultiHits)//check for double or multiple crystals in one hit and ignore them
	{
		Echo("Remove multiple hits.");
		//check first event:
		if (!(tmpHitEvents[0].Event == tmpHitEvents[1].Event &&  tmpHitEvents[0].Filename == tmpHitEvents[1].Filename))
			HitEvents.push_back(tmpHitEvents[0]);
		//check events [1:end-1]
		for (unsigned int i = 1; i < tmpHitEvents.size() - 1; i++)
		{
			if ((tmpHitEvents[i].Event == tmpHitEvents[i - 1].Event &&  tmpHitEvents[i].Filename == tmpHitEvents[i - 1].Filename) || (tmpHitEvents[i].Event == tmpHitEvents[i + 1].Event && tmpHitEvents[i].Filename == tmpHitEvents[i + 1].Filename))
				continue;
			HitEvents.push_back(tmpHitEvents[i]);
		}
		//check last event
		if (!(tmpHitEvents[tmpHitEvents.size() - 1].Event == tmpHitEvents[tmpHitEvents.size() - 2].Event &&  tmpHitEvents[tmpHitEvents.size() - 1].Filename == tmpHitEvents[tmpHitEvents.size() - 2].Filename))
			HitEvents.push_back(tmpHitEvents[tmpHitEvents.size() - 1]);
		if (echo)
			std::cout << HitEvents.size() << " single hits within " << tmpHitEvents.size() << " total hits loaded.\n";
	}
	else
	{
		HitEvents = tmpHitEvents;
		if (echo)
			std::cout << HitEvents.size() << " total hits loaded.\n";
	}
	Echo("Done reading stream-file.\n");
}



void Settings::Echo(std::string output)
{
	if (echo)
		std::cout << output << "\n";

}



//OpenCL Stuff

void Settings::checkErr(cl_int err, const char * name)
{
	if (err != CL_SUCCESS) {
		std::cerr << "ERROR: " << name
			<< " (" << err << ")" << std::endl;
		exit(EXIT_FAILURE);
	}
}

void Settings::SetUp_OpenCL()
{
	cl_int err;
	cl::Platform::get(&platforms);

	//Find NVIDIA Plattform
	int pl_NVIDIA_Num = -1;
	for (int i = 0; i<(int)platforms.size(); i++)
	{
		if (platforms[i].getInfo<CL_PLATFORM_NAME>()[0] == 'N')
		{
			if (echo)
				std::cout << "OpenCL GPU Plattform found: '" << platforms[i].getInfo<CL_PLATFORM_NAME>() << "'\n";
			pl_NVIDIA_Num = i;
		}
	}
	if (pl_NVIDIA_Num == -1)
	{
		std::cerr << "ERROR: No NVIDIA GPUs found\n";
		std::cerr << "    -> in Settings::SetUp_OpenCL()\n";
		throw;
	}

	//create Context
	cl_context_properties cprops[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[pl_NVIDIA_Num])(),0 };
	cl::Context context(CL_DEVICE_TYPE_GPU, cprops, NULL, NULL, NULL);

	//Devices
	devices = context.getInfo<CL_CONTEXT_DEVICES>();
	if(echo)
		std::cout << "Number of OpenCL devices: " << devices.size() << "\n";
	int NumberOfDevices = (int) devices.size();
	if (NumberOfDevices == 0)
	{
		std::cerr << "ERROR: No OpenCL Devices found\n";
		std::cerr << "    -> in Settings::SetUp_OpenCL()\n";
		throw;
	}

	//Create OCL Device Pool
	OCL_Available.clear();
	for (unsigned int i = 0; i < devices.size(); i++)
	{
		OCL_Available.push_back(true);
	}

	Checklist.OpenCL = true;
}

int Settings::OCL_ReserveDevice()
{
	if (!Checklist.OpenCL)
	{
		std::cerr << "ERROR: OpenCL Setup required. Launch 'SetUp_OpenCL()' first. \n";
		std::cerr << "    -> Settings::OCL_ReserveDevice()\n";
		throw;
	}

	int index = -1;
	#pragma omp critical(OCLPoolManagement) //needs to be serial (avoid raceconditions!)
	{
		for (unsigned int i = 0; i < OCL_Available.size(); i++)
		{
			if (OCL_Available[i] && index == -1)
			{
				index = i;
				OCL_Available[index] = false;
			}
		}
	}

	return index; // if index = -1 => no device available
}

void Settings::OCL_FreeDevice(int DeviceIndex)
{
	if (!Checklist.OpenCL)
	{
		std::cerr << "ERROR: OpenCL Setup required. Launch 'SetUp_OpenCL()' first. \n";
		std::cerr << "    -> Settings::OCL_ReserveDevice()\n";
		throw;
	}

	#pragma omp critical(OCLPoolManagement) //needs to be serial (avoid raceconditions!)
	{
		OCL_Available[DeviceIndex] = true;
	}
}

