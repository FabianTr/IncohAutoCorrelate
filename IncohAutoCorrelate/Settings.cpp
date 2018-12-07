#define CL_ENABLE_EXCEPTIONS

#include "Settings.h"
#include <string>
#include <math.h>
#include <Eigen/SVD>
#include <fstream>
#include "H5Cpp.h"
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/foreach.hpp>
#include <set>

#include "ArrayOperators.h"


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

			while (getline(File, line))
			{
				if (line.find("Cell parameters") == 0) break;
			}
			Eigen::Matrix<float, 3, 3> Mprime;
			for (size_t i = 0; getline(File, line) && i<3; ++i) {
				float x, y, z;
				stringstream ss(line);
				ss.seekg(7);
				ss >> x >> y >> z;
				//         cout << line << endl;

				Mprime(0, i) = x;
				Mprime(1, i) = y;
				Mprime(2, i) = z;
			}
			//Kabsch algorithm, see https://en.wikipedia.org/wiki/Kabsch_algorithm 
			Eigen::JacobiSVD<Eigen::Matrix<float, 3, 3>> svd(Mprime.transpose() * MReference.inverse(), Eigen::ComputeFullU | Eigen::ComputeFullV);

			float det;
			det = (svd.matrixV()*svd.matrixU().transpose()).determinant();

			Eigen::Matrix<float, 3, 3> Diag;
			Diag << 1, 0, 0, 0, 1, 0, 0, 0, det;

			Eigen::Matrix<float, 3, 3> Rot;
			Rot = svd.matrixV()*Diag*svd.matrixU().transpose();

			//currEvent.RotMatrix[i * 3 + 0] = x;
			//currEvent.RotMatrix[i * 3 + 1] = y;
			//currEvent.RotMatrix[i * 3 + 2] = z;


			for (int k = 0; k < 3; k++)
			{
				for (int l = 0; l < 3; l++)
				{
					currEvent.RotMatrix[l + 3*k] = Rot(k, l);
				}
			}
			//std::cout << "\n\n";
			//std::cout <<"Kabsch Rot\n" << Rot << "\n\n";

			//int stop;
			//std::cin >> stop;

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


std::array<unsigned int, 2> Settings::ScanH5Files(std::vector<std::string> Filenames, std::vector<std::string> Datasets, bool ResumeOnError)
{
	if (Filenames.size() != Datasets.size())
	{
		std::cerr << "ERROR: Filenames and Datasets must have the same amount of Entries\n";
		std::cerr << "     -> in Settings::ScanH5Files()\n";
		throw;
	}

	std::vector<HitEvent> tmpHitEvents;
	std::array<unsigned int,2> DetectorSize = { 0, 0 };

	unsigned int EventCount = 0;
	//Crawl through H5 Files and Create Events
	for (unsigned int i = 0; i < (unsigned int)Filenames.size(); i++)
	{
		H5::H5File file(Filenames[i], H5F_ACC_RDONLY);
		H5::DataSet dataset = file.openDataSet(Datasets[i]);

		//Check plausibility
		if (dataset.getTypeClass() != H5T_FLOAT)
		{
			std::cerr << "ERROR: Intensity data is not stored as floating point numbers.\n";
			std::cerr << "     -> in Settings::ScanH5Files()\n";
			throw;
		}
		H5::DataSpace DS = dataset.getSpace();
		if (DS.getSimpleExtentNdims() != 3) //check if shape is [nE][nx][ny] ( or [ny][nx][nE] ) nE =^ Number of Slices(Events) // here only [nE][nx][ny] will be accepted
		{
			std::cerr << "ERROR: Intensity data dimension is not 3, but " << DS.getSimpleExtentNdims() << " => shape is not (N, nx, ny)\n";
			std::cerr << "     -> in Settings::ScanH5Files()\n";
			throw;
		}

		hsize_t dims[3];
		DS.getSimpleExtentDims(dims, NULL);

		if (i == 0)
		{
			//Set Detector size
			DetectorSize[0] = (unsigned int)dims[1];
			DetectorSize[1] = (unsigned int)dims[2];
		}
		else
		{
			//Check Detector size
			if (DetectorSize[0] != (unsigned int)dims[1] || DetectorSize[1] != (unsigned int)dims[2])
			{
				if (ResumeOnError)
				{
					std::cerr << "WARNING: Detector Size needs to be constant for all Files.\n";
					std::cerr << "     -> Ignore File \"" << Filenames[i] << "\" and continue\n";
					continue;
				}
				else
				{
					std::cerr << "ERROR: Detector Size needs to be constant for all Files.\n";
					std::cerr << "     -> in Settings::ScanH5Files()\n";
					throw;
				}
			}
		}


		for (unsigned int j = 0; j < (unsigned int)dims[0]; j++)
		{
			HitEvent newEvent;
			newEvent.Filename = Filenames[i];
			newEvent.Dataset = Datasets[i];

			newEvent.Event = j;
			newEvent.SerialNumber = j;

			newEvent.RotMatrix[0] = 1.0f;
			newEvent.RotMatrix[1] = 0.0f;
			newEvent.RotMatrix[2] = 0.0f;
			newEvent.RotMatrix[3] = 0.0f;
			newEvent.RotMatrix[4] = 1.0f;
			newEvent.RotMatrix[5] = 0.0f;
			newEvent.RotMatrix[6] = 0.0f;
			newEvent.RotMatrix[7] = 0.0f;
			newEvent.RotMatrix[8] = 1.0f;

			tmpHitEvents.push_back(newEvent);

			EventCount++;
		}

		DS.close();

	}


	if (EventCount > 0)
	{
		HitEvents = tmpHitEvents;
		if (echo)
			std::cout << EventCount << " events found\n";
	}
	else
	{
		std::cout << "No Events found\n";
	}

	return DetectorSize;

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
	Echo("Setup OpenCL - devices");
	cl_int err;
	cl::Platform::get(&CL_platforms);

	//Find NVIDIA Plattform
	int pl_NVIDIA_Num = -1;
	for (int i = 0; i<(int)CL_platforms.size(); i++)
	{
		if (CL_platforms[i].getInfo<CL_PLATFORM_NAME>()[0] == 'N')
		{
			if (echo)
				std::cout << "OpenCL GPU Plattform found: '" << CL_platforms[i].getInfo<CL_PLATFORM_NAME>() << "'\n";
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
	cl_context_properties cprops[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(CL_platforms[pl_NVIDIA_Num])(),0 };
	CL_context = cl::Context(CL_DEVICE_TYPE_GPU, cprops, NULL, NULL, NULL);

	//Devices
	CL_devices = CL_context.getInfo<CL_CONTEXT_DEVICES>();
	if(echo)
		std::cout << "Number of OpenCL devices: " << CL_devices.size() << "\n";
	int NumberOfDevices = (int) CL_devices.size();
	if (NumberOfDevices == 0)
	{
		std::cerr << "ERROR: No OpenCL Devices found\n";
		std::cerr << "    -> in Settings::SetUp_OpenCL()\n";
		throw;
	}


	

	//load Kernel FIle
	std::ifstream file("kernel.cl");
	std::string prog(std::istreambuf_iterator<char>(file), (std::istreambuf_iterator<char>()));
	file.close();

	if (prog == "")
	{
		std::cerr << "ERROR: kernel.cl not found\n";
		std::cerr << "   ->  in Settings::SetUp_OpenCL()\n";
		throw;
	}

	//compile Kernel
	Echo("Compile OpenCL kernels");

	cl::Program::Sources source(1, std::make_pair(prog.c_str(), prog.length() + 1));
	CL_Program = cl::Program(CL_context, source);
	err = CL_Program.build(CL_devices);
	if (err != 0)
	{
		for (cl::Device dev : CL_devices)
		{
			// Check the build status
			cl_build_status status = CL_Program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(dev);
			if (status != CL_BUILD_ERROR)
				continue;

			// Get the build log
			std::string name = dev.getInfo<CL_DEVICE_NAME>();
			std::string buildlog = CL_Program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev);
			std::cerr << "Build log for " << name << ":" << std::endl
				<< buildlog << std::endl;
		}
	}
	checkErr(err, "Kernel compile error. Check kernel.cl!");




	//Create OCL Device Pool
	OCL_Available.clear();
	for (unsigned int i = 0; i < CL_devices.size(); i++)
	{
		OCL_Available.push_back(true);
	}

	Checklist.OpenCL = true;

	std::cout << "OpenCL kernels successful compiled.\n";
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

void Settings::SafeHitEventListToFile(char * Filename)
{
	//HitEventList is stored in xml format
	using boost::property_tree::ptree;
	ptree pt;

	pt.put("root.Info.Size", HitEvents.size());
	for (unsigned int i = 0; i < HitEvents.size(); i++)
	{
		std::string path = "root.Content.";
		path = path + std::to_string(i);

		pt.put(path + ".Filename", HitEvents[i].Filename);
		pt.put(path + ".Dataset", HitEvents[i].Dataset);
		pt.put(path + ".Event", HitEvents[i].Event);
		pt.put(path + ".SerialNumber", HitEvents[i].SerialNumber);
		pt.put(path + ".MeanIntensity", HitEvents[i].MeanIntensity);
		pt.put(path + ".PhotonCount", HitEvents[i].PhotonCount);

		pt.put(path + ".R0", HitEvents[i].RotMatrix[0]);
		pt.put(path + ".R1", HitEvents[i].RotMatrix[1]);
		pt.put(path + ".R2", HitEvents[i].RotMatrix[2]);
		pt.put(path + ".R3", HitEvents[i].RotMatrix[3]);
		pt.put(path + ".R4", HitEvents[i].RotMatrix[4]);
		pt.put(path + ".R5", HitEvents[i].RotMatrix[5]);
		pt.put(path + ".R6", HitEvents[i].RotMatrix[6]);
		pt.put(path + ".R7", HitEvents[i].RotMatrix[7]);
		pt.put(path + ".R8", HitEvents[i].RotMatrix[8]);
	}
	boost::property_tree::write_xml(Filename,pt);
}
void Settings::SafeHitEventListToFile(std::string Filename)
{
	SafeHitEventListToFile(Filename, HitEvents);
}

void Settings::SafeHitEventListToFile(std::string Filename, std::vector<Settings::HitEvent> &HitEventList)
{

	//HitEventList is stored in xml format

	SafeHitEventListToFile(Filename, HitEventList, false);
}

void Settings::SafeHitEventListToFile(std::string Filename, std::vector<Settings::HitEvent> &HitEventList, bool AdditionalInformations, std::unordered_map<std::string, std::string> AdditioInfoMap)
{

	//HitEventList is stored in xml format
	using boost::property_tree::ptree;
	ptree pt;

	pt.put("root.Info.Size", HitEventList.size());
	pt.put("root.Info.Version", XML_HITLIST_VERSION);   //ACTIVATE for Version 0.02

	if (AdditionalInformations)
	{
		for (auto it = AdditioInfoMap.begin(); it != AdditioInfoMap.end(); it++)
		{
			std::string key, val;
			key= it->first;
			val = it->second;

			pt.put("root.Info." + key, val);
		}
	}


	for (unsigned int i = 0; i < HitEventList.size(); i++)
	{
		std::string path = "root.Content.N"; //std::string path = "root.Content.N"; //ACTIVATE for Version 0.02
		path = path + std::to_string(i);

		pt.put(path + ".Filename", HitEventList[i].Filename);
		pt.put(path + ".Dataset", HitEventList[i].Dataset);
		pt.put(path + ".Event", HitEventList[i].Event);
		pt.put(path + ".SerialNumber", HitEventList[i].SerialNumber);
		pt.put(path + ".MeanIntensity", HitEventList[i].MeanIntensity);
		pt.put(path + ".PhotonCount", HitEventList[i].PhotonCount);

		////Version 0.01
		//pt.put(path + ".R0", HitEventList[i].RotMatrix[0]);
		//pt.put(path + ".R1", HitEventList[i].RotMatrix[1]);
		//pt.put(path + ".R2", HitEventList[i].RotMatrix[2]);
		//pt.put(path + ".R3", HitEventList[i].RotMatrix[3]);
		//pt.put(path + ".R4", HitEventList[i].RotMatrix[4]);
		//pt.put(path + ".R5", HitEventList[i].RotMatrix[5]);
		//pt.put(path + ".R6", HitEventList[i].RotMatrix[6]);
		//pt.put(path + ".R7", HitEventList[i].RotMatrix[7]);
		//pt.put(path + ".R8", HitEventList[i].RotMatrix[8]);
		////

		std::string RMOutput = std::to_string(HitEventList[i].RotMatrix[0]);
		for (int j = 1; j < 9; j++)
		{
			RMOutput += "; " + std::to_string(HitEventList[i].RotMatrix[j]);
		}
		pt.put(path + ".Rotation", RMOutput);

	}
	boost::property_tree::write_xml(Filename, pt);
}


void Settings::InvertRotationMatrices()
{

	for (unsigned int i = 0; i < HitEvents.size(); i++)
	{
		Eigen::Matrix3d RotM;
		RotM << HitEvents[i].RotMatrix[0], HitEvents[i].RotMatrix[1], HitEvents[i].RotMatrix[2], HitEvents[i].RotMatrix[3], HitEvents[i].RotMatrix[4], HitEvents[i].RotMatrix[5], HitEvents[i].RotMatrix[6], HitEvents[i].RotMatrix[7], HitEvents[i].RotMatrix[8];
	
		Eigen::Matrix3d RotInv = RotM.inverse();

		HitEvents[i].RotMatrix[0] = RotInv(0, 0);
		HitEvents[i].RotMatrix[1] = RotInv(0, 1);
		HitEvents[i].RotMatrix[2] = RotInv(0, 2);
		HitEvents[i].RotMatrix[3] = RotInv(1, 0);
		HitEvents[i].RotMatrix[4] = RotInv(1, 1);
		HitEvents[i].RotMatrix[5] = RotInv(1, 2);
		HitEvents[i].RotMatrix[6] = RotInv(2, 0);
		HitEvents[i].RotMatrix[7] = RotInv(2, 1);
		HitEvents[i].RotMatrix[8] = RotInv(2, 2);
	}

}

void Settings::LoadHitEventListFromFile(char * Filename)
{
	std::string Path;
	Path = Filename;
	LoadHitEventListFromFile(Path);
}

void Settings::LoadHitEventListFromFile(std::string Filename)
{
	using boost::property_tree::ptree;
	ptree pt;
	boost::property_tree::read_xml(Filename, pt);

	unsigned int Size = 0;
	unsigned int Version = 1;
	Size = pt.get<unsigned int>("root.Info.Size", -1);
	Version = pt.get<unsigned int>("root.Info.Version", 1);

	if (Size == -1)
	{
		std::cerr << "ERROR: Empty or not readable xml HitEvents File\n";
		std::cerr << "    -> in  Settings::LoadHitEventListFromFile()\n";
		throw;
	}
	if (Size == 0)
		return;

	HitEvents.clear();

	std::string pathbase;
	if (pt.get<std::string>("root.content.N0.Filename", "_FAIL:OLD-VER") == "_FAIL:OLD-VER" && pt.get<std::string>("root.Content.N0.Filename", "_FAIL:OLD-VER") == "_FAIL:OLD-VER")
	{
		pathbase = "root.content.";
	}
	else
	{
		pathbase = "root.content.N";
		if (Version >= 2)
			pathbase = "root.Content.N";
	}

	for (unsigned int i = 0; i < Size; i++)
	{
		std::string path = pathbase + std::to_string(i);

		HitEvent tmp;
		tmp.Filename = pt.get<std::string>(path + ".Filename");
		tmp.Dataset = pt.get<std::string>(path + ".Dataset");
		tmp.Event = pt.get<int>(path + ".Event");
		tmp.SerialNumber = pt.get<int>(path + ".SerialNumber");
		tmp.MeanIntensity = pt.get<float>(path + ".MeanIntensity");
		tmp.PhotonCount = pt.get<int>(path + ".PhotonCount");

		switch (Version)
		{
		case 1:
			tmp.RotMatrix[0] = pt.get<float>(path + ".R0",1.0);
			tmp.RotMatrix[1] = pt.get<float>(path + ".R1",0.0);
			tmp.RotMatrix[2] = pt.get<float>(path + ".R2",0.0);
			tmp.RotMatrix[3] = pt.get<float>(path + ".R3",0.0);
			tmp.RotMatrix[4] = pt.get<float>(path + ".R4",1.0);
			tmp.RotMatrix[5] = pt.get<float>(path + ".R5",0.0);
			tmp.RotMatrix[6] = pt.get<float>(path + ".R6",0.0);
			tmp.RotMatrix[7] = pt.get<float>(path + ".R7",0.0);
			tmp.RotMatrix[8] = pt.get<float>(path + ".R8",1.0);
			break;
		case 2:
		default:
			std::string RotationMatrixInput = "";
			RotationMatrixInput = pt.get<std::string>(path + ".Rotation", "1.0; 0.0; 0.0; 0.0; 1.0; 0.0; 0.0; 0.0; 1.0");

			std::string delimiter = ";";


			float Val[9];

			size_t pos = 0;
			std::string token;
			int count = 0;
			bool ParsingAlive = true;
			while ((pos = RotationMatrixInput.find(delimiter)) != std::string::npos) 
			{

				if (count > 7)
				{
					std::cerr << "WARNING: Invalid rotation matrix format! use \"a1; a2; a3; b1; b2; b3; c1; c2; c3\" \n";
					std::cerr << "    -> Rotation matrix is set to \"1.0; 0.0; 0.0; 0.0; 1.0; 0.0; 0.0; 0.0; 1.0 \"\n";
					ParsingAlive = false;
					continue;
				}

				token = RotationMatrixInput.substr(0, pos);

				Val[count] = atof(token.data());

				//std::cout << count << " :" << Val << std::endl;
				RotationMatrixInput.erase(0, pos + delimiter.length());
				count++;
			}
			if (ParsingAlive)
			{
				Val[count] = atof(RotationMatrixInput.data());
				for (int i = 0; i < 9; i++)
				{
					tmp.RotMatrix[i] = Val[i];
				}
			}
			else
			{
				tmp.RotMatrix[0] = 1.0;
				tmp.RotMatrix[1] = 0.0;
				tmp.RotMatrix[2] = 0.0;
				tmp.RotMatrix[3] = 0.0;
				tmp.RotMatrix[4] = 1.0;
				tmp.RotMatrix[5] = 0.0;
				tmp.RotMatrix[6] = 0.0;
				tmp.RotMatrix[7] = 0.0;
				tmp.RotMatrix[8] = 1.0;
			}
			//std::cout << count << " :" << Val << std::endl;

		}

		HitEvents.push_back(tmp);
	}

	//Old Versions
	{
		////Version 0.01
		//void Settings::LoadHitEventListFromFile(std::string Filename)
		//{
		//	using boost::property_tree::ptree;
		//	ptree pt;
		//	boost::property_tree::read_xml(Filename, pt);

		//	unsigned int Size = 0;
		//	Size = pt.get<unsigned int>("root.Info.Size", -1);

		//	if (Size == -1)
		//	{
		//		std::cerr << "ERROR: Empty or not readable xml HitEvents File\n";
		//		std::cerr << "    -> in  Settings::LoadHitEventListFromFile()\n";
		//		throw;
		//	}
		//	if (Size == 0)
		//		return;

		//	HitEvents.clear();

		//	std::string pathbase;
		//	if (pt.get<std::string>("root.content.N0.Filename", "_FAIL:OLD-VER") == "_FAIL:OLD-VER")
		//	{
		//		pathbase = "root.content.";
		//	}
		//	else
		//	{
		//		pathbase = "root.content.N";
		//	}

		//	for (unsigned int i = 0; i < Size; i++)
		//	{
		//		std::string path = pathbase + std::to_string(i);

		//		HitEvent tmp;
		//		tmp.Filename = pt.get<std::string>(path + ".Filename");
		//		tmp.Dataset = pt.get<std::string>(path + ".Dataset");
		//		tmp.Event = pt.get<int>(path + ".Event");
		//		tmp.SerialNumber = pt.get<int>(path + ".SerialNumber");
		//		tmp.MeanIntensity = pt.get<float>(path + ".MeanIntensity");
		//		tmp.PhotonCount = pt.get<int>(path + ".PhotonCount");

		//		tmp.RotMatrix[0] = pt.get<float>(path + ".R0");
		//		tmp.RotMatrix[1] = pt.get<float>(path + ".R1");
		//		tmp.RotMatrix[2] = pt.get<float>(path + ".R2");
		//		tmp.RotMatrix[3] = pt.get<float>(path + ".R3");
		//		tmp.RotMatrix[4] = pt.get<float>(path + ".R4");
		//		tmp.RotMatrix[5] = pt.get<float>(path + ".R5");
		//		tmp.RotMatrix[6] = pt.get<float>(path + ".R6");
		//		tmp.RotMatrix[7] = pt.get<float>(path + ".R7");
		//		tmp.RotMatrix[8] = pt.get<float>(path + ".R8");

		//		HitEvents.push_back(tmp);
		//	}


		//}
	}
}


std::vector<unsigned int> Settings::SortHitEventsByIntensity()
{

	std::vector<std::pair<float, unsigned int>> IntIndPair;
	IntIndPair.resize(HitEvents.size());

	#pragma omp parallel for
	for (unsigned int i = 0; i < HitEvents.size(); i++)
	{
		IntIndPair[i] = { HitEvents[i].MeanIntensity ,i };
	}

	std::sort(IntIndPair.begin(), IntIndPair.end(), [](const std::pair<float, unsigned int> & a, const std::pair<float, unsigned int> & b) {return a.first > b.first; });
	
	std::vector<unsigned int> SortedInd;
	SortedInd.resize(HitEvents.size());

	#pragma omp parallel for
	for (unsigned int i = 0; i < HitEvents.size(); i++)
	{
		SortedInd[i] = IntIndPair[i].second;
	}


	return SortedInd;
}