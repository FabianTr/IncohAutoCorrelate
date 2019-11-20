#include "Settings.h"
#include "IniParser.h"


Settings::Settings()
{
}


Settings::~Settings()
{
}

void Settings::LoadDetectorSettings(std::string File)
{
	IniParser parser;
	parser.LoadFile(File);

	//Det Parameter
	DetSize = parser.GetValue<int>("DetectorSize", 1024);
	SuSa = parser.GetValue<int>("SubSampling",12);
	ChargeSharingSigma = parser.GetValue<float>("CsSigma", 0.22f);
	DarkNoise = parser.GetValue<float>("DarkNoise",0.0f);

	//file stuff
	OutputPath = parser.GetValue<std::string>("OutputPath","SimulationOutput.h5");
	OutputDataset = parser.GetValue<std::string>("OutputDataset", "data");

	//statistics
	MeanIntensity = parser.GetValue<float>("MeanIntensity",1.0f);
	Modes = parser.GetValue<float>("Modes",1.0f);
	Pattern = parser.GetValue<unsigned int>("Pattern",1);
}

void Settings::SafeExampleSettings(std::string File)
{
	IniParser parser;

	//Det Parameter
	parser.SetValue<int>("DetectorSize", 1024);
	parser.SetValue<int>("SubSampling", 12);
	parser.SetValue<float>("CsSigma", 0.22f);
	parser.SetValue<float>("DarkNoise", 0.0f);

	//file stuff
	parser.SetValue<std::string>("OutputPath", "SimulationOutput.h5");
	parser.SetValue<std::string>("OutputDataset", "data");

	//statistics
	parser.SetValue<float>("MeanIntensity", 1.0f);
	parser.SetValue<float>("Modes", 1.0f);
	parser.SetValue<unsigned int>("Pattern", 1);


	parser.SafeFile(File);
}

