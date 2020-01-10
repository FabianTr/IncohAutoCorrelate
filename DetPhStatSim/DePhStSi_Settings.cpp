#include "DePhStSi_Settings.h"
#include "IniParser.h"


DePhStSi_Settings::DePhStSi_Settings()
{
}


DePhStSi_Settings::~DePhStSi_Settings()
{
}

void DePhStSi_Settings::LoadDetectorDePhStSi_Settings(std::string File, std::string ArgStream) //argstream in "Key1=Val1;Key2=Val2;..."
{
	//parse File
	IniParser parser;
	parser.LoadFile(File);
	//parse ArgStream

	if (ArgStream != "")
	{
		std::vector<std::string> AddArgs = IniParser::StringSplitter(ArgStream, ";");
		for (size_t i = 0; i < AddArgs.size(); i++)
		{
			parser.SetValueStr(AddArgs[i]);
		}
	}

	//Det Parameter
	DetSize = parser.GetValue<int>("DetectorSize", 1024);
	SuSa = parser.GetValue<int>("SubSampling",12);
	ChargeSharingSigma = parser.GetValue<float>("CsSigma", 0.22f);
	DarkNoise = parser.GetValue<float>("DarkNoise",0.0f);

	//file stuff
	OutputPath = parser.GetValue<std::string>("OutputPath","SimulationOutput.h5");
	OutputDataset = parser.GetValue<std::string>("OutputDataset", "data");
	GroundTruthDataset = parser.GetValue<std::string>("GroundTruthDataset", "GroundTruth");

	//statistics
	MeanIntensity = parser.GetValue<float>("MeanIntensity",1.0f);
	Modes = parser.GetValue<float>("Modes",1.0f);
	Pattern = parser.GetValue<unsigned int>("Pattern",1);

	//run Parameter
	Dark = parser.GetValue<bool>("Dark", false);
}

void DePhStSi_Settings::SafeExampleDePhStSi_Settings(std::string File)
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
	parser.SetValue<std::string>("GroundTruthDataset", "GroundTruth");

	//statistics
	parser.SetValue<float>("MeanIntensity", 1.0f);
	parser.SetValue<float>("Modes", 1.0f);
	parser.SetValue<unsigned int>("Pattern", 1);

	//run Parameter
	parser.SetValue<bool>("Dark", false);

	parser.SafeFile(File);
}

