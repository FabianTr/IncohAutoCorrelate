#include "MainRunModes.h"

#include <iostream>

#include "ProfileTime.h"
#include "Detector.h"


//Helper
std::vector<std::string> MainRunModes::CSV_Splitter(std::string Input, std::string delimiter)
{
	//Splits "delimiter"-seperated Values within string to vector
	std::vector<std::string> Output;
	size_t pos = 0;
	std::string token;
	bool ParsingAlive = true;
	while ((pos = Input.find(delimiter)) != std::string::npos)
	{
		token = Input.substr(0, pos);
		Output.push_back(token);
		Input.erase(0, pos + delimiter.length());
	}
	Output.push_back(Input);
	return Output;
}


//Load settings
RunIAC::CreateDataEval_Settings MainRunModes::LoadEvaluationSettings(std::string Filename, Settings &Options)
{
	//Load Evaluation settings from XML-File
	RunIAC::CreateDataEval_Settings EVS;

	using boost::property_tree::ptree;
	ptree pt;
	boost::property_tree::read_xml(Filename, pt);

	unsigned int Version = pt.get<unsigned int>("root.Info.Version", Options.INTERNAL_VERSION);

	//->Input Files
	EVS.XML_Path = pt.get<std::string>("root.EvalSettings.InputFiles.EventList_Path", "");
	EVS.PixelMap_Path = pt.get<std::string>("root.EvalSettings.InputFiles.PixelMap_Path", "");
	EVS.PixelMap_DataSet = pt.get<std::string>("root.EvalSettings.InputFiles.PixelMap_Dataset", "");
	EVS.PixelMask_Path = pt.get<std::string>("root.EvalSettings.InputFiles.PixelMask_Path", "");
	EVS.PixelMask_Dataset = pt.get<std::string>("root.EvalSettings.InputFiles.PixelMask_Dataset", "");

	//EVS.UseExistingAvInt requires EVS.Out_AvIntensity_Path also requires the right mean intensity and photon count in xml-event list
	EVS.UseExistingAvInt = pt.get<bool>("root.EvalSettings.InputFiles.UseExistingAvInt", false); // requires EVS.Out_AvIntensity_Path also requires the right mean intensity and photon count in xml-event list
	
	//->Output Files
	EVS.Out_Final_AC_Path = pt.get<std::string>("root.EvalSettings.OutputFiles.AC_Final_Path", "");
	EVS.Out_AvIntensity_Path = pt.get<std::string>("root.EvalSettings.OutputFiles.AvIntensity_Path", "");
	EVS.Out_Cq_Path = pt.get<std::string>("root.EvalSettings.OutputFiles.Cq_Merged_Path", "");
	EVS.Out_Cq_small_Path = pt.get<std::string>("root.EvalSettings.OutputFiles.Cq_small_Path", "");
	EVS.Out_ACuw_Path = pt.get<std::string>("root.EvalSettings.OutputFiles.ACuw_Path", "");
	EVS.Out_Q_Vector = pt.get<std::string>("root.EvalSettings.OutputFiles.Q_Vector", "");

	//-> Output Mesh
	EVS.MeshSize = pt.get<int>("root.EvalSettings.OutputMesh.MeshSize", 201);
	EVS.QZoom = pt.get<float>("root.EvalSettings.OutputMesh.QZoom",1.0f);
	EVS.AngularAveraged = pt.get<bool>("root.EvalSettings.OutputMesh.AngularAveraged", false);
	EVS.DoubleMap = pt.get<bool>("root.EvalSettings.OutputMesh.DoubleMap",true);
	EVS.AC_FirstMap_Flags.InterpolationMode = (Settings::Interpolation)pt.get<int>("root.EvalSettings.OutputMesh.AC_FirstMap_Flags.InterpolationMode", Settings::Interpolation::NearestNeighbour);
	EVS.AC_SecondMap_Flags.InterpolationMode = (Settings::Interpolation)pt.get<int>("root.EvalSettings.OutputMesh.AC_SecondMap_Flags.InterpolationMode", Settings::Interpolation::NearestNeighbour);

	//->PhotonThreshold
	EVS.PhotonOffset = pt.get<float>("root.EvalSettings.PhotonThreshold.PhotonOffset", 0.0f);
	EVS.PhotonStep = pt.get<float>("root.EvalSettings.PhotonThreshold.PhotonStep", 1.0f);

	//->Misc
	EVS.RestrictStackToBoundaries = pt.get<bool>("root.EvalSettings.Misc.RestrictStackToBoundaries", false);
	EVS.LowerBoundary = pt.get<unsigned int>("root.EvalSettings.Misc.LowerBoundary", 0);
	EVS.UpperBoundary = pt.get<unsigned int>("root.EvalSettings.Misc.UpperBoundary", 10);
	EVS.FractionalCq = pt.get<bool>("root.EvalSettings.Misc.FractionalCq",false);
	EVS.SizeOfCqFraction = pt.get<unsigned int>("root.EvalSettings.Misc.SizeOfCqFraction",10);
	EVS.InvertRotMatrix = pt.get<bool>("root.EvalSettings.Misc.InvertRotMatrix", false);
	EVS.EchoLevel = pt.get<int>("root.EvalSettings.Misc.EchoLevel", 3);


	//switch off echo if wanted
	if (EVS.EchoLevel == 0)
		Options.echo = false;

	return EVS;
}

Statistics::StatisticsSettings MainRunModes::LoadStatisticSettings(std::string Filename, Settings & Options)
{
	Statistics::StatisticsSettings StatSet;

	using boost::property_tree::ptree;
	ptree pt;
	boost::property_tree::read_xml(Filename, pt);

	unsigned int Version = pt.get<unsigned int>("root.Info.Version", Options.INTERNAL_VERSION);

	//PixelHisto
	{
		StatSet.PixelHistogramSettings.OutputPath = pt.get<std::string>("root.StatisticsSettings.PixelHistograms.OutputPath", "");

		StatSet.PixelHistogramSettings.Bins = pt.get<unsigned int>("root.StatisticsSettings.PixelHistograms.Bins", 100);
		StatSet.PixelHistogramSettings.SmalestValue = pt.get<double>("root.StatisticsSettings.PixelHistograms.SmallestValue", 0.0);
		StatSet.PixelHistogramSettings.LargestValue = pt.get<double>("root.StatisticsSettings.PixelHistograms.LargestValue", 100.0);

		StatSet.PixelHistogramSettings.Normalized = pt.get<bool>("root.StatisticsSettings.PixelHistograms.Normalize", true);
	}

	return StatSet;
}

PPP::Create_LAPSettings MainRunModes::LoadPPPLAPSettings(std::string Filename, Settings & Options)
{
	PPP::Create_LAPSettings LAPS;

	using boost::property_tree::ptree;
	ptree pt;
	boost::property_tree::read_xml(Filename, pt);

	unsigned int Version = pt.get<unsigned int>("root.Info.Version", Options.INTERNAL_VERSION);

	//LAP
	{
		//Load general content
		LAPS.ADU_perPhoton = pt.get<float>("root.PatternPreProcessing.LargestAdjacentPixel.ADUperPhoton", 100);
		LAPS.SeedThershold = pt.get<float>("root.PatternPreProcessing.LargestAdjacentPixel.SeedThershold", 0.5f);
		LAPS.CombinedThershold = pt.get<float>("root.PatternPreProcessing.LargestAdjacentPixel.CombinedThershold", 0.9f);

		LAPS.Output_NewXML = pt.get<std::string>("root.PatternPreProcessing.LargestAdjacentPixel.Output_NewXML", "");
		LAPS.Output_Path = pt.get<std::string>("root.PatternPreProcessing.LargestAdjacentPixel.Output_Path", "");
		LAPS.Output_Dataset = pt.get<std::string>("root.PatternPreProcessing.LargestAdjacentPixel.Output_Dataset", "");

		LAPS.DetPanels_Num = pt.get<unsigned int>("root.PatternPreProcessing.LargestAdjacentPixel.DetPanels_Num", 1);

		LAPS.GainMapPath = pt.get<std::string>("root.PatternPreProcessing.LargestAdjacentPixel.GainMapPath", "");
		LAPS.DatasetGain = pt.get<std::string>("root.PatternPreProcessing.LargestAdjacentPixel.DatasetGain", "");
		LAPS.DatasetOffset = pt.get<std::string>("root.PatternPreProcessing.LargestAdjacentPixel.DatasetOffset", "");

		//Load detector panels
		for (unsigned int i = 0; i < LAPS.DetPanels_Num; i++)
		{
			PPP::DetectorPanel currPan;
			std::string XMLKey = "root.PatternPreProcessing.LargestAdjacentPixel.DetPanels.N" + std::to_string(i);
			currPan.FirstInd = pt.get<unsigned int>(XMLKey + ".FirstInd", 0);
			currPan.Scans[0] = pt.get<unsigned int>(XMLKey + ".FastScan", 0);
			currPan.Scans[1] = pt.get<unsigned int>(XMLKey + ".SlowScan", 0);

			LAPS.DetectorPanels.push_back(currPan);
		}

	}

	return LAPS;
}

MainRunModes::AllSettings MainRunModes::LoadSettings(std::string Filename, Settings & Options)
{
	MainRunModes::AllSettings SettingsStack;

	SettingsStack.EvaluationSettings = MainRunModes::LoadEvaluationSettings(Filename, Options); //Evaluation & General
	SettingsStack.StatisticsSettings = MainRunModes::LoadStatisticSettings(Filename, Options); //Statistics
	SettingsStack.PPPLAPSettings = MainRunModes::LoadPPPLAPSettings(Filename, Options); //PPP.LAP

	return SettingsStack;
}


//Create and save example settings
int MainRunModes::Create_Example_Evaluation_Config_File(std::string Filename, Settings &Options)
{
	//saves a example evaluations settings file
	using boost::property_tree::ptree;
	ptree pt;
	pt = Example_Evaluation_Config_PT(pt, Options);


	//save to File
	boost::property_tree::write_xml(Filename, pt);

	//report
	if (Options.echo)
	{
		std::cout << "\n\n\n****************************\n     Report:\n\n";
		std::cout << "Example Evaluation Setting File have been created and saved as \"" << Filename << "\".\n";
		
	}


	return 0;
}

int MainRunModes::Create_Example_Config_File(std::string Filename, Settings & Options)
{
	//saves a example evaluations settings file
	using boost::property_tree::ptree;
	ptree pt;
	pt = Example_Evaluation_Config_PT(pt, Options);
	pt = Example_Statistics_Config_PT(pt, Options);
	pt = Example_PatternPreProcessing_LAP(pt, Options);
	//save to File
	boost::property_tree::write_xml(Filename, pt);

	//report
	if (Options.echo)
	{
		std::cout << "\n\n\n****************************\n     Report:\n\n";
		std::cout << "Example Config File have been created and saved as \"" << Filename << "\".\n";

	}


	return 0;
}

boost::property_tree::ptree MainRunModes::Example_Evaluation_Config_PT(boost::property_tree::ptree pt, Settings &Options)
{
	//Example Settings Evaluation
	RunIAC::CreateDataEval_Settings EES; //Eval-Example-Settings
	{
		//->Input Files
		EES.XML_Path = "EventList.xml";
		EES.PixelMap_Path = "Pixelmap.h5";
		EES.PixelMap_DataSet = "Pixelmap";
		EES.PixelMask_Path = "Pixelmask.h5";
		EES.PixelMap_DataSet = "";

		EES.UseExistingAvInt = false;

		//->Output Files
		EES.Out_Final_AC_Path = "AC_Final.bin";
		EES.Out_AvIntensity_Path = "AvIntensity.bin";
		EES.Out_Cq_Path = "CQ_Merged.bin";
		EES.Out_Cq_small_Path = "CQ_small.bin";
		EES.Out_ACuw_Path = "AC_unweighted.bin";
		EES.Out_Q_Vector = "QVector.bin";

		//-> Output Mesh
		EES.MeshSize = 201;
		EES.QZoom = 1.0;
		EES.AngularAveraged = false;
		EES.DoubleMap = true;
		EES.AC_FirstMap_Flags.InterpolationMode = Settings::Interpolation::NearestNeighbour;
		EES.AC_SecondMap_Flags.InterpolationMode = Settings::Interpolation::NearestNeighbour;


		//->PhotonThreshold
		EES.PhotonOffset = 0.0;
		EES.PhotonStep = 1.0;

		//->Misc
		EES.RestrictStackToBoundaries = false;
		EES.LowerBoundary = 0;
		EES.UpperBoundary = 10;

		EES.FractionalCq = false;
		EES.SizeOfCqFraction = 10;

		EES.InvertRotMatrix = false;

		EES.EchoLevel = 3;

		//->DisturbationForSimulation
		EES.DetDisturb = false;
		EES.DetDisturb_Rot = 0.0;
		EES.DetDisturb_Shift = 0.0;
	}

	//Store Evaluation Settings in PT
	{
		pt.put("root.Info.Info_Evaluation", "In this file are all settings for data-evaluation (autocorrelation) stored. This is an example file, just edit it as you need. This file can be created with the argument \"-ec\" (evaluation config).");
		pt.put("root.Info.Version", Options.INTERNAL_VERSION);

		//->Input Files
		pt.put("root.EvalSettings.InputFiles.EventList_Path", EES.XML_Path);
		pt.put("root.EvalSettings.InputFiles.PixelMap_Path", EES.PixelMap_Path);
		pt.put("root.EvalSettings.InputFiles.PixelMap_Dataset", EES.PixelMap_DataSet);
		pt.put("root.EvalSettings.InputFiles.PixelMask_Path", EES.PixelMask_Path);
		pt.put("root.EvalSettings.InputFiles.PixelMask_Dataset", EES.PixelMask_Dataset);
		pt.put("root.EvalSettings.InputFiles.UseExistingAvInt", EES.UseExistingAvInt);
		//->Output Files
		pt.put("root.EvalSettings.OutputFiles.AC_Final_Path", EES.Out_Final_AC_Path);
		pt.put("root.EvalSettings.OutputFiles.AvIntensity_Path", EES.Out_AvIntensity_Path);
		pt.put("root.EvalSettings.OutputFiles.Cq_Merged_Path", EES.Out_Cq_Path);
		pt.put("root.EvalSettings.OutputFiles.Cq_small_Path", EES.Out_Cq_small_Path);
		pt.put("root.EvalSettings.OutputFiles.ACuw_Path", EES.Out_ACuw_Path);
		pt.put("root.EvalSettings.OutputFiles.Q_Vector", EES.Out_Q_Vector);

		//-> Output Mesh
		pt.put("root.EvalSettings.OutputMesh.MeshSize", EES.MeshSize);
		pt.put("root.EvalSettings.OutputMesh.QZoom", EES.QZoom);
		pt.put("root.EvalSettings.OutputMesh.AngularAveraged", EES.AngularAveraged);
		pt.put("root.EvalSettings.OutputMesh.DoubleMap", EES.DoubleMap);
		pt.put("root.EvalSettings.OutputMesh.AC_FirstMap_Flags.InterpolationMode", EES.AC_FirstMap_Flags.InterpolationMode);
		pt.put("root.EvalSettings.OutputMesh.AC_SecondMap_Flags.InterpolationMode", EES.AC_SecondMap_Flags.InterpolationMode);

		//->PhotonThreshold
		pt.put("root.EvalSettings.PhotonThreshold.PhotonOffset", EES.PhotonOffset);
		pt.put("root.EvalSettings.PhotonThreshold.PhotonStep", EES.PhotonStep);

		//->Misc
		pt.put("root.EvalSettings.Misc.RestrictStackToBoundaries", EES.RestrictStackToBoundaries);
		pt.put("root.EvalSettings.Misc.LowerBoundary", EES.LowerBoundary);
		pt.put("root.EvalSettings.Misc.UpperBoundary", EES.UpperBoundary);

		pt.put("root.EvalSettings.Misc.FractionalCq", EES.FractionalCq);
		pt.put("root.EvalSettings.Misc.SizeOfCqFraction", EES.SizeOfCqFraction);

		pt.put("root.EvalSettings.Misc.InvertRotMatrix", EES.InvertRotMatrix);
		pt.put("root.EvalSettings.Misc.EchoLevel", EES.EchoLevel);
	}
	return pt;
}

boost::property_tree::ptree MainRunModes::Example_Statistics_Config_PT(boost::property_tree::ptree pt, Settings & Options)
{
	//Example Statistic Settings
	Statistics::Create_PixelHistogramSettings EPHS;
	{
		EPHS.Bins = 100;
		EPHS.SmalestValue = 0.0;
		EPHS.LargestValue = 100.0;
		EPHS.OutputPath = "PixelwiseHistograms.bin";

		EPHS.Normalized = true;
	}
	//Store Statistic Settings in PT
	{
		pt.put("root.StatisticsSettings.PixelHistograms.OutputPath", EPHS.OutputPath);
		pt.put("root.StatisticsSettings.PixelHistograms.Bins", EPHS.Bins);
		pt.put("root.StatisticsSettings.PixelHistograms.SmallestValue", EPHS.SmalestValue);
		pt.put("root.StatisticsSettings.PixelHistograms.LargestValue", EPHS.LargestValue);
		pt.put("root.StatisticsSettings.PixelHistograms.Normalize", EPHS.Normalized);


	}
	return pt;
}

boost::property_tree::ptree MainRunModes::Example_PatternPreProcessing_LAP(boost::property_tree::ptree pt, Settings & Options)
{
	//Example PPP_LAP Settings
	PPP::Create_LAPSettings LAPS;
	PPP::DetectorPanel SamplePanel;
	{
		LAPS.ADU_perPhoton = 100.0f;
		LAPS.SeedThershold = 0.5f;
		LAPS.CombinedThershold = 0.9f;

		LAPS.Output_NewXML = "NewEventList.xml";
		LAPS.Output_Path = "NewH5_File.h5";
		LAPS.Output_Dataset = "data";

		LAPS.DetPanels_Num = 1;

		LAPS.GainMapPath = "";
		LAPS.DatasetGain = "gain";
		LAPS.DatasetOffset = "offset";

		// Detector Panels
		SamplePanel.FirstInd = 0;
		SamplePanel.Scans[0] = 100;
		SamplePanel.Scans[1] = 100;
	}
	//Store Statistic Settings in PT
	{
		pt.put("root.PatternPreProcessing.LargestAdjacentPixel.ADUperPhoton", LAPS.ADU_perPhoton);
		pt.put("root.PatternPreProcessing.LargestAdjacentPixel.SeedThershold", LAPS.SeedThershold);
		pt.put("root.PatternPreProcessing.LargestAdjacentPixel.CombinedThershold", LAPS.CombinedThershold);

		pt.put("root.PatternPreProcessing.LargestAdjacentPixel.Output_NewXML", LAPS.Output_NewXML);
		pt.put("root.PatternPreProcessing.LargestAdjacentPixel.Output_Path", LAPS.Output_Path);
		pt.put("root.PatternPreProcessing.LargestAdjacentPixel.Output_Dataset", LAPS.Output_Dataset);

		pt.put("root.PatternPreProcessing.LargestAdjacentPixel.DetPanels_Num", LAPS.DetPanels_Num);

		pt.put("root.PatternPreProcessing.LargestAdjacentPixel.GainMapPath", LAPS.GainMapPath);
		pt.put("root.PatternPreProcessing.LargestAdjacentPixel.DatasetGain", LAPS.DatasetGain);
		pt.put("root.PatternPreProcessing.LargestAdjacentPixel.DatasetOffset", LAPS.DatasetOffset);


		pt.put("root.PatternPreProcessing.LargestAdjacentPixel.DetPanels.N0.FirstInd", SamplePanel.FirstInd);
		pt.put("root.PatternPreProcessing.LargestAdjacentPixel.DetPanels.N0.FastScan", SamplePanel.Scans[0]);
		pt.put("root.PatternPreProcessing.LargestAdjacentPixel.DetPanels.N0.SlowScan", SamplePanel.Scans[1]);

	}
	return pt;
}


//Run Evaluations

//Get averaged Intensity
int MainRunModes::AverageIntensity(std::string EvaluationConfigFile, Settings &Options, bool UpdateEventXML)
{
	//load Settings & check stuff
	RunIAC::CreateDataEval_Settings EvalSettings = LoadEvaluationSettings(EvaluationConfigFile, Options);
	if (EvalSettings.XML_Path == "")
	{
		std::cerr << "ERROR: No EventList-path found. Please modify the evaluation config file accordingly.\n";
		return -1;
	}
	if (EvalSettings.Out_AvIntensity_Path == "")
	{
		std::cerr << "ERROR: No Averaged-Intensity-path found. Please modify the evaluation config file accordingly.\n";
		return -1;
	}
	if (EvalSettings.EchoLevel == 0)
		Options.echo = false;


	//Set up Detector
	Detector Det;

	if (EvalSettings.PixelMap_Path == "" || EvalSettings.PixelMap_DataSet == "")
	{
		Options.LoadHitEventListFromFile(EvalSettings.XML_Path); //Hard hack!, not beautiful, improve!!!
		//No Pixelmap -> need to get Detector Size out of Event H5-File
		std::array<unsigned int, 2> DetSize = Options.ScanH5Files({ Options.HitEvents[0].Filename}, {Options.HitEvents[0].Dataset});
		Det.DetectorSize[0] = DetSize[0];
		Det.DetectorSize[1] = DetSize[1];
	}
	else
	{
		//Obtain DetSize via PixelMap
		Det.LoadPixelMap(EvalSettings.PixelMap_Path, EvalSettings.PixelMap_DataSet);
	}
	Det.LoadPixelMask(EvalSettings.PixelMask_Path); //load Pixelmask (generates default (1)-Mask if path == "")
	
	//load Events
	Options.LoadHitEventListFromFile(EvalSettings.XML_Path);
	if (Options.HitEvents.size() == 0)
	{
		std::cerr << "ERROR: No Events found\n";
		return -1;
	}
	if (Options.echo)
		std::cout << Options.HitEvents.size() << "Events found\n";

	//Load and average Intensities, update XML, if wanted.
	
	RunIAC::Load_and_average_Intensities(Options, Det, EvalSettings.PhotonOffset, EvalSettings.PhotonStep, EvalSettings.XML_Path, EvalSettings.Out_AvIntensity_Path, UpdateEventXML);
	return 0;
}

int MainRunModes::GainCorrectionAndLAP(std::string Arg1, Settings & Options)
{
	AllSettings AS = LoadSettings(Arg1, Options);

	Detector Det;
	Det.LoadPixelMap(AS.EvaluationSettings.PixelMap_Path, AS.EvaluationSettings.PixelMap_DataSet);
	Det.LoadPixelMask(AS.EvaluationSettings.PixelMask_Path, AS.EvaluationSettings.PixelMask_Dataset);
	
	ProfileTime Profiler;
	Profiler.Tic();

	PPP::ProcessData_PF_LAP(Det, AS.PPPLAPSettings, AS.EvaluationSettings.XML_Path);
	std::cout << "DONE in " << Profiler.Toc(false) << "\n";

	return 0;
}

int MainRunModes::CreateAllPixelHistograms(std::string ConfigFile, Settings & Options)
{
	RunIAC::CreateDataEval_Settings EVS = LoadEvaluationSettings(ConfigFile, Options);
	Statistics::StatisticsSettings STS = MainRunModes::LoadStatisticSettings(ConfigFile, Options);

	Detector Det;
	Det.LoadPixelMap(EVS.PixelMap_Path, EVS.PixelMap_DataSet);
	Det.LoadPixelMask(EVS.PixelMask_Path, EVS.PixelMask_Dataset);
	Options.LoadHitEventListFromFile(EVS.XML_Path);
	Statistics::CreateAndSaveAllPixelHistograms(STS.PixelHistogramSettings, Det, Options);

	//Statistics::Histogram Hist = Statistics


	return 0;
}


//Create XML Event Lists
int MainRunModes::Create_XMLHitlist_from_H5Stack_script(std::string Arg1, std::string Arg2, Settings &Options)
{
	std::vector<std::string> Dublett = CSV_Splitter(Arg1, ";");
	if (Dublett.size() != 2)
	{
		std::cerr << "ERROR: while parsing argument for XML-Event list from H5-files.\n";
		std::cerr << "Input need to be in the shape \"Input1.h5, Input2.h5, ...;Dataset1, Dataset2, ...\". \n";
		return -1;
	}
	std::vector<std::string> H5_Paths = CSV_Splitter(Dublett[0], ",");
	std::vector<std::string> H5_Datasets = CSV_Splitter(Dublett[1], ",");

	if (H5_Paths.size() != H5_Datasets.size())
	{
		std::cerr << "ERROR: while parsing argument for XML-Event list from H5-files.\n";
		std::cerr << "Unequal amont of hdf5-paths and datasets. \n";
		return -1;
	}
	return Create_XMLHitlist_from_H5Stack(H5_Paths, H5_Datasets, Arg2, Options);
}
int MainRunModes::Create_XMLHitlist_from_H5Stack(std::vector<std::string> H5_Paths, std::vector<std::string> H5_Datasets, std::string XML_Output_Path, Settings &Options)
{
	std::array<unsigned int, 2 > DetSize;
	DetSize = Options.ScanH5Files(H5_Paths, H5_Datasets);
	Options.SafeHitEventListToFile(XML_Output_Path);


	std::cout << "\n\n\n****************************\n     Report:\n\n";
	std::cout << H5_Paths.size() << " HDF5-files scanned and a total of " << Options.HitEvents.size() << " entries found.\n";
	std::cout << "Size of the 2D array (single entrie ^= detector size) is " << DetSize[0] << " x " << DetSize[1] << " pixel\n";
	std::cout << "Because of no orientation information, rotation matrix was set to identity matrix for each entry.\n";
	std::cout << "XML-Hit-List has been saved as \"" << XML_Output_Path << "\".\n";

	return 0;
}



// --- EVALUATE ---
//Auto correlate
int MainRunModes::AutoCorrelateData(std::string ConfigFile, Settings & Options)
{
	//Setup Open CL Stuff
	Options.SetUp_OpenCL();
	//load Config File
	RunIAC::CreateDataEval_Settings EVS = LoadEvaluationSettings(ConfigFile, Options);

	RunIAC::Run_AutoCorr_DataEval(Options, EVS);


	return 0;
}









