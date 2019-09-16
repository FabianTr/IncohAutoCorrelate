#include "MainRunModes.h"

#include <iostream>
#include <fstream>

#include "ProfileTime.h"
#include "Detector.h"
#include <boost/algorithm/string.hpp>


//Helper
std::vector<std::string> MainRunModes::CSV_Splitter(std::string Input, std::string delimiter)
{
	//Splits "delimiter"-seperated Values within string to vector
	std::vector<std::string> Output;
	size_t pos = 0;
	std::string token;
	//bool ParsingAlive = true;
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

	//unsigned int Version = pt.get<unsigned int>("root.Info.Version", Options.INTERNAL_VERSION);

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
	EVS.UsePixelMask_as_Flatfield = pt.get<bool>("root.EvalSettings.Misc.UsePixelmaskAsFlatField", false);
	EVS.FractionalCq = pt.get<bool>("root.EvalSettings.Misc.FractionalCq",false);
	EVS.SizeOfCqFraction = pt.get<unsigned int>("root.EvalSettings.Misc.SizeOfCqFraction",10);
	EVS.InvertRotMatrix = pt.get<bool>("root.EvalSettings.Misc.InvertRotMatrix", false);
	EVS.EchoLevel = pt.get<int>("root.EvalSettings.Misc.EchoLevel", 3);

	EVS.Out_Report = pt.get<std::string>("root.EvalSettings.Misc.Report_Path", "");

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

	//unsigned int Version = pt.get<unsigned int>("root.Info.Version", Options.INTERNAL_VERSION);

	
	{
		//PixelHisto
		StatSet.PixelHistogramSettings.OutputPath = pt.get<std::string>("root.StatisticsSettings.PixelHistograms.OutputPath", "");

		StatSet.PixelHistogramSettings.Bins = pt.get<unsigned int>("root.StatisticsSettings.PixelHistograms.Bins", 100);
		StatSet.PixelHistogramSettings.SmalestValue = pt.get<double>("root.StatisticsSettings.PixelHistograms.SmallestValue", 0.0);
		StatSet.PixelHistogramSettings.LargestValue = pt.get<double>("root.StatisticsSettings.PixelHistograms.LargestValue", 100.0);

		StatSet.PixelHistogramSettings.Normalized = pt.get<bool>("root.StatisticsSettings.PixelHistograms.Normalize", true);

		//Speckle Contrast
		StatSet.SpeckleContrastSettings.CSVOutputPath = pt.get<std::string>("root.StatisticsSettings.SpeckleContrastStatistics.CSVOutputPath", "SpeckleContrast.csv");
	
		//IsolatedPhotonChargeSharing
		StatSet.ChargeSharingSettings.Seed = pt.get<double>("root.StatisticsSettings.IsoPhotonChargeSharingFit.ADUSeed", 0.5);//ADU seed for one photon hit
		StatSet.ChargeSharingSettings.MaxADU = pt.get<double>("root.StatisticsSettings.IsoPhotonChargeSharingFit.MaxADUSum", 1.5);//maximum ADU sum to prevent two photon hits

		StatSet.ChargeSharingSettings.IsolationRadius = pt.get<unsigned int>("root.StatisticsSettings.IsoPhotonChargeSharingFit.IsolationRadius", 2);//min distance to next hit
		StatSet.ChargeSharingSettings.PixelMaskRadius = pt.get<unsigned int>("root.StatisticsSettings.IsoPhotonChargeSharingFit.MaskRadius", 1);//min distance to masked pixel and panel edge

		StatSet.ChargeSharingSettings.OutputBinaryPath = pt.get<std::string>("root.StatisticsSettings.IsoPhotonChargeSharingFit.BinOutputPath", "ChareSharingFit.bin");
	}

	return StatSet;
}

PPP::Create_LAPSettings MainRunModes::LoadPPPLAPSettings(std::string Filename, Settings & Options)
{
	PPP::Create_LAPSettings LAPS;

	using boost::property_tree::ptree;
	ptree pt;
	boost::property_tree::read_xml(Filename, pt);

	//unsigned int Version = pt.get<unsigned int>("root.Info.Version", Options.INTERNAL_VERSION);

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

PPP::CreateDarkSettings MainRunModes::LoadPPPDarkSettings(std::string Filename, Settings & Options)
{
	PPP::CreateDarkSettings Dark;

	using boost::property_tree::ptree;
	ptree pt;
	boost::property_tree::read_xml(Filename, pt);

	//unsigned int Version = pt.get<unsigned int>("root.Info.Version", Options.INTERNAL_VERSION);

	//LAP
	{
		//Load general content
		Dark.Dark_Path = pt.get<std::string>("root.PatternPreProcessing.DarkFieldCorrection.DarkField_Path", "");
		Dark.Dark_Dataset = pt.get<std::string>("root.PatternPreProcessing.DarkFieldCorrection.DarkField_Dataset", "");

		Dark.Output_Path = pt.get<std::string>("root.PatternPreProcessing.DarkFieldCorrection.Output_Path", "");
		Dark.Output_Dataset = pt.get<std::string>("root.PatternPreProcessing.DarkFieldCorrection.Output_Dataset", "");

		Dark.Output_NewXML = pt.get<std::string>("root.PatternPreProcessing.DarkFieldCorrection.Output_XML", "");

		Dark.RestrictToDataSource = pt.get<bool>("root.PatternPreProcessing.DarkFieldCorrection.RestrictToDataSource", false);
		Dark.DataSource_Path = pt.get<std::string>("root.PatternPreProcessing.DarkFieldCorrection.DataSourcePath", "");
	}

	return Dark;
}


MainRunModes::AllSimSettings MainRunModes::LoadSimulationSettings(std::string Filename, Settings & Options)
{
	MainRunModes::AllSimSettings SimS;

	using boost::property_tree::ptree;
	ptree pt;
	boost::property_tree::read_xml(Filename, pt);

	//unsigned int Version = pt.get<unsigned int>("root.Info.Version", Settings::INTERNAL_VERSION);


	{
		SimS.SimSettings.Filename_XML = pt.get<std::string>("root.Simulation.OutputPath_EventList", "SimOut.xml");
		SimS.SimSettings.Filename_Intensity = pt.get<std::string>("root.Simulation.OutputPath_Intensities", "SimOut.h5");
		SimS.SimSettings.Dataset = pt.get<std::string>("root.Simulation.OutputDataset_Intensities", "data");
		SimS.SimSettings.SaveResults = pt.get<bool>("root.Simulation.SaveResults", true);

		SimS.SimSettings.Wavelength = pt.get<double>("root.Simulation.Wavelength", 1.94);
		SimS.SimSettings.NumberOfSimulations = pt.get<unsigned int>("root.Simulation.NumberOfPattern", 10);
		SimS.SimSettings.Modes = pt.get<unsigned int>("root.Simulation.NumberOfModes", 1);
		SimS.SimSettings.CrystSettings.FlYield = pt.get<float>("root.Simulation.FluorescenceYield", 1.0f);
		SimS.SimSettings.CrystSettings.Incoherent = pt.get<bool>("root.Simulation.Incoherent", true);
		SimS.SimSettings.CrystSettings.RandOrientation = pt.get<bool>("root.Simulation.RandomOriented", false);
		SimS.SimSettings.CrystSettings.Isotropie = pt.get<float>("root.Simulation.OrientationIsotropie", 1.0f);

		SimS.SimSettings.AveragePhotonesPerEmitterOnDetector = pt.get<float>("root.Simulation.AveragePhotonesPerEmitterOnDetector", 1.0f);
		SimS.SimSettings.SubSampling = pt.get<unsigned int>("root.Simulation.SubSampling", 1);
		SimS.SimSettings.PoissonSample = pt.get<bool>("root.Simulation.PoissonSample", true);
		SimS.SimSettings.Value_per_Photon = pt.get<float>("root.Simulation.ValuePerPhoton", 1.0f);
		SimS.SimSettings.ADUNoise = pt.get<float>("root.Simulation.ADUNoise", 0.0f);

		SimS.SimSettings.AutoPixelOrientation = pt.get<bool>("root.Simulation.AutoPixelOrientation", true);
		SimS.SimSettings.AutoPixelSize = pt.get<bool>("root.Simulation.AutoPixelSize", true);


		std::string delimiter = ";"; // for array parsing


		Settings::SplitString(pt.get<std::string>("root.Simulation.PixelOrientationVectors", "0.0; 1.0; 0.0; 0.0; 0.0; 1.0"), SimS.SimSettings.PixelOrientationVectors, 6, delimiter);

		Settings::SplitString(pt.get<std::string>("root.Simulation.PixelSize", "50.0; 50.0"), SimS.SimSettings.PixelSize, 2, delimiter);

		Settings::SplitString(pt.get<std::string > ("root.Simulation.UnitCells", "10; 10; 10"), SimS.SimSettings.UnitCells, 3, delimiter);

		//Parse LatticeVectors
		Settings::SplitString(pt.get<std::string >("root.Simulation.LatticeVector_1", "1.0; 0.0; 0.0"), SimS.LatticeVector[0], 3, delimiter);
		Settings::SplitString(pt.get<std::string >("root.Simulation.LatticeVector_2", "0.0; 1.0; 0.0"), SimS.LatticeVector[1], 3, delimiter);
		Settings::SplitString(pt.get<std::string >("root.Simulation.LatticeVector_3", "0.0; 0.0; 1.0"), SimS.LatticeVector[2], 3, delimiter);


		unsigned int NOE = pt.get<unsigned int>("root.Simulation.UnitCell.NumberOfEmittersPerUnitcell", 0);
		std::array<double, 3> UcEm;
		SimS.UnitCell.clear();
		for (unsigned int i = 0; i < NOE; i++)
		{
			Settings::SplitString(pt.get<std::string >("root.Simulation.UnitCell.Emitter_" + std::to_string(i+1), "0.0; 0.0; 0.0"), UcEm.data(), 3, delimiter);
			SimS.UnitCell.push_back(UcEm);
		}


		SimS.GPMSettings.Filename = pt.get<std::string>("root.GeneratePixelMap.Filename", "");
		SimS.GPMSettings.Dataset = pt.get<std::string>("root.GeneratePixelMap.Dataset", "");
		SimS.GPMSettings.SizeA = pt.get<int>("root.GeneratePixelMap.SizeA", 0);
		SimS.GPMSettings.SizeB = pt.get<int>("root.GeneratePixelMap.SizeB", 0);
		SimS.GPMSettings.PixelSize = pt.get<int>("root.GeneratePixelMap.PixelSize", 0.0f);
		Settings::SplitString(pt.get<std::string >("root.GeneratePixelMap.Center", "0.0; 0.0; 0.0"), SimS.GPMSettings.Center.data(), 3, delimiter);
		Settings::SplitString(pt.get<std::string >("root.GeneratePixelMap.VecA", "1.0; 0.0; 0.0"), SimS.GPMSettings.VecA.data(), 3, delimiter);
		Settings::SplitString(pt.get<std::string >("root.GeneratePixelMap.VecB", "0.0; 1.0; 0.0"), SimS.GPMSettings.VecB.data(), 3, delimiter);
	}


	return SimS;
}

MainRunModes::AllSettings MainRunModes::LoadSettings(std::string Filename, Settings & Options)
{
	MainRunModes::AllSettings SettingsStack;

	SettingsStack.EvaluationSettings = MainRunModes::LoadEvaluationSettings(Filename, Options); //Evaluation & General
	SettingsStack.StatisticsSettings = MainRunModes::LoadStatisticSettings(Filename, Options); //Statistics
	SettingsStack.PPPLAPSettings = MainRunModes::LoadPPPLAPSettings(Filename, Options); //PPP.LAP
	SettingsStack.AllSimulationSettings = MainRunModes::LoadSimulationSettings(Filename, Options); //Simulation stuff
	SettingsStack.PPPDarkSettings = MainRunModes::LoadPPPDarkSettings(Filename, Options); //PPP.Dark

	//combine redundant settings
	{
		SettingsStack.StatisticsSettings.ChargeSharingSettings.DetectorPanels = SettingsStack.PPPLAPSettings.DetectorPanels;
		SettingsStack.StatisticsSettings.ChargeSharingSettings.RestrictToBoundaries = SettingsStack.EvaluationSettings.RestrictStackToBoundaries;
		SettingsStack.StatisticsSettings.ChargeSharingSettings.LowerLimit = SettingsStack.EvaluationSettings.LowerBoundary;
		SettingsStack.StatisticsSettings.ChargeSharingSettings.UpperLimit = SettingsStack.EvaluationSettings.UpperBoundary;
	}

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
	pt = Example_PatternPreProcessing_Dark(pt, Options);
	pt = Example_Simulation_Config_PT(pt, Options);
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

		EES.UsePixelMask_as_Flatfield = false;

		EES.FractionalCq = false;
		EES.SizeOfCqFraction = 10;

		EES.InvertRotMatrix = false;

		EES.EchoLevel = 3;

		EES.Out_Report = "Report.txt";

		//->DisturbationForSimulation
		EES.DetDisturb = false;
		EES.DetDisturb_Rot = 0.0;
		EES.DetDisturb_Shift = 0.0;
	}

	//Store Evaluation Settings in PT
	{
		pt.put("root.Info.Info_Evaluation", "In this file are all settings for data-evaluation (autocorrelation) stored. This is an example file, just edit it as you need. This file can be created with the argument \"-ec\" (evaluation config).");
		pt.put("root.Info.Version", Settings::INTERNAL_VERSION);
		pt.put("root.Info.VersionRevision", Settings::INTERNAL_VERSION_Revision);

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

		pt.put("root.EvalSettings.Misc.UsePixelmaskAsFlatField", EES.UsePixelMask_as_Flatfield);

		pt.put("root.EvalSettings.Misc.FractionalCq", EES.FractionalCq);
		pt.put("root.EvalSettings.Misc.SizeOfCqFraction", EES.SizeOfCqFraction);

		pt.put("root.EvalSettings.Misc.InvertRotMatrix", EES.InvertRotMatrix);
		pt.put("root.EvalSettings.Misc.EchoLevel", EES.EchoLevel);

		pt.put("root.EvalSettings.Misc.Report_Path", EES.Out_Report);

		
	}
	return pt;
}

boost::property_tree::ptree MainRunModes::Example_Statistics_Config_PT(boost::property_tree::ptree pt, Settings & Options)
{
	//Example Statistic Settings
	Statistics::Create_PixelHistogramSettings EPHS;
	Statistics::Create_SpeckleContrastSettings SCSS;
	Statistics::Create_ChargeSharingSettings CSS;

	{
		//Pixel Histogram
		EPHS.Bins = 100;
		EPHS.SmalestValue = 0.0;
		EPHS.LargestValue = 100.0;
		EPHS.OutputPath = "PixelwiseHistograms.bin";

		EPHS.Normalized = true;
		//Speckle Contrast
		SCSS.CSVOutputPath = "SpecleContrastStatistics.csv";

		//charge sharing (isolated photon hit)
		CSS.Seed = 0.5;
		CSS.MaxADU = 1.5;
		CSS.IsolationRadius = 2;
		CSS.PixelMaskRadius = 1;
		CSS.OutputBinaryPath = "ChareSharingFit.bin";
	}
	//Store Statistic Settings in PT
	{
		//Pixel Histogram
		pt.put("root.StatisticsSettings.PixelHistograms.OutputPath", EPHS.OutputPath);
		pt.put("root.StatisticsSettings.PixelHistograms.Bins", EPHS.Bins);
		pt.put("root.StatisticsSettings.PixelHistograms.SmallestValue", EPHS.SmalestValue);
		pt.put("root.StatisticsSettings.PixelHistograms.LargestValue", EPHS.LargestValue);
		pt.put("root.StatisticsSettings.PixelHistograms.Normalize", EPHS.Normalized);

		//Speckle Contrast
		pt.put("root.StatisticsSettings.SpeckleContrastStatistics.CSVOutputPath", SCSS.CSVOutputPath);

		//Charge sharing (isolated photon hit)
		pt.put("root.StatisticsSettings.IsoPhotonChargeSharingFit.InfoText", "For isolated photon charge sharing radius fit. Also requires Panel settings from Evaluation (XML, PixelMap, PixelMask, Boundaries) and PatternPreProcessing.LargestAdjugatPixel.");

		pt.put("root.StatisticsSettings.IsoPhotonChargeSharingFit.ADUSeed", CSS.Seed);
		pt.put("root.StatisticsSettings.IsoPhotonChargeSharingFit.MaxADUSum", CSS.MaxADU);

		pt.put("root.StatisticsSettings.IsoPhotonChargeSharingFit.IsolationRadius", CSS.IsolationRadius);
		pt.put("root.StatisticsSettings.IsoPhotonChargeSharingFit.MaskRadius", CSS.PixelMaskRadius);

		pt.put("root.StatisticsSettings.IsoPhotonChargeSharingFit.BinOutputPath", CSS.OutputBinaryPath);

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

boost::property_tree::ptree MainRunModes::Example_Simulation_Config_PT(boost::property_tree::ptree pt, Settings & Options)
{
	Simulator::SimulationSettings SimS;
	std::string SimS_InfoText = "Settings for Simulation.\nWavelength has the same unit as the lattice verctors and the emitter position. This unit can be choosen arbitrarily.";

	SimS.Filename_XML = "OutputEventList.xml";
	SimS.Filename_Intensity = "OutputIntensities.h5";
	SimS.Dataset = "data";
	SimS.SaveResults = true;

	SimS.Wavelength = 1.94;
	SimS.NumberOfSimulations = 10;
	SimS.Modes = 1;
	SimS.CrystSettings.FlYield = 1.0f;
	SimS.CrystSettings.Incoherent = true;
	SimS.CrystSettings.RandOrientation = false;
	SimS.CrystSettings.Isotropie = 1.0f;

	SimS.AveragePhotonesPerEmitterOnDetector = 1.0f;
	SimS.SubSampling = 1;
	SimS.PoissonSample = true;
	SimS.Value_per_Photon = 1.0f;
	SimS.ADUNoise = 0;

	SimS.AutoPixelOrientation = true;
	SimS.AutoPixelSize = true;

	SimS.PixelOrientationVectors[0] = 0.0;
	SimS.PixelOrientationVectors[1] = 1.0;
	SimS.PixelOrientationVectors[2] = 0.0;
	SimS.PixelOrientationVectors[3] = 0.0;
	SimS.PixelOrientationVectors[4] = 0.0;
	SimS.PixelOrientationVectors[5] = 1.0;

	SimS.PixelSize[0] = 50;
	SimS.PixelSize[1] = 50;


	SimS.UnitCells[0] = 10;
	SimS.UnitCells[1] = 10;
	SimS.UnitCells[2] = 10;


	//Options.MReference << 1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0;

	double LatticeVector[3][3];
	LatticeVector[0][0] = 1.0;
	LatticeVector[0][1] = 0.0;
	LatticeVector[0][2] = 0.0;

	LatticeVector[1][0] = 0.0;
	LatticeVector[1][1] = 1.0;
	LatticeVector[1][2] = 0.0;

	LatticeVector[2][0] = 0.0;
	LatticeVector[2][1] = 0.0;
	LatticeVector[2][2] = 1.0;

	unsigned int NumberOfEmittersPerUnitcell = 1;

	std::vector<std::array<double, 3>> UnitCell;
	std::array<double, 3> t_pos;
	t_pos = { 0 , 0 , 0 };
	UnitCell.push_back(t_pos);


	//Generate Pixelmap
	Simulator::GeneratePMSettings GPMS;
	std::string GPMS_InfoText = "Settings to create a simple pixel map (-SGP -SimulateGeneratePixelMap). PixelSize and Center in mum, SizeA & SizeB in pixel (integer), VecA & VecB for pixel orientation does not need to be normalized.";

	GPMS.Filename = "NewPixelMap.h5";
	GPMS.Dataset = "PixelMap";

	GPMS.SizeA = 100;
	GPMS.SizeB = 100;
	GPMS.PixelSize = 50.0;

	GPMS.Center[0] = 0.0f;
	GPMS.Center[1] = 0.0f;
	GPMS.Center[3] = 100000.0f;

	GPMS.VecA[0] = 1.0f;
	GPMS.VecA[1] = 0.0f;
	GPMS.VecA[2] = 0.0f;

	GPMS.VecB[0] = 0.0f;
	GPMS.VecB[1] = 1.0f;
	GPMS.VecB[2] = 0.0f;


	{
		pt.put("root.Simulation.InfoText", GPMS_InfoText);
		pt.put("root.Simulation.OutputPath_EventList", SimS.Filename_XML);
		pt.put("root.Simulation.OutputPath_Intensities", SimS.Filename_Intensity);
		pt.put("root.Simulation.OutputDataset_Intensities", SimS.Dataset);
		//pt.put("root.Simulation.SaveResults", SimS.SaveResults);

		pt.put("root.Simulation.Wavelength", SimS.Wavelength);
		pt.put("root.Simulation.NumberOfPattern", SimS.NumberOfSimulations);
		pt.put("root.Simulation.NumberOfModes", SimS.Modes);
		pt.put("root.Simulation.FluorescenceYield", SimS.CrystSettings.FlYield);
		pt.put("root.Simulation.Incoherent", SimS.CrystSettings.Incoherent);
		pt.put("root.Simulation.RandomOriented", SimS.CrystSettings.RandOrientation);
		pt.put("root.Simulation.OrientationIsotropie", SimS.CrystSettings.Isotropie);

		pt.put("root.Simulation.AveragePhotonesPerEmitterOnDetector", SimS.AveragePhotonesPerEmitterOnDetector);
		pt.put("root.Simulation.SubSampling", SimS.SubSampling);
		pt.put("root.Simulation.PoissonSample", SimS.PoissonSample);
		pt.put("root.Simulation.ValuePerPhoton", SimS.Value_per_Photon);
		pt.put("root.Simulation.ADUNoise", SimS.ADUNoise);

		pt.put("root.Simulation.AutoPixelOrientation", SimS.AutoPixelOrientation);
		pt.put("root.Simulation.AutoPixelSize", SimS.AutoPixelSize);

		std::string POV = std::to_string(SimS.PixelOrientationVectors[0]);
		for (int i = 1; i < 6; i++)
		{
			POV = POV + "; " + std::to_string(SimS.PixelOrientationVectors[i]);
		}
		pt.put("root.Simulation.PixelOrientationVectors", POV);
		
		std::string PXS = std::to_string(SimS.PixelSize[0]) + "; " + std::to_string(SimS.PixelSize[1]);
		pt.put("root.Simulation.PixelSize", PXS);

		std::string UC = std::to_string(SimS.UnitCells[0]) + "; " + std::to_string(SimS.UnitCells[1]) + "; " + std::to_string(SimS.UnitCells[2]);
		pt.put("root.Simulation.UnitCells", UC);

		std::string LAV = std::to_string(LatticeVector[0][0]) + "; " + std::to_string(LatticeVector[0][1])  + "; " + std::to_string(LatticeVector[0][2]);
		pt.put("root.Simulation.LatticeVector_1", LAV);
		LAV = std::to_string(LatticeVector[1][0]) + "; " + std::to_string(LatticeVector[1][1]) + "; " + std::to_string(LatticeVector[1][2]);
		pt.put("root.Simulation.LatticeVector_2", LAV);
		LAV = std::to_string(LatticeVector[2][0]) + "; " + std::to_string(LatticeVector[2][1]) + "; " + std::to_string(LatticeVector[2][2]);
		pt.put("root.Simulation.LatticeVector_3", LAV);

		pt.put("root.Simulation.UnitCell.NumberOfEmittersPerUnitcell", NumberOfEmittersPerUnitcell);

		std::string EM = std::to_string(UnitCell[0][0]) + "; " + std::to_string(UnitCell[0][1]) + "; " + std::to_string(UnitCell[0][2]);
		pt.put("root.Simulation.UnitCell.Emitter_1", EM);

		//Generate Pixel Map
		pt.put("root.GeneratePixelMap.InfoText", GPMS_InfoText);

		pt.put("root.GeneratePixelMap.Filename", GPMS.Filename);
		pt.put("root.GeneratePixelMap.Dataset", GPMS.Dataset);

		pt.put("root.GeneratePixelMap.SizeA", GPMS.SizeA);
		pt.put("root.GeneratePixelMap.SizeB", GPMS.SizeB);

		pt.put("root.GeneratePixelMap.PixelSize", GPMS.PixelSize);

		pt.put("root.GeneratePixelMap.Center", std::to_string(GPMS.Center[0]) + "; " + std::to_string(GPMS.Center[1]) + "; " + std::to_string(GPMS.Center[2]));

		pt.put("root.GeneratePixelMap.VecA", std::to_string(GPMS.VecA[0]) + "; " + std::to_string(GPMS.VecA[1]) + "; " + std::to_string(GPMS.VecA[2]));
		pt.put("root.GeneratePixelMap.VecB", std::to_string(GPMS.VecB[0]) + "; " + std::to_string(GPMS.VecB[1]) + "; " + std::to_string(GPMS.VecB[2]));

	}

	return pt;
}

boost::property_tree::ptree MainRunModes::Example_PatternPreProcessing_Dark(boost::property_tree::ptree pt, Settings & Options)
{
	//Example PPP_LAP Settings
	PPP::CreateDarkSettings Dark;
	PPP::DetectorPanel SamplePanel;
	{
		Dark.Dark_Path = "Darkfield.h5";
		Dark.Dark_Dataset = "data/data";

		Dark.Output_Path = "Output.h5";
		Dark.Output_Dataset = "data/data";
		
		Dark.Output_NewXML = "NewHitList.xml";

		Dark.RestrictToDataSource = false;
		Dark.DataSource_Path = "Source.h5";
	}
	//Store Statistic Settings in PT
	{
		pt.put("root.PatternPreProcessing.DarkFieldCorrection.DarkField_Path", Dark.Dark_Path);
		pt.put("root.PatternPreProcessing.DarkFieldCorrection.DarkField_Dataset", Dark.Dark_Dataset);

		pt.put("root.PatternPreProcessing.DarkFieldCorrection.Output_Path", Dark.Output_Path);
		pt.put("root.PatternPreProcessing.DarkFieldCorrection.Output_Dataset", Dark.Output_Dataset);

		pt.put("root.PatternPreProcessing.DarkFieldCorrection.Output_XML", Dark.Output_NewXML);

		pt.put("root.PatternPreProcessing.DarkFieldCorrection.RestrictToDataSource", Dark.RestrictToDataSource);
		pt.put("root.PatternPreProcessing.DarkFieldCorrection.DataSourcePath", Dark.DataSource_Path);
	}
	return pt;
}


//Run Evaluations

//Get averaged Intensity
int MainRunModes::AverageIntensity(std::string EvaluationConfigFile, Settings &Options, bool UpdateEventXML)// default:  bool UpdateEventXML
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
	Det.LoadPixelMask(EvalSettings.PixelMask_Path, EvalSettings.PixelMask_Dataset); //load Pixelmask (generates default (1)-Mask if path == "")
	
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
//Calibrations
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

int MainRunModes::GainCorrection(std::string Arg1, Settings & Options)
{
	AllSettings AS = LoadSettings(Arg1, Options);

	Detector Det;
	Det.LoadPixelMap(AS.EvaluationSettings.PixelMap_Path, AS.EvaluationSettings.PixelMap_DataSet);
	Det.LoadPixelMask(AS.EvaluationSettings.PixelMask_Path, AS.EvaluationSettings.PixelMask_Dataset);

	ProfileTime Profiler;
	Profiler.Tic();

	PPP::ProcessData_PF_LAP(Det, AS.PPPLAPSettings, AS.EvaluationSettings.XML_Path,true);
	std::cout << "DONE in " << Profiler.Toc(false) << "\n";
	return 0;
}

int MainRunModes::DarkCalibration(std::string Arg1, Settings & Options)
{
	AllSettings AS = LoadSettings(Arg1, Options);
	Detector Det;
	Det.LoadPixelMap(AS.EvaluationSettings.PixelMap_Path, AS.EvaluationSettings.PixelMap_DataSet);
	Det.LoadPixelMask(AS.EvaluationSettings.PixelMask_Path, AS.EvaluationSettings.PixelMask_Dataset);

	ProfileTime Profiler;
	Profiler.Tic();

	PPP::ProcessData_DarkFieldCorrection(Det, AS.PPPDarkSettings, AS.EvaluationSettings.XML_Path, Options);
	std::cout << "DONE in " << Profiler.Toc(false) << "\n";

	return 0;
}


//Statistics (obtain information in order to correct data and more ...)
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

int MainRunModes::GenerateSpeckleContrastStatistics(std::string ConfigFile, Settings & Options)
{

	MainRunModes::AllSettings AllSet = LoadSettings(ConfigFile,Options);

	if (AllSet.StatisticsSettings.SpeckleContrastSettings.CSVOutputPath == "")
	{
		std::cerr << "ERROR: No output path set\n";
		std::cerr << "Please set \"StatisticsSettings.SpeckleContrastStatistics.CSVOutputPath\" to an valid value in the configuration file." << std::endl;
		return -1;
	}
	
	//Load HitList
	Options.LoadHitEventListFromFile(AllSet.EvaluationSettings.XML_Path);

	//Create Detector
	Detector RefDet;
	RefDet.LoadPixelMap(AllSet.EvaluationSettings.PixelMap_Path, AllSet.EvaluationSettings.PixelMap_DataSet);
	RefDet.LoadPixelMask(AllSet.EvaluationSettings.PixelMask_Path, AllSet.EvaluationSettings.PixelMask_Dataset); 

	unsigned int LowBound = 0;
	unsigned int UpBound = (unsigned int)Options.HitEvents.size();

	if (AllSet.EvaluationSettings.RestrictStackToBoundaries)
	{
		LowBound = AllSet.EvaluationSettings.LowerBoundary;
		UpBound = AllSet.EvaluationSettings.UpperBoundary;
	}

	//Get SC-Statistics
	Statistics::SpeckleContrastStatistics SCS;
	SCS = Statistics::GetSpeckleContrastStatistics(Options, RefDet, LowBound, UpBound, AllSet.EvaluationSettings.PhotonOffset, AllSet.EvaluationSettings.PhotonStep);

	//Write results to File
	std::ofstream File(AllSet.StatisticsSettings.SpeckleContrastSettings.CSVOutputPath, std::ofstream::out);

	File << "#IntMean, IntVar";
	for (unsigned int i = 1; i <= SCS.Nmax; i++) //starts with 1 photon
	{
		File << ", P" << i;
	}
	File << std::endl;

	for (unsigned int i = 0; i < (unsigned int)SCS.SCC_Statistics.size(); i++)
	{
		File << SCS.SCC_Statistics[i].MeanPhotonDensity << ", " << SCS.SCC_Statistics[i].VariancePhotonDensity;
		for (unsigned int  j = 0; j < SCS.Nmax; j++)
		{
			if (j < SCS.SCC_Statistics[i].Probability.size())
			{
				File << ", " << SCS.SCC_Statistics[i].Probability[j];
			}
			else
			{
				File << ", 0.0";
			}
		}
		File << std::endl;
	}

	File.close();
	if (Options.echo)
		std::cout << "Speckle statistics saved as \"" << AllSet.StatisticsSettings.SpeckleContrastSettings.CSVOutputPath << "\"" << std::endl;

	return 0;
}

int MainRunModes::IsolatedPhotonChargeSharingFit(std::string ConfigFile, Settings & Options)
{
	//Requires Settings:
	//	EvalSettings/InputFiles
	//	EvalSettings/Misc (Boundaries)
	//	PatternPreProcessing/LargestAdjacentPixel/DetPanels

	//load settings
	MainRunModes::AllSettings AllSet = LoadSettings(ConfigFile, Options);

	// create Detector
	Detector Det;
	Det.LoadPixelMap(AllSet.EvaluationSettings.PixelMap_Path, AllSet.EvaluationSettings.PixelMap_DataSet);
	Det.LoadPixelMask(AllSet.EvaluationSettings.PixelMask_Path, AllSet.EvaluationSettings.PixelMask_Dataset);

	//load HitList
	Options.LoadHitEventListFromFile(AllSet.EvaluationSettings.XML_Path);

	//run photon fitting
	Statistics::GetChargeSharingByIsolatedPhotonHits(Options, Det, AllSet.StatisticsSettings.ChargeSharingSettings);

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
	//RunIAC::CreateDataEval_Settings EVS = LoadEvaluationSettings(ConfigFile, Options);

	std::cout << "load settings\n";
	MainRunModes::AllSettings EVS = LoadSettings(ConfigFile, Options);

	ProfileTime Profiler;
	Profiler.Tic();
	RunIAC::IAC_Report Report;
	std::cout << "start evaluation\n";
	Report = RunIAC::Run_AutoCorr_DataEval(Options, EVS.EvaluationSettings);
	double TimeNeeded = Profiler.Toc(false);

	//save Report
	if (EVS.EvaluationSettings.Out_Report != "")
	{
		std::ofstream file;
		file.open(EVS.EvaluationSettings.Out_Report);
		
		file << " *** IAC - Report (autocorrelation mode) *** " << std::endl;
		file << "Ver: " << Settings::GetVersion() << "\n" << std::endl;
		file << "From settings file: \"" << EVS.XMLSetting_Path << "\"\n" << std::endl;

		if (EVS.EvaluationSettings.AngularAveraged) //Aav
		{
			file << "Angular averaged (1D - Mode)\n";
			file << "Vector Length: " << Report.FinalMeshSize << "\n";
			file << "dq/dx = " << Report.dQperVox << "  (dq/dx is normalized to be 2 for 180° between k1 and k2)\n";

		}
		else //3D
		{
			file << "AC-Volume (3D - Mode)\n";
			file << "Final Mesh Size: " << Report.FinalMeshSize << " x "<< Report.FinalMeshSize << " x " << Report.FinalMeshSize << "\n";
			file << "dq/dVox = " << Report.dQperVox << "  (dq/dVox is normalized to be 2 for 180° between k1 and k2)\n";
			file << "small C(q) Mesh Size: " << Report.SmallCqMeshSize[0] << " x " << Report.SmallCqMeshSize[1] << " x " << Report.SmallCqMeshSize[2] << "\n";

			file << "Mean photon count per pattern: " << Report.MeanPhotonCount << "\n";

			file << "\n Further Information (from Settings file):\n";
			file << "Input paths:\n";
			file << "Event-XML-file path: \"" << EVS.EvaluationSettings.XML_Path << "\"\n";
			file << "Pixel-map path: \"" << EVS.EvaluationSettings.PixelMap_Path << "\"; Dataset: \"" << EVS.EvaluationSettings.PixelMap_DataSet << "\"\n";
			file << "Pixel-mask path: \"" << EVS.EvaluationSettings.PixelMask_Path << "\"; Dataset: \"" << EVS.EvaluationSettings.PixelMask_Path << "\"\n";

			file << "Output paths:\n";
			file << "AC (merged) Output path: \"" << EVS.EvaluationSettings.Out_Final_AC_Path << "\"\n";
			if (EVS.EvaluationSettings.Out_Cq_Path != "")
				file << "CQ (merged) path: \"" << EVS.EvaluationSettings.Out_Cq_Path << "\"\n";
			if(EVS.EvaluationSettings.Out_ACuw_Path != "")
				file << "AC (uw) Output path: \"" << EVS.EvaluationSettings.Out_ACuw_Path << "\"\n";
			if (EVS.EvaluationSettings.Out_Cq_small_Path != "")
				file << "CQ (small) Output path: \"" << EVS.EvaluationSettings.Out_Cq_small_Path << "\"\n";
			if (EVS.EvaluationSettings.Out_AvIntensity_Path != "")
				file << "Averaged intensity (flat) path: \"" << EVS.EvaluationSettings.Out_AvIntensity_Path << "\"\n";

			file << "Other Settings:\n";
			file << "Evaluated Stack: Events: " << Report.LowerBound << " - " << Report.UpperBound << "\n";


		}
		file << "\nEvaluation performed within " << TimeNeeded << "s (" << TimeNeeded / (3600) << "h)\n" << std::endl;


		file.close();
	}

	return 0;
}



//Sort Events
int MainRunModes::SortHitsByMeanIntensity(std::string Arg1, std::string Arg2, Settings &Options)
{
	std::cout << "Sort XML-EventList by mean intensity and save output as csv\n";
	Options.LoadHitEventListFromFile(Arg1);
	std::vector<unsigned int> SortedInd = Options.SortHitEventsByIntensity();

	std::ofstream f;
	f.open(Arg2);

	f << "XMLIndex, MeanIntensity, Photons, H5-Path, Dataset, H5Index \n";

	for (unsigned int i = 0; i < SortedInd.size(); i++)
	{
		f << SortedInd[i] << ", " << Options.HitEvents[SortedInd[i]].MeanIntensity << ", " << Options.HitEvents[SortedInd[i]].PhotonCount << ", " << Options.HitEvents[SortedInd[i]].Filename << ", " << Options.HitEvents[SortedInd[i]].Dataset << ", " << Options.HitEvents[SortedInd[i]].Event <<"\n";
	}

	f.close();
	std::cout << "Sorted List saved as \"" << Arg2 << "\"\n";
	std::cout << "Done.\n";
	return 0;
}

int MainRunModes::SortXMLHitsByMeanIntensity(std::string Arg1, std::string Arg2, Settings & Options)
{
	std::cout << "Sort XML-EventList by mean intensity and save output new XML-list\n";
	Options.LoadHitEventListFromFile(Arg1);
	std::vector<unsigned int> SortedInd = Options.SortHitEventsByIntensity();

	Settings NewList;
	NewList.HitEvents.reserve(SortedInd.size());

	for (unsigned int i = 0; i < SortedInd.size(); i++)
	{
		NewList.HitEvents.push_back(Options.HitEvents[SortedInd[i]]);
	}

	NewList.SafeHitEventListToFile(Arg2);

	std::cout << "Sorted Hit-Event-List (XML) saved as \"" << Arg2 << "\"\n";
	std::cout << "Done.\n";
	return 0;

	return 0;
}

int MainRunModes::MergeXMLHitLits(std::string Arg1, std::string Arg2, std::string Arg3, Settings & Options)
{
	std::vector<std::string> XML_in = CSV_Splitter(Arg2, ",");
	std::vector<std::string> SupplInfoList;
	if (Arg3 == "")
	{
		for (unsigned int i = 0; i < XML_in.size(); i++)
			SupplInfoList.push_back("");
	}
	else
	{
		SupplInfoList = CSV_Splitter(Arg3, ",");
		if (SupplInfoList.size() != XML_in.size())
		{
			std::cerr << "ERROR: XML input path list must have the same length as supplementary information list.\n";
			std::cerr << "Note: \"supplementary information list\" is optional, but if set the size needs to match"<< std::endl;
			return -1;
		}
	}

	Options.HitEvents.clear();
	for (unsigned int i = 0; i < XML_in.size(); i++)
	{
		std::cout << "Copy " << i + 1 << " of " << XML_in.size() << " Hit-Lits to new list." << std::endl;
		Settings Set_In;
		Set_In.LoadHitEventListFromFile(XML_in[i]);
		for (unsigned int j = 0; j < Set_In.HitEvents.size(); j++)
		{
			Settings::HitEvent t_Event;
			t_Event = Set_In.HitEvents[j];

			t_Event.SupplementInfo += SupplInfoList[i];

			Options.HitEvents.push_back(t_Event);
		}
	}
	Options.SafeHitEventListToFile(Arg1);
	std::cout << "Saved new Hit-List as \"" << Arg1 << "\"" << std::endl;
	return 0;
}

int MainRunModes::GetHitListFromCSVFile(std::string Arg1, std::string Arg2, Settings & Options)
{
	Settings OptOut;
	OptOut.HitEvents.clear();

	std::ifstream csvFile(Arg1);
	for (std::string line; std::getline(csvFile,line); )
	{
		std::vector<std::string> LineFrag;
		LineFrag = CSV_Splitter(line, ";");
		if (LineFrag.size() != 6)
		{
			std::cerr << "ERROR: CSV-HitList need to be in the format \"H5-Filename; H5-Dataset; Event; MeanInt; PhotonCount; RotationMatrix (x1,y1,z1,x2,...)\"" << std::endl;
			std::cerr << "    -> in MainRunModes::GetHitListFromCSVFile()" << std::endl;
			throw;
		}
		Settings::HitEvent currEvent;

		currEvent.Filename =  LineFrag[0];
		boost::trim(currEvent.Filename);
		currEvent.Dataset = LineFrag[1];
		boost::trim(currEvent.Dataset);
		currEvent.Event = atoi(LineFrag[2].data());
		currEvent.SerialNumber = currEvent.Event;
		currEvent.MeanIntensity = (float)atof(LineFrag[3].data());
		currEvent.PhotonCount = atoi(LineFrag[4].data());

		float RM[9];
		Settings::SplitString<float>(LineFrag[5], RM, 9, ",");

		for (int i = 0; i < 9; i++)
			currEvent.RotMatrix[i] = RM[i];

		OptOut.HitEvents.push_back(currEvent);
	}
	OptOut.SafeHitEventListToFile(Arg2);
	std::cout << "New XML-HitList saved as \"" << Arg2 <<"\""<< std::endl;
	return 0;
}

int MainRunModes::GetHitListFromStreamFile(std::string Arg1, std::string Arg2, std::string Arg3, Settings & Options)
{
	//Settings OptOut;
	Options.HitEvents.clear();

	Options.LoadStreamFile(Arg1, Arg2, false);
	Options.SafeHitEventListToFile(Arg3);

	std::cout << "Saved XML event-list as:\n\"" << Arg3 << "\"" << std::endl;

	return 0;
}

int MainRunModes::Simulate(std::string ConfigFile, Settings & Options)
{
	//Setup Open CL Stuff
	Options.SetUp_OpenCL();

	MainRunModes::AllSettings SIS = LoadSettings(ConfigFile, Options);

	//Check Stuff
	{
		if (SIS.AllSimulationSettings.UnitCell.size() == 0)
		{
			std::cerr << "ERROR: No amtoms within unit cell" << std::endl;
			std::cerr << "    -> in MainRunModes::Simulate()" << std::endl;
			throw;
		}
	
	}

	ProfileTime Profiler;
	Profiler.Tic();

	SIS.AllSimulationSettings.SimSettings.CrystalSize[0] = SIS.AllSimulationSettings.SimSettings.UnitCells[0] * sqrt(SIS.AllSimulationSettings.LatticeVector[0][0] * SIS.AllSimulationSettings.LatticeVector[0][0] + SIS.AllSimulationSettings.LatticeVector[1][0] * SIS.AllSimulationSettings.LatticeVector[1][0] + SIS.AllSimulationSettings.LatticeVector[2][0] * SIS.AllSimulationSettings.LatticeVector[2][0]);
	SIS.AllSimulationSettings.SimSettings.CrystalSize[1] = SIS.AllSimulationSettings.SimSettings.UnitCells[1] * sqrt(SIS.AllSimulationSettings.LatticeVector[0][1] * SIS.AllSimulationSettings.LatticeVector[0][1] + SIS.AllSimulationSettings.LatticeVector[1][1] * SIS.AllSimulationSettings.LatticeVector[1][1] + SIS.AllSimulationSettings.LatticeVector[2][1] * SIS.AllSimulationSettings.LatticeVector[2][1]);
	SIS.AllSimulationSettings.SimSettings.CrystalSize[2] = SIS.AllSimulationSettings.SimSettings.UnitCells[2] * sqrt(SIS.AllSimulationSettings.LatticeVector[0][2] * SIS.AllSimulationSettings.LatticeVector[0][2] + SIS.AllSimulationSettings.LatticeVector[1][2] * SIS.AllSimulationSettings.LatticeVector[1][2] + SIS.AllSimulationSettings.LatticeVector[2][2] * SIS.AllSimulationSettings.LatticeVector[2][2]);
	
	//Set up crystal
	Crystal Cryst(SIS.AllSimulationSettings.LatticeVector,  SIS.AllSimulationSettings.SimSettings.UnitCells, SIS.AllSimulationSettings.UnitCell);
	
	//Set up detector
	Detector Sim_Det;
	Sim_Det.LoadPixelMap(SIS.EvaluationSettings.PixelMap_Path, SIS.EvaluationSettings.PixelMap_DataSet);
	Sim_Det.Calc_kMap();

	Simulator Sim;
	Simulator::SimulationOutput Sim_Output;

	Sim.ParSimulate(Cryst, Sim_Det, SIS.AllSimulationSettings.SimSettings, Sim_Output, Options);

	std::cout << "Simulation performed within " << Profiler.Toc() / 3600 << "h" << std::endl;

	return 0;
}

int MainRunModes::GeneratePixelMap(std::string Arg1, Settings & Options)
{
	MainRunModes::AllSettings SIS = LoadSettings(Arg1, Options);

	if (Options.echo)
	{
		std::cout << "\nSettings:\n";
		std::cout << "Size: " << SIS.AllSimulationSettings.GPMSettings.SizeA << " x " << SIS.AllSimulationSettings.GPMSettings.SizeB << " pixel\n";
		std::cout << "Center: (" << SIS.AllSimulationSettings.GPMSettings.Center[0] << ", " << SIS.AllSimulationSettings.GPMSettings.Center[1] << ", " << SIS.AllSimulationSettings.GPMSettings.Center[2] << ")\n";
		std::cout << "Pixel edge-size: " << SIS.AllSimulationSettings.GPMSettings.PixelSize << "\n";
		std::cout << "Fast scan vector: (" << SIS.AllSimulationSettings.GPMSettings.VecA[0] << ", " << SIS.AllSimulationSettings.GPMSettings.VecA[1] << ", " << SIS.AllSimulationSettings.GPMSettings.VecA[2] << ")\n";
		std::cout << "Slow scan vector: (" << SIS.AllSimulationSettings.GPMSettings.VecB[0] << ", " << SIS.AllSimulationSettings.GPMSettings.VecB[1] << ", " << SIS.AllSimulationSettings.GPMSettings.VecB[2] << ")\n";
		std::cout << std::endl;
	}
	Simulator Sim;
	Sim.GeneratePixelMap(SIS.AllSimulationSettings.GPMSettings);

	std::cout << "Pixel-map saved as \"" << SIS.AllSimulationSettings.GPMSettings.Filename << "\"\n";
	std::cout << "Dataset: \"" << SIS.AllSimulationSettings.GPMSettings.Dataset << "\"" << std::endl;

	return 0;
}






