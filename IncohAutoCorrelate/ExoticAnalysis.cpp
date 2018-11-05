#include <iostream>
#include <omp.h>

#include "ExoticAnalysis.h"
#include "ProfileTime.h"
#include "ArrayOperators.h"
#include <fstream>




void ExoticAnalysis::CompareClustersInAC_PatternByPattern(Settings & PrgSettings, RunIAC::CreateDataEval_Settings EvalSettings, std::vector<std::vector<unsigned int>> Clusters, std::string OutputFilePath,  std::vector<std::string> HeaderNames,bool IncludeHeader)
{
	unsigned int NumberOfClusters = Clusters.size();
	if (NumberOfClusters == 0)
	{
		std::cerr << "ERROR: No Clusters defined\n";
		std::cerr << "in CompareClustersInAC_PatternByPattern()\n";
		throw;
	}

	ProfileTime ProfileLevel_0;
	ProfileTime ProfileLevel_1;

	Detector Det;

	ACMesh Mesh_CurrAC(&PrgSettings);
	ACMesh Mesh_CQ(&PrgSettings);

	unsigned int StackSize = 0;
	unsigned int lowerBound = 0;
	unsigned int upperBound = 0;

	//Initialize Stuff
	{
		if (EvalSettings.PixelMap_Path == "")
		{
			std::cerr << "ERROR: No pixel map path set => can't load Hit Events\n";
			std::cerr << "    -> in Run_AutoCorr_DataEval()\n";
			throw;
		}
		if (EvalSettings.EchoLevel > 0)
			std::cout << "Load pixel map\n";

		//load pixelmap and, if exists, pixel mask
		Det.LoadPixelMap(EvalSettings.PixelMap_Path, EvalSettings.PixelMap_DataSet);

		Det.Calc_kMap();
		Det.LoadPixelMask(EvalSettings.PixelMask_Path, EvalSettings.PixelMask_Dataset);
	}


	//Create Output Field

	std::vector<std::vector<double>> OutputMatrix(StackSize, std::vector<double>(NumberOfClusters + 1));


	//


	//Load Events
	if (EvalSettings.XML_Path == "")
	{
		std::cerr << "ERROR: No XML path set => can't load Hit Events\n";
		std::cerr << "    -> in Run_AutoCorr_DataEval()\n";
		throw;
	}
	if (EvalSettings.EchoLevel > 0)
		std::cout << "Load Hit Event List\n";
	{
		PrgSettings.LoadHitEventListFromFile(EvalSettings.XML_Path);

		if (EvalSettings.InvertRotMatrix)
		{
			if (EvalSettings.EchoLevel > 0)
				std::cout << "Invert rotation matrices from Hit_Event_List\n";
			PrgSettings.InvertRotationMatrices();
		}

		if (EvalSettings.RestrictStackToBoundaries)
		{
			if (EvalSettings.UpperBoundary > PrgSettings.HitEvents.size())
			{
				std::cerr << "ERROR: Upper Hit boundary exeeds number of hits in loaded XML file\n";
				std::cerr << "    -> in Run_AutoCorr_DataEval()\n";
				throw;
			}
			else
			{
				StackSize = EvalSettings.UpperBoundary - EvalSettings.LowerBoundary;
				lowerBound = EvalSettings.LowerBoundary;
				upperBound = EvalSettings.UpperBoundary;
			}
		}
		else
		{
			StackSize = PrgSettings.HitEvents.size();
			lowerBound = 0;
			upperBound = StackSize;
		}
	}

	//Load or Create averaged intensity
	if (EvalSettings.UseExistingAvInt && !EvalSettings.UsePixelMask_as_Flatfield) //load existing avIntensity (flat field)
	{
		PrgSettings.Echo("Load existing averaged intensity");
		//Load avIntensity
		delete[] Det.Intensity;
		Det.Intensity = new float[Det.DetectorSize[0] * Det.DetectorSize[1]]();
		ArrayOperators::LoadArrayFromFile(EvalSettings.Out_AvIntensity_Path, Det.Intensity, Det.DetectorSize[0] * Det.DetectorSize[1]);
		//Apply Pixelmask
		Det.ApplyPixelMask();
	}
	else if (!EvalSettings.UsePixelMask_as_Flatfield)//Create avIntensity
	{
		PrgSettings.Echo("Load and averaged all Intensities");
		Det.LoadAndAverageIntensity(PrgSettings.HitEvents, EvalSettings.PhotonOffset, EvalSettings.PhotonStep, lowerBound, upperBound, true);

		if (EvalSettings.Out_AvIntensity_Path != "")//save averaged intensity
		{
			ArrayOperators::SafeArrayToFile(EvalSettings.Out_AvIntensity_Path, Det.Intensity, Det.DetectorSize[0] * Det.DetectorSize[1], ArrayOperators::Binary);
			if (EvalSettings.EchoLevel > 0)
				std::cout << "-> Averaged intensity saved as \"" << EvalSettings.Out_AvIntensity_Path << "\".\n";
		}

		//Apply Pixelmask
		Det.ApplyPixelMask();
	}
	else //Use Pixelmask as avIntensity (flat field)
	{
		PrgSettings.Echo("Use pixel-mask as flat-field (instead of averaged intensity)");

		delete[] Det.Intensity;
		Det.Intensity = new float[Det.DetectorSize[1] * Det.DetectorSize[0]]();
		for (unsigned int i = 0; i < Det.DetectorSize[0] * Det.DetectorSize[1]; i++)
		{
			Det.Intensity[i] = (float)Det.PixelMask[i];
		}
		Det.Checklist.Intensity = true;


	}






	//C(q)

	ACMesh smallMesh(&PrgSettings);
	smallMesh.CreateSmallMesh_CofQ_ForDetector(Det, EvalSettings.MeshSize, EvalSettings.QZoom);

	Detector::AutoCorrFlags Flags;
	Flags.InterpolationMode = Settings::NearestNeighbour; //TODO implement more modes
	Det.AutoCorrelate_CofQ_SmallMesh(smallMesh, Flags, PrgSettings);

	// -> <Merge CQ>
	ProfileLevel_1.Tic();
	if (EvalSettings.EchoLevel > 0)
		std::cout << "Weight and merge C(q)\n";

	Mesh_CQ.CreateBigMesh_CofQ_ForDetector(Det, EvalSettings.MeshSize, EvalSettings.QZoom);

	Det.Merge_smallCofQ(Mesh_CQ, smallMesh, PrgSettings.HitEvents, lowerBound, upperBound, PrgSettings, Flags);


	//Autocorrelate
	ProfileTime ACProfiler1;
	ProfileTime ACProfiler2;

	ProfileLevel_1.Tic();
	ACMesh AC_uw(&PrgSettings);

	if (EvalSettings.EchoLevel > 0)
		std::cout << "Start auto correlation for sparsificated intensities\n";

	Det.LoadIntensityData(&PrgSettings.HitEvents[0]);

	unsigned int counter = 0;


	std::ofstream file;
	file.open(OutputFilePath, std::ios::app);

	if (IncludeHeader)
	{
		//include header for output csv

		file << "mean intensity";
		for (unsigned int n = 1; n <= NumberOfClusters; n++)
		{
			if (HeaderNames.size() != Clusters.size())
			{
				std::cout << "Heade size doesn't macht cluster size, ... fix that!\n";
				file << ", cluster " << n;
			}
			else
			{
				file << ", " << HeaderNames[n-1];
			}
		}
		file << std::endl;
	}

	ACProfiler1.Tic();
	for (unsigned int i = lowerBound; i < upperBound; i++)
	{
		
		//Reset Mesh
		AC_uw.CreateBigMeshForDetector(Det, EvalSettings.MeshSize, EvalSettings.QZoom);

		if (EvalSettings.EchoLevel > 4)
			std::cout << "AC event " << i << "/" << (upperBound) << std::endl;

		Det.LoadIntensityData(&PrgSettings.HitEvents[i]);

		ArrayOperators::ParMultiplyElementwise(Det.Intensity, Det.PixelMask, Det.DetectorSize[0] * Det.DetectorSize[1]);

		Det.CreateSparseHitList(EvalSettings.PhotonOffset, EvalSettings.PhotonStep); //Sparsificate

		//if (EvalSettings.EchoLevel > 4)
		//	std::cout << i << ": Pixels with hits: " << Det.SparseHitList.size()*100.0 / (Det.DetectorSize[0] * Det.DetectorSize[1]) << "%" << "    Mean intensity: " << PrgSettings.HitEvents[i].MeanIntensity << "\n";

		Det.AutoCorrelateSparseList(AC_uw, Flags, EvalSettings.DoubleMap, PrgSettings);
	
		file << PrgSettings.HitEvents[i].MeanIntensity;

		OutputMatrix[counter][0] = PrgSettings.HitEvents[i].MeanIntensity;
		for (unsigned int n = 1; n <= NumberOfClusters; n++)
		{
			OutputMatrix[counter][n] = 0.0;
			
			for (unsigned int m = 0; m < Clusters[n].size(); m++)
			{
				double tmpVal = (double)PrgSettings.IntToFloat(AC_uw.Mesh[Clusters[n][m]]);
				tmpVal = tmpVal / Mesh_CQ.CQMesh[Clusters[n][m]];
				OutputMatrix[counter][n] += tmpVal;
			}
			file << ", " << OutputMatrix[counter][n];
		}
		file << "\n";
		std::cout << counter + 1  << " / " << StackSize << ": MeanInt = " << PrgSettings.HitEvents[i].MeanIntensity << "   "<< ACProfiler1.Toc(false)/60.0 << "min elapsed." << std::endl;

		counter++;
	}


	file.close();

	std::cout << "Saved results in \"" << OutputFilePath << "\"\n";
	std::cout << "Done in " << ACProfiler1.Toc(false) / 3600.0 << "h.\n";
}
