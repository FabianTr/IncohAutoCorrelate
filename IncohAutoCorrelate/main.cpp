#include <cstdio>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <array>
#include <omp.h>
#include <cmath>

#include "Settings.h"

//my Classes
#include "Detector.h"

#include "ArrayOperators.h"
#include "ProfileTime.h"

//Variables







int main()
{
	ProfileTime profiler;
	Settings Options;
	Options.echo = true;

	Options.F_I_Conversion.Step = 0.01f;
	


	//TestDetectorSparseList();

	Detector TestDet;

	TestDet.LoadPixelMap("/home/trostfab/LR17Stuff/PixelMap_X123.h5", "PixelMap_X123");

	

	Options.Echo("Load Streamfile");
	Options.LoadStreamFile("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/indexing/3fs_JF.stream","entry_1/instrument_1/detector_2/detector_corrected/data",false);
	//Options.AlternateDetectorSourcePath("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/cheetah/hdf5");

	//for (int i = 0; i <10; i++)
	//{
	//	std::cout << Options.HitEvents[i].Filename << " index: " << Options.HitEvents[i].SerialNumber <<" Event: " << Options.HitEvents[i].Event << "\n";
	//}

	
	Options.Echo("Load intensity data for Hit 0");
	TestDet.LoadIntensityData(&Options.HitEvents[0]);

	std::cout << "\n";
	for (int iy = 0; iy <20; iy++)
	{
		for (int ix = 0; ix <20; ix++)
		{
			std::cout << TestDet.Intensity[ix + TestDet.DetectorSize[1] * iy] << "   ";
		}
		std::cout << "\n";
	}

	Options.Echo("Create k-map");
	TestDet.Calc_kMap();

	Options.Echo("Create BigMesh");
	ACMesh BigMesh;
	BigMesh.Options = &Options;
	BigMesh.CreateBigMeshForDetector(TestDet, TestDet.DetectorSize[0] + 1);

	Options.Echo("Create sparese detector");
	TestDet.CreateSparseHitList(3.2f, 6.4f);//
	Detector::AutoCorrFlags flags;
	flags.InterpolationMode = Settings::Interpolation::NearestNeighbour;
	TestDet.AutoCorrelateSparseList(BigMesh, flags);


	//Options.Echo("AutoCorrelate");

	//profiler.Tic();

	//for (int i = 0; i < 1000; i++)
	//{
	//	TestDet.LoadIntensityData(&Options.HitEvents[i]);
	//	TestDet.CreateSparseHitList(3.0);

	//	Detector::AutoCorrFlags flags;
	//	flags.InterpolationMode = Settings::Interpolation::NearestNeighbour;
	//	TestDet.AutoCorrelateSparseList(BigMesh, flags);

	//}
	//
	//profiler.Toc(true);

	Options.Echo("Load and average intensities");
	profiler.Tic();
	TestDet.LoadAndAverageIntensity(Options.HitEvents, 3.2,6.4, 0, 5000);
	profiler.Toc(true);
	std::cout << "\n";

	int ind = 0;
	for (int i = 0; i < 100; i++)
	{
		for (int j = 0; j < 10; j++)
		{
			std::cout << Options.HitEvents[ind].PhotonCount << "\t";
			ind++;
		}
		std::cout << "\n";
	}
	

	/*for (int iy = 0; iy <20; iy++)
	{
		for (int ix = 0; ix <20; ix++)
		{
			std::cout << TestDet.Intensity[ix + TestDet.DetectorSize[1] * iy] << "   ";
		}
		std::cout << "\n";
	}*/


	int x;
	std::cout << "Program ended\n";
	std::cin >> x;
    return 0;
}