#include <cstdio>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <array>
#include <omp.h>

#include "Settings.h"

//my Classes
#include "Detector.h"

//Variables




void TestDetectorSparseList()
{

	Detector TestDet;


	TestDet.DetectorSize[0] = 10;
	TestDet.DetectorSize[1] = 12;


	TestDet.PixelMap = new float[3 * TestDet.DetectorSize[0] * TestDet.DetectorSize[1]];


	for (int i_y = 0; i_y < TestDet.DetectorSize[1]; i_y++)
	{
		for (int i_x = 0; i_x < TestDet.DetectorSize[0]; i_x++)
		{
			TestDet.PixelMap[0 + 3 * i_x + 3 * TestDet.DetectorSize[0] * i_y] = i_x -(float) 0.45*TestDet.DetectorSize[0];
			TestDet.PixelMap[1 + 3 * i_x + 3 * TestDet.DetectorSize[0] * i_y] = i_y - (float) 0.65*TestDet.DetectorSize[1];
			TestDet.PixelMap[2 + 3 * i_x + 3 * TestDet.DetectorSize[0] * i_y] = 75;
		}
	}


	TestDet.Calc_kMap();


}



int main()
{
	Settings Options;
	Options.echo = true;

	Options.Echo("hello from IncohAutoCorrelate!");


	

	//TestDetectorSparseList();

	Detector TestDet;

	TestDet.LoadPixelMap("/home/trostfab/LR17Stuff/PixelMap_X123.h5", "PixelMap_X123");

	Options.LoadStreamFile("/gpfs/cfel/cxi/scratch/data/2018/LCLS-2018-Chapman-Mar-LR17/indexing/hemSh_all2_local.stream","entry_1/instrument_1/detector_2/detector_corrected/data",false);
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
	Options.Echo("Create sparese detector");
	TestDet.CreateSparseHitList(3.0);

	std::cout << "Number of entrys above threshold: " <<TestDet.SparseHitList.size() << "\n";

	Options.Echo("Calc mean Intensity");
	TestDet.CalculateMeanIntensity(true);

	std::cout << "Mean intensity: " << Options.HitEvents[0].MeanIntensity << "\n";


	int x;
	std::cout << "Program ended\n";
	std::cin >> x;
    return 0;
}