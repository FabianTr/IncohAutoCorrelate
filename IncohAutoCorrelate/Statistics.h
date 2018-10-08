#pragma once

//#include "Detector.h"
#include "Settings.h"
#include "ArrayOperators.h"
#include "Detector.h"


#include <vector>
#include <string.h>


namespace Statistics
{

	
	struct Create_PixelHistogramSettings
	{
		unsigned int Bins = 100;
		double SmalestValue = 0;
		double LargestValue = 100;

		bool Normalized = true;

		std::string OutputPath = "";
	};

	struct StatisticsSettings
	{
		Statistics::Create_PixelHistogramSettings PixelHistogramSettings;
	};

	void Get_OrientationSphere(float*& Vectors, std::vector<Settings::HitEvent> EventList);


	struct PatternStackStatistics
	{
		double PhotonCount_Mean = 0;
		double PhotonCount_StAbw = 0;
	};

	class Histogram
	{
	public:
		Histogram(unsigned int Size, double BinSize=1.0, double FirstBin=0);

		unsigned int Size = 0;
		double FirstBin = 0;
		double BinSize = 1;
		std::vector<unsigned long> HistogramContent;
		unsigned long UnderflowBin = 0;
		unsigned long OverflowBin = 0;

		unsigned long long Entries = 0;

		void AddValue(double Value);
		void SafeToFile(std::string Filename);
	};



	Histogram Make_AllPixel_Histogram(Settings & Options, Detector &RefDet, unsigned int Bins, double SmallestVal, double HighestVal);
	std::vector<Histogram> MakePixelHistogramStack(Settings & Options, Detector &RefDet, unsigned int Bins, double SmallestVal, double HighestVal);

	void CreateAndSaveAllPixelHistograms(Create_PixelHistogramSettings HistSettings, Detector &RefDet, Settings & Options);
	


	double GetAverageRatioOfPixelsWithHits(Settings & Options, Detector &RefDet, float Offset, unsigned int LowerBound, unsigned int UpperBound, double & StDev);
	double GetAverageRatioOfPixelsWithHits(Settings & Options, Detector &RefDet, float Offset, unsigned int LowerBound, unsigned int UpperBound);
	double GetAverageRatioOfPixelsWithHits(Settings & Options, Detector &RefDet, float Offset, double & StDev);
	double GetAverageRatioOfPixelsWithHits(Settings & Options, Detector &RefDet, float Offset);
	


}