#pragma once

#include "Forward.h"
#include <vector>
#include <array>
#include "H5Cpp.h"
#include "Settings.h"
#include "ACMesh.h"

//NOTATION:
//
//Arrays: [n_y][n_x][dim(0,3)]
//logic (def):
//dimension of pixel: fastest
//x run is faster than y run (here x and y as detector intern coordinates)

class Detector
{
private:
	

	//Pixelmap Stuff
	struct XMapExtremeV { float max_x, min_x, max_y, min_y, max_z, min_z; };
	XMapExtremeV PixelMapExtend;
	inline float GetPixelPos(int dimension, int i_x, int i_y);
	inline float GetkVal(int dimension, int i_x, int i_y);
	void Calc_PixelMapExtremeValues();


	void GetSliceOutOfHDFCuboid(float* data, H5std_string Path, H5std_string DataSet, int SlicePosition);

	//k-Map Stuff

	//HelpFunctions
	inline float DiscretizeToPhotones(float Value, float Threshold, float PhotonSamplingStep);
	

	//Autocorrelation
	enum KernelMode	{	CQ_creator = 0, FullAutoCorr = 1, SparseAutoCorr = 2, AngularAveragedAutoCorr = 3	};
	

	enum DataType { NONE = -1, INT = 1, LONG = 2, FLOAT = 3, DOUBLE = 4 };

	struct DetChecklist { bool SparseHitList = false; bool Event = false; };
	DetChecklist Checklist;

public:
	struct AutoCorrFlags { Settings::Interpolation InterpolationMode; };

	Detector();
	~Detector();

//Functions
	void Calc_kMap();

	void LoadPixelMap(H5std_string Path, H5std_string DataSet);
	
	void LoadIntensityData(Settings::HitEvent* Event);
	void LoadIntensityData();
	void LoadAndAverageIntensity(std::vector<Settings::HitEvent>& Events, float Threshold);
	void LoadAndAverageIntensity(std::vector<Settings::HitEvent>& Events, float Threshold, int LowerBound, int UpperBound);
	void LoadAndAverageIntensity(std::vector<Settings::HitEvent>& Events, float Threshold, float PhotonSamplingStep);
	void LoadAndAverageIntensity(std::vector<Settings::HitEvent>& Events, float Threshold, float PhotonSamplingStep, bool Pixelmask);
	void LoadAndAverageIntensity(std::vector<Settings::HitEvent>& Events, float Threshold, float PhotonSamplingStep, int LowerBound, int UpperBound);
	void LoadAndAverageIntensity(std::vector<Settings::HitEvent>& Events, float Threshold, float PhotonSamplingStep, int LowerBound, int UpperBound, bool Pixelmask);

	void CreateSparseHitList(float Threshold);
	void CreateSparseHitList(float Threshold, float PhotonSamplingStep);

	float CalculateMeanIntensity(bool FromSparse);

	//Function "wraps":

	void InitializeDetector(H5std_string PixelMap_Path, H5std_string PixelMap_DataSet, float Pixel_Threshold);
	

	//Correlation Kernels:
	void AutoCorrelateSparseList(ACMesh & BigMesh, AutoCorrFlags Flags);
	void AutoCorrelate_CofQ(ACMesh & BigMesh, AutoCorrFlags Flags, std::vector<Settings::HitEvent>& Events, unsigned int LowerBound, unsigned int UpperBound, Settings& Options);

	void AutoCorrelate_CofQ_SmallMesh(ACMesh & SmallMesh, AutoCorrFlags Flags, Settings& Options);

//Vars
	unsigned int DetectorSize[2]; // Number of Pixels in n_y (0) and n_x (1) dimension: [n_y][n_x] => DetectorSize[0] ^= slow-scan; DetectorSize[1] ^= fast-scan

	//Intensity
	float* Intensity = NULL; // COntent of Detector

	int* PixelMask = NULL; //Binarymask to mask bad Pixel
	//Pixelmap Stuff
	float* PixelMap = NULL;    // Pixelmap with vitually three dimensions [n_y][n_x][3]
	//k-map
	float* kMap = NULL;// k-Map with vitually three dimensions [n_y][n_x][3]
	float Max_k[3];
	float Min_k[3];// to estimate required Q Volume
	float Max_q[3];

	std::vector <std::array< float, 4>> SparseHitList;



//Event for Detecctor
	Settings::HitEvent* DetectorEvent=NULL;

};
