#pragma once

#include "Forward.h"
#include "Detector.h"
#include "Settings.h"
#include "RunIAC.h"

//In contrast to ACMesh, AC1D represents the angular averaged autocorrelation in 1 dimension. Also both, AC_UW and C(q) is stored 
//within one class and combination functions are added. //ToDo: Maybe also implement this "double use" for ACMesh.
class AC1D
{
private:
	
public:
	AC1D();
	~AC1D();


	double * CQ = NULL;
	double * AC_UW = NULL;
	double * AC = NULL;
	double * Q = NULL;

	int EchoLevel = 1;

	struct MeshShape {
		unsigned int Size = 1000;
		float dq_per_Step;
		float Max_Q; // sqrt(qx^2 + qy^2 + qz^2) calculate it from pixelmap
	};

	MeshShape Shape;

	void Initialize();
	void Initialize(Detector &Det, unsigned int ArraySize);
	void Initialize(Detector & Det, unsigned int ArraySize, float QZoom);

	//kernel handel
	void Calculate_CQ(Detector &Det, Settings &Options, Settings::Interpolation IterpolMode);


	//Sparse or Dense Autocorrelate dyn switch
	void Calculate_AC_UW_MR(Settings & Options, Detector & RefDet, Settings::Interpolation IterpolMode, float PhotonOffset, float PhotonStep, int Threads = 200);
	void Calculate_AC_UW_MR(Settings &Options, Detector & RefDet, Settings::Interpolation IterpolMode, std::array<float, 2> Photonisation, bool JungfrauDet, int Threads = 200);



};

