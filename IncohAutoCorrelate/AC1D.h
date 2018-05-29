#pragma once

#include "Forward.h"
#include "Detector.h"
#include "Settings.h"

//In contrast to ACMesh, AC1D represents the angular averaged autocorrelation in 1 dimension. Also both, AC_UW and C(q) is stored 
//within one class and combination functions are added. //ToDo: Maybe also implement this "double use" for ACMesh.
class AC1D
{
private:
	void Calculate_AC_UW_Mapped(Settings & Options, double * AC_M, unsigned int LowerBound, unsigned int UpperBound, Settings::Interpolation IterpolMode);
public:
	AC1D();
	~AC1D();


	double * CQ = NULL;
	double * AC_UW = NULL;
	double * AC = NULL;
	double * Q = NULL;

	struct MeshShape {
		unsigned int Size = 1000;
		float dq_per_Step;
		float Max_Q; // sqrt(qx^2 + qy^2 + qz^2) calculate it from pixelmap
	};

	MeshShape Shape;

	void Initialize();
	void Initialize(Detector &Det, unsigned int ArraySize);


	//kernel handel
	void Calculate_CQ(Detector &Det, Settings &Options, Settings::Interpolation IterpolMode);


	//Sparse Autocorrelate
	void Calculate_AC_UW_MR(Settings &Options, Settings::Interpolation IterpolMode);
	
};

