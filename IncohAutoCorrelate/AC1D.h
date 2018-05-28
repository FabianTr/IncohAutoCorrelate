#pragma once

#include "Forward.h"
#include "Detector.h"
#include "Settings.h"

//In contrast to ACMesh, AC1D represents the angular averaged autocorrelation in 1 dimension. Also both, AC_UW and C(q) is stored 
//within one class and combination functions are added. //ToDo: Maybe also implement this "double use" for ACMesh.
class AC1D
{
public:
	AC1D();
	~AC1D();


	double * CQ = NULL;
	double * AC_UW = NULL;
	double * AC = NULL;
};

