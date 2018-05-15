#pragma once

//#include "Detector.h"
#include "Settings.h"
#include "ArrayOperators.h"
#include <vector>


namespace Statistics
{
	void Get_OrientationSphere(float*& Vectors, std::vector<Settings::HitEvent> EventList);

}