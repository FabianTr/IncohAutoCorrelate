#pragma once

#include <vector>
#include <string.h>


#include "Detector.h"
#include "Settings.h"
#include "RunIAC.h"
#include "ACMesh.h"

namespace ExoticAnalysis
{

	void CompareClustersInAC_PatternByPattern(Settings &PrgSettings, RunIAC::CreateDataEval_Settings EvalSettings, std::vector<std::vector<unsigned int>> Clusters, std::string OutputFilePath, std::vector<std::string> HeaderNames, bool IncludeHeader = true);


}