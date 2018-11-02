#pragma once

#include <vector>
#include <string.h>


#include "Detector.h"
#include "Settings.h"
#include "RunIAC.h"
#include "ACMesh.h"


class ExoticAnalysis
{ //Class for fast hacks :D
public:
	ExoticAnalysis();
	~ExoticAnalysis();


	void CompareClustersInAC_PatternByPattern(Settings &PrgSettings, RunIAC::CreateDataEval_Settings EvalSettings, std::vector<std::vector<unsigned int>> Clusters, std::string OutputFilePath, bool IncludeHeader = true);

};

