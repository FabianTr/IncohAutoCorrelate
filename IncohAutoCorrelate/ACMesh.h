#pragma once

#include "Forward.h"
#include "Detector.h"
#include "Settings.h"

class ACMesh
{
private:
	

public:
	ACMesh(Settings* PrgSettings);
	~ACMesh();

	double* CQMesh = nullptr;
	unsigned long* Mesh = nullptr; // "virt" 3DArray [a,b,z] dimensions are not the same as Detector Geometry dimensions 


	struct MeshShape { 
		int k_A, k_B, k_C; //the Dimension alignement k_A is, by definition, the biggest, and K_C the smallest |:| K_A is fast scan and K_C is slow scan
		int Size_AB, Size_C; //Mesh size with padding (+1 in each direction)
		int Center[3]; //Voxel where q is (0, 0, 0) in [q_A, q_B, q_C]
		float dq_per_Voxel;

		float Max_Q; // is not dq_per_Voxel*Size_AB because of the rotation factor sqrt(2.00001)
	};

//Settings Pointer
	Settings* Options = NULL; //important to obtain the float to int mapping (functionpointer)
	

	MeshShape Shape;
//Functions
	void CreateSmallMeshForDetector(Detector & Det, int PerpSize);
	void CreateSmallMeshForDetector(Detector & Det, int PerpSize, float q_Zoom);

	void CreateBigMeshForDetector(Detector & Det, int EdgeSize);
	void CreateBigMeshForDetector(Detector & Det, int EdgeSize, float q_Zoom);

	void CreateBigMesh_CofQ_ForDetector(Detector & Det, int EdgeSize);
	void CreateBigMesh_CofQ_ForDetector(Detector & Det, int EdgeSize, float q_Zoom);

	void CreateSmallMesh_CofQ_ForDetector(Detector & Det, int PerpSize);
	void CreateSmallMesh_CofQ_ForDetector(Detector & Det, int PerpSize, float q_Zoom);

	void Atomic_Add_q_Entry(float q[3], float Value, Settings::Interpolation InterpolationMode);
	void Atomic_Add_q_Entry(float q_local[3], float RotationM[9], float Value, Settings::Interpolation InterpolationMode );
	void Atomic_Add_q_Entry(float q_local[3], float RotationM[9], float Value, Settings::Interpolation InterpolationMode, bool DoubleBinning);
	


	struct ACMesh_Checklist {
		bool SmallMesh = false;
		bool BigMesh = false;
		bool CofQBigMesh = false;
		bool CofQSmallMesh = false;
	};
	ACMesh_Checklist Checklist;
};

