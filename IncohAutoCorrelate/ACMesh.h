#pragma once

#include "Forward.h"
#include "Detector.h"
#include "Settings.h"

class ACMesh
{
private:
	

public:
	ACMesh();
	~ACMesh();

	double* CQMesh = NULL;
	unsigned int* Mesh = NULL; // "virt" 3DArray [a,b,z] dimensions are not the same as Detector Geometry dimensions 


	struct MeshShape { 
		int k_A, k_B, k_C; //the Dimension alignement k_A is, by definition, the biggest, and K_C the smallest |:| K_A is fast scan and K_C is slow scan
		int Size_AB, Size_C; //Mesh size with padding (+1 in each direction)
		int Center[3]; //Voxel where q is (0, 0, 0) in [q_A, q_B, q_C]
		float dq_per_Voxel;
	};

//Settings Pointer
	Settings* Options;
	

	MeshShape Shape;
//Functions
	void CreateSmallMeshForDetector(Detector Det, int PerpSize);
	void CreateBigMeshForDetector(Detector Det, int EdgeSize);

	void CreateBigMesh_CofQ_ForDetector(Detector Det, int EdgeSize);

	void Atomic_Add_q_Entry(float q[3], float Value, Settings::Interpolation InterpolationMode);
	void Atomic_Add_q_Entry(float q_local[3], float RotationM[9], float Value, Settings::Interpolation InterpolationMode);


	struct ACMesh_Checklist {
		bool SmallMesh = false;
		bool BigMesh = false;
		bool CofQMesh = false;
	};
	ACMesh_Checklist Checklist;
};

