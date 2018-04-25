#pragma once

#include "Forward.h"
#include "Detector.h"

class ACMesh
{
private:
	

public:
	ACMesh();
	~ACMesh();


	float* Mesh; // "virt" 3DArray [a,b,z] dimensions are not the same as Detector Geometry dimensions 
	struct MeshShape { 
		int k_A, k_B, k_C; //the Dimension alignement k_A is, by definition, the biggest, and K_C the smallest
		int Size_AB, Size_C; //Mesh size with padding (+1 in each direction)
		int Center[3]; //Voxel where q is (0, 0, 0) in [q_A, q_B, q_C]
		float dq_per_Voxel;
	};
	MeshShape Shape;
//Functions
	void CreateSmallMeshForDetector(Detector Det, int PerpSize);


};

