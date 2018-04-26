#include "ACMesh.h"
#include <math.h>
#include <stdexcept>
#include <omp.h>
#include "ArrayOperators.h"

ACMesh::ACMesh()
{
}


ACMesh::~ACMesh()
{
}

void ACMesh::CreateSmallMeshForDetector(Detector Det, int PerpSize)
{
	if (PerpSize % 2 == 0)//check if PerpSize is even or odd 
	{
		std::cerr << "ERROR: Meshsize must be odd for each dimension.\n";
		throw std::invalid_argument("PerpSize must be odd");
	}
	//Find largest q-axis:
	if (Det.Max_q[0] > Det.Max_q[1] && Det.Max_q[0] > Det.Max_q[2])
		Shape.k_A = 0;
	else if (Det.Max_q[1] > Det.Max_q[2] && Det.Max_q[1] > Det.Max_q[0])
		Shape.k_A = 1;
	else if (Det.Max_q[2] > Det.Max_q[0] && Det.Max_q[2] > Det.Max_q[1])
		Shape.k_A = 2;
	//Find smallest q-axis:
	if (Det.Max_q[0] < Det.Max_q[1] && Det.Max_q[0] < Det.Max_q[2])
		Shape.k_C = 0;
	else if (Det.Max_q[1] < Det.Max_q[2] && Det.Max_q[1] < Det.Max_q[0])
		Shape.k_C = 1;
	else if (Det.Max_q[2] < Det.Max_q[0] && Det.Max_q[2] < Det.Max_q[1])
		Shape.k_C = 2;
	// calculate the second largest q-axis
	Shape.k_B = 3 - Shape.k_A - Shape.k_C;
    //Set Size
	Shape.Size_AB = PerpSize + 2;//+2 padding
	Shape.Size_C = (int) floor((Det.Max_q[Shape.k_C] / Det.Max_q[Shape.k_A])*PerpSize + 2.5); //+ 2 padding
	if (Shape.Size_C % 2 == 0)//check if PerpSize is even or odd 
	{
		Shape.Size_C += 1;
	}
	Shape.Center[0] = (Shape.Size_AB - 1) / 2;
	Shape.Center[1] = (Shape.Size_AB - 1) / 2;
	Shape.Center[2] = (Shape.Size_C - 1) / 2;

	Shape.dq_per_Voxel = Det.Max_q[Shape.k_A] / (((Shape.Size_AB - 1) / 2) - 1); //Calculate Voxel Size (the last -1 takes care of zero padding)
	
	delete Mesh;
	Mesh = new unsigned int[Shape.Size_AB*Shape.Size_AB*Shape.Size_C]();
}

void ACMesh::CreateBigMeshForDetector(Detector Det, int EdgeSize)
{
	if (EdgeSize % 2 == 0)//check if PerpSize is even or odd 
	{
		std::cerr << "ERROR: Meshsize must be odd for each dimension.\n";
		std::cerr << "   ->: ACMesh::CreateBigMeshForDetector()\n";
		throw std::invalid_argument("PerpSize must be odd");
	}
	
	//Shape is trivial (k-map native):
	Shape.k_A = 0;
	Shape.k_B = 1;
	Shape.k_C = 2;

	//Set Size (cubic)
	Shape.Size_AB = EdgeSize + 2;//+2 padding
	Shape.Size_C = Shape.Size_AB;

	Shape.Center[0] = (Shape.Size_AB - 1) / 2;
	Shape.Center[1] = (Shape.Size_AB - 1) / 2;
	Shape.Center[2] = (Shape.Size_C - 1) / 2;

	float MaxQ = std::max(std::max(Det.Max_q[0], Det.Max_q[1]), Det.Max_q[2]);

	Shape.dq_per_Voxel = MaxQ / (((Shape.Size_AB - 1)/2)  - 2); //Calculate Voxel Size (the last -2 takes care of zero padding)

	delete Mesh;
	Mesh = new unsigned int[Shape.Size_AB*Shape.Size_AB*Shape.Size_C]();
}

void ACMesh::Atomic_Add_q_Entry(float q[3], float Value, Settings::Interpolation InterpolationMode)
{
	//re normalize q to the Mesh spacing
//	std::cout << "q: " << q[0] << ", " << q[1] << ", " << q[2] << " \tdq_per_Voxel: " << Shape.dq_per_Voxel << "\n";

	ArrayOperators::MultiplyScalar(q, 1.0 / Shape.dq_per_Voxel, 3);
	switch (InterpolationMode)
	{
	case  Settings::Interpolation::NearestNeighbour:
		int fs, ms, ss;
		fs = (int)floorf(q[0] + 0.5) + Shape.Center[0];
		ms = (int)floorf(q[1] + 0.5) + Shape.Center[1];
		ss = (int)floorf(q[2] + 0.5) + Shape.Center[2];

		int val;
		val = Options->FloatToInt(Value);

		#pragma omp atomic
		Mesh[fs + ms*fs + ss*ms*fs] += val;
	
		//if (std::max(std::max(fabs(q[0]), fabs(q[1])),fabs(q[2])) > 300)
		//	std::cout << "q: " << q[0] << ", " << q[1] << ", " << q[2] << ";\t V = " << Value << " -> " << val << ";\t fs,ms,ss: " << fs << " " << ms << " " << ss << "\n";

		break;
	case  Settings::Interpolation::Linear:

		break;
	default:
		std::cerr << "ERROR: No valid interpolation mode choosen\n";
		std::cerr << "   ->: in ACMesh::Atomic_Add_q_Entry\n";
		throw;
	}
}

void ACMesh::Atomic_Add_q_Entry(float q_local[3], float RotationM[9], float Value, Settings::Interpolation InterpolationMode)
{
	ArrayOperators::Rotate(q_local, RotationM);
	Atomic_Add_q_Entry(q_local,  Value,  InterpolationMode);
}
