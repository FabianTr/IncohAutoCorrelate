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
	//delete[] CQMesh;
	//delete[] Mesh;
}

void ACMesh::CreateSmallMeshForDetector(Detector Det, int PerpSize)
{
	CreateSmallMeshForDetector(Det, PerpSize, 1.0f);
}
void ACMesh::CreateSmallMeshForDetector(Detector Det, int PerpSize, float q_Zoom)
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
	Shape.Size_C = (int)floor((Det.Max_q[Shape.k_C] / Det.Max_q[Shape.k_A])*PerpSize + 2.5); //+ 2 padding
	if (Shape.Size_C % 2 == 0)//check if PerpSize is even or odd 
	{
		Shape.Size_C += 1;
	}
	Shape.Center[0] = (Shape.Size_AB - 1) / 2;
	Shape.Center[1] = (Shape.Size_AB - 1) / 2;
	Shape.Center[2] = (Shape.Size_C - 1) / 2;

	float MaxQ = Det.Max_q[Shape.k_A] / q_Zoom;


	Shape.Max_Q = MaxQ;
	Shape.dq_per_Voxel = (MaxQ / (((Shape.Size_AB - 1) / 2) - 2));// * sqrt(2.00001) //Calculate Voxel Size (the last -1 takes care of zero padding);sqrt(2.00001) is factor to ensure every rotation fits in mesh


	delete Mesh;
	Mesh = new unsigned int[Shape.Size_AB*Shape.Size_AB*Shape.Size_C]();
	Checklist.SmallMesh = true;
}

void ACMesh::CreateBigMeshForDetector(Detector Det, int EdgeSize)
{
	CreateBigMeshForDetector(Det, EdgeSize, 1.0f);
}
void ACMesh::CreateBigMeshForDetector(Detector Det, int EdgeSize, float q_Zoom)
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

	float MaxQ = std::max(std::max(Det.Max_q[0], Det.Max_q[1]), Det.Max_q[2]) / q_Zoom;

	Shape.Max_Q = MaxQ;
	Shape.dq_per_Voxel = MaxQ / (((Shape.Size_AB - 1) / 2) - 2);//* sqrt(2.00001) //Calculate Voxel Size (the last -2 takes care of zero padding)//sqrt(2.00001) is factor to ensure every rotation fits in mesh


	delete[] Mesh;

	Mesh = new unsigned int[Shape.Size_AB*Shape.Size_AB*Shape.Size_C]();
	Checklist.BigMesh = true;
}

void ACMesh::CreateBigMesh_CofQ_ForDetector(Detector Det, int EdgeSize)
{
	CreateBigMesh_CofQ_ForDetector(Det, EdgeSize, 1.0f);
} 
void ACMesh::CreateBigMesh_CofQ_ForDetector(Detector Det, int EdgeSize, float q_Zoom)
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

	float MaxQ = std::max(std::max(Det.Max_q[0], Det.Max_q[1]), Det.Max_q[2])  / q_Zoom;//* sqrt(2.00001)//sqrt(2.00001) is factor to ensure every rotation fits in mesh
	
	Shape.Max_Q = MaxQ;
	Shape.dq_per_Voxel = MaxQ / (((Shape.Size_AB - 1) / 2) - 2); //Calculate Voxel Size (the last -2 takes care of zero padding)

	delete CQMesh;
	CQMesh = new double[Shape.Size_AB*Shape.Size_AB*Shape.Size_C]();
	Checklist.CofQBigMesh = true;
}

void ACMesh::CreateSmallMesh_CofQ_ForDetector(Detector Det, int PerpSize)
{
	CreateSmallMesh_CofQ_ForDetector(Det, PerpSize, 1.0f);
}
void ACMesh::CreateSmallMesh_CofQ_ForDetector(Detector Det, int PerpSize, float q_Zoom)
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



	if (q_Zoom == 1)
	{
		Shape.Size_C = (int)floor((Det.Max_q[Shape.k_C] / Det.Max_q[Shape.k_A])*PerpSize + 2.5); //+ 2 padding
	}
	else
	{
		Shape.Size_C= Shape.Size_AB;
	}



	if (Shape.Size_C % 2 == 0)//check if PerpSize is even or odd 
	{
		Shape.Size_C += 1;
	}
	Shape.Center[0] = (Shape.Size_AB - 1) / 2;
	Shape.Center[1] = (Shape.Size_AB - 1) / 2;
	Shape.Center[2] = (Shape.Size_C - 1) / 2;


	float MaxQ = Det.Max_q[Shape.k_A] / q_Zoom;

	Shape.Max_Q = MaxQ;
	Shape.dq_per_Voxel = (MaxQ / (((Shape.Size_AB - 1) / 2) - 2));//* sqrt(2.00001) //Calculate Voxel Size (the last -1 takes care of zero padding);sqrt(2.00001) is factor to ensure every rotation fits in mesh
	
	delete CQMesh;
	CQMesh = new double[Shape.Size_AB*Shape.Size_AB*Shape.Size_C]();

	Checklist.CofQSmallMesh = true;
}


void ACMesh::Atomic_Add_q_Entry(float q_local[3], float RotationM[9], float Value, Settings::Interpolation InterpolationMode)
{
	Atomic_Add_q_Entry(q_local, RotationM, Value, InterpolationMode, false);
}
void ACMesh::Atomic_Add_q_Entry(float q_local[3], float RotationM[9], float Value, Settings::Interpolation InterpolationMode, bool DoubleBinning)
{
	if (DoubleBinning)
	{
		if (InterpolationMode == Settings::Interpolation::Linear)
		{
			std::cerr << "ERROR: Linear interpolation not implemented for double binning yet.\nContinue with nearest neighbor.\n";
		}

		if (sqrtf(q_local[0] * q_local[0] + q_local[1] * q_local[1] + q_local[2] * q_local[2]) > Shape.Max_Q)
		{
			return;
		}

		ArrayOperators::MultiplyScalar(q_local, 1.0 / Shape.dq_per_Voxel, 3);
		//First binning
		int fs_l, ms_l, ss_l;
		fs_l = (int)floorf(q_local[0] + 0.5) + Shape.Center[0];
		ms_l = (int)floorf(q_local[1] + 0.5) + Shape.Center[1];
		ss_l = (int)floorf(q_local[2] + 0.5) + Shape.Center[2];


		//float q0 = q_local[0];
		//float q1 = q_local[1];
		//float q2 = q_local[2];

		q_local[0] = (float)(fs_l - Shape.Center[0]);
		q_local[1] = (float)(ms_l - Shape.Center[1]);
		q_local[2] = (float)(ss_l - Shape.Center[2]);

		//float qs0 = q_local[0];
		//float qs1 = q_local[1];
		//float qs2 = q_local[2];

		//float r1 = RotationM[0];
		//float r2 = RotationM[1];
		//float r3 = RotationM[2];
		//float r4 = RotationM[3];
		//float r5 = RotationM[4];
		//float r6 = RotationM[5];
		//float r7 = RotationM[6];
		//float r8 = RotationM[7];
		//float r9 = RotationM[8];


		ArrayOperators::Rotate(q_local, RotationM);

		//float qr0 = q_local[0];
		//float qr1 = q_local[1];
		//float qr2 = q_local[2];
		
		//second binning
		int fs, ms, ss;
		fs = (int)floorf(q_local[0] + 0.5) + Shape.Center[0];
		ms = (int)floorf(q_local[1] + 0.5) + Shape.Center[1];
		ss = (int)floorf(q_local[2] + 0.5) + Shape.Center[2];

		int s = Shape.Size_AB;

		unsigned int val;
		val = Options->FloatToInt(Value);

#pragma omp atomic
		Mesh[fs + ms * s + ss * s*s] += val;
	}
	else
	{
		ArrayOperators::Rotate(q_local, RotationM);
		Atomic_Add_q_Entry(q_local, Value, InterpolationMode);
	}
}
void ACMesh::Atomic_Add_q_Entry(float q[3], float Value, Settings::Interpolation InterpolationMode)
{
	//re normalize q to the Mesh spacing
	//	std::cout << "q: " << q[0] << ", " << q[1] << ", " << q[2] << " \tdq_per_Voxel: " << Shape.dq_per_Voxel << "\n";


	//Check if q is in Range
	if (sqrtf(q[0] * q[0] + q[3] * q[3] + q[2] * q[2]) > Shape.Max_Q)
	{
		return;
	}

	ArrayOperators::MultiplyScalar(q, 1.0 / Shape.dq_per_Voxel, 3);
	switch (InterpolationMode)
	{
	case  Settings::Interpolation::NearestNeighbour:
	{
		int fs, ms, ss;
		fs = (int)floorf(q[0] + 0.5) + Shape.Center[0];
		ms = (int)floorf(q[1] + 0.5) + Shape.Center[1];
		ss = (int)floorf(q[2] + 0.5) + Shape.Center[2];

		int s = Shape.Size_AB;

		unsigned int val;
		val = Options->FloatToInt(Value);

#pragma omp atomic
		Mesh[fs + ms * s + ss * s*s] += val;

		//if (std::max(std::max(fabs(q[0]), fabs(q[1])),fabs(q[2])) > 300)
		//	std::cout << "q: " << q[0] << ", " << q[1] << ", " << q[2] << ";\t V = " << Value << " -> " << val << ";\t fs,ms,ss: " << fs << " " << ms << " " << ss << "\n";
	}
	break;
	case  Settings::Interpolation::Linear:
	{
		int fsf, msf, ssf; //fast-scan-, medium-scan-, slow-scan- floor
		fsf = (int)floorf(q[0]) + Shape.Center[0];
		msf = (int)floorf(q[1]) + Shape.Center[1];
		ssf = (int)floorf(q[2]) + Shape.Center[2];
		float SepF, SepM, SepS; //according Seperators
		SepF = q[0] - floorf(q[0]);
		SepM = q[1] - floorf(q[1]);
		SepS = q[2] - floorf(q[2]);

		int s = Shape.Size_AB;

		int val;
		val = Options->FloatToInt(Value);


#pragma omp atomic
		Mesh[fsf + msf * s + ssf * s*s] += val * (1 - SepF)*(1 - SepM)*(1 - SepS); // A + 0

#pragma omp atomic
		Mesh[(fsf + 1) + (msf + 0) * s + (ssf + 0) * s*s] += val * (SepF)*(1 - SepM)*(1 - SepS); //ssf + 1
#pragma omp atomic
		Mesh[(fsf + 0) + (msf + 1) * s + (ssf + 0) *s*s] += val * (1 - SepF)*(SepM)*(1 - SepS); //msf + 1
#pragma omp atomic
		Mesh[(fsf + 0) + (msf + 0) * s + (ssf + 1) * s*s] += val * (1 - SepF)*(1 - SepM)*(SepS); //ssf + 1

#pragma omp atomic
		Mesh[(fsf + 1) + (msf + 1) * s + (ssf + 0) * s*s] += val * (SepF)*(SepM)*(1 - SepS); //ffs + 1 ; msf + 1
#pragma omp atomic
		Mesh[(fsf + 0) + (msf + 1) * s + (ssf + 1) * s*s] += val * (1 - SepF)*(SepM)*(SepS); //msf + 1 ; ssf + 1
#pragma omp atomic
		Mesh[(fsf + 1) + (msf + 0) * s + (ssf + 1) * s*s] += val * (SepF)*(1 - SepM)*(SepS); //ffs + 1 ; ssf + 1

#pragma omp atomic
		Mesh[(fsf + 1) + (msf + 1) * s + (ssf + 1) * s*s] += val * (SepF)*(SepM)*(SepS); // A + 1

	}
	break;
	default:
		std::cerr << "ERROR: No valid interpolation mode choosen\n";
		std::cerr << "   ->: in ACMesh::Atomic_Add_q_Entry\n";
		throw;
	}
}

