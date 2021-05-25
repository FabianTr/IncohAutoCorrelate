#include "ACMesh.h"
#include <math.h>
#include <stdexcept>
#include <omp.h>
#include "ArrayOperators.h"






ACMesh::ACMesh(Settings *PrgSettings)
{
	if (CQMesh == nullptr)
		CQMesh = new double[1];
	if (Mesh == nullptr)
		Mesh = new unsigned long[1];
	Options = PrgSettings;
}

ACMesh::~ACMesh()
{
	delete[] CQMesh;
	delete[] Mesh;
}

void ACMesh::CreateSmallMeshForDetector(Detector & Det, int PerpSize, float q_Zoom)
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
	//regularize
	Shape.Voxel_per_dq = (unsigned int)floor(1 / Shape.dq_per_Voxel);
	Shape.dq_per_Voxel = 1.0f / ((float)Shape.Voxel_per_dq);

	delete[] Mesh;
	Mesh = new unsigned long[Shape.Size_AB*Shape.Size_AB*Shape.Size_C]();
	Checklist.SmallMesh = true;
}
void ACMesh::CreateBigMeshForDetector(Detector & Det, int EdgeSize, float q_Zoom)
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
	Shape.Voxel_per_dq = (unsigned int)floor(1 / Shape.dq_per_Voxel);
	Shape.dq_per_Voxel = 1.0f / ((float)Shape.Voxel_per_dq);

	delete[] Mesh;
	Mesh = new unsigned long[Shape.Size_AB*Shape.Size_AB*Shape.Size_C]();
	Checklist.BigMesh = true;
}
void ACMesh::CreateBigMesh_CofQ_ForDetector(Detector & Det, int EdgeSize, float q_Zoom)
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
	Shape.dq_per_Voxel = (MaxQ / (((Shape.Size_AB - 1) / 2) - 2)); //Calculate Voxel Size (the last -2 takes care of zero padding)
	Shape.Voxel_per_dq = (unsigned int)floor(1 / Shape.dq_per_Voxel);
	Shape.dq_per_Voxel = 1.0f / ((float)Shape.Voxel_per_dq);

	delete[] CQMesh;
	CQMesh = new double[Shape.Size_AB*Shape.Size_AB*Shape.Size_C]();
	Checklist.CofQBigMesh = true;
}
void ACMesh::CreateSmallMesh_CofQ_ForDetector(Detector & Det, int PerpSize, float q_Zoom)
{
	if (PerpSize % 2 == 0)//check if PerpSize is even or odd 
	{
		std::cerr << "ERROR: Meshsize must be odd for each dimension.\n";
		throw std::invalid_argument("PerpSize must be odd");
	}
	//Find largest q-axis:

	Shape.k_A = 0;
	Shape.k_B = 1;
	Shape.k_C = 2;

	if (Det.Max_q[Shape.k_B] > Det.Max_q[Shape.k_A])
	{
		std::swap(Shape.k_A,Shape.k_B);
	}

	if (Det.Max_q[Shape.k_C] > Det.Max_q[Shape.k_B])
	{
		std::swap(Shape.k_C, Shape.k_B);
	}

	if (Det.Max_q[Shape.k_B] > Det.Max_q[Shape.k_A])
	{
		std::swap(Shape.k_A, Shape.k_B);
	}

	Shape.Size_AB = PerpSize + 2;//+2 padding

	
	if (q_Zoom == 1)
	{
		//float t_1 = Det.Max_q[Shape.k_C] / Det.Max_q[Shape.k_A];
		//float t_2 = t_1 * PerpSize + 2.5;
		//float t_3 = floor(t_2);
		//int t_4 = (int)t_3;

		Shape.Size_C = (int)floor( (Det.Max_q[Shape.k_C] / Det.Max_q[Shape.k_A]) *PerpSize + 2.5); //+ 2 padding
	}
	else
	{
		Shape.Size_C = Shape.Size_AB;
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
	Shape.Voxel_per_dq = (unsigned int)floor(1 / Shape.dq_per_Voxel);
	Shape.dq_per_Voxel = 1.0f / ((float)Shape.Voxel_per_dq);

	delete[] CQMesh;
	CQMesh = new double[Shape.Size_AB*Shape.Size_AB*Shape.Size_C]();

	Checklist.CofQSmallMesh = true;
}


void ACMesh::Atomic_Add_q_Entry(float q_local[3], float RotationM[9], float Value, Settings::Interpolation InterpolationMode)
{
	Atomic_Add_q_Entry(q_local, RotationM, Value, InterpolationMode, InterpolationMode, false);
}
void ACMesh::Atomic_Add_q_Entry(float q_local[3], float RotationM[9], float Value, Settings::Interpolation FirstInterpolationMode, Settings::Interpolation SecondInterpolationMode, bool DoubleBinning)
{

	if (sqrtf(q_local[0] * q_local[0] + q_local[1] * q_local[1] + q_local[2] * q_local[2]) > Shape.Max_Q)
	{
		return;
	}

	ArrayOperators::MultiplyScalar(q_local, (float)Shape.Voxel_per_dq, 3);

	if (DoubleBinning)
	{
		if (FirstInterpolationMode == Settings::Interpolation::NearestNeighbour)
		{
			//First binning
			int fs_l, ms_l, ss_l;
			fs_l = (int)round(q_local[0]) + Shape.Center[0];
			ms_l = (int)round(q_local[1]) + Shape.Center[1];
			ss_l = (int)round(q_local[2]) + Shape.Center[2];

			q_local[0] = (float)(fs_l - (int)Shape.Center[0]);
			q_local[1] = (float)(ms_l - (int)Shape.Center[1]);
			q_local[2] = (float)(ss_l - (int)Shape.Center[2]);

			ArrayOperators::Rotate(q_local, RotationM);

			Atomic_Add_rotated_q_Entry(q_local, Value, SecondInterpolationMode);
		}
		if (FirstInterpolationMode == Settings::Interpolation::Linear)
		{
			double Sep_fs = 1.0 - (q_local[0] - floor(q_local[0]));
			double Sep_ms = 1.0 - (q_local[1] - floor(q_local[1]));
			double Sep_ss = 1.0 - (q_local[2] - floor(q_local[2]));

			int fs_l, ms_l, ss_l;
			fs_l = (int)floorf(q_local[0] ) + Shape.Center[0];
			ms_l = (int)floorf(q_local[1] ) + Shape.Center[1];
			ss_l = (int)floorf(q_local[2] ) + Shape.Center[2];

			//fff
			q_local[0] = (float)((fs_l + 0) - (int)Shape.Center[0]);
			q_local[1] = (float)((ms_l + 0) - (int)Shape.Center[1]);
			q_local[2] = (float)((ss_l + 0) - (int)Shape.Center[2]);
			ArrayOperators::Rotate(q_local, RotationM);
			Atomic_Add_rotated_q_Entry(q_local, Value * (Sep_fs)*(Sep_ms)*(Sep_ss), SecondInterpolationMode);

			//tff
			q_local[0] = (float)((fs_l + 1) - (int)Shape.Center[0]);
			q_local[1] = (float)((ms_l + 0) - (int)Shape.Center[1]);
			q_local[2] = (float)((ss_l + 0) - (int)Shape.Center[2]);
			ArrayOperators::Rotate(q_local, RotationM);
			Atomic_Add_rotated_q_Entry(q_local, Value * (1 - Sep_fs)*(Sep_ms)*(Sep_ss), SecondInterpolationMode);

			//fft
			q_local[0] = (float)((fs_l + 0) - (int)Shape.Center[0]);
			q_local[1] = (float)((ms_l + 0) - (int)Shape.Center[1]);
			q_local[2] = (float)((ss_l + 1) - (int)Shape.Center[2]);
			ArrayOperators::Rotate(q_local, RotationM);
			Atomic_Add_rotated_q_Entry(q_local, Value * (Sep_fs)*(Sep_ms)*(1 - Sep_ss), SecondInterpolationMode);

			//tft
			q_local[0] = (float)((fs_l + 1) - (int)Shape.Center[0]);
			q_local[1] = (float)((ms_l + 0) - (int)Shape.Center[1]);
			q_local[2] = (float)((ss_l + 1) - (int)Shape.Center[2]);
			ArrayOperators::Rotate(q_local, RotationM);
			Atomic_Add_rotated_q_Entry(q_local, Value * (1 - Sep_fs)*(Sep_ms)*(1 - Sep_ss), SecondInterpolationMode);

			//ftf
			q_local[0] = (float)((fs_l + 0) - (int)Shape.Center[0]);
			q_local[1] = (float)((ms_l + 1) - (int)Shape.Center[1]);
			q_local[2] = (float)((ss_l + 0) - (int)Shape.Center[2]);
			ArrayOperators::Rotate(q_local, RotationM);
			Atomic_Add_rotated_q_Entry(q_local, Value * (Sep_fs)*(1 - Sep_ms)*(Sep_ss), SecondInterpolationMode);

			//ttf
			q_local[0] = (float)((fs_l + 1) - (int)Shape.Center[0]);
			q_local[1] = (float)((ms_l + 1) - (int)Shape.Center[1]);
			q_local[2] = (float)((ss_l + 0) - (int)Shape.Center[2]);
			ArrayOperators::Rotate(q_local, RotationM);
			Atomic_Add_rotated_q_Entry(q_local, Value * (1 - Sep_fs)*(1 - Sep_ms)*(Sep_ss), SecondInterpolationMode);

			//ftt
			q_local[0] = (float)((fs_l + 0) - (int)Shape.Center[0]);
			q_local[1] = (float)((ms_l + 1) - (int)Shape.Center[1]);
			q_local[2] = (float)((ss_l + 1) - (int)Shape.Center[2]);
			ArrayOperators::Rotate(q_local, RotationM);
			Atomic_Add_rotated_q_Entry(q_local, Value * (Sep_fs)*(1 - Sep_ms)*(1 - Sep_ss), SecondInterpolationMode);

			//ttt
			q_local[0] = (float)((fs_l + 1) - (int)Shape.Center[0]);
			q_local[1] = (float)((ms_l + 1) - (int)Shape.Center[1]);
			q_local[2] = (float)((ss_l + 1) - (int)Shape.Center[2]);
			ArrayOperators::Rotate(q_local, RotationM);
			Atomic_Add_rotated_q_Entry(q_local, Value * (1 - Sep_fs)*(1 - Sep_ms)*(1 - Sep_ss), SecondInterpolationMode);
		}

	}
	else
	{
		ArrayOperators::Rotate(q_local, RotationM);
		Atomic_Add_rotated_q_Entry(q_local, Value, SecondInterpolationMode);
	}

	
}

void ACMesh::Atomic_Add_rotated_q_Entry(float q_local[3], float Value, Settings::Interpolation SecondInterpolationMode)
{

	if (SecondInterpolationMode == 0) //neares neighbour
	{
		int fs, ms, ss;
		fs = (int)round(q_local[0]) + Shape.Center[0];
		ms = (int)round(q_local[1]) + Shape.Center[1];
		ss = (int)round(q_local[2]) + Shape.Center[2];

		int s = Shape.Size_AB;

		unsigned int val;
		val = Options->FloatToInt(Value);

#pragma omp atomic
		Mesh[fs + ms * s + ss * s*s] += val;
	}
	if (SecondInterpolationMode == 1) //linear
	{
		double Sep_fs = 1 - (q_local[0] - floor(q_local[0]));
		double Sep_ms = 1 - (q_local[1] - floor(q_local[1]));
		double Sep_ss = 1 - (q_local[2] - floor(q_local[2]));
		int fs, ms, ss;

		fs = (int)floorf(q_local[0]) + Shape.Center[0];
		ms = (int)floorf(q_local[1]) + Shape.Center[1];
		ss = (int)floorf(q_local[2]) + Shape.Center[2];

		int s = Shape.Size_AB;
		unsigned int val;
		//val = Options->FloatToInt(Value);

		//fff
		val = Options->FloatToInt(Value * ((Sep_fs) * (Sep_ms)  * (Sep_ss)));
#pragma omp atomic
		Mesh[fs + ms * s + ss * s*s] += val;
		//tff
		val = Options->FloatToInt(Value * ((1 - Sep_fs) * (Sep_ms)  * (Sep_ss)));
#pragma omp atomic
		Mesh[(fs + 1) + ms * s + ss * s*s] += val;
		//fft
		val = Options->FloatToInt(Value * ((Sep_fs) * (Sep_ms)  * (1 - Sep_ss)));
#pragma omp atomic
		Mesh[(fs + 0) + ms * s + (ss + 1) * s*s] += val;
		//tft
		val = Options->FloatToInt(Value * ((1 - Sep_fs) * (Sep_ms)  * (1 - Sep_ss)));
#pragma omp atomic
		Mesh[(fs + 1) + ms * s + (ss + 1) * s*s] += val;
		//ftf
		val = Options->FloatToInt(Value * ((Sep_fs) * (1 - Sep_ms)  * (Sep_ss)));
#pragma omp atomic
		Mesh[(fs + 0) + (ms + 1) * s + (ss + 0) * s*s] += val;
		//ttf
		val = Options->FloatToInt(Value * ((1 - Sep_fs) * (1 - Sep_ms)  * (Sep_ss)));
#pragma omp atomic
		Mesh[(fs + 1) + (ms + 1) * s + (ss + 0) * s*s] += val;
		//ftt
		val = Options->FloatToInt(Value * ((Sep_fs) * (1 - Sep_ms)  * (1 - Sep_ss)));
#pragma omp atomic
		Mesh[(fs + 0) + (ms + 1) * s + (ss + 1) * s*s] += val;
		//ttt
		val = Options->FloatToInt(Value * ((1 - Sep_fs) * (1 - Sep_ms)  * (1 - Sep_ss)));
#pragma omp atomic
		Mesh[(fs + 1) + (ms + 1) * s + (ss + 1) * s*s] += val;
	}
}

