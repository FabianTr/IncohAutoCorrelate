#include "ACMesh.h"
#include <math.h>
#include <stdexcept>

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

	
	delete Mesh;
	Mesh = new float(Shape.Size_AB*Shape.Size_AB*Shape.Size_C);
}
