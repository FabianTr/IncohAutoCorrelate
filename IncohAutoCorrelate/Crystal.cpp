#include "Crystal.h"
#include <omp.h>
#include <iostream>



Crystal::Crystal(double LatticeVector[3][3], unsigned int CrystalSize[3], std::vector<std::array<double, 3>> UnitCell) :mt(std::random_device{}())
{
	//Setup Crystal-lattice
	Create_Lattice(LatticeVector,CrystalSize[0], CrystalSize[1], CrystalSize[2]); //create lattice
	Create_Crystal(CrystalSize[0], CrystalSize[1], CrystalSize[2], UnitCell);
}


Crystal::~Crystal()
{
}

void Crystal::RandomGen_ReSeed()
{
	mt.seed(std::random_device{}());
}

std::vector<Crystal::Emitter> Crystal::GetEmitters(EmittingCrystSettings Settings, std::array<float, 9>& RotationMatrix)
{
	return GetEmitters(Settings, RotationMatrix, false);
}

std::vector<Crystal::Emitter> Crystal::GetEmitters(EmittingCrystSettings Settings,  std::array<float, 9> & RotationMatrix, bool UseGivenRotation)
{
	//Temporary Atom Positions
	std::vector<std::array<double, 3>> At_temp;

	//roll fluorescing atoms with respect to the fluorescence yield 
	if (Settings.FlYield < 1)
	{
		At_temp.reserve((std::size_t)(AtomPositions.size()*Settings.FlYield*1.1));
		for (unsigned int i = 0; i < (unsigned int)AtomPositions.size(); i++)
		{
			if (Drand() <= Settings.FlYield)
				At_temp.push_back(AtomPositions[i]);
		}
	}
	else
	{
		At_temp = AtomPositions;
	}


	if (Settings.RandOrientation && !UseGivenRotation)
	{
		//Roll Rotation
		std::array<double, 3> V;
		V[0] = Drand() * 2. - 1.;
		V[1] = (Drand() * 2. - 1.) * Settings.Isotropie;
		V[2] = (Drand() * 2. - 1.) * Settings.Isotropie;

		while (V[0] * V[0] + V[1] * V[1] + V[2] * V[2] > 1)
		{
			V[0] = Drand() * 2. - 1.;
			V[1] = (Drand() * 2. - 1.) * Settings.Isotropie;
			V[2] = (Drand() * 2. - 1.) * Settings.Isotropie;
		}
		double alpha = Drand() * 2 * M_PIl;

		//Rotate
		std::array<float, 9> tRM;

		tRM=RotateVectors(At_temp, alpha, V);
		for (int i = 0; i < 9; i++)
		{
			RotationMatrix[i] = tRM[i];
		}
	}
	else if (UseGivenRotation)
	{
		RotateVectors(At_temp, RotationMatrix);
	}
	else
	{
		//return unity matrix
		RotationMatrix[0] = 1;
		RotationMatrix[1] = 0;
		RotationMatrix[2] = 0;

		RotationMatrix[3] = 0;
		RotationMatrix[4] = 1;
		RotationMatrix[5] = 0;

		RotationMatrix[6] = 0;
		RotationMatrix[7] = 0;
		RotationMatrix[8] = 1;
	}

	//Copy remaining positions and roll phase, if requested
	std::vector<Emitter> Ret;
	Ret.resize(At_temp.size());

	//#pragma omp parallel for
	for (unsigned int i = 0; i < Ret.size(); i++)
	{		
		Emitter EM;
		EM.Position[0] = At_temp[i][0];
		EM.Position[1] = At_temp[i][1];
		EM.Position[2] = At_temp[i][2];
		if (Settings.Incoherent)
		{
			//EM.Phase = (ThrRand(mt) * 2 * M_PIl);
			EM.Phase = Drand() * 2.0 * M_PIl;
			
		}
		else
		{
			EM.Phase = 0.0;
		}
		Ret[i]=EM;
	}

	return Ret;
}




std::array<float, 9> Crystal::RotateVectors(std::vector<std::array<double, 3>> &Vectors, const double angle, std::array<double, 3> axis)
{
	//normalize axis vector
	double N = sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]);
	axis[0] = axis[0] / N;
	axis[1] = axis[1] / N;
	axis[2] = axis[2] / N;
	//rotation matrix {{a1, a2, a3}, {b1, b2, b3}, {c1, c2, c3}}
	double a1 = axis[0] * axis[0] * (1 - cos(angle)) + cos(angle);
	double a2 = axis[1] * axis[0] * (1 - cos(angle)) - axis[2] * sin(angle);
	double a3 = axis[2] * axis[0] * (1 - cos(angle)) + axis[1] * sin(angle);

	double b1 = axis[0] * axis[1] * (1 - cos(angle)) + axis[2] * sin(angle);
	double b2 = axis[1] * axis[1] * (1 - cos(angle)) + cos(angle);
	double b3 = axis[2] * axis[1] * (1 - cos(angle)) - axis[0] * sin(angle);

	double c1 = axis[0] * axis[2] * (1 - cos(angle)) - axis[1] * sin(angle);
	double c2 = axis[1] * axis[2] * (1 - cos(angle)) + axis[0] * sin(angle);
	double c3 = axis[2] * axis[2] * (1 - cos(angle)) + cos(angle);

	std::array<float, 9> RotationMatrix;
	RotationMatrix[0] = a1;
	RotationMatrix[1] = a2;
	RotationMatrix[2] = a3;

	RotationMatrix[3] = b1;
	RotationMatrix[4] = b2;
	RotationMatrix[5] = b3;

	RotationMatrix[6] = c1;
	RotationMatrix[7] = c2;
	RotationMatrix[8] = c3;

#pragma omp parallel for
	for (int i = 0; i < (int)Vectors.size(); i++)
	{
		std::array<double, 3> V = Vectors[i];
		//Rotate
		Vectors[i][0] = a1 * V[0] + a2 * V[1] + a3 * V[2];
		Vectors[i][1] = b1 * V[0] + b2 * V[1] + b3 * V[2];
		Vectors[i][2] = c1 * V[0] + c2 * V[1] + c3 * V[2];
	}

	return RotationMatrix;
}

std::array<float, 9> Crystal::RotateVectors(std::vector<std::array<double, 3>>& Vectors, std::array<float, 9> RotMatrix)
{
#pragma omp parallel for
	for (int i = 0; i < (int)Vectors.size(); i++)
	{
		std::array<double, 3> V = Vectors[i]; // copy one vector
		//Rotate
		Vectors[i][0] = RotMatrix[0] * V[0] + RotMatrix[1] * V[1] + RotMatrix[2] * V[2];
		Vectors[i][1] = RotMatrix[3] * V[0] + RotMatrix[4] * V[1] + RotMatrix[5] * V[2];
		Vectors[i][2] = RotMatrix[6] * V[0] + RotMatrix[7] * V[1] + RotMatrix[8] * V[2];
	}

	return RotMatrix; 
}

inline double Crystal::Drand()
{
	return rnd(mt);
}


void Crystal::Create_Lattice(double LatticeVector[3][3], unsigned int l, unsigned int m, unsigned int n)
{
	Lattice.clear();
	Lattice.resize(l*m*n);//adjust array size

#pragma omp parallel for
	for (unsigned int x = 0; x < l; x++)
	{
		for (unsigned int y = 0; y < m; y++)
		{
			for (unsigned int z = 0; z < n; z++)
			{
				Lattice[z + n * y + n * m*x][0] = (x - l / 2.0 + 0.5) * LatticeVector[0][0] + (y - m / 2.0 + 0.5) * LatticeVector[1][0] + (z - n / 2.0 + 0.5) * LatticeVector[2][0]; //Lattice centered around (0,0,0)'
				Lattice[z + n * y + n * m*x][1] = (x - l / 2.0 + 0.5) * LatticeVector[0][1] + (y - m / 2.0 + 0.5) * LatticeVector[1][1] + (z - n / 2.0 + 0.5) * LatticeVector[2][1];
				Lattice[z + n * y + n * m*x][2] = (x - l / 2.0 + 0.5) * LatticeVector[0][2] + (y - m / 2.0 + 0.5) * LatticeVector[1][2] + (z - n / 2.0 + 0.5) * LatticeVector[2][2];
			}
		}
	}
}

void Crystal::Create_Crystal(unsigned int l, unsigned int m, unsigned int n, std::vector<std::array<double, 3>> UnitCell)
{
	AtomPositions.clear();
	AtomPositions.resize(l*m*n*UnitCell.size());

	uint k = UnitCell.size();
#pragma omp parallel for
	for (unsigned int i = 0; i < l*m*n; i++)
	{
		for (unsigned int j = 0; j < k; j++)
		{
			AtomPositions[i*k + j][0] = Lattice[i][0] + UnitCell[j][0];
			AtomPositions[i*k + j][1] = Lattice[i][1] + UnitCell[j][1];
			AtomPositions[i*k + j][2] = Lattice[i][2] + UnitCell[j][2];
		}
	}

}
