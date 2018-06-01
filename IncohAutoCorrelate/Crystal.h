#pragma once

#include <vector>
#include <array>
#include <random> 


class Crystal
{
private:
	//Random stuff
	inline double Drand();
	std::uniform_real_distribution<double> rnd;
	std::mt19937_64 mt;
	//Lattice
	std::vector<std::array<double, 3>> Lattice;
	void Create_Lattice(double LatticeVector[3][3], int l, int m, int n);
	void Create_Crystal(int l, int m, int n, std::vector<std::array<double, 3>> UnitCell);

public:
	Crystal(double LatticeVector[3][3], unsigned int CrystalSize[3], std::vector<std::array<double, 3>> UnitCell);
	~Crystal();

	static std::array<float, 9> RotateVectors(std::vector<std::array<double, 3>> &Vectors, const double angle, std::array<double, 3> axis);
	static std::array<float, 9> RotateVectors(std::vector<std::array<double, 3>> &Vectors, std::array<float, 9> RotMatrix);

	std::vector<std::array<double, 3>> AtomPositions; //Atom Positions of the whole structure //x_1,y_1,z_1, ..., x_n, y_n, z_
	
	struct Emitter
	{
		std::array<double, 3> Position;
		double Phase = 0.0f;
	};

	struct EmittingCrystSettings
	{
		bool Incoherent = true; //Random phases
		float FlYield = 1.0f; // 0 < FlY <= 1: Fluorescence probability

		bool RandOrientation = false;//Roll orientation for each exposure
		float Isotropie = 1.0f; // 0 <= Isotr <= 1: 1 istotropic, 0 only one direction
	};

	std::vector<Emitter> GetEmitters(EmittingCrystSettings Settings, std::array<float,9> & RotationMatrix);
	std::vector<Emitter> GetEmitters(EmittingCrystSettings Settings, std::array<float, 9> & RotationMatrix, bool UseGivenRotation); //needed because of possible modes > 1


};

