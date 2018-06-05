#pragma once
#include <chrono>
class ProfileTime
{
private:
	std::chrono::high_resolution_clock::time_point start;
	std::chrono::high_resolution_clock::time_point end;
public:
	ProfileTime();
	~ProfileTime();

	void Tic();
	double Toc(bool print);
	double Toc();
};

