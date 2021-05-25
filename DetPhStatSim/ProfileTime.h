#pragma once
#include <chrono>
#include <iomanip>
#include <iostream>
#include <cmath>

class ProfileTime
{
private:
	std::chrono::high_resolution_clock::time_point start;
	std::chrono::high_resolution_clock::time_point end;
public:
	ProfileTime();
	~ProfileTime();

	void Tic();
	double Toc(bool print = false, bool hms_format = true);

	static void PrintTime(double secs)
	{
		int h = (int)std::floor(secs / 3600.0);
		secs = secs - h * 3600.0;
		int min = (int)std::floor(secs / 60.0);
		secs = secs - min * 60.0;

		char s[7];
		std::sprintf(s, "%06.3f", secs);
		std::cout << h << ":" << std::setfill('0') << std::setw(2) << min << ":" << s << std::endl;
	}
};