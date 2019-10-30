#include "ProfileTime.h"
#include <chrono>
#include <iostream>
#include <cmath>
#include <iomanip>

using namespace std;


ProfileTime::ProfileTime()
{
}


ProfileTime::~ProfileTime()
{
}

void ProfileTime::Tic()
{
	start = chrono::high_resolution_clock::now();
}
double ProfileTime::Toc(bool print, bool hms_format)
{
	end = chrono::high_resolution_clock::now();
	std::chrono::milliseconds interv  = std::chrono::duration_cast< std::chrono::milliseconds>(end - start);
	if (print)
	{
		if (hms_format)
		{
			double secs = (double)interv.count() / 1000.0;
			int h = (int)std::floor(secs / 3600.0);
			secs = secs - h * 3600.0;
			int min = (int)std::floor(secs / 60.0);
			secs = secs - min * 60.0;

			char s[7];
			std::sprintf(s, "%06.3f", secs);
			cout << h << ":" << std::setfill('0') << std::setw(2) << min << ":" << s << std::endl;

		}
		else
		{
			cout << (double)interv.count() / 1000. << " sec" << std::endl;
		}
	}
	return (double)interv.count()/1000;

}