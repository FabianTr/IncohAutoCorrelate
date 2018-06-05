#include "ProfileTime.h"
#include <chrono>
#include <iostream>
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
double ProfileTime::Toc(bool print)
{
	end = chrono::high_resolution_clock::now();
	std::chrono::milliseconds interv  = std::chrono::duration_cast< std::chrono::milliseconds>(end - start);
	if (print)
		cout << (double)interv.count() / 1000. << " sec\n";
	return (double)interv.count()/1000;

}

double ProfileTime::Toc()
{
	return Toc(true);
}