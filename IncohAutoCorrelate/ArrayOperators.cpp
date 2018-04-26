#include <omp.h>
#include <thread>
#include <iostream>
#include <vector>
#include <stdlib.h> 
#include <chrono>

#include<mutex>

namespace ArrayOperators
{
	//Temporary exercise stuff
	std::mutex mutex;

	int t_add[5] = {0, 0, 0, 0, 0};
	void AddSomething(int x)
	{
		

	//	std::lock_guard<std::mutex> lockGuard(mutex);
		for (int i = 0; i < x; i++)
		{
			std::this_thread::sleep_for(std::chrono::nanoseconds(i));
			{
				for (int j = 0; j < 5; j++)
				{
					t_add[j]++;
				}
				
			}
			
		}
		//
	}

	void hello() 
	{
	//	int x = t_add;
	//	std::this_thread::sleep_for(std::chrono::milliseconds(5000- t_add*100));
	//	std::cout  << x << std::endl;
	}

	void FunWithThreads()
	{
		std::cout << "\n\n\n\n Today: Fun with THREADS:\n\n\n";

		std::vector<std::thread> threads;
		for (int i = 0; i < 1000; ++i) {
			threads.push_back(std::thread(AddSomething, 10000));
		}

		for (auto& thread : threads) 
		{
			thread.join();
		}
		std::cout << t_add[0]/1000 << " ; " << t_add[1] / 1000 << " ; " << t_add[2] / 1000 << " ; " << t_add[3] / 1000 << " ; " << t_add[4] / 1000  << "\n";

	}
	
}