#include "AC1D.h"
#include <thread>
#include <vector>
#include "ArrayOperators.h"
#include "ProfileTime.h"
#include <mutex>
#include <functional>
#include <unordered_map>

AC1D::AC1D()
{
	//Needed for destructor to not crash if one of this isn't needed
	//Yes its waste of memory, ... think of a better way ...
	//CQ = new double[1];
	//AC_UW = new double[1];
	//AC = new double[1];
	//Q = new double[1];
}


AC1D::~AC1D()
{
	//delete[] CQ;
	//delete[] AC_UW;
	//delete[] AC;
	//delete[] Q;
}

void AC1D::Initialize()
{
	delete[] CQ;
	delete[] AC_UW;
	delete[] AC;
	delete[] Q;

	CQ = new double[Shape.Size]();
	AC_UW = new double[Shape.Size]();
	AC = new double[Shape.Size]();
	Q = new double[Shape.Size]();

	for (unsigned int i = 0; i < Shape.Size; i++)
	{
		Q[i] = (float)i * Shape.dq_per_Step;
	}
}
void AC1D::Initialize(Detector & Det, unsigned int ArraySize)
{
	Shape.Size = ArraySize;
	Shape.Max_Q = (float)(sqrt(Det.Max_q[0] * Det.Max_q[0] + Det.Max_q[1] * Det.Max_q[1] + Det.Max_q[2] * Det.Max_q[2]));
	Shape.dq_per_Step = Shape.Max_Q / ((float)(Shape.Size + 1));
	Initialize();
}

void AC1D::Calculate_CQ(Detector & Det, Settings & Options, Settings::Interpolation IterpolMode)
{
	//profiler stuff
	ProfileTime Profiler;
	//reserve OpenCL Device
	int OpenCLDeviceNumber = -1;
	cl_int err;

	while ((OpenCLDeviceNumber = Options.OCL_ReserveDevice()) == -1)
	{
		std::this_thread::sleep_for(std::chrono::microseconds(Options.ThreadSleepForOCLDev));
	}

	//obtain Device
	cl::Device CL_Device = Options.CL_devices[OpenCLDeviceNumber];


	double Multiplicator = 1;

	float Min_I = 0, Max_I = 0, Mean_I = 0;
	ArrayOperators::Min_Max_Mean_Value(Det.Intensity, Det.DetectorSize[0] * Det.DetectorSize[1], Min_I, Max_I, Mean_I);

	for (; 1 > Mean_I*Mean_I * Multiplicator; )
	{
		Multiplicator *= 10;
	}
	Multiplicator = Multiplicator * 100000;

	unsigned int MapAndReduce_Factor = 10000; //Take care of Device Memory !!!!!!!!!
	unsigned int TempArraySize = MapAndReduce_Factor * Shape.Size;

	double Params[7];
	Params[0] = (double)(Det.DetectorSize[0] * Det.DetectorSize[1]);
	Params[1] = (double)Shape.dq_per_Step;
	Params[2] = (double)Shape.Size;
	Params[3] = (double)IterpolMode;
	Params[4] = (double)Shape.Max_Q;
	Params[5] = (double)Multiplicator;
	Params[6] = (double)MapAndReduce_Factor; //Map and reduce: sub sections


	//DEBUG BULLSHIT
	std::cout << "Parameter for 1D C(q):\n";
	for (int i = 0; i < 7; i++)
	{
		std::cout << Params[i] << "\n";
	}
	//END DEBUG BULLSHIT

	uint64_t * TempArray = new uint64_t[TempArraySize]();

	std::cout << "TempArraySize: " << TempArraySize << "\n";

	//Setup Queue
	cl::CommandQueue queue(Options.CL_context, CL_Device, 0, &err);
	Options.checkErr(err, "Setup CommandQueue in AC1D::Calculate_CQ() ");
	cl::Event cl_event;



	//Define Kernel Buffers
	//Output
	size_t ACsize = sizeof(uint64_t) * TempArraySize;
	cl::Buffer CL_CQ(Options.CL_context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, ACsize, TempArray, &err);
	//Input:
	size_t Intsize = sizeof(float) * Det.DetectorSize[0] * Det.DetectorSize[1];
	cl::Buffer CL_Intensity(Options.CL_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Intsize, Det.Intensity, &err);
	size_t KMapsize = sizeof(float) * 3 * Det.DetectorSize[0] * Det.DetectorSize[1];
	cl::Buffer CL_kMap(Options.CL_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, KMapsize, Det.kMap, &err);
	cl::Buffer CL_Params(Options.CL_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(Params), &Params, &err);

	//Setup Kernel
	cl::Kernel kernel(Options.CL_Program, "AutoCorr_CQ_AV", &err);
	Options.checkErr(err, "Setup AutoCorr_CQ in AC1D::Calculate_CQ() ");

	//Set Arguments
	kernel.setArg(0, CL_Intensity);
	kernel.setArg(1, CL_kMap);
	kernel.setArg(2, CL_Params);
	kernel.setArg(3, CL_CQ);
	const size_t &global_size = Det.DetectorSize[0] * Det.DetectorSize[1];

	//launch Kernel
	Options.Echo("Launch kernel ... \n");
	Profiler.Tic();
	err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(global_size), cl::NullRange, NULL, &cl_event);

	Options.checkErr(err, "Launch Kernel in AC1D::Calculate_CQ() ");
	cl_event.wait();
	Options.Echo("C(q)-AV kernel finished in");
	Profiler.Toc(Options.echo);

	err = queue.enqueueReadBuffer(CL_CQ, CL_TRUE, 0, ACsize, TempArray);
	Options.checkErr(err, "OpenCL kernel, launched in AC1D::Calculate_CQ() ");


	
	//Free Device
	Options.OCL_FreeDevice(OpenCLDeviceNumber);

	//convert to Double and Reduce
	delete[] CQ;
	CQ = new double[Shape.Size]();

	int j = 0;
	for (unsigned int i = 0; i < TempArraySize; i++)
	{
		if (j == Shape.Size)
			j = 0;

		CQ[j] += ((double)TempArray[i] / (double)Multiplicator);
		j++;
	}
	//Free memory
	delete[] TempArray;
}


std::mutex g_loadFile_mutex;
std::mutex g_echo_mutex;
//std::unordered_map<std::string, std::mutex> H5Mutex;
void Calculate_AC_UW_Mapped(Settings & Options,Detector & RefDet, double * AC_M,unsigned int LowerBound, unsigned int UpperBound, Settings::Interpolation IterpolMode, std::array<float, 2> Photonisation, float MaxQ, float dqdx, bool JungfrauDet)
{
	//g_echo_mutex.lock();
	//std::cout << "Thread from " << LowerBound <<" launched\n";
	//g_echo_mutex.unlock();

	Detector Det(RefDet);
	if (JungfrauDet)
	{
		Det.Intensity = new float[1];
	}
	else
	{
		Det.Intensity = new float[Det.DetectorSize[0]* Det.DetectorSize[1]];
	}

	for (unsigned int i = LowerBound; i < UpperBound; i++)
	{
		//
		if ((i- LowerBound) % 100 == 0 && i > LowerBound)
		{
			g_echo_mutex.lock();
			std::cout << "AC (uw)-Thread " << LowerBound << " - " << UpperBound << " : " << i- LowerBound << "/" << (UpperBound - LowerBound) << std::endl;
			g_echo_mutex.unlock();
		}


		//Load Intensity
		g_loadFile_mutex.lock();
		if (JungfrauDet)
		{
			Det.LoadIntensityData_PSANA_StyleJungfr(Options.HitEvents[i].Filename, Options.HitEvents[i].Dataset, Options.HitEvents[i].Event);
		}
		else
		{
			Det.LoadIntensityData_EPIX(Det.Intensity, Options.HitEvents[i].Filename, Options.HitEvents[i].Dataset, Options.HitEvents[i].Event);
		}
		g_loadFile_mutex.unlock();

		ArrayOperators::MultiplyElementwise(Det.Intensity, Det.PixelMask, Det.DetectorSize[0] * Det.DetectorSize[1]);

		//Create sparse List (and Photonize)
		Det.CreateSparseHitList(Photonisation[0], Photonisation[1],false); //suppress OMP parallelization within Threadpool!
		//iterate over all combinations
		for (unsigned int j = 0; j < Det.SparseHitList.size(); j++)
		{
			for (unsigned int k = j; k < Det.SparseHitList.size(); k++)
			{
				if (k == j)
					continue;

				float q = sqrt(
					(Det.SparseHitList[j][0] - Det.SparseHitList[k][0]) * (Det.SparseHitList[j][0] - Det.SparseHitList[k][0]) +
					(Det.SparseHitList[j][1] - Det.SparseHitList[k][1]) * (Det.SparseHitList[j][1] - Det.SparseHitList[k][1]) +
					(Det.SparseHitList[j][2] - Det.SparseHitList[k][2]) * (Det.SparseHitList[j][2] - Det.SparseHitList[k][2]) 
					);

				if (q > MaxQ)
				{
					continue;
				}
				q = q / dqdx;

				if (IterpolMode == Settings::Interpolation::NearestNeighbour)
				{
					unsigned int sc;
					sc = (unsigned int)(floor(q + 0.5));
					AC_M[sc] += 2.0f * (Det.SparseHitList[j][3] * Det.SparseHitList[k][3]); //factor 2 because of both orientations (k = j; ...)
				}
				else if (IterpolMode == Settings::Interpolation::Linear)
				{
					unsigned int sc1, sc2;
					sc1 = (unsigned int)(floor(q));
					sc2 = sc1 + 1;

					float Sep = q - sc1; //separator

					float Val1 = 2.0f * ((Det.SparseHitList[j][3] * Det.SparseHitList[k][3])*(1 - Sep));
					float Val2 = 2.0f * ((Det.SparseHitList[j][3] * Det.SparseHitList[k][3])*(Sep));

					AC_M[sc1] += Val1;
					AC_M[sc2] += Val2;
				}
			}
		}

	}

	//Free in thread distributed Det memory:
	delete[] Det.Intensity;


	g_echo_mutex.lock();
	std::cout << "AC (uw)-Thread " << LowerBound << " - " << UpperBound << " Finished." << std::endl;
	g_echo_mutex.unlock();
	

}

void AC1D::Calculate_AC_UW_MR(Settings & Options, Detector & RefDet, Settings::Interpolation IterpolMode, std::array<float,2> Photonisation, bool JungfrauDet)
{
	const int Threads = 200; //set higher than expectred threads because of waitingtimes for read from file

	if (Options.HitEvents.size() <= 0)
	{
		std::cerr << "ERROR: No entrys in HitEvents\n";
		std::cerr << "    -> in AC1D::Calculate_AC_M()\n";
		throw;
	}

	int WorkerSize = (int)Options.HitEvents.size() / (Threads - 1);
	std::vector<std::array<unsigned int,2>> WorkerBounds;
	for (unsigned int i = 0; i < Threads; i++)
	{
		std::array<unsigned int, 2> t;
		t[0] = i * WorkerSize;
		if (t[0] + WorkerSize < Options.HitEvents.size())
		{
			t[1] = t[0] + WorkerSize;
		}
		else
		{
			t[1] = Options.HitEvents.size();
		}
	//	std::cout << t[0] << " - " << t[1] << "\n";
		WorkerBounds.push_back(t);
	}

	std::vector<double *> AC_Map;
	for (int i = 0; i < Threads; i++)
	{
		AC_Map.push_back(new double[Shape.Size]());
	}


	//Run AutoCorrelations (Map)
	std::vector<std::thread> AC_Threads;
	for (int i = 0; i < Threads; i++)
	{
		AC_Threads.push_back(std::thread(Calculate_AC_UW_Mapped,std::ref(Options), std::ref(RefDet), std::ref(AC_Map[i]), WorkerBounds[i][0], WorkerBounds[i][1], IterpolMode, Photonisation, Shape.Max_Q, Shape.dq_per_Step, JungfrauDet));
	}

	for (int i = 0; i < Threads; i++)
	{
		AC_Threads[i].join();
	}
	//ProfileTime profiler;
	//std::cout << "launch Worker 0 testwise\n";
	//profiler.Tic();
	//Calculate_AC_UW_Mapped(std::ref(Options), std::ref(RefDet), std::ref(AC_Map[0]), WorkerBounds[0][0], WorkerBounds[0][1], IterpolMode, Photonisation, Shape.Max_Q, Shape.dq_per_Step);
	//profiler.Toc(true);
	//std::cout << "done\n";


	//create AC_UW
	delete[] AC_UW;
	AC_UW = new double[Shape.Size]();

	//Reduce
	for (int i = 0; i < Threads; i++)
	{
		ArrayOperators::ParAdd(AC_UW, AC_Map[i], Shape.Size);
	}

	//Free Memory
	for (int i = 0; i < AC_Map.size(); i++)
	{
		delete[] AC_Map[i];
	}
}

