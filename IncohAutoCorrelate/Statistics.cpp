#include "Statistics.h"
#include <omp.h>

#include "ArrayOperators.h"
#include "ProfileTime.h"


namespace Statistics
{
	void Get_OrientationSphere(float *& Vectors, std::vector<Settings::HitEvent> EventList)
	{
		unsigned int Size = EventList.size();
		Vectors = new float[3 * Size]();


		for (unsigned int i = 0; i < Size; i++)
		{
			float* t_Vec = new float[3]();
			t_Vec[0] = 1;
			t_Vec[1] = 0;
			t_Vec[2] = 0;


			ArrayOperators::Rotate(t_Vec, EventList[i].RotMatrix);


			Vectors[3 * i + 0] = t_Vec[0];
			Vectors[3 * i + 1] = t_Vec[1];
			Vectors[3 * i + 2] = t_Vec[2];
		}


	}
	//
	Histogram Make_AllPixel_Histogram(Settings & Options, Detector & RefDet, unsigned int Bins, double SmallestVal, double HighestVal)
	{
		Histogram Hist(Bins, (HighestVal - SmallestVal) / Bins, SmallestVal);
		Detector Det(RefDet, true);

		ProfileTime profiler;
		profiler.Tic();

		double CounterStep = Options.HitEvents.size() / 100.0;
		int Prog = 0;

		for (unsigned int i = 0; i < Options.HitEvents.size(); i++)
		{
			Det.LoadIntensityData(&Options.HitEvents[i]);
			if (Det.Checklist.PixelMask)
				Det.ApplyPixelMask();

			for (unsigned int j = 0; j < Det.DetectorSize[0]* Det.DetectorSize[1]; j++)
			{
				Hist.AddValue(Det.Intensity[j]);
			}
			if (Options.echo)
			{
				if (i / CounterStep > Prog)
				{
					std::cout << "Pattern " << i << " / " << Options.HitEvents.size() << " \t^= " << Prog << "%\n";
					profiler.Toc(true);
					Prog++;
				}
			}
		}
		profiler.Toc(Options.echo);
		return Hist;
	}

	std::vector<Histogram> MakePixelHistogramStack(Settings & Options, Detector & RefDet, unsigned int Bins, double SmallestVal, double HighestVal)
	{
		std::vector<Histogram> HistStack;
		HistStack.reserve(RefDet.DetectorSize[0] * RefDet.DetectorSize[1]);
		//create empty Histograms:
		for (unsigned int i = 0; i < RefDet.DetectorSize[0]*RefDet.DetectorSize[1]; i++)
		{
			Histogram Hist(Bins, (HighestVal - SmallestVal) / Bins, SmallestVal);
			HistStack.push_back(Hist);
		}
		Detector Det(RefDet, true);

		ProfileTime profiler;
		profiler.Tic();

		double CounterStep = Options.HitEvents.size() / 100.0;
		int Prog = 0;

		for (unsigned int i = 0; i < Options.HitEvents.size(); i++)
		{
			Det.LoadIntensityData(&Options.HitEvents[i]);
			if (Det.Checklist.PixelMask)
				Det.ApplyPixelMask();

			//Do NOT parallelize this!!! 
			for (unsigned int j = 0; j < Det.DetectorSize[0] * Det.DetectorSize[1]; j++)
			{
				HistStack[j].AddValue(Det.Intensity[j]);
			}

			if (Options.echo)
			{
				if (i / CounterStep > Prog)
				{
					std::cout << "Pattern " << i << " / " << Options.HitEvents.size() << " \t^= " << Prog << "%"<<std::endl;
					profiler.Toc(true);
					Prog++;
				}
			}

		}
		profiler.Toc(Options.echo);

		return HistStack;
	}

	void CreateAndSaveAllPixelHistograms(Create_PixelHistogramSettings HistSettings, Detector & RefDet, Settings & Options)
	{
		std::vector<Statistics::Histogram> HistStack = Statistics::MakePixelHistogramStack(Options, RefDet, HistSettings.Bins, HistSettings.SmalestValue, HistSettings.LargestValue); //Main Work happens here:
		
		double * FinalHistStack = new double[HistSettings.Bins * RefDet.DetectorSize[0] * RefDet.DetectorSize[1]]();
	
		for (unsigned int i = 0; i < RefDet.DetectorSize[0] * RefDet.DetectorSize[1]; i++)
		{
			unsigned long AllBinCount = 0;
			for (unsigned int j = 0; j < HistSettings.Bins; j++) //sum up all Bins (without over and underflow)
			{
				AllBinCount += HistStack[i].HistogramContent[j];
			}
			if (AllBinCount > 0)
			{
				for (unsigned int j = 0; j < HistSettings.Bins; j++) //normalize
				{
					if (HistSettings.Normalized)
					{
						FinalHistStack[HistSettings.Bins*i + j] = ((double)HistStack[i].HistogramContent[j] / ((double)AllBinCount));
					}
					else
					{
						FinalHistStack[HistSettings.Bins*i + j] = (double)HistStack[i].HistogramContent[j];
					}
				}
			}
		}
		ArrayOperators::SafeArrayToFile(HistSettings.OutputPath, FinalHistStack, HistSettings.Bins * RefDet.DetectorSize[0] * RefDet.DetectorSize[1], ArrayOperators::Binary);

		delete[] FinalHistStack;
	}
	//
	SpeckleContrastStatistics GetSpeckleContrastStatistics(Settings & Options, Detector & RefDet, unsigned int LowerBound, unsigned int UpperBound, float Offset, float Step)
	{
		SpeckleContrastStatistics SCS;

		//Get number of masked pixel
		SCS.NumberOfUnmaskedPixels = 0;
		for (unsigned int i = 0; i < RefDet.DetectorSize[0] * RefDet.DetectorSize[1]; i++)
		{
			SCS.NumberOfUnmaskedPixels += RefDet.PixelMask[i];
		}

		//Preallocate outer vector
		SCS.SCC_Statistics.resize(UpperBound - LowerBound);

		//Loop through patterns --- can be parallized therefore clone det, ...
		unsigned int counter = 0.0;
		const unsigned int modulo = (UpperBound - LowerBound + 50 )/100;
		#pragma omp parallel for
		for (unsigned int i = 0; i < UpperBound - LowerBound; i++)
		{
			//Load Intensity
			Detector Det(RefDet, true);
			#pragma omp critical
			{
				Det.LoadIntensityData(&Options.HitEvents[i + LowerBound]);
			}
			Det.ApplyPixelMask();
			//Loop through Pixel
			ArrayOperators::DiscretizeToPhotons(Det.Intensity, Offset, Step, Det.DetectorSize[0] * Det.DetectorSize[1]);
			for (unsigned int j = 0; j < Det.DetectorSize[0] * Det.DetectorSize[1]; j++)
			{
				unsigned int Photons = Det.Intensity[j];
				if (Photons == 0)
					continue;
				//Check if propability-vector is large enought and expand if needed
				while (Photons > SCS.SCC_Statistics[i].Probability.size())
				{
					SCS.SCC_Statistics[i].Probability.push_back(0);
					if (Photons > SCS.Nmax)
						SCS.Nmax = Photons; // store maximum of photons found per pixel
				}
				//Adds entry to photon-count histogram
				SCS.SCC_Statistics[i].Probability[Photons - 1] += 1.0/((double)SCS.NumberOfUnmaskedPixels);
				//updates mean number of photons
				SCS.SCC_Statistics[i].MeanPhotonDensity += (double)Photons / ((double)SCS.NumberOfUnmaskedPixels);
			}
			//Calculate Variance
			double Var = 0.0;
			for (unsigned int j = 0; j < Det.DetectorSize[0] * Det.DetectorSize[1]; j++)
			{
				Var += ((double)Det.Intensity[j] - SCS.SCC_Statistics[i].MeanPhotonDensity)*((double)Det.Intensity[j] - SCS.SCC_Statistics[i].MeanPhotonDensity);		
			}
			SCS.SCC_Statistics[i].VariancePhotonDensity = Var / ((double)SCS.NumberOfUnmaskedPixels);

			#pragma omp critical
			{
				if ((++counter) % modulo == 0) {
					std::cout << counter/modulo << "%" << std::endl;
				}
			}
		}

		return SCS;
	}
	//

	void GetChargeSharingByIsolatedPhotonHits(Settings & Options, Detector & RefDet, Create_ChargeSharingSettings CS_Settings)
	{
		std::cout << "Start charge sharing fits on isolated photons ..." << std::endl;

		//function to fit
		class Gauss3x3FitTarget
		{
		public:
			bool UseConstraints = false;
			double MaxAbsXY = 0.5;
			double MaxSigma = 0.25;

			double operator()(const std::pair<int, int> XY, const std::vector<double> Parameter, std::vector<double> &ParGrad) const
			{
				//Parameter(3): [x0, y0, sigma]
				const double x0 = Parameter[0];
				const double y0 = Parameter[1];
				const double s = Parameter[2];

				//check sigma = 0:
				if (s == 0)
				{
					ParGrad[0] = 0.0;
					ParGrad[1] = 0.0;
					ParGrad[2] = 0.0;
					return 0.0;
				}

				// XY Integration limits
				const double Xbot = (double)XY.first - 1.5;
				const double Xtop = (double)XY.first - 0.5;
				const double Ybot = (double)XY.second - 1.5;
				const double Ytop = (double)XY.second - 0.5;

				//construction kit
				const double ErfX = std::erf((Xbot - x0) / (std::sqrt(2)*s)) - std::erf((Xtop - x0) / (std::sqrt(2)*s));
				const double ErfY = std::erf((Ybot - y0) / (std::sqrt(2)*s)) - std::erf((Ytop - y0) / (std::sqrt(2)*s));

				const double LogExpXb = -std::pow((x0 - Xbot), 2) / (2.0 * s*s);
				const double ExpXb = LogExpXb < -700 ? 0 : std::exp(LogExpXb);

				const double LogExpXt = -std::pow((x0 - Xtop), 2) / (2.0 * s*s);
				const double ExpXt = LogExpXt < -700 ? 0 : std::exp(LogExpXt);

				const double LogExpYb = -std::pow((y0 - Ybot), 2) / (2.0 * s*s);
				double ExpYb = LogExpYb < -700 ? 0 : std::exp(LogExpYb);

				const double LogExpYt = -std::pow((y0 - Ytop), 2) / (2.0 * s*s);
				double ExpYt = LogExpYt < -700 ? 0 : std::exp(LogExpYt);


				//function value
				const double ret = (1.0 / 4.0) * (ErfX * ErfY);


				// d/dx0
				ParGrad[0] = (1.0 / (2.0 * s * std::sqrt(2.0 * M_PI))) * (ExpXt - ExpXb)*ErfY;
				// d/dy0
				ParGrad[1] = (1.0 / (2.0 * s * std::sqrt(2.0 * M_PI))) * (ExpYt - ExpYb)*ErfX;
				// d/ds
				ParGrad[2] = (1.0 / (2.0 * s * s * std::sqrt(2.0 * M_PI))) *   (
					(ExpYb * (y0 - Ybot) + ExpYt * (Ytop - y0)) * ErfX +
					(ExpXb * (x0 - Xbot) + ExpXt * (Xtop - x0)) * ErfY);

				//if (isnan(ret) || isnan(ParGrad[0]) || isnan(ParGrad[1]) || isnan(ParGrad[2]))
				//{
				//	std::cout << "x0 = " << x0 << ";\t y0 = " << y0 << "\t s = " << s << std::endl;

				//	int wait;
				//	std::cin >> wait;
				//}
				return ret;
			}
		};

		struct FittedPhoton{
			float Sigma = 0.0;
			float X0 = 0.0;
			float Y0 = 0.0;

			float integratedADUs = 0;

			std::array<std::array<float, 3>, 3> CovMat;

			std::array<float, 9> Image;
		};

		Gauss3x3FitTarget Gauss; 

		std::vector<FittedPhoton> FittedPhotons;

		if (!CS_Settings.RestrictToBoundaries)
		{
			CS_Settings.LowerLimit = 0;
			CS_Settings.UpperLimit = Options.HitEvents.size();
		}

		ProfileTime Profiler;
		long StackSize = CS_Settings.UpperLimit - CS_Settings.LowerLimit;
		long Progress = 0;
		int t_prog = -1;
		Profiler.Tic();

		#pragma omp parallel for 
		for (size_t idx = CS_Settings.LowerLimit; idx < CS_Settings.UpperLimit; idx++)//loop over all Events
		{
			#pragma omp critical
			{
				if (((Progress * 100) / (StackSize - 1)) > t_prog)
				{
					std::cout << "Running CSF: " << ((Progress) * 100) / (StackSize - 1) << "% \t";
					Profiler.Toc(true, true);
					t_prog++;
				}
				Progress++;
			}

			Detector Det = Detector(RefDet, true);
			Det.LoadIntensityData(&Options.HitEvents[idx]);

			std::vector<FittedPhoton> FittedPhotons_Map;

			for (unsigned int i_pan = 0; i_pan < CS_Settings.DetectorPanels.size(); i_pan++)//loop over all Panels
			{
				int FS = CS_Settings.DetectorPanels[i_pan].Scans[0]; //size in fs-direction
				int SS = CS_Settings.DetectorPanels[i_pan].Scans[1]; //size in ss-direction
				int PanSize = FS * SS; //Size of current Panel
				int FrameInd = CS_Settings.PixelMaskRadius*(1 + FS); //Pixels at the edges to be ignored due to Pixel Mask Radius (PMR)

				for (int ind = CS_Settings.DetectorPanels[i_pan].FirstInd + FrameInd; ind < (CS_Settings.DetectorPanels[i_pan].FirstInd + PanSize - FrameInd); ind++) //loop over all Pixel in Panel without upper left frame of PMR and bottom frame of PMR
				{
					bool IsolatedHit = true;
					if (Det.Intensity[ind] < CS_Settings.Seed) //if value is smaller than seed threshold skip next steps 
					{
						IsolatedHit = false;
						continue;
					}

					//std::cout << ind << ":"<< Det.Intensity[ind] << " > " << CS_Settings.Seed << std::endl;

					//ind is the index of Intensity and i the index within the Panel!
					int i = ind - CS_Settings.DetectorPanels[i_pan].FirstInd;

					int fs = i % FS;
					int ss = i / FS;

					if (fs >= FS - CS_Settings.PixelMaskRadius) //check for right boarder
						continue;

					//Check PMR- criterium
					
					for (int sss = -(int)CS_Settings.PixelMaskRadius; sss <= (int)CS_Settings.PixelMaskRadius; sss++)
					{
						for (int fff = -(int)CS_Settings.PixelMaskRadius; fff <= (int)CS_Settings.PixelMaskRadius; fff++)
						{
							int t_fs = fs + fff;
							int t_ss = ss + sss;
							if (t_fs < 0 || t_fs > FS || t_ss < 0 || t_ss > SS) //check if inside boundaries
							{
								IsolatedHit = false;
								break;
							}

							int t_ind = t_fs + t_ss * FS + CS_Settings.DetectorPanels[i_pan].FirstInd;

							if (Det.PixelMask[t_ind] == 0)
							{
								IsolatedHit = false;
								break;
							}
						}
					}
					if (!IsolatedHit)
						continue;

					

					//Check SingleHit- criterium
					for (int sss = -(int)CS_Settings.IsolationRadius; sss <= (int)CS_Settings.IsolationRadius; sss++)
					{
						if (!IsolatedHit)
							continue;
						for (int fff = -(int)CS_Settings.IsolationRadius; fff <= (int)CS_Settings.IsolationRadius; fff++)
						{
							//ignore current pixel
							if (sss == 0 && fff == 0)
								continue;

							int t_fs = fs + fff;
							int t_ss = ss + sss;
							if (t_fs < 0 || t_fs > FS || t_ss < 0 || t_ss > SS) //check if inside boundaries
							{
								IsolatedHit = false;
								break;
							}

							int t_ind = t_fs + t_ss * FS + CS_Settings.DetectorPanels[i_pan].FirstInd;

							if (Det.Intensity[t_ind] >= CS_Settings.Seed) //check if hit is the only one within 'IsolationRadius'
							{
								IsolatedHit = false;
								break;
							}
								
						}
					}
					if (!IsolatedHit)
						continue;
					
					// Here we have an isolated hit.
					FittedPhoton FPhoton;

					//Load hit with adjugate pixels in 3x3 matrix
					std::vector<std::pair<std::pair<int, int>, double>> IsoHit(9);
					int j = 0;
					double IntADU = 0.0;
					for (int Y = 0; Y < 3; Y++)
					{
						for (int X = 0; X < 3; X++)
						{
							int t_fs = fs + X - 1;
							int t_ss = ss + Y - 1;
							int t_ind = t_fs + t_ss * FS + CS_Settings.DetectorPanels[i_pan].FirstInd;


							//std::cout << t_fs << ";" << t_ss << ":" << t_ind << "\t";
							////std::cout << t_ind <<":"<< Det.Intensity[t_ind] << "\t";

							IsoHit[j] = { {X,Y}, Det.Intensity[t_ind] };
							FPhoton.Image[j] = Det.Intensity[t_ind];
							IntADU += Det.Intensity[t_ind];
							FPhoton.integratedADUs = IntADU;
							j++;
						}
						//std::cout << std::endl;
					}

					//std::cout << std::endl;


					if (IntADU <= CS_Settings.Seed)
						continue;
					//sort out two photon hits
					if (IntADU >= CS_Settings.MaxADU)
						continue;
					//normalize to one
					for (int i = 0; i < 9; i++)
						IsoHit[i].second = IsoHit[i].second / IntADU;


					// Fit symmetric 2D gaussian
					std::vector<double> StartParams(3);
					StartParams[0] = 0;
					StartParams[1] = 0;
					StartParams[2] = 0.5;

					std::vector<double> FittedParams(3);
					Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Cov;
					FittedParams = detail::GaussNewton(Gauss, IsoHit, StartParams, Cov);
					
					
					FPhoton.X0 = (float)FittedParams[0];
					FPhoton.Y0 = (float)FittedParams[1];
					FPhoton.Sigma = std::abs( (float)FittedParams[2]);
					for (int i = 0; i < 3; i++)
						for (int j = 0; j < 3; j++)
							FPhoton.CovMat[i][j] = Cov(i, j);
					FittedPhotons_Map.push_back(FPhoton); //add fitted photon (mapped, therefore no need for critical)

					////DEBUG
					//std::cout << IsoHit[0].second << " \t" << IsoHit[1].second << " \t" << IsoHit[2].second << "\n";
					//std::cout << IsoHit[3].second << " \t" << IsoHit[4].second << " \t" << IsoHit[5].second << "\n";
					//std::cout << IsoHit[6].second << " \t" << IsoHit[7].second << " \t" << IsoHit[8].second << "\n";
					//std::cout << "X0 = " << FPhoton.X0 << "; \tY0 = " << FPhoton.Y0 << "; \tS = " << FPhoton.Sigma << std::endl;
					//std::cout << ind << std::endl;

					////\DEBUG
				}
			}

			//reduce
			#pragma omp critical
			{
				FittedPhotons.insert(FittedPhotons.end(),FittedPhotons_Map.begin(), FittedPhotons_Map.end());
			}
		}


		//save results to file

		// Structure: 4 + 9 + 9 = 22 (* float)
		float* DataOut = new float[FittedPhotons.size()*22];

		for (size_t i = 0; i < FittedPhotons.size(); i++)
		{
			DataOut[22 * i + 0] = FittedPhotons[i].X0;
			DataOut[22 * i + 1] = FittedPhotons[i].Y0;
			DataOut[22 * i + 2] = FittedPhotons[i].Sigma;
			DataOut[22 * i + 3] = FittedPhotons[i].integratedADUs;

			DataOut[22 * i +  4] = FittedPhotons[i].CovMat[0][0];
			DataOut[22 * i +  5] = FittedPhotons[i].CovMat[0][1];
			DataOut[22 * i +  6] = FittedPhotons[i].CovMat[0][2];
			DataOut[22 * i +  7] = FittedPhotons[i].CovMat[1][0];
			DataOut[22 * i +  8] = FittedPhotons[i].CovMat[1][1];
			DataOut[22 * i +  9] = FittedPhotons[i].CovMat[1][2];
			DataOut[22 * i + 10] = FittedPhotons[i].CovMat[2][0];
			DataOut[22 * i + 11] = FittedPhotons[i].CovMat[2][1];
			DataOut[22 * i + 12] = FittedPhotons[i].CovMat[2][2];

			for (int jj = 0; jj < 9; jj++)
				DataOut[22 * i + 13 + jj] = FittedPhotons[i].Image[jj];
		}
		ArrayOperators::SafeArrayToFile(CS_Settings.OutputBinaryPath,DataOut, FittedPhotons.size() * 22, ArrayOperators::Binary);

		std::cout << "\n" << FittedPhotons.size() << " isolated photons found.\n" << std::endl;
		std::cout << "Results saved as " << FittedPhotons.size() << " * 12 float32-array.\n";
		std::cout << "X0; Y0; Sigma; integrated ADUs; CovarianceMatrix as: C00, C01, C02, C10, C11, C12, C20, C21, C22; image of photon hit (3x3) \n" << std::endl;
		std::cout << "Results saved as: \"" << CS_Settings.OutputBinaryPath << "\"" << std::endl;


		delete [] DataOut;
	}


	// ***************************
	Histogram::Histogram(unsigned int size, double binSize, double firstBin)
	{
		Size = size;
		BinSize = binSize;
		FirstBin = firstBin;

		HistogramContent.clear();
		HistogramContent.resize(Size, 0);

		Entries = 0;
	}
	void Histogram::AddValue(double Value)
	{
		//#pragma omp critical
		Entries ++;
		Value = Value - FirstBin;
		if (Value < 0)
		{
			//#pragma omp critical
			UnderflowBin++;
			return;
		}
		unsigned int ind = (unsigned int)floor((Value / BinSize) + 0.5);
		if (ind >= Size)
		{
			//#pragma omp critical
			OverflowBin++;
			return;
		}
		#pragma omp critical
		HistogramContent[ind] ++;
	}
	void Histogram::CalcEntries()
	{
		Entries = 0;
		for (size_t i = 0; i < HistogramContent.size(); i++)
		{
			Entries += HistogramContent[i];
		} 
		Entries += (UnderflowBin + UnderflowBin);
	}

	void Histogram::SafeToFile(std::string Filename)
	{
		double * Hist = new double[Size + 2]();

		Hist[0]     = (double)UnderflowBin;
		Hist[Size + 1] = (double)OverflowBin;

		for (unsigned int i = 0; i < HistogramContent.size(); i++)
		{
			Hist[i+1] = (double)HistogramContent[i];
		}
		
		ArrayOperators::SafeArrayToFile(Filename, Hist, Size + 2, ArrayOperators::Binary);

		delete[] Hist;
	}

}

