//#pragma once
#ifndef STATISTICS_H
#define STATISTICS_H

//#include "Detector.h"
#include "Settings.h"
#include "ArrayOperators.h"
#include "Detector.h"
#include "PPP.h"

#include <vector>
#include <string.h>
#include <Eigen/Core>
#include <Eigen/Dense>


namespace Statistics
{

	
	struct Create_PixelHistogramSettings
	{
		unsigned int Bins = 100;
		double SmalestValue = 0;
		double LargestValue = 100;

		bool Normalized = true;

		std::string OutputPath = "";
	};

	struct Create_SpeckleContrastSettings 
	{
		std::string CSVOutputPath = "";
	};

	struct Create_ChargeSharingSettings
	{
		std::vector<PPP::DetectorPanel> DetectorPanels;

		double Seed = 0.5; //minimum ADU to seed a hit
		double MaxADU = 1.5; //maximum ADU to prevent two photon hits
		unsigned int IsolationRadius = 2; //minimum distance in pixel towards the next hit
		unsigned int PixelMaskRadius = 1; //minimum distance to bad pixel

		bool RestrictToBoundaries = false;
		size_t LowerLimit = 0;
		size_t UpperLimit = 0;
	
		std::string OutputBinaryPath = "";
	};

	struct StatisticsSettings
	{
		Statistics::Create_PixelHistogramSettings PixelHistogramSettings;
		Statistics::Create_SpeckleContrastSettings SpeckleContrastSettings;
		Statistics::Create_ChargeSharingSettings ChargeSharingSettings;
	};

	void Get_OrientationSphere(float*& Vectors, std::vector<Settings::HitEvent> EventList);


	struct PatternStackStatistics
	{
		double PhotonCount_Mean = 0;
		double PhotonCount_StAbw = 0;
	};

	class Histogram
	{
	public:
		Histogram(unsigned int Size, double BinSize = 1.0, double FirstBin = 0);

		unsigned int Size = 0;
		double FirstBin = 0;
		double BinSize = 1;
		std::vector<unsigned long> HistogramContent;
		unsigned long UnderflowBin = 0;
		unsigned long OverflowBin = 0;

		unsigned long long Entries = 0;

		void AddValue(double Value);
		void CalcEntries();
		void SafeToFile(std::string Filename);
	};



	Histogram Make_AllPixel_Histogram(Settings & Options, Detector &RefDet, unsigned int Bins, double SmallestVal, double HighestVal);
	std::vector<Histogram> MakePixelHistogramStack(Settings & Options, Detector &RefDet, unsigned int Bins, double SmallestVal, double HighestVal);

	void CreateAndSaveAllPixelHistograms(Create_PixelHistogramSettings HistSettings, Detector &RefDet, Settings & Options);
	

 
	struct SpeckleContrastStatistics
	{
		struct SCC_Pattern
		{
			std::vector<double> Probability;
			double MeanPhotonDensity = 0.0;
			double VariancePhotonDensity = 0.0;
		};
		std::vector<SCC_Pattern> SCC_Statistics;
		unsigned int Nmax = 1;

		unsigned int NumberOfUnmaskedPixels = 0;
	};
	SpeckleContrastStatistics GetSpeckleContrastStatistics(Settings & Options, Detector &RefDet, unsigned int LowerBound, unsigned int UpperBound, float Offset = 0.0f, float Step = 1.0f);


	void GetChargeSharingByIsolatedPhotonHits(Settings & Options, Detector & RefDet, Create_ChargeSharingSettings CS_Settings);

	namespace detail
	{

		//Target:: f(std::pair<int,int> XY, std::array<double, 3> Parameter, std::array<double, 3> &ParGrad)
		template <typename Target, typename KeyType, typename Real>
		std::vector<Real> GaussNewton(const Target &f, const std::vector<std::pair<KeyType, Real>> &data, std::vector<Real> A0, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> &Cov, std::pair<unsigned int, Real> Limit = {9999,1.0E-15})
		{
			int DataSize = (int)data.size();
			int ParameterSize = (int)A0.size();

			Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> J(DataSize, ParameterSize); //rows, cols //Jacobi matrix
			Eigen::Matrix<Real, Eigen::Dynamic, 1> F_new(DataSize, 1);

			double LastF;
			for (unsigned int i_step = 0; i_step < Limit.first; i_step++)
			{		
				Eigen::Matrix<Real, Eigen::Dynamic, 1> A = Eigen::Map<Eigen::Matrix<Real, Eigen::Dynamic, 1>>(A0.data(), ParameterSize,1);

				//get Jacobi matrix
				Eigen::Matrix<Real, Eigen::Dynamic, 1> F(DataSize, 1);
				for (int r = 0; r < DataSize; r++)
				{
					std::vector<Real> DerivV(ParameterSize);
					Real d = f(data[r].first, A0, DerivV);
					F(r) = (d - data[r].second);
					for (int c = 0; c < ParameterSize; c++)
					{
						Real jj = DerivV[c];
						J(r, c) = jj;	
					}
				}
				
				double alpha = 1.0;
				
				for (size_t i = 0; i < DataSize; i++)
					F_new(i) = std::numeric_limits<Real>::infinity();

				//optimization step
				Eigen::Matrix<Real, Eigen::Dynamic, 1> dA = ((J.transpose()*J).inverse() * J.transpose()) * F;

				if (isnan(dA.sum()))
					break;

				Eigen::Matrix<Real, Eigen::Dynamic, 1> A_new(DataSize);
				while ((F.norm() < F_new.norm()) && (alpha > 0))
				{
					A_new = A - alpha*dA;
					std::vector<Real> DerivV(ParameterSize);

					for (size_t j = 0; j < ParameterSize; j++)
						A0[j] = A_new(j);

					for (size_t j = 0; j < DataSize; j++)
						F_new(j) = (f(data[j].first, A0, DerivV) - data[j].second);

					alpha *= std::exp(-1.0);
				}

				if ( std::abs( F.norm() - F_new.norm()) <= Limit.second)
					break;

				LastF = F_new.norm();
			}

			//calculate Covariance 
			Cov = (J.transpose()*J).inverse() * (F_new.squaredNorm()/(DataSize-ParameterSize));
			return A0;
		}
		

	}
}
#endif