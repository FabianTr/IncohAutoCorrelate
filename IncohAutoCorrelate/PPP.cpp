#include "PPP.h"

#include <omp.h>
#include <hdf5.h>
#include "H5Cpp.h"


#include "ArrayOperators.h"

namespace PPP
{



	void PhotonFinder_LargestAdjacentPixel(float * Intensity, std::vector<DetectorPanel> DetectorPanels,int FullDetSize, float ADU_perPhoton, float SeedThershold, float CombinedThershold )
	{
		//This method combines two pixel for the photon detection.
		//Therefore the pixels with an ADU abouth the photon threshold are selected, and the "full photons segmented".
		//These "full photon ADUs" were substracted  


		float * FullPhotons = new float[FullDetSize]();
		float * FragmentedPhotons = new float[FullDetSize]();

		//find full photons
		#pragma omp parallel for
		for (int i = 0; i < FullDetSize; i++)
		{
			if (Intensity[i] <= 0) //ignore nevative ADUs
				continue;

			FullPhotons[i] = floorf(Intensity[i] / ADU_perPhoton); //full photon (integer value)
			FragmentedPhotons[i] = (Intensity[i] / ADU_perPhoton) - FullPhotons[i]; //remaining ADU (float value < 1)
		}



		for (int i_pan = 0; i_pan < DetectorPanels.size(); i_pan++)
		{
			int fs = DetectorPanels[i_pan].Scans[0];
			int ss = DetectorPanels[i_pan].Scans[1];
			#pragma omp parallel for
			for (int ind = DetectorPanels[i_pan].FirstInd; ind < (DetectorPanels[i_pan].FirstInd + DetectorPanels[i_pan].Size); ind++)
			{
				//ind is the index of Intensity and i the index within the Panel!
				int i = ind - DetectorPanels[i_pan].FirstInd; 

				if (FragmentedPhotons[ind] >= SeedThershold) //check if value is high enough to search for neighbours to match
				{
					float LAP = 0; //Value of Largest Adjacet Pixel

					if (i%fs != 0) // check for left boundary
					{
						if (FragmentedPhotons[ind - 1] > LAP)
							LAP = FragmentedPhotons[ind - 1];
					} 
					if ((i+1)%fs != 0 && i < DetectorPanels[i_pan].Size) // check for right boundary
					{
						if (FragmentedPhotons[ind + 1] > LAP)
							LAP = FragmentedPhotons[ind + 1];
					}
					if (i-fs > 0) // check for upper boundary
					{
						if (FragmentedPhotons[ind - fs] > LAP)
							LAP = FragmentedPhotons[ind - fs];
					}
					if (i + fs < DetectorPanels[i_pan].Size) // check for lower boundary
					{
						if (FragmentedPhotons[ind + fs] > LAP)
							LAP = FragmentedPhotons[ind + fs];
					}

					//check and merge results
					if (FragmentedPhotons[ind] + LAP >= CombinedThershold)
					{
						Intensity[ind] = FullPhotons[ind] + 1; // +1 Photon count by LAP
					}
					else
					{
						Intensity[ind] = FullPhotons[ind]; // no LAP addition, return number of full photons
					}
				}
				else
				{
					Intensity[ind] = FullPhotons[ind];
				}
			}
		}
		delete[] FullPhotons;
		delete[] FragmentedPhotons;
	}

	void ProcessData_PF_LAP(std::string XML_In, std::string XML_Out, std::string H5_Out,std::string Dataset, std::vector<DetectorPanel> DetectorPanels,Detector &Det, int FullDetSize, float ADU_perPhoton, float SeedThershold, float CombinedThershold)
	{
		Settings OptionsIn;

		OptionsIn.LoadHitEventListFromFile(XML_In);


		Settings OptionsOut(OptionsIn);
		OptionsOut.HitEvents.clear();
		OptionsOut.HitEvents.reserve(OptionsIn.HitEvents.size());

		H5::H5File file(H5_Out, H5F_ACC_TRUNC);

		hsize_t dims[3];
		dims[0] = OptionsIn.HitEvents.size();
		dims[1] = Det.DetectorSize[0];
		dims[2] = Det.DetectorSize[1];
		H5::DataSpace dataspace(3, dims);

		H5::DataSet dataset = file.createDataSet(Dataset, H5::PredType::NATIVE_FLOAT, dataspace);

		hsize_t start[3] = { 0, 0, 0 };  // Start of hyperslab, offset
		hsize_t stride[3] = { 1, 1, 1 }; // Stride of hyperslab
		hsize_t count[3] = { 1, 1, 1 };  // Block count
		hsize_t block[3] = { 1, dims[1], dims[2] }; // Block sizes
		
		H5::DataSpace mspace(3, block);

		float CounterStep = ((float)OptionsIn.HitEvents.size())/100.0f;
		float Counter = 0;
		
		for (unsigned int i = 0; i < OptionsIn.HitEvents.size(); i++)
		{
			Det.LoadIntensityData(&OptionsIn.HitEvents[i]);
			Det.ApplyPixelMask();

			PhotonFinder_LargestAdjacentPixel(Det.Intensity, DetectorPanels, FullDetSize, ADU_perPhoton, SeedThershold, CombinedThershold);
			
			Settings::HitEvent t_Event;
			t_Event.Dataset = Dataset;
			t_Event.Event = i;
			t_Event.Filename = H5_Out;
			t_Event.SerialNumber = i;
			for (int j = 0; j < 9; j++)
				t_Event.RotMatrix[j] = OptionsIn.HitEvents[i].RotMatrix[j];
			t_Event.PhotonCount = (int)ArrayOperators::Sum(Det.Intensity, Det.DetectorSize[0] * Det.DetectorSize[1]);
			t_Event.MeanIntensity = (float)t_Event.PhotonCount / ((float)(Det.DetectorSize[0] * Det.DetectorSize[1]));

			OptionsOut.HitEvents.push_back(t_Event);

			start[0] = i;
			dataspace.selectHyperslab(H5S_SELECT_SET, count, start, stride, block);
			dataset.write(Det.Intensity, H5::PredType::NATIVE_FLOAT, mspace, dataspace);

			if ((float)i >= Counter)
			{
				std::cout << i << "/" << OptionsIn.HitEvents.size() << "  ^= " << Counter/ CounterStep << "%\n";
				Counter += CounterStep;
			}

		}

		mspace.close();
		dataspace.close();
		dataset.close();
		file.close();


		//Save XML File
		OptionsOut.SafeHitEventListToFile(XML_Out);

	}

	void ProcessData_PF_LAP_SM(std::string XML_In, std::string XML_Out, std::string H5_Out, std::string Dataset, std::vector<DetectorPanel> DetectorPanels, Detector &Det, int FullDetSize, float ADU_perPhoton, float SeedThershold, float CombinedThershold, bool Jungfrau)
	{
		Settings OptionsIn;

		OptionsIn.LoadHitEventListFromFile(XML_In);


		Settings OptionsOut(OptionsIn);
		OptionsOut.HitEvents.clear();
		OptionsOut.HitEvents.reserve(OptionsIn.HitEvents.size());

		H5::H5File file(H5_Out, H5F_ACC_TRUNC);

		hsize_t dims[3];
		dims[0] = OptionsIn.HitEvents.size();
		dims[1] = Det.DetectorSize[0];
		dims[2] = Det.DetectorSize[1];
		H5::DataSpace dataspace(3, dims);

		H5::DataSet dataset = file.createDataSet(Dataset, H5::PredType::NATIVE_FLOAT, dataspace);

		hsize_t start[3] = { 0, 0, 0 };  // Start of hyperslab, offset
		hsize_t stride[3] = { 1, 1, 1 }; // Stride of hyperslab
		hsize_t count[3] = { 1, 1, 1 };  // Block count
		hsize_t block[3] = { 1, dims[1], dims[2] }; // Block sizes

		H5::DataSpace mspace(3, block);

		float CounterStep = ((float)OptionsIn.HitEvents.size()) / 100.0f;
		float Counter = 0;

		for (unsigned int i = 0; i < OptionsIn.HitEvents.size(); i++)
		{
			//Det.LoadIntensityData(&OptionsIn.HitEvents[i]);

			if (Jungfrau)
			{
				Det.LoadIntensityData_PSANA_StyleJungfr(OptionsIn.HitEvents[i].Filename, OptionsIn.HitEvents[i].Dataset, OptionsIn.HitEvents[i].Event);
			}
			else
			{
				delete[] Det.Intensity;
				Det.Intensity = new float[Det.DetectorSize[0] * Det.DetectorSize[1]];
				Det.LoadIntensityData_EPIX(Det.Intensity,OptionsIn.HitEvents[i].Filename, OptionsIn.HitEvents[i].Dataset, OptionsIn.HitEvents[i].Event);
			}

			Det.ApplyPixelMask();

			PhotonFinder_LargestAdjacentPixel(Det.Intensity, DetectorPanels, FullDetSize, ADU_perPhoton, SeedThershold, CombinedThershold);

			Settings::HitEvent t_Event;
			t_Event.Dataset = Dataset;
			t_Event.Event = i;
			t_Event.Filename = H5_Out;
			t_Event.SerialNumber = i;
			for (int j = 0; j < 9; j++)
				t_Event.RotMatrix[j] = OptionsIn.HitEvents[i].RotMatrix[j];
			t_Event.PhotonCount = (int)ArrayOperators::Sum(Det.Intensity, Det.DetectorSize[0] * Det.DetectorSize[1]);
			t_Event.MeanIntensity = (float)t_Event.PhotonCount / ((float)(Det.DetectorSize[0] * Det.DetectorSize[1]));

			OptionsOut.HitEvents.push_back(t_Event);


			start[0] = i;
			dataspace.selectHyperslab(H5S_SELECT_SET, count, start, stride, block);
			dataset.write(Det.Intensity, H5::PredType::NATIVE_FLOAT, mspace, dataspace);

			if ((float)i >= Counter)
			{
				std::cout << i << "/" << OptionsIn.HitEvents.size() << "  ^= " << Counter / CounterStep << "%\n";
				Counter += CounterStep;
			}

		}

		mspace.close();
		dataspace.close();
		dataset.close();
		file.close();


		//Save XML File
		OptionsOut.SafeHitEventListToFile(XML_Out);

	}





}