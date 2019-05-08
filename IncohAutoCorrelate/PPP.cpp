#include "PPP.h"

#include <omp.h>
#include <hdf5.h>
#include "H5Cpp.h"


#include "ArrayOperators.h"

namespace PPP
{
	//ShortTerms:
	//	LAP = Largest Adjacent Pixel
	//	SM = SingleMolecule  (here Nanostar)
	//	PF = Photon Finder


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
			int DetSize = fs * ss;
			#pragma omp parallel for
			for (int ind = DetectorPanels[i_pan].FirstInd; ind < (DetectorPanels[i_pan].FirstInd + DetSize); ind++)
			{
				//ind is the index of Intensity and i the index within the Panel!
				int i = ind - DetectorPanels[i_pan].FirstInd; 

				if (FragmentedPhotons[ind] >= SeedThershold) //check if value is high enough to search for neighbours to match
				{
					float LAP = 0; //Value of Largest Adjacet Pixel

					if (i % fs != 0) // check for left boundary
					{
						if (FragmentedPhotons[ind - 1] > LAP)
							LAP = FragmentedPhotons[ind - 1];
					} 
					if ((i + 1)%fs != 0 && i < DetSize) // check for right boundary
					{
						if (FragmentedPhotons[ind + 1] > LAP)
							LAP = FragmentedPhotons[ind + 1];
					}
					if (i - fs > 0) // check for upper boundary
					{
						if (FragmentedPhotons[ind - fs] > LAP)
							LAP = FragmentedPhotons[ind - fs];
					}
					if (i + fs < DetSize) // check for lower boundary
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
		Create_LAPSettings LAPSettings;
		LAPSettings.ADU_perPhoton = ADU_perPhoton;
		LAPSettings.CombinedThershold = CombinedThershold;
		LAPSettings.DetectorPanels = DetectorPanels;
		LAPSettings.DetPanels_Num = DetectorPanels.size();
		LAPSettings.Output_Dataset = Dataset;
		LAPSettings.Output_Path = H5_Out;
		LAPSettings.Output_NewXML = XML_Out;
		LAPSettings.SeedThershold = SeedThershold;
		ProcessData_PF_LAP(Det, LAPSettings, XML_In);
	}

	void ProcessData_PF_LAP(Detector & Det, Create_LAPSettings LAPSettings, std::string XML_In, bool GainOnly)
	{
		std::string XML_Out = LAPSettings.Output_NewXML;
		std::string H5_Out = LAPSettings.Output_Path;
		std::string Dataset = LAPSettings.Output_Dataset;
		std::vector<DetectorPanel> DetectorPanels = LAPSettings.DetectorPanels;
	
		float ADU_perPhoton = LAPSettings.ADU_perPhoton;
		float SeedThershold = LAPSettings.SeedThershold;
		float CombinedThershold = LAPSettings.CombinedThershold;

		if (true)
		{
			if (GainOnly)
			{
				std::cout << "\nRun gain correction only\n";
			}
			else {
				std::cout << "\nLAP correction - Parameter:\n";
				std::cout << "ADU per photon: " << LAPSettings.ADU_perPhoton << "\n";
				std::cout << "Seed threshold: " << LAPSettings.SeedThershold << "\n";
				std::cout << "Combined threshold: " << LAPSettings.CombinedThershold << "\n";
				std::cout << "Number of detector panels: " << LAPSettings.DetPanels_Num << "\n";
				for (unsigned int i = 0; i < LAPSettings.DetectorPanels.size(); i++)
				{
					std::cout << "Panel " << i << ": Start index = " << LAPSettings.DetectorPanels[i].FirstInd << "; ss = " << LAPSettings.DetectorPanels[i].Scans[1] << "; fs = " << LAPSettings.DetectorPanels[i].Scans[0] << "\n";
				}
			}
		std::cout << std::endl;
		}

		unsigned FullDetSize = 0; 
		for (unsigned int j = 0; j < LAPSettings.DetPanels_Num; j++)
		{
			FullDetSize += DetectorPanels[j].Scans[0] * DetectorPanels[j].Scans[1];
		}

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
			Det.LoadIntensityData(&OptionsIn.HitEvents[i]);//Load Intensity
			Det.ApplyPixelMask();//Apply Pxelmask
			if (LAPSettings.GainMapPath != "") //Check if gain correction is intendet
			{
				GainCorrection(Det, LAPSettings.GainMapPath, LAPSettings.DatasetOffset, LAPSettings.DatasetGain, OptionsIn); //Perform Gain Correction
			}
			//Run LAP
			if (!GainOnly)
			{
				PhotonFinder_LargestAdjacentPixel(Det.Intensity, DetectorPanels, FullDetSize, ADU_perPhoton, SeedThershold, CombinedThershold);
			}
			//Create Event 
			Settings::HitEvent t_Event;

			t_Event.PhotonCount = ArrayOperators::Sum(Det.Intensity, Det.DetectorSize[0] * Det.DetectorSize[1]);
			t_Event.MeanIntensity = (float)t_Event.PhotonCount / (Det.DetectorSize[0] * Det.DetectorSize[1]);

			t_Event.Dataset = Dataset;
			t_Event.Event = i;
			t_Event.Filename = H5_Out;
			t_Event.SerialNumber = i;
			for (int j = 0; j < 9; j++)
				t_Event.RotMatrix[j] = OptionsIn.HitEvents[i].RotMatrix[j];
			t_Event.PhotonCount = (int)ArrayOperators::Sum(Det.Intensity, Det.DetectorSize[0] * Det.DetectorSize[1]);
			t_Event.MeanIntensity = (float)t_Event.PhotonCount / ((float)(Det.DetectorSize[0] * Det.DetectorSize[1]));

			OptionsOut.HitEvents.push_back(t_Event);
			//Write new Intensity to H5
			start[0] = i;
			dataspace.selectHyperslab(H5S_SELECT_SET, count, start, stride, block);
			dataset.write(Det.Intensity, H5::PredType::NATIVE_FLOAT, mspace, dataspace);

			//Count for status update
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
		std::cout << "New H5-File saved as \"" << LAPSettings.Output_Path << "\" with dataset \"" << LAPSettings.Output_Dataset << "\"\n";
		std::cout << "New XML-EventList saved as \"" << XML_Out << "\"" << std::endl;
	}

	void GainCorrection(Detector & Det, std::string GainCorr_Path, std::string Dataset_Offset, std::string Dataset_Gain, Settings & Options)
	{
		float * GM_Offset = new float[Det.DetectorSize[0] * Det.DetectorSize[1]]();
		float * GM_Gain = new float[Det.DetectorSize[0] * Det.DetectorSize[1]]();

		H5::H5File file(GainCorr_Path, H5F_ACC_RDONLY);

		H5::DataSet dataset_O = file.openDataSet(Dataset_Offset);

		
		//std::cout << dataset_O.getTypeClass() << "  " << H5T_IEEE_F32LE << "\n";
		if (dataset_O.getTypeClass() != H5T_FLOAT)
		{
			std::cerr << "ERROR: Gain-map (offset) is not stored as float array.\n";
			std::cerr << "     -> in PPP::GainCorrection()\n";
			throw;
		}

		H5::DataSpace DS_O = dataset_O.getSpace();
		if (DS_O.getSimpleExtentNdims() != 2) //check if shape is [nE][nx][ny] or [ny][nx][nE]  nE =^ Number of Slices(Events)
		{
			std::cerr << "ERROR: Gain-map (offset) data dimension is not 2, but " << DS_O.getSimpleExtentNdims() << " => shape is not (nx, ny)\n";
			std::cerr << "     -> in PPP::GainCorrection()\n";
			throw;
		}
		hsize_t dims[2];
		DS_O.getSimpleExtentDims(dims, NULL);

		if (dims[0] != Det.DetectorSize[0] || dims[1] != Det.DetectorSize[1])
		{
			std::cerr << "ERROR: Gain-map (offset) size does not match pixle-map size.\n";
			std::cerr << "     -> in PPP::GainCorrection()\n";
			throw;
		}

		//Get Subset 
		hsize_t offset[2], count[2], stride[2], block[2];
		hsize_t dimsm[2];

		offset[0] = 0;
		offset[1] = 0;

		count[0] = Det.DetectorSize[0];
		count[1] = Det.DetectorSize[1];

		block[0] = 1;
		block[1] = 1;

		stride[0] = 1;
		stride[1] = 1;

		dimsm[0] = Det.DetectorSize[0];
		dimsm[1] = Det.DetectorSize[1];

		//Load Data (offset)
		H5::DataSpace mspace_O(3, dimsm, NULL);
		DS_O.selectHyperslab(H5S_SELECT_SET, count, offset, stride, block);

		H5::PredType type = H5::PredType::NATIVE_FLOAT;
		dataset_O.read(GM_Offset, type, mspace_O, DS_O);
		//Close DS, dataset, mspace
		DS_O.close();
		dataset_O.close();
		mspace_O.close();

		//Load Gain
		H5::DataSet dataset_G = file.openDataSet(Dataset_Gain);


		if (dataset_G.getTypeClass() != H5T_FLOAT)
		{
			std::cerr << "ERROR: Gain-map (gain) is not stored as integer array.\n";
			std::cerr << "     -> in PPP::GainCorrection()\n";
			throw;
		}

		H5::DataSpace DS_G = dataset_G.getSpace();
		if (DS_G.getSimpleExtentNdims() != 2) //check if shape is [nE][nx][ny] or [ny][nx][nE]  nE =^ Number of Slices(Events)
		{
			std::cerr << "ERROR: Gain-map (gain) data dimension is not 2, but " << DS_G.getSimpleExtentNdims() << " => shape is not (nx, ny)\n";
			std::cerr << "     -> in PPP::GainCorrection()\n";
			throw;
		}

		DS_G.getSimpleExtentDims(dims, NULL);
		if (dims[0] != Det.DetectorSize[0] || dims[1] != Det.DetectorSize[1])
		{
			std::cerr << "ERROR: Gain-map (gain) size does not match pixle-map size.\n";
			std::cerr << "     -> in PPP::GainCorrection()\n";
			throw;
		}

		//Load Data (gain)
		H5::DataSpace mspace_G(3, dimsm, NULL);
		DS_G.selectHyperslab(H5S_SELECT_SET, count, offset, stride, block);

		dataset_G.read(GM_Gain, type, mspace_G, DS_G);
		//Close DS, dataset, mspace
		DS_G.close();
		dataset_G.close();
		mspace_G.close();
		file.close();


		//for (int i = 0; i < 7000; i++)
		//{
		//	std::cout << GM_Offset[i] << " : " << GM_Gain[i] << "\n";
		//}

		//Gain and Offset should be loaded by now -> start correcting

		ArrayOperators::MultiplyScalar(GM_Offset, -1.0, Det.DetectorSize[0] * Det.DetectorSize[1]); //Negate Offset
		ArrayOperators::ParAdd(Det.Intensity, GM_Offset, Det.DetectorSize[0] * Det.DetectorSize[1]); //subtract offset from Intensity
		
		//Devide by Gain
		#pragma omp parallel for
		for (unsigned int i = 0; i < Det.DetectorSize[0] * Det.DetectorSize[1]; i++)
		{
			if (Det.Intensity[i] >= 0 && GM_Gain[i] > 0 && Det.PixelMask[i] == 1)
			{
				Det.Intensity[i] = Det.Intensity[i] / GM_Gain[i];
				if (Det.Intensity[i] > 100)
					std::cout <<"Warning: over 100 Photones at "<< i<<" : " << Det.Intensity[i] << " (Gain: " << GM_Gain[i] << " ) " << "\n";
			}
			else
			{
				Det.Intensity[i] = 0;
			}
		}

		//Clean Up
		delete[] GM_Offset;
		delete[] GM_Gain;
	}

	void ProcessData_DarkFieldCorrection(Detector & Det, CreateDarkSettings DarkSettings, std::string XML_In, Settings & Options)
	{
		Settings Set_In;
		Settings Set_Out;
		Options.Echo("Load event-list");
		Set_In.LoadHitEventListFromFile(XML_In);

		Options.Echo("Load dark field");
		if (DarkSettings.Dark_Path == "" || DarkSettings.Dark_Dataset == "")
		{
			std::cerr << "ERROR: Invalid dark-field path and/or dataset. Check XML-config file.\n";
			std::cerr << "     -> in PPP::ProcessData_DarkFieldCorrection()\n";
			throw;
		}

		//load Dark Field
		float * Dark = new float[Det.DetectorSize[0] * Det.DetectorSize[1]]();
		ArrayOperators::MultiplyScalar(Dark, -1.0f, Det.DetectorSize[0] * Det.DetectorSize[1]); //invert Darkfield

		//H5 Darkfiled loading stuff
		{
			H5::H5File file(DarkSettings.Dark_Path, H5F_ACC_RDONLY);
			H5::DataSet dataset = file.openDataSet(DarkSettings.Dark_Dataset);
			if (dataset.getTypeClass() != H5T_FLOAT)
			{
				std::cerr << "ERROR: Darkfield is not stored as float array.\n";
				std::cerr << "     -> in PPP::ProcessData_DarkFieldCorrection()\n";
				throw;
			}

			H5::DataSpace DS = dataset.getSpace();
			if (DS.getSimpleExtentNdims() != 2) //check if shape is [nE][nx][ny] or [ny][nx][nE]  nE =^ Number of Slices(Events)
			{
				std::cerr << "ERROR: DarkField data dimension is not 2, but " << DS.getSimpleExtentNdims() << " => shape is not (nx, ny)\n";
				std::cerr << "     -> in PPP::ProcessData_DarkFieldCorrection()\n";
				throw;
			}


			hsize_t dims[2];
			DS.getSimpleExtentDims(dims, NULL);

			if (dims[0] != Det.DetectorSize[0] || dims[1] != Det.DetectorSize[1])
			{
				std::cerr << "ERROR: Dark field size does not match pixle-map size.\n";
				std::cerr << "     -> in PPP::ProcessData_DarkFieldCorrection()\n";
				throw;
			}

			//Get Subset 
			hsize_t offset[2], count[2], stride[2], block[2];
			hsize_t dimsm[2];

			offset[0] = 0;
			offset[1] = 0;

			count[0] = Det.DetectorSize[0];
			count[1] = Det.DetectorSize[1];

			block[0] = 1;
			block[1] = 1;

			stride[0] = 1;
			stride[1] = 1;

			dimsm[0] = Det.DetectorSize[0];
			dimsm[1] = Det.DetectorSize[1];

			//Load Data Darkfield
			H5::DataSpace mspace(3, dimsm, NULL);
			DS.selectHyperslab(H5S_SELECT_SET, count, offset, stride, block);

			H5::PredType type = H5::PredType::NATIVE_FLOAT;
			dataset.read(Dark, type, mspace, DS);
			//Close DS, dataset, mspace
			DS.close();
			dataset.close();
			mspace.close();
		}


	//Correct DarkField ...
		Options.Echo("Apply dark field correction ...");

		unsigned int RelStackSize = 0;
		if (DarkSettings.RestrictToDataSource)
		{
			for (unsigned int i = 0; i < Set_In.HitEvents.size(); i++)
			{
				if (DarkSettings.DataSource_Path == Set_In.HitEvents[i].Filename)
				{
					RelStackSize++;
				}
			}
		}
		else
		{
			RelStackSize = Set_In.HitEvents.size();
		}

		if (Options.echo && DarkSettings.RestrictToDataSource) std::cout <<  "Restriction to single data source!" << std::endl;
		if (Options.echo) std::cout << RelStackSize << " events to be corrected." << std::endl;

		//H5 Stuff
		H5::H5File file(DarkSettings.Output_Path, H5F_ACC_TRUNC);
		hsize_t dims[3];
		dims[0] = RelStackSize;
		dims[1] = Det.DetectorSize[0];
		dims[2] = Det.DetectorSize[1];
		H5::DataSpace dataspace(3, dims);

		H5::DataSet dataset = file.createDataSet(DarkSettings.Output_Dataset, H5::PredType::NATIVE_FLOAT, dataspace);

		hsize_t start[3] = { 0, 0, 0 };  // Start of hyperslab, offset
		hsize_t stride[3] = { 1, 1, 1 }; // Stride of hyperslab
		hsize_t count[3] = { 1, 1, 1 };  // Block count
		hsize_t block[3] = { 1, dims[1], dims[2] }; // Block sizes

		H5::DataSpace mspace(3, block);

		float CounterStep = ((float)RelStackSize) / 100.0f;
		float Counter = 0;

		//iterate over Events
		unsigned int EventCounter = 0;
		Set_Out.HitEvents.clear();
		for (unsigned int i = 0; i < Set_In.HitEvents.size(); i++)
		{
			//Check (if restrict to one data-source), datasource path and if it doesn't match, continue
			if (DarkSettings.RestrictToDataSource && (DarkSettings.DataSource_Path != Set_In.HitEvents[i].Filename))
				continue;
			//load intensity
			Det.LoadIntensityData(&Set_In.HitEvents[i]);
			
			//Subtract Dark field (already inverted, therefore add)
			ArrayOperators::ParAdd(Det.Intensity, Dark, Det.DetectorSize[0] * Det.DetectorSize[1]);
			
			//store in H5 File
			start[0] = EventCounter;
			dataspace.selectHyperslab(H5S_SELECT_SET, count, start, stride, block);
			dataset.write(Det.Intensity, H5::PredType::NATIVE_FLOAT, mspace, dataspace);

			//Create Event
			{
				Settings::HitEvent t_Event;
				t_Event = Set_In.HitEvents[i];

				t_Event.Dataset = DarkSettings.Output_Dataset;
				t_Event.Event = EventCounter;
				t_Event.Filename = DarkSettings.Output_Path;
				t_Event.HitsPixelRatio = Set_In.HitEvents[i].HitsPixelRatio;
				t_Event.MeanIntensity = Set_In.HitEvents[i].MeanIntensity;
				t_Event.PhotonCount = Set_In.HitEvents[i].PhotonCount;
				t_Event.SerialNumber = Set_In.HitEvents[i].SerialNumber;

				for (int r = 0; r < 9; r++)
					t_Event.RotMatrix[r] = Set_In.HitEvents[i].RotMatrix[r];

				Set_Out.HitEvents.push_back(t_Event);
			}
			if ((float)EventCounter >= Counter)
			{
				std::cout << EventCounter << "/" << RelStackSize << "  ^= " << Counter / CounterStep << "%\n";
				Counter += CounterStep;
			}
			EventCounter++;
		}

		mspace.close();
		dataspace.close();
		dataset.close();
		file.close();

		Set_Out.SafeHitEventListToFile(DarkSettings.Output_NewXML);

		//CleanUp Memory
		delete[] Dark;
	}





	
}