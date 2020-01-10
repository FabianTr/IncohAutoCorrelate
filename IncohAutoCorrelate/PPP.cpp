#include "PPP.h"

#include <omp.h>
#include <hdf5.h>
#include "H5Cpp.h"


#include "ArrayOperators.h"
#include "ProfileTime.h"

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



		for (unsigned int i_pan = 0; i_pan < DetectorPanels.size(); i_pan++)
		{
			int fs = DetectorPanels[i_pan].Scans[0];
			int ss = DetectorPanels[i_pan].Scans[1];
			int DetSize = fs * ss;
			#pragma omp parallel for
			for (unsigned int ind = DetectorPanels[i_pan].FirstInd; ind < (DetectorPanels[i_pan].FirstInd + DetSize); ind++)
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


	namespace
	{
		const float pi = 3.141592653589793;

		template<typename T0, typename T1, typename T2>
		constexpr T0 clip(const T0& n, const T1& l, const T2& h) 
		{
			return n<l ? l : n>h ? h : n;
		}

		//gradient of gaussion in fs direction
		const double inline pixel_gauss_dfs(const float& fs, const float& ss,	const float& s	)
		{
			const float a = erf((fs + 1) / sqrt(2 * s * s));
			const float b = erf((fs) / sqrt(2 * s * s));
			const float c = erf((ss + 1) / sqrt(2 * s * s));
			const float d = erf((ss) / sqrt(2 * s * s));
			return 0.5 * (c - d) * (exp(-0.5 * (fs + 1) * (fs + 1) / (s * s)) - exp(-0.5 * (fs) * (fs) / (s * s))) / sqrt(2 * pi * s * s);
		}

		//gradient of gaussion in ss direction
		const double inline pixel_gauss_dss(const float& fs, const float& ss,	const float& s)
		{
			const float a = erf((fs + 1) / sqrt(2 * s * s));
			const float b = erf((fs) / sqrt(2 * s * s));
			const float c = erf((ss + 1) / sqrt(2 * s * s));
			const float d = erf((ss) / sqrt(2 * s * s));
			return 0.5 * (a - b) * (exp(-0.5 * (ss + 1) * (ss + 1) / (s * s)) - exp(-0.5 * (ss) * (ss) / (s * s))) / sqrt(2 * pi * s * s);
		}

		//renders gaussian with x0 = y0 = 0 with sigma s
		const double inline pixel_gauss(const float& fs, const float& ss, const float& s	)
		{
			const float a = erf((fs + 1) / sqrt(2 * s * s));
			const float b = erf((fs) / sqrt(2 * s * s));
			const float c = erf((ss + 1) / sqrt(2 * s * s));
			const float d = erf((ss) / sqrt(2 * s * s));
			return 0.25 * (a - b) * (c - d);
		}

		// a = [-1, 1] 
		void inline render_photon(const float a, const float x, const float y, const float sigma, const size_t nfs, const size_t nss, float* data)
		{
			for (size_t ss = clip(int(floor(y - 1 - 3 * sigma)), 0, nss); ss != clip(int(ceil(y + 1 + 3 * sigma)), 0, nss); ++ss) 
			{
				for (size_t fs = clip(int(floor(x - 1 - 3 * sigma)), 0, nfs); fs != clip(int(ceil(x + 1 + 3 * sigma)), 0, nfs);	++fs) 
				{
					data[nfs * ss + fs] += a * pixel_gauss(fs - x, ss - y, sigma);
				}
			}
		}
		
		//Fitness function
		float photon_target(const float a,const float x,const float y,const float sigma,const size_t nfs,const size_t nss,	const float* data,	const float* tmp,	const float noise_sigma = 0.044322f	)
		{
			float p = 0;
			for (size_t ss = clip(int(floor(y - 1 - 3 * sigma)), 0, nss);ss != clip(int(ceil(y + 1 + 3 * sigma)), 0, nss);	++ss) 
			{
				for (size_t fs = clip(int(floor(x - 1 - 3 * sigma)), 0, nfs);fs != clip(int(ceil(x + 1 + 3 * sigma)), 0, nfs);++fs) 
				{
					const size_t i = nfs * ss + fs;
					p += pow(tmp[i] + a * pixel_gauss(fs - x, ss - y, sigma) - data[i], 2u)	- pow(tmp[i] - data[i], 2);
				}
			}
			return p / pow(noise_sigma, 2);
		}

		float optimize_photon(const float a,float& x,float& y,const float charge_sigma,const size_t nfs,const size_t nss,	const float* data,	const float* tmp,const float noise_sigma = 1.0)
		{
			float eps = charge_sigma; //movement of photon center (guess for "wrongness" of origin)

			float t0 = photon_target(a, x, y, charge_sigma, nfs, nss, data, tmp, noise_sigma);
			for (size_t i = 0; i != 16; ++i) 
			{
				//calculate derivation
				float dx = 0;
				float dy = 0;
				for (size_t ss = clip(int(floor(y - 1 - 3 * charge_sigma)), 0, nss);ss != clip(int(ceil(y + 1 + 3 * charge_sigma)), 0, nss); ++ss)
				{	
					for (size_t fs = clip(int(floor(x - 1 - 3 * charge_sigma)), 0, nfs);fs != clip(int(ceil(x + 1 + 3 * charge_sigma)), 0, nfs);++fs) 
					{
						const size_t i = nfs * ss + fs;
						const float p = a * pixel_gauss(fs - x, ss - y, charge_sigma);
						dx += 2 * (data[i] - p - tmp[i]) * pixel_gauss_dfs(fs - x, ss - y, charge_sigma); //2*(daten-gauss)*grad*gaus
						dy += 2 * (data[i] - p - tmp[i]) * pixel_gauss_dss(fs - x, ss - y, charge_sigma);
					}			
				}
				dx *= a / pow(noise_sigma, 2);
				dy *= a / pow(noise_sigma, 2);
				// \ deriv.

				float norm = sqrt(pow(dx, 2u) + pow(dy, 2u)); //normalize deriv.
				if (norm < 1e-30) break;
				dx /= norm;
				dy /= norm;
				bool dir = true; //direction false if step is getting increased true for decrease
				float t;
				while (eps < sqrt(pow(nfs, 2u) + pow(nss, 2u))) 
				{
					t = photon_target(a, x - eps * dx, y - eps * dy, charge_sigma, nfs, nss, data, tmp, noise_sigma);
					if (t < t0) 
					{
						eps *= 2;
						dir = false;
					}
					else 
					{
						eps /= 2;
						break;
					}
				}
				if (dir) 
				{
					while (eps > 1e-10) 
					{
						t = photon_target(a, x - eps * dx, y - eps * dy, charge_sigma, nfs, nss, data, tmp, noise_sigma);
						if (t < t0)
							break;
						eps *= 0.5;
					}
				}
				//cerr << "# " << x << " " << y << " " << eps << endl;
				x -= eps * dx;
				y -= eps * dy;
				if (eps * 1e-3 < charge_sigma) break;
				//cerr << "# " << x << " " << y << " " << eps << endl;
			}
			return t0;
		}

		void seed_photons(std::vector<std::array<float, 2>>& photons,const float a,const float sigma,const size_t nfs,const size_t nss,	const float* data,float* tmp,const float noise_sigma = 0,const float min_reduction = 16.0f	)//min_reduction = 16.0f:: ^= 4sigma 
		{
			for (size_t ss = 0; ss != nss; ++ss) {
				for (size_t fs = 0; fs != nfs; ++fs) 
				{
					if (tmp[fs + ss * nfs] > data[fs + ss * nfs])
						continue;
					float min_x;
					float min_y;
					float min_t = -min_reduction;
					for (size_t j = 0; j != 2; ++j) 
					{
						for (size_t i = 0; i != 2; ++i) 
						{
							float x = fs + i / 2.0;
							float y = ss + j / 2.0;
							const float t =	optimize_photon(a, x, y, sigma, nfs, nss, data, tmp, noise_sigma);
							if (t < min_t) 
							{
								min_t = t;
								min_x = x;
								min_y = y;
							}
						}
					}
					if (min_t < -min_reduction) 
					{
						render_photon(a, min_x, min_y, sigma, nfs, nss, tmp);
						photons.push_back({ min_x,min_y });
						continue;
					}
				}
			}
		}

		void photonize(const float sigma, const size_t nfs, const size_t nss, const float* data, float* ret,float noise_sigma = 0.04432)
		{
			const float a = 1.0f;
			std::vector<std::array<float, 2UL>> photons;
			float* tmp = new float[nfs * nss]();

			// seed
			const float min_reduction = 4.0f; //sqrt(min_reduction) = sigma photon needs to get better
			seed_photons(photons, 1.0f, sigma, nfs, nss, data, tmp, noise_sigma, min_reduction);

			//optimisaion steps (´max  hier 64)
			for (size_t i = 0; i != 8; ++i)
			{
				double max_step = 0;
				for (size_t i = 0; i != nfs * nss; ++i) tmp[i] = 0;
				for (size_t i = 0; i != photons.size(); ++i) {
					const auto& [x, y] = photons[i];
					render_photon(1.0, x, y, sigma, nfs, nss, tmp);
				}

				//Loop über photonen
				for (size_t j = 0; j != photons.size(); ++j)
				{
					auto& [x, y] = photons[j]; // x = photons[j][0];  x = photons[j][1]; 
					render_photon(-a, x, y, sigma, nfs, nss, tmp); //adds the photon (at x,y) to tmp (if a = -1 subtracts)
					optimize_photon(a, x, y, sigma, nfs, nss, data, tmp, noise_sigma); //optimize x,y - position of one photon (changes x and y)
					render_photon(a, x, y, sigma, nfs, nss, tmp);
				}

				std::vector<std::array<float, 2UL>> _photons;
				for (size_t j = 0; j != photons.size(); ++j) //checks if all photons are necessary
				{
					auto& [x, y] = photons[j];
					render_photon(-a, x, y, sigma, nfs, nss, tmp);
					if (photon_target(a, x, y, sigma, nfs, nss, data, tmp, noise_sigma)	< -min_reduction) 
					{
						_photons.push_back({ x,y });
						render_photon(a, x, y, sigma, nfs, nss, tmp);
						continue;
					}
				}

				seed_photons(_photons, a, sigma, nfs, nss, data, tmp, noise_sigma, min_reduction); //fills potential new photons to tmp
				photons = _photons;
			}

			//photons need to be binned to pixels again
			ArrayOperators::MultiplyScalar(ret, 0.0f, nfs* nss);
			for (size_t j = 0; j < photons.size(); j++)
			{
				ret[(size_t)floor(photons[j][1]) * nfs + (size_t)floor(photons[j][0])] += 1.0;
			}

			delete[] tmp;
		}

		

	}

	void PhotonFinder_GaussFit(float* Intensity, const unsigned int FullDetSize, const Create_GaussPhotonizeSettings GaussPhotonizeSettings)
	{

		//std::cout <<"Panels: "<< GaussPhotonizeSettings.DetectorPanels[0].FirstInd << " :: " << GaussPhotonizeSettings.DetectorPanels[0].Scans[0] << " :: " << GaussPhotonizeSettings.DetectorPanels[0].Scans[1] << std::endl;

		//iterate over panels
		for (unsigned int i_pan = 0; i_pan < GaussPhotonizeSettings.DetectorPanels.size(); i_pan++)
		{
			int fs = GaussPhotonizeSettings.DetectorPanels[i_pan].Scans[0];
			int ss = GaussPhotonizeSettings.DetectorPanels[i_pan].Scans[1];
			int DetSize = fs * ss;

			
			size_t t_ind = 0;
			float* temp = new float[DetSize];
			for (unsigned int ind = GaussPhotonizeSettings.DetectorPanels[i_pan].FirstInd; ind < (GaussPhotonizeSettings.DetectorPanels[i_pan].FirstInd + DetSize); ind++)
			{
				temp[t_ind] = Intensity[ind]/GaussPhotonizeSettings.ADU_perPhoton; //copy data of panel into temp and normalize such that 1 photon should be 1.0f.
				t_ind++;
			}

			float* ret = new float[DetSize]();
			photonize(GaussPhotonizeSettings.ChargeSharingSigma, fs, ss, temp, ret);//retrive photons by GaussFitting

			
			//copy retrived photons back to intensity
			t_ind = 0;
			for (unsigned int ind = GaussPhotonizeSettings.DetectorPanels[i_pan].FirstInd; ind < (GaussPhotonizeSettings.DetectorPanels[i_pan].FirstInd + DetSize); ind++)
			{
				Intensity[ind] = ret[t_ind] * GaussPhotonizeSettings.ADU_perPhoton; //copy data of panel into temp and normalize such that 1 photon should be 1.0f.
				t_ind++;
			}

			delete[] ret;
			delete[] temp;
		}


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
		if (GainOnly)
		{
			FullDetSize = Det.DetectorSize[0] * Det.DetectorSize[1];
		}
		else
		{
			for (unsigned int j = 0; j < LAPSettings.DetPanels_Num; j++)
			{
				FullDetSize += DetectorPanels[j].Scans[0] * DetectorPanels[j].Scans[1];
			}
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

		hsize_t start[3] = { 0, 0, 0 };  // Start of hyperslab, offset
		hsize_t stride[3] = { 1, 1, 1 }; // Stride of hyperslab
		hsize_t count[3] = { 1, 1, 1 };  // Block count
		hsize_t block[3] = { 1, dims[1], dims[2] }; // Block sizes

		H5::DSetCreatPropList plist = H5::DSetCreatPropList();
		plist.setChunk(3, block);
		plist.setDeflate(6); //compression

		H5::DataSet dataset = file.createDataSet(Dataset, H5::PredType::NATIVE_FLOAT, dataspace,plist);

		H5::DataSpace mspace(3, block);

		float CounterStep = ((float)OptionsIn.HitEvents.size()) / 100.0f;
		float Counter = 0;

		for (unsigned int i = 0; i < OptionsIn.HitEvents.size(); i++)
		{
			Det.LoadIntensityData(&OptionsIn.HitEvents[i]);//Load Intensity
			Det.ApplyPixelMask();//Apply Pxelmask
			if (LAPSettings.GainMapPath != "") //Check if gain correction is intendet
			{
				GainCorrection(Det, LAPSettings.GainMapPath, LAPSettings.DatasetOffset, LAPSettings.DatasetGain, OptionsIn, GainOnly); //Perform Gain Correction
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

	void ProcessData_GaussFit(Detector& RefDet, Create_GaussPhotonizeSettings GaussPhotonizeSettings, std::string XML_In)
	{

		//Load Hits
		Settings OptionsIn;
		OptionsIn.LoadHitEventListFromFile(XML_In);

		size_t NumOfEvents = 0;
		size_t FirstEvent = 0;
		size_t LastEvent = 0;
		if (GaussPhotonizeSettings.RestrictToLimits)
		{
			if (GaussPhotonizeSettings.UpperLimit > OptionsIn.HitEvents.size())
			{
				std::cout << "WARNING: requested upper limit of Event-List exceeds number of Events loaded. Upper limit is set to number of events." << std::endl;
				LastEvent = OptionsIn.HitEvents.size();
			}
			else
			{
				LastEvent = GaussPhotonizeSettings.UpperLimit;
			}

			if (GaussPhotonizeSettings.LowLimit <= LastEvent)
			{
				FirstEvent = GaussPhotonizeSettings.LowLimit;
			}
			else
			{
				std::cout << "WARNING: requested lower limit of Event-List exceeds or is equal upper limit. Lower limit is set to zero" << std::endl;
				FirstEvent = 0;
			}
		}
		else
		{
			FirstEvent = 0;
			LastEvent = OptionsIn.HitEvents.size();
		}
		NumOfEvents = LastEvent - FirstEvent;

		Settings OptionsOut(OptionsIn);
		OptionsOut.HitEvents.clear();
		OptionsOut.HitEvents.reserve(NumOfEvents);


		// Hdf5 Stuff
		H5::H5File file(GaussPhotonizeSettings.Output_Path, H5F_ACC_TRUNC);

		hsize_t dims[3];
		dims[0] = NumOfEvents;
		dims[1] = RefDet.DetectorSize[0];
		dims[2] = RefDet.DetectorSize[1];
		H5::DataSpace dataspace(3, dims);

		hsize_t start[3] = { 0, 0, 0 };  // Start of hyperslab, offset
		hsize_t stride[3] = { 1, 1, 1 }; // Stride of hyperslab
		hsize_t count[3] = { 1, 1, 1 };  // Block count
		hsize_t block[3] = { 1, dims[1], dims[2] }; // Block sizes

		H5::DSetCreatPropList plist = H5::DSetCreatPropList();
		plist.setChunk(3, block);
		plist.setDeflate(6); //compression

		H5::DataSet dataset = file.createDataSet(GaussPhotonizeSettings.Output_Dataset, H5::PredType::NATIVE_FLOAT, dataspace, plist);

		H5::DataSpace mspace(3, block);

		// \ HDF5

		float CounterStep = ((float)NumOfEvents) / 100.0f;
		float Counter = 0;

		ProfileTime Profiler;
		
		std::cout << "Pattern to be analyzed: " << NumOfEvents << std::endl;
		std::cout << "Start photonization ..." << std::endl;
		Profiler.Tic();
		#pragma omp parallel for
		for (size_t i = FirstEvent; i < LastEvent; i++)
		{
			Detector Det = Detector(RefDet, true);


			Det.LoadIntensityData(&OptionsIn.HitEvents[i]);//Load Intensity
			Det.ApplyPixelMask();//Apply Pxelmask

			//Run Gauss fit

			// Photonize
			PhotonFinder_GaussFit(Det.Intensity, Det.DetectorSize[0] * Det.DetectorSize[1], GaussPhotonizeSettings);

			// \ 
			
			//Create new Event 
			Settings::HitEvent t_Event;

			t_Event.PhotonCount = ArrayOperators::Sum(Det.Intensity, Det.DetectorSize[0] * Det.DetectorSize[1]);
			t_Event.MeanIntensity = (float)t_Event.PhotonCount / (Det.DetectorSize[0] * Det.DetectorSize[1]);

			t_Event.Dataset = GaussPhotonizeSettings.Output_Dataset;
			t_Event.Event = i;
			t_Event.Filename = GaussPhotonizeSettings.Output_Path;
			t_Event.SerialNumber = i;
			for (int j = 0; j < 9; j++)
				t_Event.RotMatrix[j] = OptionsIn.HitEvents[i].RotMatrix[j];
			t_Event.PhotonCount = (int)ArrayOperators::Sum(Det.Intensity, Det.DetectorSize[0] * Det.DetectorSize[1]);
			t_Event.MeanIntensity = (float)t_Event.PhotonCount / ((float)(Det.DetectorSize[0] * Det.DetectorSize[1]));


			//write stuff
			#pragma omp critical
			{
				OptionsOut.HitEvents.push_back(t_Event);
				//Write new Intensity to H5
				start[0] = i;
				dataspace.selectHyperslab(H5S_SELECT_SET, count, start, stride, block);
				dataset.write(Det.Intensity, H5::PredType::NATIVE_FLOAT, mspace, dataspace);

				//Count for status update
				if ((float)i >= Counter)
				{
					std::cout << i-FirstEvent << "/" << NumOfEvents << "  ^= " << Counter / CounterStep << "%\n";
					Counter += CounterStep;
				}

			}

		}

		mspace.close();
		dataspace.close();
		dataset.close();
		file.close();

		std::cout << "Done in ";
		Profiler.Toc(true, true);
	}

	void GainCorrection(Detector & Det, std::string GainCorr_Path, std::string Dataset_Offset, std::string Dataset_Gain, Settings & Options, bool AllowNegativeValues)
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
		H5::DataSpace mspace_O(2, dimsm, NULL);
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
		H5::DataSpace mspace_G(2, dimsm, NULL);
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
			if (((Det.Intensity[i] >= 0)||(AllowNegativeValues)) && GM_Gain[i] > 0 && Det.PixelMask[i] == 1)
			{
				Det.Intensity[i] = Det.Intensity[i] / GM_Gain[i];
				//if (Det.Intensity[i] > 100 )
				//	std::cout << "Warning: over 100 Photones at " << i << " : " << Det.Intensity[i] << " (Gain: " << GM_Gain[i] << " ) " << "\n";
				
			
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
		
		//std::cout << Det.DetectorSize[0] << " x " << Det.DetectorSize[0] << std::endl;
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
			H5::DataSpace mspace(2, dimsm, NULL);
			DS.selectHyperslab(H5S_SELECT_SET, count, offset, stride, block);

			H5::PredType type = H5::PredType::NATIVE_FLOAT;
			dataset.read(Dark, type, mspace, DS);
			//Close DS, dataset, mspace
			DS.close();
			dataset.close();
			mspace.close();
		}

		//invert Dark
		ArrayOperators::MultiplyScalar(Dark, -1.0f, Det.DetectorSize[0] * Det.DetectorSize[1]); //invert Darkfield

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

		H5::DataSet dataset;
		try
		{
			H5::Exception::dontPrint();
			dataset = file.createDataSet(DarkSettings.Output_Dataset, H5::PredType::NATIVE_FLOAT, dataspace);
		}
		catch (H5::FileIException e)
		{
			std::cerr << "HDF5 is retarded and does not allow '/' in dataset names sometimes. changed dataset to 'data' and continue ..." << std::endl;
			DarkSettings.Output_Dataset = "data";
			dataset = file.createDataSet(DarkSettings.Output_Dataset, H5::PredType::NATIVE_FLOAT, dataspace);
		}

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