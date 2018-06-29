#include "PPP.h"

#include <omp.h>


namespace PPP
{



	void PhotonFinder_LargestAdjacentPixel(float * Intensity, std::vector<DetectorPanel> DetectorPanels,int FullDetSize, float ADU_perPhoton, float SeedThershold, float CombinedThershold )
	{
		float * FullPhotons = new float[FullDetSize]();
		float * FragmentedPhotons = new float[FullDetSize]();

		//find full photons
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
			}
		}
	}
}