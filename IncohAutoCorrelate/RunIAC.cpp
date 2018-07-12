#include <iostream>
#include <omp.h>

#include "RunIAC.h"
#include "Detector.h"
#include "ArrayOperators.h"
#include"ProfileTime.h"
#include "H5Cpp.h"

namespace RunIAC
{



	void Create_CQ_Mesh(ACMesh & CQ, CreateCQ_Settings CQ_Settings, Settings & PrgSettings)
	{
		Create_CQ_Mesh(CQ, CQ_Settings, PrgSettings, 0, PrgSettings.HitEvents.size());
	}
	void Create_CQ_Mesh(ACMesh & CQ, CreateCQ_Settings CQ_Settings, Settings & PrgSettings, unsigned int LowerBound, unsigned int UpperBound)
	{
		if (CQ_Settings.echo)
			std::cout << CQ_Settings.ThreadName << ": start C(q) calculation.\n";
		ProfileTime profiler;
		profiler.Tic();
		Detector Det;
		if (CQ_Settings.echo)
			std::cout << CQ_Settings.ThreadName << ": Set up detector ...\n";

		//Load PixelMap
		Det.LoadPixelMap(CQ_Settings.PixelMap_Path, CQ_Settings.PixelMap_DataSet);
		//Create k-Map
		Det.Calc_kMap();
		//Load Pixelmask
		Det.PixelMask = new int[Det.DetectorSize[0] * Det.DetectorSize[1]]();
		if (CQ_Settings.PixelMask_Path != "")
		{
			ArrayOperators::LoadArrayFromFile<int>(CQ_Settings.PixelMask_Path, Det.PixelMask, Det.DetectorSize[0] * Det.DetectorSize[1]);
		}
		else
		{
			// No Pixelmask given => set every entry to one
#pragma omp parallel for
			for (unsigned int i = 0; i < Det.DetectorSize[0] * Det.DetectorSize[1]; i++)
			{
				Det.PixelMask[i] = 1;
			}
		}

		//load integrated/averaged intensity
		Det.Intensity = new float[Det.DetectorSize[0] * Det.DetectorSize[1]]();
		ArrayOperators::LoadArrayFromFile(CQ_Settings.AVIntensity_Path, Det.Intensity, Det.DetectorSize[0] * Det.DetectorSize[1]);
		//apply Pixelmask
		ArrayOperators::ParMultiplyElementwise(Det.Intensity, Det.PixelMask, Det.DetectorSize[0] * Det.DetectorSize[1]);


		//Setup dense Correlation
		if (CQ_Settings.echo)
			std::cout << CQ_Settings.ThreadName << ": Start dense autocorrelating (small Mesh) ...\n";

		ACMesh smallMesh;
		smallMesh.CreateSmallMesh_CofQ_ForDetector(Det, CQ_Settings.MeshSize, CQ_Settings.QZoom);

		if (CQ_Settings.echo)
			std::cout << CQ_Settings.ThreadName << ": Calc small C(q) ...\n";

		Det.AutoCorrelate_CofQ_SmallMesh(smallMesh, CQ_Settings.AC_Small_Flags, PrgSettings);

		if (CQ_Settings.SaveSmall_CQ)
		{
			ArrayOperators::SafeArrayToFile(CQ_Settings.SmallCQ_Path, smallMesh.CQMesh, smallMesh.Shape.Size_AB*smallMesh.Shape.Size_AB*smallMesh.Shape.Size_C, ArrayOperators::FileType::Binary);
			if (CQ_Settings.echo)
				std::cout << CQ_Settings.ThreadName << ": Small C(q) saved as: " << CQ_Settings.SmallCQ_Path << "\n";
		}

		//Weight and Merge
		if (CQ_Settings.echo)
			std::cout << CQ_Settings.ThreadName << ": Weight and merge C(q) ...\n";

		CQ.CreateBigMesh_CofQ_ForDetector(Det, CQ_Settings.MeshSize, CQ_Settings.QZoom);
		Det.Merge_smallCofQ(CQ, smallMesh, PrgSettings.HitEvents, LowerBound, UpperBound, PrgSettings, CQ_Settings.AC_Merge_Flags);

		if (CQ_Settings.SaveBig_CQ)
		{
			ArrayOperators::SafeArrayToFile(CQ_Settings.BigCQ_Path, CQ.CQMesh, CQ.Shape.Size_AB*CQ.Shape.Size_AB*CQ.Shape.Size_C, ArrayOperators::FileType::Binary);
			if (CQ_Settings.echo)
				std::cout << CQ_Settings.ThreadName << ": Big C(q) saved as: " << CQ_Settings.BigCQ_Path << "\n";
		}

		if (CQ_Settings.echo)
		{
			std::cout << CQ_Settings.ThreadName << ": C(q) createt in ";
			profiler.Toc(true);
			std::cout << "\n";

			std::cout << CQ_Settings.ThreadName << ": Finished.\n";
		}

	}

	void Run_AC_UW(ACMesh & AC, CreateAC_Settings AC_Settings, Settings & PrgSettings)
	{
		Run_AC_UW(AC, AC_Settings, PrgSettings, 0, PrgSettings.HitEvents.size());
	}
	void Run_AC_UW(ACMesh & AC, CreateAC_Settings AC_Settings, Settings & PrgSettings, unsigned int LowerBound, unsigned int UpperBound)
	{
		ProfileTime profiler;
		{
			if (AC_Settings.echo)
				std::cout << AC_Settings.ThreadName << ": start autocorrelation for events: " << LowerBound << " - " << UpperBound << "\n";
		}


		profiler.Tic();
		Detector Det;

		if (AC_Settings.echo)
			std::cout << AC_Settings.ThreadName << ": Set up detector ...\n";

		//Load PixelMap
		Det.LoadPixelMap(AC_Settings.PixelMap_Path, AC_Settings.PixelMap_DataSet);
		//Create k-Map
		Det.Calc_kMap();
		//Load Pixelmask
		delete[] Det.PixelMask;
		Det.PixelMask = new int[Det.DetectorSize[0] * Det.DetectorSize[1]]();
		if (AC_Settings.PixelMask_Path != "")
		{
			ArrayOperators::LoadArrayFromFile<int>(AC_Settings.PixelMask_Path, Det.PixelMask, Det.DetectorSize[0] * Det.DetectorSize[1]);
		}
		else
		{
			// No Pixelmask given => set every entry to one
#pragma omp parallel for
			for (unsigned int i = 0; i < Det.DetectorSize[0] * Det.DetectorSize[1]; i++)
			{
				Det.PixelMask[i] = 1;
			}
		}

		//Setup  Correlation
		AC.CreateBigMeshForDetector(Det, AC_Settings.MeshSize, AC_Settings.QZoom);
		AC.Options = &PrgSettings;


		if (AC_Settings.echo)
			std::cout << AC_Settings.ThreadName << ": Start dense autocorrelating (small Mesh) ...\n";



		//Det.LoadIntensityData(&PrgSettings.HitEvents[0]);
		for (unsigned int i = LowerBound; i < UpperBound; i++)
		{
			if (i % AC_Settings.StatusEcho == 0 && AC_Settings.echo)
				std::cout << AC_Settings.ThreadName << ": AC Event " << i << "/" << (UpperBound) << std::endl;


			Det.LoadIntensityData(&PrgSettings.HitEvents[i]);

			ArrayOperators::ParMultiplyElementwise(Det.Intensity, Det.PixelMask, Det.DetectorSize[0] * Det.DetectorSize[1]);

			Det.CreateSparseHitList(AC_Settings.PhotonOffset, AC_Settings.PhotonStep); //Sparsificate
			std::cout << i << ": Pixels with hits: " << Det.SparseHitList.size()*100.0 / (Det.DetectorSize[0]* Det.DetectorSize[1]) << "%"<< "    Mean intensity: " << PrgSettings.HitEvents[i].MeanIntensity << "\n";

			Det.AutoCorrelateSparseList(AC, AC_Settings.AC_FirstMap_Flags, AC_Settings.DoubleMap, PrgSettings);
		}




		if (AC_Settings.echo)
		{
			std::cout << AC_Settings.ThreadName << ": All events autocorrelated in ";
			profiler.Toc(true);
			std::cout << "\n";

			std::cout << "Additional Information:\n";
			std::cout << "Max Q in Mesh: +/- " << AC.Shape.Max_Q << "\n";


		}

		if (AC_Settings.SaveBig_AC)
		{
			double* ACMesh = new double[AC.Shape.Size_AB*AC.Shape.Size_AB*AC.Shape.Size_C]();
#pragma omp parallel for
			for (int i = 0; i < AC.Shape.Size_AB*AC.Shape.Size_AB*AC.Shape.Size_C; i++)
			{
				ACMesh[i] = PrgSettings.IntToFloat(AC.Mesh[i]);
			}
			ArrayOperators::SafeArrayToFile(AC_Settings.BigAC_Path, ACMesh, AC.Shape.Size_AB*AC.Shape.Size_AB*AC.Shape.Size_C, ArrayOperators::FileType::Binary);

			if (AC_Settings.echo)
				std::cout << AC_Settings.ThreadName << ": Saved AC (double) as " << AC_Settings.BigAC_Path << "\n";

			delete[] ACMesh;
		}


		if (AC_Settings.echo)
			std::cout << AC_Settings.ThreadName << ": Finished.\n";
		//end of function
	}

	void Merge_ACandCQ(double *& Output, ACMesh & AC, ACMesh & CQ, Settings & PrgSettings)
	{
		double* ACMesh = new double[AC.Shape.Size_AB*AC.Shape.Size_AB*AC.Shape.Size_C]();
#pragma omp parallel for
		for (int i = 0; i < AC.Shape.Size_AB*AC.Shape.Size_AB*AC.Shape.Size_C; i++)
		{
			ACMesh[i] = PrgSettings.IntToFloat(AC.Mesh[i]);
		}

		Merge_ACandCQ(Output, ACMesh, CQ, PrgSettings);

		delete[] ACMesh;
	}
	void Merge_ACandCQ(double *& Output, double * AC, ACMesh & CQ, Settings & PrgSettings)
	{
		unsigned int Size = CQ.Shape.Size_AB*CQ.Shape.Size_AB*CQ.Shape.Size_C;

		Output = new double[Size]();

#pragma omp parallel for
		for (unsigned int i = 0; i < Size; i++)
		{
			if (CQ.CQMesh[i] <= 0)
				Output[i] = 0;

			Output[i] = AC[i] / CQ.CQMesh[i];
		}

	}


	//Single Molecule
	unsigned int GetH5StackSize(std::string Filename, std::string Dataset)
	{
		H5::H5File file(Filename, H5F_ACC_RDONLY);
		H5::DataSet dataset = file.openDataSet(Dataset);

		H5::DataSpace DS = dataset.getSpace();

		hsize_t dims[4];
		DS.getSimpleExtentDims(dims, NULL);

		return (unsigned int)dims[0];
	}

	void Run_AC_SM_Full(AC1D & AC, CreateSM_Settings SM_Settings, Settings & PrgSettings)
	{
		ProfileTime Profiler;
		ProfileTime Profiler_All;
		Profiler_All.Tic();
		std::cout << "Set up Detector.\n";
		Detector Det;

		//Load PixelMap
		Det.LoadPixelMap(SM_Settings.PixelMap_Path, SM_Settings.PixelMap_DataSet);
		//Create k-Map
		Det.Calc_kMap();
		//Load Pixelmask
		Det.PixelMask = new int[Det.DetectorSize[0] * Det.DetectorSize[1]]();
		if (SM_Settings.PixelMask_Path != "")
		{
			ArrayOperators::LoadArrayFromFile<int>(SM_Settings.PixelMask_Path, Det.PixelMask, Det.DetectorSize[0] * Det.DetectorSize[1]);
		}
		else
		{
			// No Pixelmask given => set every entry to one
#pragma omp parallel for
			for (unsigned int i = 0; i < Det.DetectorSize[0] * Det.DetectorSize[1]; i++)
			{
				Det.PixelMask[i] = 1;
			}
		}
		Det.Checklist.PixelMask = true;

		//Set up AC1D - Container
		AC.Initialize(Det, SM_Settings.ArraySize);
		//AC1D AC;

		AC.Shape.Max_Q = (float)(sqrt(Det.Max_q[0] * Det.Max_q[0] + Det.Max_q[1] * Det.Max_q[1] + Det.Max_q[2] * Det.Max_q[2]));
		AC.Shape.dq_per_Step = AC.Shape.Max_Q / ((float)(AC.Shape.Size - 1));

		std::cout << "Max Q: " << AC.Shape.Max_Q << "    Size: " << AC.Shape.Size << "    dq/dx: " << AC.Shape.dq_per_Step << "\n";

		//Get number of events 
		std::cout << "Calculate full stack size (number of events).\n";
		PrgSettings.HitEvents.clear();
		PrgSettings.HitEvents.reserve(100000);
		unsigned int StackSize = 0;
		for (unsigned int i = 0; i < SM_Settings.Files.size(); i++)
		{
			unsigned int t_size = GetH5StackSize(SM_Settings.Files[i], SM_Settings.H5Dataset[i]);
			StackSize += t_size;
			for (unsigned int j = 0; j < t_size; j++)
			{
				Settings::HitEvent t_Event;
				t_Event.Event = j;
				t_Event.Filename = SM_Settings.Files[i];
				t_Event.Dataset = SM_Settings.H5Dataset[i];
				PrgSettings.HitEvents.push_back(t_Event);
			}
		}
		std::cout << "Full stack size: " << StackSize << "\n\n";


		//Average intensity of events
		std::cout << "Average intensity\n";
		Det.Intensity = new float[Det.DetectorSize[0] * Det.DetectorSize[1]]();
		{
			Detector t_Int(Det, true);
			delete[] t_Int.Intensity;
			t_Int.Intensity = new float[1](); // otherwise it deletes pointer of Det in LoadIntensityData_PSANA_StyleJungfr ...
			Profiler.Tic();
			for (unsigned int i = 0; i < StackSize; i++)
			{
				if (SM_Settings.JungfrDet)
				{
					t_Int.LoadIntensityData_PSANA_StyleJungfr(PrgSettings.HitEvents[i].Filename, PrgSettings.HitEvents[i].Dataset, PrgSettings.HitEvents[i].Event);
				}
				else
				{
					t_Int.Intensity = new float[Det.DetectorSize[0] * Det.DetectorSize[1]];
					//that solutions is not elegant => improve readabillity and code coherence!!
					t_Int.LoadIntensityData_EPIX(t_Int.Intensity, PrgSettings.HitEvents[i].Filename, PrgSettings.HitEvents[i].Dataset, PrgSettings.HitEvents[i].Event);
					t_Int.Checklist.Intensity = true;
				}

				//apply PixelMask
				ArrayOperators::ParMultiplyElementwise(Det.Intensity, Det.PixelMask, Det.DetectorSize[0] * Det.DetectorSize[1]);

				ArrayOperators::DiscretizeToPhotons(t_Int.Intensity, SM_Settings.PhotonOffset, SM_Settings.PhotonStep, Det.DetectorSize[0] * Det.DetectorSize[1]);
				ArrayOperators::ParAdd(Det.Intensity, t_Int.Intensity, Det.DetectorSize[0] * Det.DetectorSize[1]);
				//Save Mean Int of exposure
				PrgSettings.HitEvents[i].PhotonCount = (int)(ArrayOperators::Sum(t_Int.Intensity, Det.DetectorSize[0] * Det.DetectorSize[1]));
				PrgSettings.HitEvents[i].MeanIntensity = (ArrayOperators::Sum(t_Int.Intensity, Det.DetectorSize[0] * Det.DetectorSize[1]) / ((float)(Det.DetectorSize[0] * Det.DetectorSize[1])));
			}
			ArrayOperators::MultiplyScalar(Det.Intensity, (1.0 / ((float)StackSize)), Det.DetectorSize[0] * Det.DetectorSize[1]);
			Profiler.Toc(true);
		}

		ArrayOperators::SafeArrayToFile(SM_Settings.Output_AV_Int_Path, Det.Intensity, Det.DetectorSize[0] * Det.DetectorSize[1], ArrayOperators::FileType::Binary);
		std::cout << "Saved averaged Intensity as: " << SM_Settings.Output_AV_Int_Path << "\n";

		// Calculate C(q)
		PrgSettings.Echo("\nCalculate C(q) - AV:");
		Profiler.Tic();
		AC.Calculate_CQ(Det, PrgSettings, Settings::Interpolation::Linear);
		Profiler.Toc(true);

		ArrayOperators::SafeArrayToFile(SM_Settings.Output_CQ_Path, AC.CQ, AC.Shape.Size, ArrayOperators::FileType::Binary);
		std::cout << "Saved angular averaged C(q) as: " << SM_Settings.Output_CQ_Path << "\n";

		//Calculate AC_UW

		PrgSettings.Echo("\nCalculate AC uw - AAV:");
		Profiler.Tic();
		std::array<float, 2> t_Photonis;
		t_Photonis[0] = SM_Settings.PhotonOffset;
		t_Photonis[1] = SM_Settings.PhotonStep;
		AC.Calculate_AC_UW_MR(PrgSettings, Det, Settings::Interpolation::Linear, t_Photonis, SM_Settings.JungfrDet);
		Profiler.Toc(true);

		ArrayOperators::SafeArrayToFile(SM_Settings.Output_ACUW_Path, AC.AC_UW, AC.Shape.Size, ArrayOperators::FileType::Binary);
		std::cout << "Saved angular averaged unweighted AC as: " << SM_Settings.Output_ACUW_Path << "\n";

		// Merge
		delete[] AC.AC;
		AC.AC = new double[AC.Shape.Size]();
		for (unsigned int i = 0; i < AC.Shape.Size; i++)
		{
			AC.AC[i] = AC.AC_UW[i] / AC.CQ[i];
		}

		ArrayOperators::SafeArrayToFile(SM_Settings.Output_AC_Path, AC.AC, AC.Shape.Size, ArrayOperators::FileType::Binary);
		std::cout << "Saved angular averaged AC as: " << SM_Settings.Output_AC_Path << "\n";


		//Q axis
		delete[] AC.Q;
		AC.Q = new double[AC.Shape.Size]();
		for (unsigned int i = 0; i < AC.Shape.Size; i++)
		{
			double step = AC.Shape.Max_Q / ((double)(AC.Shape.Size - 1));
			AC.Q[i] = i * step / 1.94; // 1.94 is the wavelength for a 6.4keV photon
		}


		ArrayOperators::SafeArrayToFile(SM_Settings.Output_Q_Path, AC.Q, AC.Shape.Size, ArrayOperators::FileType::Binary);
		std::cout << "Saved angular averaged Q as: " << SM_Settings.Output_Q_Path << "\n";



		std::cout << "\n Whole angular averaged, single molecule evaluation performed within\n";
		Profiler_All.Toc(true);
	}


	void Run_AC_SM_fractionalCQ(AC1D & AC, CreateSM_Settings SM_Settings, Settings & PrgSettings, unsigned int FractionSize, bool PPPdata)
	{
		//Warning: this Method ignores OpenCL Device pool, ... fix this problem as soon as possible!!!

		ProfileTime Profiler;
		ProfileTime Profiler_All;
		Profiler_All.Tic();
		std::cout << "Set up Detector.\n";
		Detector Det;

		//Setup Detector and AC1D:
		{
			//Load PixelMap
			Det.LoadPixelMap(SM_Settings.PixelMap_Path, SM_Settings.PixelMap_DataSet);
			//Create k-Map
			Det.Calc_kMap();
			//Load Pixelmask
			Det.PixelMask = new int[Det.DetectorSize[0] * Det.DetectorSize[1]]();
			if (SM_Settings.PixelMask_Path != "")
			{
				ArrayOperators::LoadArrayFromFile<int>(SM_Settings.PixelMask_Path, Det.PixelMask, Det.DetectorSize[0] * Det.DetectorSize[1]);
			}
			else
			{
				// No Pixelmask given => set every entry to one
				#pragma omp parallel for
				for (unsigned int i = 0; i < Det.DetectorSize[0] * Det.DetectorSize[1]; i++)
				{
					Det.PixelMask[i] = 1;
				}
			}
			Det.Checklist.PixelMask = true;

			//Set up AC1D - Container
			AC.Initialize(Det, SM_Settings.ArraySize);

			//AC1D AC;
			AC.Shape.Max_Q = (float)(sqrt(Det.Max_q[0] * Det.Max_q[0] + Det.Max_q[1] * Det.Max_q[1] + Det.Max_q[2] * Det.Max_q[2]));
			AC.Shape.dq_per_Step = AC.Shape.Max_Q / ((float)(AC.Shape.Size - 1));

			std::cout << "Max Q: " << AC.Shape.Max_Q << "    Size: " << AC.Shape.Size << "    dq/dx: " << AC.Shape.dq_per_Step << "\n";
		}

		//Get number of events 
		
		unsigned int StackSize = 0;
		if (!PPPdata)
		{
			std::cout << "Calculate full stack size (number of events).\n";
			PrgSettings.HitEvents.clear();
			PrgSettings.HitEvents.reserve(100000);
			
			for (unsigned int i = 0; i < SM_Settings.Files.size(); i++)
			{
				unsigned int t_size = GetH5StackSize(SM_Settings.Files[i], SM_Settings.H5Dataset[i]);
				StackSize += t_size;
				for (unsigned int j = 0; j < t_size; j++)
				{
					Settings::HitEvent t_Event;
					t_Event.Event = j;
					t_Event.Filename = SM_Settings.Files[i];
					t_Event.Dataset = SM_Settings.H5Dataset[i];
					PrgSettings.HitEvents.push_back(t_Event);
				}
			}
		}
		else
		{
			StackSize = PrgSettings.HitEvents.size();
		}
		unsigned int Fractions = StackSize / FractionSize;
		std::cout << "Full stack size: " << StackSize << " \t FractionSize: " << FractionSize << " => Fractions: " << Fractions << "\n\n";

		AC.AC = new double[AC.Shape.Size]();

		Settings CurrSettingsList(PrgSettings);		for (unsigned int CurrFracNum = 0; CurrFracNum < Fractions; CurrFracNum++)
		{
			AC1D AC_t; //AC1D for current fraction
			std::cout << "Run Events of fraction " << CurrFracNum << ":\n     *****     \n";
			Det.Intensity = new float[Det.DetectorSize[0] * Det.DetectorSize[1]]();
			Det.Checklist.Intensity = true;
			
			CurrSettingsList.echo = false;
			CurrSettingsList.HitEvents.clear();
			CurrSettingsList.HitEvents.reserve(FractionSize);
			{
				AC_t.Initialize(Det, SM_Settings.ArraySize);
				AC_t.Shape.Max_Q = (float)(sqrt(Det.Max_q[0] * Det.Max_q[0] + Det.Max_q[1] * Det.Max_q[1] + Det.Max_q[2] * Det.Max_q[2]));
				AC_t.Shape.dq_per_Step = AC_t.Shape.Max_Q / ((float)(AC_t.Shape.Size - 1));
			}

			std::cout << "Average intensity ...\n";
			{
				Detector t_Int(Det, true);
				delete[] t_Int.Intensity;
				t_Int.Intensity = new float[1](); // otherwise it deletes pointer of Det in LoadIntensityData_PSANA_StyleJungfr ...
				Profiler.Tic();
				for (unsigned int i = FractionSize * CurrFracNum; i < FractionSize * (CurrFracNum + 1); i++)//i < 
				{
					if (PPPdata)
					{
						t_Int.LoadIntensityData(&PrgSettings.HitEvents[i]);
					}
					else
					{
						if (SM_Settings.JungfrDet)
						{
							t_Int.LoadIntensityData_PSANA_StyleJungfr(PrgSettings.HitEvents[i].Filename, PrgSettings.HitEvents[i].Dataset, PrgSettings.HitEvents[i].Event);
						}
						else
						{
							delete[] t_Int.Intensity;
							t_Int.Intensity = new float[Det.DetectorSize[0] * Det.DetectorSize[1]];
							//that solutions is not elegant => improve readabillity and code coherence!!
							t_Int.LoadIntensityData_EPIX(t_Int.Intensity, PrgSettings.HitEvents[i].Filename, PrgSettings.HitEvents[i].Dataset, PrgSettings.HitEvents[i].Event);
							t_Int.Checklist.Intensity = true;
						}
					}
					//apply PixelMask
					ArrayOperators::ParMultiplyElementwise(t_Int.Intensity, Det.PixelMask, Det.DetectorSize[0] * Det.DetectorSize[1]);

					ArrayOperators::DiscretizeToPhotons(t_Int.Intensity, SM_Settings.PhotonOffset, SM_Settings.PhotonStep, Det.DetectorSize[0] * Det.DetectorSize[1]);
					ArrayOperators::ParAdd(Det.Intensity, t_Int.Intensity, Det.DetectorSize[0] * Det.DetectorSize[1]);
					//Save Mean Int of exposure
					PrgSettings.HitEvents[i].PhotonCount = (int)(ArrayOperators::Sum(t_Int.Intensity, Det.DetectorSize[0] * Det.DetectorSize[1]));
					PrgSettings.HitEvents[i].MeanIntensity = (ArrayOperators::Sum(t_Int.Intensity, Det.DetectorSize[0] * Det.DetectorSize[1]) / ((float)(Det.DetectorSize[0] * Det.DetectorSize[1])));
					CurrSettingsList.HitEvents.push_back(PrgSettings.HitEvents[i]);
				}
				ArrayOperators::MultiplyScalar(Det.Intensity, (1.0 / ((float)StackSize)), Det.DetectorSize[0] * Det.DetectorSize[1]);
				Profiler.Toc(true);
			}
			
			
			// Calculate C(q)
			PrgSettings.Echo("\nCalculate C(q) - AV:");
			Profiler.Tic();
			AC_t.Calculate_CQ(Det, CurrSettingsList, Settings::Interpolation::Linear);
			Profiler.Toc(true);

			//Calculate AC_UW
			PrgSettings.Echo("\nCalculate AC uw - AAV:");
			Profiler.Tic();
			std::array<float, 2> t_Photonis;
			t_Photonis[0] = SM_Settings.PhotonOffset;
			t_Photonis[1] = SM_Settings.PhotonStep;
			AC_t.Calculate_AC_UW_MR(CurrSettingsList, Det, Settings::Interpolation::Linear, t_Photonis,false); //implement for non PPP, maybe!
			Profiler.Toc(true);

			//Merge


			for (unsigned int i = 0; i < AC.Shape.Size; i++)
			{
				AC.AC[i] += AC_t.AC_UW[i] / AC_t.CQ[i];
				AC.Q[i] = AC_t.Q[i];
			}




			//clean up
			delete[] Det.Intensity;
			delete[] AC_t.AC;
			delete[] AC_t.AC_UW;
			delete[] AC_t.CQ;
			delete[] AC_t.Q;
		}

		ArrayOperators::SafeArrayToFile(SM_Settings.Output_Q_Path, AC.Q, AC.Shape.Size, ArrayOperators::FileType::Binary);
		std::cout << "Saved angular averaged Q as: " << SM_Settings.Output_Q_Path << "\n";

		ArrayOperators::SafeArrayToFile(SM_Settings.Output_AC_Path, AC.AC, AC.Shape.Size, ArrayOperators::FileType::Binary);
		std::cout << "Saved angular averaged AC as: " << SM_Settings.Output_AC_Path << "\n";

	}



		//Statistics
	void Print_Statistics_SM(CreateSM_Settings SM_Settings, Settings & PrgSettings)
	{
		ProfileTime Profiler;
		ProfileTime Profiler_All;
		Profiler_All.Tic();
		std::cout << "Set up Detector.\n";
		Detector Det;

		//Load PixelMap
		Det.LoadPixelMap(SM_Settings.PixelMap_Path, SM_Settings.PixelMap_DataSet);
		//Create k-Map
		Det.Calc_kMap();
		//Load Pixelmask
		Det.PixelMask = new int[Det.DetectorSize[0] * Det.DetectorSize[1]]();
		if (SM_Settings.PixelMask_Path != "")
		{
			ArrayOperators::LoadArrayFromFile<int>(SM_Settings.PixelMask_Path, Det.PixelMask, Det.DetectorSize[0] * Det.DetectorSize[1]);
		}
		else
		{
			// No Pixelmask given => set every entry to one
#pragma omp parallel for
			for (unsigned int i = 0; i < Det.DetectorSize[0] * Det.DetectorSize[1]; i++)
			{
				Det.PixelMask[i] = 1;
			}
		}
		Det.Checklist.PixelMask = true;


		//Get number of events 
		std::cout << "Calculate full stack size (number of events).\n";
		PrgSettings.HitEvents.clear();
		PrgSettings.HitEvents.reserve(100000);
		unsigned int StackSize = 0;
		for (unsigned int i = 0; i < SM_Settings.Files.size(); i++)
		{
			unsigned int t_size = GetH5StackSize(SM_Settings.Files[i], SM_Settings.H5Dataset[i]);
			StackSize += t_size;
			for (unsigned int j = 0; j < t_size; j++)
			{
				Settings::HitEvent t_Event;
				t_Event.Event = j;
				t_Event.Filename = SM_Settings.Files[i];
				t_Event.Dataset = SM_Settings.H5Dataset[i];
				PrgSettings.HitEvents.push_back(t_Event);
			}
		}
		std::cout << "Full stack size: " << StackSize << "\n\n";


		//Average intensity of events
		std::cout << "Average intensity\n";
		Det.Intensity = new float[Det.DetectorSize[0] * Det.DetectorSize[1]]();
		{
			Detector t_Int(Det, true);
			delete[] t_Int.Intensity;
			t_Int.Intensity = new float[1](); // otherwise it deletes pointer of Det in LoadIntensityData_PSANA_StyleJungfr ...
			Profiler.Tic();
			for (unsigned int i = 0; i < StackSize; i++)
			{
				if (SM_Settings.JungfrDet)
				{
					t_Int.LoadIntensityData_PSANA_StyleJungfr(PrgSettings.HitEvents[i].Filename, PrgSettings.HitEvents[i].Dataset, PrgSettings.HitEvents[i].Event);
				}
				else
				{
					t_Int.Intensity = new float[Det.DetectorSize[0] * Det.DetectorSize[1]];
					//that solutions is not elegant => improve readabillity and code coherence!!
					t_Int.LoadIntensityData_EPIX(t_Int.Intensity, PrgSettings.HitEvents[i].Filename, PrgSettings.HitEvents[i].Dataset, PrgSettings.HitEvents[i].Event);
					t_Int.Checklist.Intensity = true;
				}

				//apply PixelMask
				ArrayOperators::ParMultiplyElementwise(Det.Intensity, Det.PixelMask, Det.DetectorSize[0] * Det.DetectorSize[1]);

				ArrayOperators::DiscretizeToPhotons(t_Int.Intensity, SM_Settings.PhotonOffset, SM_Settings.PhotonStep, Det.DetectorSize[0] * Det.DetectorSize[1]);
				ArrayOperators::ParAdd(Det.Intensity, t_Int.Intensity, Det.DetectorSize[0] * Det.DetectorSize[1]);
				//Save Mean Int of exposure
				PrgSettings.HitEvents[i].PhotonCount = (int)(ArrayOperators::Sum(t_Int.Intensity, Det.DetectorSize[0] * Det.DetectorSize[1]));
				PrgSettings.HitEvents[i].MeanIntensity = (((float)PrgSettings.HitEvents[i].PhotonCount) / ((float)(Det.DetectorSize[0] * Det.DetectorSize[1])));
			}
			ArrayOperators::MultiplyScalar(Det.Intensity, (1.0 / ((float)StackSize)), Det.DetectorSize[0] * Det.DetectorSize[1]);
			Profiler.Toc(true);
		}


		double MeanIntensity = 0;
		double StAbwIntensity = 0;

		for (unsigned int i = 0; i < PrgSettings.HitEvents.size(); i++)
		{
			MeanIntensity += PrgSettings.HitEvents[i].MeanIntensity;
		}
		MeanIntensity = MeanIntensity / PrgSettings.HitEvents.size();

		for (unsigned int i = 0; i < PrgSettings.HitEvents.size(); i++)
		{
			StAbwIntensity += (MeanIntensity - PrgSettings.HitEvents[i].MeanIntensity)*(MeanIntensity - PrgSettings.HitEvents[i].MeanIntensity);
		}
		StAbwIntensity = sqrt(StAbwIntensity / PrgSettings.HitEvents.size());


		std::cout << "Mean Intensity: " << MeanIntensity << "   StAbw: " << StAbwIntensity << "\n";


	}



	//General
	void Load_and_average_Intensities(Settings &Options, Detector &Det, float PhotonThreshold, float PhotonStep, std::string XML_Filename, std::string AvInt_Filename)
	{
		ProfileTime profiler;
		Options.Echo("Load and average intensities (all)");
		profiler.Tic();
		Det.LoadAndAverageIntensity(Options.HitEvents, PhotonThreshold, PhotonStep, true);

		//Det.LoadAndAverageIntensity(Options.HitEvents, PhotonThreshold, PhotonStep);

		profiler.Toc(true);
		std::cout << "done.\n";

		Options.Echo("Save EventList as XML");
		Options.SafeHitEventListToFile(XML_Filename);
		Options.Echo("Save averaged Intensity");
		ArrayOperators::SafeArrayToFile(AvInt_Filename, Det.Intensity, Det.DetectorSize[0] * Det.DetectorSize[1], ArrayOperators::FileType::Binary);

	}


	void GeneratePixelMapForSimulator(std::string Filename, std::string Dataset, double Distance, double PixelSize, int SizeFS, int SizeSS, std::array<float, 3> Orientation)
	{
		double * PixelMap;
		PixelMap = new double[SizeFS * SizeSS];



		std::cerr << "To Implement.\n";
		throw;


		delete[] PixelMap;
	}


}