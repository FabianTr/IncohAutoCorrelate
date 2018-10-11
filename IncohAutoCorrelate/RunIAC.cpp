#include <iostream>
#include <omp.h>

#include "RunIAC.h"
#include "Detector.h"
#include "ArrayOperators.h"
#include"ProfileTime.h"
#include "H5Cpp.h"
#include "Simulator.h"

namespace RunIAC
{
	

	
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
			std::cout << AC_Settings.ThreadName << ": Start sparse autocorrelating  ...\n";



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
			{
				Output[i] = 0;
			}
			else
			{
				Output[i] = AC[i] / CQ.CQMesh[i];
			}

			if (std::isnan(Output[i]))
				Output[i] = 0;
		}

	}


	unsigned int GetH5StackSize(std::string Filename, std::string Dataset)
	{
		H5::H5File file(Filename, H5F_ACC_RDONLY);
		H5::DataSet dataset = file.openDataSet(Dataset);

		H5::DataSpace DS = dataset.getSpace();

		hsize_t dims[4];
		DS.getSimpleExtentDims(dims, NULL);

		return (unsigned int)dims[0];
	}

	void Run_AutoCorr_DataEval(Settings & PrgSettings, CreateDataEval_Settings EvalSettings)
	{
		ProfileTime ProfileLevel_0;
		ProfileTime ProfileLevel_1;
		Detector Det;
		unsigned int StackSize = 0;
		unsigned int lowerBound = 0;
		unsigned int upperBound = 0;

		// <Output Field> (used on demand)
		AC1D Vector_AC;

		ACMesh Mesh_CurrAC(&PrgSettings);
		ACMesh Mesh_CQ(&PrgSettings);
		// </Output Field>

		//Initialize Stuff
		{
			//Load PixelMap and PixelMask
			{
				if (EvalSettings.PixelMap_Path == "")
				{
					std::cerr << "ERROR: No pixel map path set => can't load Hit Events\n";
					std::cerr << "    -> in Run_AutoCorr_DataEval()\n";
					throw;
				}
				if (EvalSettings.EchoLevel > 0)
					std::cout << "Load pixel map\n";

				//load pixelmap and if existing pixel mask
				Det.LoadPixelMap(EvalSettings.PixelMap_Path, EvalSettings.PixelMap_DataSet);
				if (EvalSettings.DetDisturb) //Disturbation of Pixelmap for simulated data
				{
					Simulator Sim;
					Sim.DisturbePixelMap(Det, EvalSettings.DetDisturb_Shift, EvalSettings.DetDisturb_Rot);
				}
				Det.Calc_kMap();
				Det.LoadPixelMask(EvalSettings.PixelMask_Path, EvalSettings.PixelMask_Dataset);
			}



			if (EvalSettings.XML_Path == "")
			{
				std::cerr << "ERROR: No XML path set => can't load Hit Events\n";
				std::cerr << "    -> in Run_AutoCorr_DataEval()\n";
				throw;
			}
			if (EvalSettings.EchoLevel > 0)
				std::cout << "Load Hit Event List\n";

			//Load Hit event list from xml-file (and invert Rot-matrix if requested)
			{
				PrgSettings.LoadHitEventListFromFile(EvalSettings.XML_Path);

				if (EvalSettings.InvertRotMatrix)
				{
					if (EvalSettings.EchoLevel > 0)
						std::cout << "Invert rotation matrices from Hit_Event_List\n";
					PrgSettings.InvertRotationMatrices();
				}

				if (EvalSettings.RestrictStackToBoundaries)
				{
					if (EvalSettings.UpperBoundary > PrgSettings.HitEvents.size())
					{
						std::cerr << "ERROR: Upper Hit boundary exeeds number of hits in loaded XML file\n";
						std::cerr << "    -> in Run_AutoCorr_DataEval()\n";
						throw;
					}
					else
					{
						StackSize = EvalSettings.UpperBoundary - EvalSettings.LowerBoundary;
						lowerBound = EvalSettings.LowerBoundary;
						upperBound = EvalSettings.UpperBoundary;
					}
				}
				else
				{
					StackSize = PrgSettings.HitEvents.size();
					lowerBound = 0;
					upperBound = StackSize;
				}
			}
		
		}
		//check if fractional C(q) eval is required
		if (EvalSettings.FractionalCq)
		{
			int Fractions_Num = StackSize / EvalSettings.SizeOfCqFraction;
			
			//To Implement
			std::cerr << "Fractional C(q) is not implemented yet\n";
			throw; //Implement remaining stuff
		}
		else
		{
			//Load or Create averaged intensity
			if (EvalSettings.UseExistingAvInt)
			{
				PrgSettings.Echo("Load existing averaged intensity");
				//Load avIntensity
				delete[] Det.Intensity;
				Det.Intensity = new float[Det.DetectorSize[0] * Det.DetectorSize[1]]();
				ArrayOperators::LoadArrayFromFile(EvalSettings.Out_AvIntensity_Path, Det.Intensity, Det.DetectorSize[0] * Det.DetectorSize[1]);
				//Apply Pixelmask
				Det.ApplyPixelMask();
			}
			else
			{
				//Create avIntensity
				Det.LoadAndAverageIntensity(PrgSettings.HitEvents, EvalSettings.PhotonOffset, EvalSettings.PhotonStep, lowerBound, upperBound, true);
				
				if (EvalSettings.Out_AvIntensity_Path != "")//save averaged intensity
				{
					ArrayOperators::SafeArrayToFile(EvalSettings.Out_AvIntensity_Path, Det.Intensity, Det.DetectorSize[0] * Det.DetectorSize[1], ArrayOperators::Binary);
					if (EvalSettings.EchoLevel > 0)
						std::cout << "-> Averaged intensity saved as \"" << EvalSettings.Out_AvIntensity_Path << "\".\n";
				}
				//Apply Pixelmask
				Det.ApplyPixelMask();
			}


			if (EvalSettings.AngularAveraged)// Angular Averaged, 1D Mode
			{
				Settings::Interpolation AvInterpol = Settings::Interpolation::Linear;

				ProfileLevel_0.Tic();
				//initialize with specified size
				Vector_AC.Initialize(Det, EvalSettings.MeshSize,EvalSettings.QZoom);
				Vector_AC.EchoLevel = EvalSettings.EchoLevel;

				// <C(q)>
				if (EvalSettings.EchoLevel > 0)
					std::cout << "Calculate C(q) - vector\n";
				Vector_AC.Calculate_CQ(Det, PrgSettings, AvInterpol);

				//save C(q)
				if (EvalSettings.Out_Cq_Path != "")
				{
					ArrayOperators::SafeArrayToFile(EvalSettings.Out_Cq_Path, Vector_AC.CQ, Vector_AC.Shape.Size, ArrayOperators::FileType::Binary);
					if (EvalSettings.EchoLevel > 0)
					{
						std::cout << " -> Saved angular averaged C(q) as: \"" << EvalSettings.Out_Cq_Path << "\"\n";
						std::cout << " --> Vetor Length = " << Vector_AC.Shape.Size << "\n";
					}
				}
				// </C(q)>

				// <AC uw>
				if (EvalSettings.EchoLevel > 0)
					std::cout << "Calculate unweighted AC - vector\n";
				
				ProfileLevel_1.Tic();

				bool RememberEchoLevel = PrgSettings.echo;
				if (EvalSettings.EchoLevel < 1)
					PrgSettings.echo = false;
				Vector_AC.Calculate_AC_UW_MR(PrgSettings, Det, AvInterpol, EvalSettings.PhotonOffset, EvalSettings.PhotonStep);
				PrgSettings.echo = RememberEchoLevel;

				if (EvalSettings.EchoLevel > 0)
				{
					std::cout << "Calculated unweighted AC - vector in\n";
					ProfileLevel_1.Toc(true);
				}
				//save Results
				if (EvalSettings.Out_ACuw_Path != "")
				{
					ArrayOperators::SafeArrayToFile(EvalSettings.Out_ACuw_Path, Vector_AC.AC_UW, Vector_AC.Shape.Size, ArrayOperators::FileType::Binary);
					if (EvalSettings.EchoLevel > 0)
					{
						std::cout << " -> Saved angular averaged unweighted AC as: \"" << EvalSettings.Out_ACuw_Path << "\"\n";
						std::cout << " --> Vetor Length = " << Vector_AC.Shape.Size << "\n";
					}
				}
				// </AC uw>

				// <AC>
				Vector_AC.CalcAC();
				if (EvalSettings.Out_Final_AC_Path != "")
				{
					ArrayOperators::SafeArrayToFile(EvalSettings.Out_Final_AC_Path, Vector_AC.AC, Vector_AC.Shape.Size, ArrayOperators::FileType::Binary);
					if (EvalSettings.EchoLevel > 0)
					{
						std::cout << " -> Saved angular Final AC as: \"" << EvalSettings.Out_Final_AC_Path << "\"\n";
						std::cout << " --> Vetor Length = " << Vector_AC.Shape.Size << "\n";
					}
				}

				// </AC>

				// <Q Vector>
				if (EvalSettings.Out_Q_Vector != "")
				{
					Vector_AC.CreateQVector();
					ArrayOperators::SafeArrayToFile(EvalSettings.Out_Q_Vector, Vector_AC.Q, Vector_AC.Shape.Size, ArrayOperators::FileType::Binary);
					if (EvalSettings.EchoLevel > 0)
					{
						std::cout << " -> Saved Q Vector as: \"" << EvalSettings.Out_Q_Vector << "\"\n";
						std::cout << " --> Vetor Length = " << Vector_AC.Shape.Size << "\n";
					}
				}
				// </Q Vector>
				if (EvalSettings.EchoLevel > 0)
				{
					std::cout << "\n--------------------------\n Evaluation done in\n";
					ProfileLevel_0.Toc(true);
				}
			}
			else //   3D Mode
			{
				ProfileLevel_0.Tic();
				if (EvalSettings.EchoLevel > 0)
					std::cout << "Start dense autocorrelating for small C(q)\n";

				// <C(q)>
				// -> <C(q)_small>
				ProfileLevel_1.Tic();
				ACMesh smallMesh(&PrgSettings);
				smallMesh.CreateSmallMesh_CofQ_ForDetector(Det, EvalSettings.MeshSize, EvalSettings.QZoom);

				Detector::AutoCorrFlags Flags;
				Flags.InterpolationMode = Settings::NearestNeighbour;
				Det.AutoCorrelate_CofQ_SmallMesh(smallMesh, Flags, PrgSettings);

				//save small C(q)
				if (EvalSettings.Out_Cq_small_Path != "")
				{
					ArrayOperators::SafeArrayToFile(EvalSettings.Out_Cq_small_Path, smallMesh.CQMesh, smallMesh.Shape.Size_AB*smallMesh.Shape.Size_AB*smallMesh.Shape.Size_C, ArrayOperators::FileType::Binary);
					if (EvalSettings.EchoLevel > 0)
					{
						std::cout << "-> C(q) small Mesh saved as \"" << EvalSettings.Out_Cq_small_Path << "\"\n";
						std::cout << "--> Mesh shape: " << smallMesh.Shape.Size_AB << " x " << smallMesh.Shape.Size_AB << " x " << smallMesh.Shape.Size_C << "\n";
					}
				}
				if (EvalSettings.EchoLevel > 0)
				{
					std::cout << "C(q) small Mesh calculation done in\n";
					ProfileLevel_1.Toc(true);
				}
				// -> </C(q)_small>

				// -> <Merge CQ>
				ProfileLevel_1.Tic();
				if (EvalSettings.EchoLevel > 0)
					std::cout << "Weight and merge C(q)\n";

				Mesh_CQ.CreateBigMesh_CofQ_ForDetector(Det, EvalSettings.MeshSize, EvalSettings.QZoom);

				Det.Merge_smallCofQ(Mesh_CQ, smallMesh, PrgSettings.HitEvents, lowerBound, upperBound, PrgSettings, Flags);

				//save big C(q)
				if(EvalSettings.Out_Cq_Path != "")
				{
					ArrayOperators::SafeArrayToFile(EvalSettings.Out_Cq_Path, Mesh_CQ.CQMesh, Mesh_CQ.Shape.Size_AB*Mesh_CQ.Shape.Size_AB*Mesh_CQ.Shape.Size_C, ArrayOperators::FileType::Binary);
					if (EvalSettings.EchoLevel > 0)
					{
						std::cout << "-> C(q) Mesh (merged) saved as \"" << EvalSettings.Out_Cq_Path << "\"\n";
						std::cout << "--> Mesh shape: " << Mesh_CQ.Shape.Size_AB << " x " << Mesh_CQ.Shape.Size_AB << " x " << Mesh_CQ.Shape.Size_C << "\n";
					}
				}
				if (EvalSettings.EchoLevel > 0)
				{
					std::cout << "C(q) merge and weight done in\n";
					ProfileLevel_1.Toc(true);
				}
				// -> </Merge CQ>
				// </C(q)>

				// <AC uw>
				ProfileLevel_1.Tic();
				ACMesh AC_uw(&PrgSettings);
				//Setup  Correlation
				AC_uw.CreateBigMeshForDetector(Det, EvalSettings.MeshSize, EvalSettings.QZoom);

				if (EvalSettings.EchoLevel > 0)
					std::cout << "Start auto correlation for sparsificated intensities\n";

				Det.LoadIntensityData(&PrgSettings.HitEvents[0]);
				for (unsigned int i = lowerBound; i < upperBound; i++)
				{
					if (EvalSettings.EchoLevel > 1)
						std::cout << "AC event " << i << "/" << (upperBound) << std::endl;

					Det.LoadIntensityData(&PrgSettings.HitEvents[i]);

					ArrayOperators::ParMultiplyElementwise(Det.Intensity, Det.PixelMask, Det.DetectorSize[0] * Det.DetectorSize[1]);

					Det.CreateSparseHitList(EvalSettings.PhotonOffset, EvalSettings.PhotonStep); //Sparsificate
				
					if (EvalSettings.EchoLevel > 2)
						std::cout << i << ": Pixels with hits: " << Det.SparseHitList.size()*100.0 / (Det.DetectorSize[0] * Det.DetectorSize[1]) << "%" << "    Mean intensity: " << PrgSettings.HitEvents[i].MeanIntensity << "\n";

					Det.AutoCorrelateSparseList(AC_uw, Flags, EvalSettings.DoubleMap, PrgSettings);
				}
				//Save
				if (EvalSettings.Out_ACuw_Path != "")
				{
					double* ACMesh = new double[AC_uw.Shape.Size_AB*AC_uw.Shape.Size_AB*AC_uw.Shape.Size_C]();
					#pragma omp parallel for
					for (int i = 0; i < AC_uw.Shape.Size_AB*AC_uw.Shape.Size_AB*AC_uw.Shape.Size_C; i++)
					{
						ACMesh[i] = PrgSettings.IntToFloat(AC_uw.Mesh[i]);
					}
					ArrayOperators::SafeArrayToFile(EvalSettings.Out_ACuw_Path, ACMesh, AC_uw.Shape.Size_AB*AC_uw.Shape.Size_AB*AC_uw.Shape.Size_C, ArrayOperators::FileType::Binary);

					if (EvalSettings.EchoLevel > 0)
					{
						std::cout << "-> Saved AC_uw (formated as double) as \"" << EvalSettings.Out_ACuw_Path << "\"\n";
						std::cout << "--> Mesh shape: " << AC_uw.Shape.Size_AB << " x " << AC_uw.Shape.Size_AB << " x " << AC_uw.Shape.Size_C << "\n";
					}
					delete[] ACMesh;
				}
				if (EvalSettings.EchoLevel > 0)
				{
					std::cout << "AC unweighted calculated in\n";
					ProfileLevel_1.Toc(true);
				}
				// </AC uw>

				// <Apply C(q) to AC_uw>
				if (EvalSettings.EchoLevel > 0)
					std::cout << "Apply C(q) to unweighted AC\n";
				double * FinalAC = nullptr;
				Merge_ACandCQ(FinalAC, AC_uw, Mesh_CQ, PrgSettings);

				if (EvalSettings.Out_Final_AC_Path != "")
				{
					ArrayOperators::SafeArrayToFile(EvalSettings.Out_Final_AC_Path, FinalAC, AC_uw.Shape.Size_AB*AC_uw.Shape.Size_AB*AC_uw.Shape.Size_AB, ArrayOperators::FileType::Binary);
					
					if (EvalSettings.EchoLevel > 0)
					{
						std::cout << "-> Saved AC (Final form, formated as double) as \"" << EvalSettings.Out_Final_AC_Path << "\"\n";
						std::cout << "--> Mesh shape: " << AC_uw.Shape.Size_AB << " x " << AC_uw.Shape.Size_AB << " x " << AC_uw.Shape.Size_C << "\n";
					}
				}
				else
				{
					std::cout << "WARNING: no path set for final AC => Results have not been saved!\n";
				}
				delete[] FinalAC;
				// </Apply C(q) to AC_uw>

				if (EvalSettings.EchoLevel > 0)
				{
					std::cout << "\n--------------------------\n Evaluation done in\n";
					ProfileLevel_0.Toc(true);
				}
				//end of eval

				return;
			}

		}

		//end of method
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
	void Load_and_average_Intensities(Settings &Options, Detector &Det, float PhotonThreshold, float PhotonStep, std::string XML_Filename, std::string AvInt_Filename, bool UpdateXML)
	{
		ProfileTime profiler;
		Options.Echo("Load and average intensities (all)");
		profiler.Tic();
		Det.LoadAndAverageIntensity(Options.HitEvents, PhotonThreshold, PhotonStep, true);

		//Det.LoadAndAverageIntensity(Options.HitEvents, PhotonThreshold, PhotonStep);

		profiler.Toc(true);
		std::cout << "done.\n";

		if (UpdateXML)
		{
			Options.Echo("Save EventList as XML");
			Options.SafeHitEventListToFile(XML_Filename);
		}
		Options.Echo("Save averaged Intensity");
		ArrayOperators::SafeArrayToFile(AvInt_Filename, Det.Intensity, Det.DetectorSize[0] * Det.DetectorSize[1], ArrayOperators::FileType::Binary);

	}


	void GeneratePixelMapForSimulator(std::string Filename, std::string Dataset, double Distance, double PixelSize, int SizeFS, int SizeSS, std::array<float, 3> Orientation)
	{
		std::cerr << "To Implement.\n";
		throw;


		double * PixelMap;
		PixelMap = new double[SizeFS * SizeSS];


		double OrientNormLength;



		delete[] PixelMap;
	}


}