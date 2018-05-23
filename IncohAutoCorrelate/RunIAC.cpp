#include <iostream>
#include <omp.h>

#include "RunIAC.h"
#include "Detector.h"
#include "ArrayOperators.h"
#include"ProfileTime.h"

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
			for (int i = 0; i <  Det.DetectorSize[0] * Det.DetectorSize[1]; i++)
			{
				Det.PixelMask[i] = 1;
			}
		}

		//load integrated/averaged intensity
		Det.Intensity = new float[Det.DetectorSize[0] * Det.DetectorSize[1]];
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
		Run_AC_UW(AC,  AC_Settings,  PrgSettings, 0, PrgSettings.HitEvents.size());
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
		Det.PixelMask = new int[Det.DetectorSize[0] * Det.DetectorSize[1]]();
		if (AC_Settings.PixelMask_Path != "")
		{
			ArrayOperators::LoadArrayFromFile<int>(AC_Settings.PixelMask_Path, Det.PixelMask, Det.DetectorSize[0] * Det.DetectorSize[1]);
		}
		else
		{
			// No Pixelmask given => set every entry to one
			#pragma omp parallel for
			for (int i = 0; i < Det.DetectorSize[0] * Det.DetectorSize[1]; i++)
			{
				Det.PixelMask[i] = 1;
			}
		}

		//Setup  Correlation
		AC.CreateBigMeshForDetector(Det, AC_Settings.MeshSize, AC_Settings.QZoom);
		AC.Options = &PrgSettings;


		if (AC_Settings.echo)
			std::cout << AC_Settings.ThreadName << ": Start dense autocorrelating (small Mesh) ...\n";

		for (int i = LowerBound; i < UpperBound; i++)
		{
			if (i % AC_Settings.StatusEcho == 0 && AC_Settings.echo)
				std::cout << AC_Settings.ThreadName << ": AC Event "<<i << "/" << (UpperBound) << std::endl;


			Det.LoadIntensityData(&PrgSettings.HitEvents[i]);
			ArrayOperators::ParMultiplyElementwise(Det.Intensity, Det.PixelMask, 1024 * 1024);
			Det.CreateSparseHitList(AC_Settings.PhotonOffset, AC_Settings.PhotonStep);
			
			Det.AutoCorrelateSparseList(AC, AC_Settings.AC_FirstMap_Flags, AC_Settings.DoubleMap);
		}

		


		if (AC_Settings.echo)
		{
			std::cout << AC_Settings.ThreadName << ": All events autocorrelated in ";
			profiler.Toc(true);
			std::cout << "\n";
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

	void Merge_ACandCQ(double *& Output, ACMesh AC, ACMesh CQ, Settings & PrgSettings)
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
	void Merge_ACandCQ(double *& Output, double * AC, ACMesh CQ, Settings & PrgSettings)
	{
		unsigned int Size = CQ.Shape.Size_AB*CQ.Shape.Size_AB*CQ.Shape.Size_C;

		Output = new double[Size]();

		#pragma omp parallel for
		for (unsigned int i = 0; i <Size; i++)
		{
			if (CQ.CQMesh[i] <= 0)
				Output[i] = 0;

			Output[i] = AC[i] / CQ.CQMesh[i];
		}

	}
}