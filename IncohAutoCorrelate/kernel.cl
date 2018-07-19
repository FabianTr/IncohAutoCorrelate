#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

__constant double PI = 3.1415926535897932;

// Atomic add of floats or double
// Adapted from:
// https://streamcomputing.eu/blog/2016-02-09/atomic-operations-for-floats-in-opencl-improved/
inline void atomic_add_float(__global double *source, const double value) {
	union {
		ulong int_val;
		double float_val;
	} next, expected, current;
	current.float_val = *source;
	do {
		expected.float_val = current.float_val;
		next.float_val = expected.float_val + value;
		current.int_val = atomic_cmpxchg((volatile __global ulong *)source, expected.int_val, next.int_val);
	} while (current.int_val != expected.int_val);
}



//__kernel void AutoCorr_CQ(__global const float *IntensityData,
//	__global const float *KMap,
//	__global const float *Rotations_and_Weights,
//	__global const double *Params,
//	__global double *CQ)
//{
//	unsigned int ind = get_global_id(0);
//	//double Params[5];
//	//Params[0] = DetSize; //Numer of pixels (size[0]*size[1])
//	//Params[1] = deltaV; //dq per Voxel
//	//Params[2] = MeshSize; //  a  (V = a*a*(a+1)/2)
//	//Params[3] = NumberOfEvents; // how many events, for roataion and weight loop
//	//Params[4] = InterpolationMode;
//
//	unsigned int DetSize = (unsigned int)Params[0];
//	float dqPerVox = (float)Params[1];
//	unsigned int MeshSize = (unsigned int)Params[2];
//	unsigned int NumEvents = (unsigned int)Params[3];
//	int InterpolMode = (unsigned int)Params[4];
//
//
//	////Debug Bullshit
//	//if (ind == 0)//ind == 0
//	//{
//	//	printf("Kernel is alive\n");
//	//	printf("Detector Size: %d\n", DetSize);
//	//	printf("Number of Events: %d\n", NumEvents);
//	//	printf("Interpolation mode: %d\n", InterpolMode);
//	//	printf("Mesh Size: %d\n", MeshSize);
//	//	printf("dq per Vox: %f\n", dqPerVox);
//	//}
//	////END
//		//local Variables
//	float k1[3];
//	float k2[3];
//	float q_RAW[3];
//
//	k1[0] = KMap[0 + 3 * ind];
//	k1[1] = KMap[1 + 3 * ind];
//	k1[2] = KMap[2 + 3 * ind];
//
//	float q[3];
//
//
//	for (unsigned int i = 0; i < DetSize; ++i) //Loop over all Pixel
//	{
//		//if (i == ind) //exclude zeroth peak
//		//{
//		//	continue;
//		//}
//
//		float Val_RAW = IntensityData[ind] * IntensityData[i];
//		k2[0] = KMap[0 + 3 * i];
//		k2[1] = KMap[1 + 3 * i];
//		k2[2] = KMap[2 + 3 * i];
//		q_RAW[0] = k1[0] - k2[0];
//		q_RAW[1] = k1[1] - k2[1];
//		q_RAW[2] = k1[2] - k2[2];
//
//		int MeshCenter = (MeshSize - 1) / 2;
//
//
//		for (unsigned int j = 0; j < NumEvents; ++j)//Loop over all Events
//		{
//			//rotate
//			q[0] = q_RAW[0] * Rotations_and_Weights[0 + 10 * j] + q_RAW[1] * Rotations_and_Weights[1 + 10 * j] + q_RAW[2] * Rotations_and_Weights[2 + 10 * j];
//			q[1] = q_RAW[0] * Rotations_and_Weights[3 + 10 * j] + q_RAW[1] * Rotations_and_Weights[4 + 10 * j] + q_RAW[2] * Rotations_and_Weights[5 + 10 * j];
//			q[2] = q_RAW[0] * Rotations_and_Weights[6 + 10 * j] + q_RAW[1] * Rotations_and_Weights[7 + 10 * j] + q_RAW[2] * Rotations_and_Weights[8 + 10 * j];
//			// HALF MESH:
//			// Meshsize: [-fs/2, fs/2][-ms/2, ms/2],[0, ss/2] needed to save memory (1000x1000x500*8Byte ~= 4GB)
//			if (q[2] < 0) //needs to be mirrowed in postprocessing on the CPU!
//			{
//				continue;
//			}
//			//weight (by square of mean intensity)
//			float Val = Val_RAW * Rotations_and_Weights[9 + 10 * j] * Rotations_and_Weights[9 + 10 * j];
//
//			//resize (for Mapping)
//			q[0] = q[0] / dqPerVox;
//			q[1] = q[1] / dqPerVox;
//			q[2] = q[2] / dqPerVox;
//
//			//Map to Mesh
//
//			if (InterpolMode == 0) //nearest Neighbor
//			{
//				unsigned int fs, ms, ss;//fast-scan-, medium-scan-, slow-scan- floor
//				fs = (unsigned int)floor(q[0] + 0.5) + MeshCenter;
//				ms = (unsigned int)floor(q[1] + 0.5) + MeshCenter;
//				ss = (unsigned int)floor(q[2] + 0.5);
//
//				atomic_add_float(&(CQ[fs + ms * MeshSize + ss * MeshSize*MeshSize]), Val);
//			}
//			if (InterpolMode == 1) //linear
//			{
//				unsigned int fsf, msf, ssf;//fast-scan-, medium-scan-, slow-scan- floor
//				fsf = (unsigned int)floor(q[0]) + MeshCenter;
//				msf = (unsigned int)floor(q[1]) + MeshCenter;
//				ssf = (unsigned int)floor(q[2]);
//				float SepF, SepM, SepS; //according Seperators
//				SepF = q[0] - floor(q[0]);
//				SepM = q[1] - floor(q[1]);
//				SepS = q[2] - floor(q[2]);
//
//				//add to mesh (8 entries each)
//				atomic_add_float(&(CQ[fsf + msf * MeshSize + ssf * MeshSize*MeshSize]), Val * (1 - SepF)*(1 - SepM)*(1 - SepS));              //A + 0
//				atomic_add_float(&(CQ[(fsf + 1) + (msf + 0) * MeshSize + (ssf + 0) * MeshSize*MeshSize]), Val * (SepF)*(1 - SepM)*(1 - SepS));//ssf + 1
//				atomic_add_float(&(CQ[(fsf + 0) + (msf + 1) * MeshSize + (ssf + 0) * MeshSize*MeshSize]), Val * (1 - SepF)*(SepM)*(1 - SepS));//msf + 1
//				atomic_add_float(&(CQ[(fsf + 0) + (msf + 0) * MeshSize + (ssf + 1) * MeshSize*MeshSize]), Val * (1 - SepF)*(1 - SepM)*(SepS));//ssf + 1
//				atomic_add_float(&(CQ[(fsf + 1) + (msf + 1) * MeshSize + (ssf + 0) * MeshSize*MeshSize]), Val * (SepF)*(SepM)*(1 - SepS));    //ffs + 1 ; msf + 1
//				atomic_add_float(&(CQ[(fsf + 0) + (msf + 1) * MeshSize + (ssf + 1) * MeshSize*MeshSize]), Val * (1 - SepF)*(SepM)*(SepS));    //msf + 1 ; ssf + 1
//				atomic_add_float(&(CQ[(fsf + 1) + (msf + 0) * MeshSize + (ssf + 1) * MeshSize*MeshSize]), Val * (SepF)*(1 - SepM)*(SepS));    //ffs + 1 ; ssf + 1
//				atomic_add_float(&(CQ[(fsf + 1) + (msf + 1) * MeshSize + (ssf + 1) * MeshSize*MeshSize]), Val * (SepF)*(SepM)*(SepS));        // A + 1
//			}
//
//
//
//		}
//	}
//}


//Autocorrelation for C(q)
__kernel void AutoCorr_CQ_small(__global const float *IntensityData,
	__global const float *KMap,
	__global const double *Params,
	__global unsigned long *CQsmall)
{
	unsigned int ind = get_global_id(0);
	//Params[0] = DetectorSize[0] * DetectorSize[1]; //Numer of pixels (size[0]*size[1])
	//Params[1] = SmallMesh.Shape.dq_per_Voxel; //dq per Voxel
	//Params[2] = SmallMesh.Shape.Size_AB; // Size perp
	//Params[3] = SmallMesh.Shape.Size_C; // Size C
	//Params[4] = SmallMesh.Shape.k_A; // Dimension Alignment
	//Params[5] = SmallMesh.Shape.k_B; // Dimension Alignment 
	//Params[6] = SmallMesh.Shape.k_C; // Dimension Alignment 
	//Params[7] = Flags.InterpolationMode;
	//Params[9] = Multiplicator; //Multiplicator for conversion to long

	unsigned int DetSize = (unsigned int)Params[0];
	float dqPerVox = (float)Params[1];
	unsigned int MeshSizeAB = (unsigned int)Params[2];
	unsigned int MeshSizeC = (unsigned int)Params[3];

	int Aind = (int)Params[4];
	int Bind = (int)Params[5];
	int Cind = (int)Params[6];

	int InterpolMode = (unsigned int)Params[7];

	float MaxQ = (float)Params[8];
	double Multiplicator = (double)Params[9];


	////Debug Bullshit
	if (ind == 0)//ind == 0
	{
		printf("Kernel is alive\n");
		printf("Detector Size: %d\n", DetSize);
		printf("Aind: %d\n", Aind);
		printf("Bind: %d\n", Bind);
		printf("Cind: %d\n", Cind);
		//printf("dq per Vox: %f\n", dqPerVox);
		//printf("Max q: %f\n", MaxQ);
	}
	////END


	//local Variables

	float k1[3];
	float k2[3];

	k1[0] = KMap[0 + 3 * ind];
	k1[1] = KMap[1 + 3 * ind];
	k1[2] = KMap[2 + 3 * ind];

	float q1[3];
	

	int MeshCenterAB = (MeshSizeAB - 1) / 2;
	int MeshCenterC = (MeshSizeC - 1) / 2;

	float INT_ind = IntensityData[ind];





	//int N = 10;
	//int n = (ind*777)%N ;
	//int i = n;
	//do{


	for (int i = 0; i < DetSize; i++) //Loop over all Pixel //< DetSize
	{
		if (i == ind) //exclude zeroth peak need to be set to infinity afterwards
		{
			continue;
		}

		double Val = INT_ind * IntensityData[i];

		if (Val < 1e-37f) //no entry shortcut
		{
			continue;
		}

		unsigned long ValConv = 0;
		ValConv = (unsigned long)(Val*Multiplicator);
		


		k2[0] = KMap[0 + 3 * i];
		k2[1] = KMap[1 + 3 * i];
		k2[2] = KMap[2 + 3 * i];
		q1[0] = (k1[0] - k2[0]);
		q1[1] = (k1[1] - k2[1]);
		q1[2] = (k1[2] - k2[2]);

		//q2[0] = (k2[0] - k1[0]);
		//q2[1] = (k2[1] - k1[1]);
		//q2[2] = (k2[2] - k1[2]);

		if (sqrt(q1[0] * q1[0] + q1[1] * q1[1] + q1[2] * q1[2]) > MaxQ)
		{
			continue;
		}

		q1[0] = q1[0] / dqPerVox;
		q1[1] = q1[1] / dqPerVox;
		q1[2] = q1[2] / dqPerVox;


		//Map to Mesh
		unsigned int fs, ms, ss;//fast-scan, medium-scan, slow-scan
		fs = (unsigned int)(floor(q1[Aind] + 0.5) + MeshCenterAB);
		ms = (unsigned int)(floor(q1[Bind] + 0.5) + MeshCenterAB);
		ss = (unsigned int)(floor(q1[Cind] + 0.5) + MeshCenterC);
		atomic_add(&(CQsmall[fs + ms * MeshSizeAB + ss * MeshSizeAB * MeshSizeAB]), ValConv);

		if (ss < 0 || ms < 0 || fs < 0 || ss >= MeshSizeC || ms >= MeshSizeAB || fs >= MeshSizeAB)//Check scans and display overflows.
		{
			printf("ME: scans:%d, %d, %d   q: %f, %f, %f\n", ss, ms, fs, q1[0], q1[1], q1[2]);
		}
	}
}


//Merge small C(q) to one big Mesh (Merge and Weight)
__kernel void Merge_CQ(__global const double *smallMesh,
	__global const float *RW,
	__global const double *Params,
	__global unsigned long *CQ)
{
	unsigned int ind = get_global_id(0);
	//Params[0] = SmallMesh.Shape.dq_per_Voxel; //dq per Voxel
	//Params[1] = SmallMesh.Shape.Size_AB; // Size perp
	//Params[2] = SmallMesh.Shape.Size_C; // Size C
	//Params[3] = SmallMesh.Shape.k_A; // Dimension Alignment
	//Params[4] = SmallMesh.Shape.k_B; // Dimension Alignment 
	//Params[5] = SmallMesh.Shape.k_C; // Dimension Alignment 
	//Params[6] = Flags.InterpolationMode;
	//Params[7] = Multiplicator; //Multiplicator for conversion to long
	//Params[8] = UpperBound - LowerBound; //Number of Rotations

	int smallMeshSize_AB = (int)Params[1];
	int smallMeshSize_C = (int)Params[2];

	int smallMeshCenterAB = (smallMeshSize_AB - 1) / 2;
	int smallMeshCenterC = (smallMeshSize_C - 1) / 2;

	int InterpolMode = (int)Params[6];

	unsigned int Aind = (unsigned int)Params[3];
	unsigned int Bind = (unsigned int)Params[4];
	unsigned int Cind = (unsigned int)Params[5];

	double Multiplicator = (double)Params[7];

	unsigned int ListSize = (unsigned int)Params[8];


	//Debug Bullshit
	if (ind == 0)//ind == 0
	{
		printf("Kernel is alive\n");
		printf("MeshShape AB: %d\n", smallMeshSize_AB);
		printf("MeshShape  C: %d\n", smallMeshSize_C);

		printf("Alingment: %d, %d, %d\n", Aind, Bind, Cind);
		printf("Multiplicator: %f\n", Multiplicator);
		printf("Listsize: %d\n", ListSize);
	}
	//END


	float q_in[3];
	int ss_in = 0, ms_in = 0, fs_in = 0;
	//retrive scans from index
	ss_in = ind / (smallMeshSize_AB*smallMeshSize_AB);
	ms_in = (ind - ss_in * smallMeshSize_AB*smallMeshSize_AB) / (smallMeshSize_AB);
	fs_in = ind - ss_in * smallMeshSize_AB*smallMeshSize_AB - ms_in * smallMeshSize_AB;
	//retrive q vectros (in voxels) fromm small C(q) mesh
	q_in[Aind] = fs_in - smallMeshCenterAB;
	q_in[Bind] = ms_in - smallMeshCenterAB;
	q_in[Cind] = ss_in - smallMeshCenterC;


	for (unsigned int i = 0; i < ListSize; i++)
	{
		float q_out[3];
		//Rotate Vector
		q_out[0] = RW[10 * i + 0] * q_in[0] + RW[10 * i + 1] * q_in[1] + RW[10 * i + 2] * q_in[2];
		q_out[1] = RW[10 * i + 3] * q_in[0] + RW[10 * i + 4] * q_in[1] + RW[10 * i + 5] * q_in[2];
		q_out[2] = RW[10 * i + 6] * q_in[0] + RW[10 * i + 7] * q_in[1] + RW[10 * i + 8] * q_in[2];




		//weight entry and transform to int64
		double Val_in = smallMesh[ind];

		if (Val_in < 1e-37f)
		{
			continue;
		}

		Val_in = Val_in * Multiplicator * (double)RW[10 * i + 9];
		unsigned long Val_out = (unsigned long)Val_in;




		if (InterpolMode == 0) //Nearest Neighbour
		{
			//get new scans
			int fs = 0, ms = 0, ss = 0;
			fs = (int)(floor(q_out[0] + 0.5) + smallMeshCenterAB);
			ms = (int)(floor(q_out[1] + 0.5) + smallMeshCenterAB);
			ss = (int)(floor(q_out[2] + 0.5) + smallMeshCenterAB);//ss = (unsigned int)(floor(q_out[2] + 0.5);


			if (ss < 0 || ms < 0 || fs < 0 || ss >= smallMeshSize_AB || ms >= smallMeshSize_AB || fs >= smallMeshSize_AB)
			{
				printf("ME. Scan: %d, %d, %d; q: %f, %f, %f; ind: %d; Scan in: %d, %d, %d;   qin: %f, %f, %f; i:%d;\n", ss, ms, fs, q_out[0], q_out[1], q_out[2], ind, ss_in, ms_in, fs_in, q_in[0], q_in[1], q_in[2], i);
			
			//	continue;
			}

			//if(fs >= smallMeshSize_AB || ms >= smallMeshSize_AB || ss >= )

			atomic_add(&(CQ[fs + ms * smallMeshSize_AB + ss * smallMeshSize_AB * smallMeshSize_AB]), Val_out);

		}
		if (InterpolMode == 1) //Linear
		{
			printf("ERROR: Linear Interpolation not implemented yet\n");
		}

	}
}


//Autocorrelates sparse Hit List (analog to CPU implementation)
__kernel void Autocor_sparseHL(__global const float *SparseHitList,
	__global const double *Params,
	__global const float *RotMatrix,
	__global unsigned long *AC)
{
	unsigned int ind = get_global_id(0);


	unsigned int ListSize = (unsigned int)Params[0]; //Entrys in sparse HitList
	float dqdV = (float)Params[1]; // dq/dV

	int MeshSize = (unsigned int)Params[2]; //Only cube Meshes allowed here
	int MeshCenter = (MeshSize - 1) / 2;

	float MaxQ = (float)Params[4];
	int DoubleMapping = (int)Params[5]; //if 1, maps two times (before and after rotation)
	double Multiplicator = Params[6]; //for conversion float -> int

	//printf("MeshSize: %d;    MeshCenter: %d\n", MeshSize, MeshCenter);

	//obtain k-vector and value given by kernel index
	float k1[3];
	k1[0] = SparseHitList[4 * ind + 0];
	k1[1] = SparseHitList[4 * ind + 1];
	k1[2] = SparseHitList[4 * ind + 2];
	double f_Val = (double)SparseHitList[4 * ind + 3];

	//printf("ind: %d; k = [%f, %f, %f]; Val = %f\n",ind, k1[0], k1[1], k1[2],f_Val);

	//printf("RotMat: {{%f, %f, %f},{%f, %f, %f},{%f, %f, %f}}\n", RotMatrix[0], RotMatrix[1], RotMatrix[2], RotMatrix[3], RotMatrix[4], RotMatrix[5], RotMatrix[6], RotMatrix[7], RotMatrix[8]);

	for (unsigned int i = 0; i < ListSize; i++)
	{
		if (i == ind) // ignore q = 0
		{
			continue;
		}

		float q[3];
		q[0] = k1[0] - SparseHitList[4 * i + 0];
		q[1] = k1[1] - SparseHitList[4 * i + 1];
		q[2] = k1[2] - SparseHitList[4 * i + 2];


		if (sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2]) > MaxQ) //check if q is in range
		{
			continue;
		}

		double t_Val = f_Val * SparseHitList[4 * i + 3];
		long Val = (long)(t_Val * Multiplicator);


		//convert q to units of voxel
		q[0] = q[0] / dqdV;
		q[1] = q[1] / dqdV;
		q[2] = q[2] / dqdV;
		

		if(DoubleMapping == 1)
		{ 
			//map to mesh
			int fs_l, ms_l, ss_l;
			fs_l = (int)(floor(q[0] + 0.5) + MeshCenter);
			ms_l = (int)(floor(q[1] + 0.5) + MeshCenter);
			ss_l = (int)(floor(q[2] + 0.5) + MeshCenter);
			//printf("t_scan = [%d, %d, %d]\n", fs_l, ms_l, ss_l);
			//convert back to q
			q[0] = (float)(fs_l - MeshCenter);
			q[1] = (float)(ms_l - MeshCenter);
			q[2] = (float)(ss_l - MeshCenter);
		}
		//printf("q = [%f, %f, %f]\n", q[0], q[1], q[2]);

		//Rotation
		float t_q[3];
		t_q[0] = q[0] * RotMatrix[0] + q[1] * RotMatrix[1] + q[2] * RotMatrix[2];
		t_q[1] = q[0] * RotMatrix[3] + q[1] * RotMatrix[4] + q[2] * RotMatrix[5];
		t_q[2] = q[0] * RotMatrix[6] + q[1] * RotMatrix[7] + q[2] * RotMatrix[8];


		q[0] = t_q[0];
		q[1] = t_q[1];
		q[2] = t_q[2];

		//map to mesh
		int fs, ms, ss;
		fs = (int)(floor(q[0] + 0.5) + MeshCenter);
		ms = (int)(floor(q[1] + 0.5) + MeshCenter);
		ss = (int)(floor(q[2] + 0.5) + MeshCenter);


		atomic_add(&(AC[fs + ms * MeshSize + ss * MeshSize * MeshSize]), Val);


	}
}



//create C(q) for angular averaged stuff (1D)
__kernel void AutoCorr_CQ_AV(__global const float *IntensityData,
	__global const float *KMap,
	__global const double *Params,
	__global unsigned long *CQ)
{
	unsigned int ind = get_global_id(0);

	unsigned int DetSize = (unsigned int)Params[0];
	float dqPerVox = (float)Params[1];
	unsigned int ArraySize = (unsigned int)Params[2];
	int InterpolMode = (unsigned int)Params[3];
	float MaxQ = (float)Params[4];
	double Multiplicator = (double)Params[5];
	unsigned int MapAndReduce = (unsigned int)Params[6];



	unsigned int MaR_ScanAdd = ((ind*MapAndReduce) / (DetSize)) * ArraySize;

	//Debug Bullshit
	if (ind == 0 && Params[7] > 0.4)//ind == 0
	{
		printf("Kernel is alive\n");
		printf("Detector Size: %d\n", DetSize);
		printf("Interpolation mode: %d\n", InterpolMode);
		printf("ArraySize: %d\n", ArraySize);
		printf("Multiplicator: %f\n", Multiplicator);
		printf("MapAndReduce: %d\n", MapAndReduce);
		printf("Max q: %f\n", MaxQ);
	}
	//END


	//local Variables

	float k1[3];
	float k2[3];

	k1[0] = KMap[0 + 3 * ind];
	k1[1] = KMap[1 + 3 * ind];
	k1[2] = KMap[2 + 3 * ind];

	float q;
	float INT_ind = IntensityData[ind];


	for (int i = 0; i < DetSize; i++) //Loop over all Pixel //< DetSize
	{
		if (i == ind) //exclude zeroth peak need to be set to infinity afterwards
		{
			continue;
		}

		double Val = INT_ind * IntensityData[i];

		if (Val < 1e-37f) //no entry shortcut
		{
			continue;
		}



		k2[0] = KMap[0 + 3 * i];
		k2[1] = KMap[1 + 3 * i];
		k2[2] = KMap[2 + 3 * i];

		q = sqrt((k1[0] - k2[0]) * (k1[0] - k2[0])
			+ (k1[1] - k2[1]) * (k1[1] - k2[1])
			+ (k1[2] - k2[2]) * (k1[2] - k2[2]));




		//if (q > MaxQ)
		//{
		//	continue;
		//}

		q = q / dqPerVox;


		if (InterpolMode == 0) //nearest neighbor interpolation
		{
			long ValConv = 0;
			ValConv = (long)(Val*Multiplicator);

			unsigned int sc;
			sc = (unsigned int)(floor(q + 0.5)) + MaR_ScanAdd;

			atomic_add(&(CQ[sc]), ValConv);
		}
		if (InterpolMode == 1) //linear interpolation
		{
			unsigned int sc1, sc2;

			sc1 = (unsigned int)(floor(q)) + MaR_ScanAdd;
			sc2 = sc1 + 1;

			double Sep = q - (floor(q)); //separator

			long ValConv1 = 0;
			long ValConv2 = 0;
			ValConv1 = (long)(Val*(1 - Sep)*Multiplicator);
			ValConv2 = (long)(Val*(Sep)*Multiplicator);

			atomic_add(&(CQ[sc1]), ValConv1);
			atomic_add(&(CQ[sc2]), ValConv2);
		}




	}//while ((i = (i + 1) % N) != n);
}



//Simulation (Emitterlist -> Intensity pattern)
__kernel void SimulateCrystal(__global const float *PixelMap,
	__global const float *EmitterList,
	__global const double *Params,
	__global float *Intensity)
{
	unsigned int ind = get_global_id(0);

	//Get Parameter
	unsigned int NumEM = (unsigned int)Params[0]; //number of emitters
	int SuSa = (int)Params[2]; //Subsampling

	float u_Step[3]; //u direction (fs)
	u_Step[0] = Params[3];
	u_Step[1] = Params[4];
	u_Step[2] = Params[5];
	float v_Step[3]; //v direction (ss)
	v_Step[0] = Params[6];
	v_Step[1] = Params[7];
	v_Step[2] = Params[8];
	//
	float Wavelength = Params[9];


	////Debug Bullshit
	//if (ind == 0)
	//{
	//	printf("\n******************\nKernel:\n");
	//	printf("NumEM: %f -> %d\n", Params[0], NumEM);
	//	printf("PoissonSample: %f \n", Params[1]);
	//	printf("SuSa: %f -> %d\n", Params[2], SuSa);
	//	printf("u-Step: (%f, %f, %f) \n", u_Step[0], u_Step[1], u_Step[2]);
	//	printf("v-Step: (%f, %f, %f) \n", v_Step[0], v_Step[1], v_Step[2]);
	//	printf("Wavelength: %f \n", Params[9]);
	//	printf("******************\n\n");
	//}
	////


	float PixelCentPos[3];
	PixelCentPos[0] = PixelMap[3 * ind + 0];
	PixelCentPos[1] = PixelMap[3 * ind + 1];
	PixelCentPos[2] = PixelMap[3 * ind + 2];

	Intensity[ind] = 0.0; //intensity (our output)

	for (int v = -SuSa; v <= SuSa; v++) //subsampling in u direction (ss)
	{
		for (int u = -SuSa; u <= SuSa; u++) //subsampling in v direction (fs)
		{
			float realPsi = 0;
			float imagPsi = 0;
			for (unsigned int i = 0; i < NumEM; i++)
			{
				//Calculate k
				float Curr_Pos[3];
				Curr_Pos[0] = PixelCentPos[0] + (v * v_Step[0]) + (u * v_Step[0]);
				Curr_Pos[1] = PixelCentPos[1] + (v * v_Step[1]) + (u * v_Step[1]);
				Curr_Pos[2] = PixelCentPos[2] + (v * v_Step[2]) + (u * v_Step[2]);
				//calculate k
				float k[3];
				float Norm_k = 1.0 / sqrt(Curr_Pos[0] * Curr_Pos[0] + Curr_Pos[1] * Curr_Pos[1] + Curr_Pos[2] * Curr_Pos[2]);
				Norm_k *= ((2.0 * PI) / Wavelength); //Norm_k converts the position vector into the according k vector.
				k[0] = Curr_Pos[0] * Norm_k;
				k[1] = Curr_Pos[1] * Norm_k;
				k[2] = Curr_Pos[2] * Norm_k;

				float arg = 0; //argument for exponent (here split in real(cos) and imag(sin) part). Psi = exp^(i arg) 
				arg += k[0] * ((float)EmitterList[4 * i + 0]);
				arg += k[1] * ((float)EmitterList[4 * i + 1]);
				arg += k[2] * ((float)EmitterList[4 * i + 2]);
				arg +=        ((float)EmitterList[4 * i + 3]); //Phase

				realPsi += cos(arg);
				imagPsi += sin(arg);

			}
			Intensity[ind] += sqrt((realPsi * realPsi) + (imagPsi * imagPsi));

			////Debug Bullshit
			//if (ind == 0)
			//{
			//	printf("Int: %f; u: %d; v:%d  \n", Intensity[ind],u,v);
			//}
			////

		}
	}

	Intensity[ind] = Intensity[ind] / ((float)((2 * SuSa + 1)*(2 * SuSa + 1)));
	//Do Poisson sampling on cpu (need to know integrated intensity first, there is no way to calculate it here (need all threads to be finished))

}



//create AC fore dense HitList
__kernel void AutoCorr_sparseHL_AV(__global const float *HitList,
	__global const double *Params,
	__global unsigned long *AC)
{
	unsigned int ind = get_global_id(0);

	unsigned int HLSize = (unsigned int)Params[0];
	int ArraySize = (int)Params[1];


	unsigned int MaR_ScanAdd = ((ind*MapAndReduce) / (HLSize)) * ArraySize;

	//Debug Bullshit
	if (ind == 0 && Params[7] > 0.4)//ind == 0
	{
		printf("Kernel is alive\n");
		//printf("Detector Size: %d\n", DetSize);
		//printf("Interpolation mode: %d\n", InterpolMode);
		//printf("ArraySize: %d\n", ArraySize);
		//printf("Multiplicator: %f\n", Multiplicator);
		//printf("MapAndReduce: %d\n", MapAndReduce);
		//printf("Max q: %f\n", MaxQ);
	}
	//END


	//local Variables

	float k1[3];
	float k2[3];

	k1[0] = KMap[0 + 3 * ind];
	k1[1] = KMap[1 + 3 * ind];
	k1[2] = KMap[2 + 3 * ind];

	float q;
	float INT_ind = IntensityData[ind];


	for (int i = 0; i < DetSize; i++) //Loop over all Pixel //< DetSize
	{
		if (i == ind) //exclude zeroth peak need to be set to infinity afterwards
		{
			continue;
		}

		double Val = INT_ind * IntensityData[i];

		if (Val < 1e-37f) //no entry shortcut
		{
			continue;
		}



		k2[0] = KMap[0 + 3 * i];
		k2[1] = KMap[1 + 3 * i];
		k2[2] = KMap[2 + 3 * i];

		q = sqrt((k1[0] - k2[0]) * (k1[0] - k2[0])
			+ (k1[1] - k2[1]) * (k1[1] - k2[1])
			+ (k1[2] - k2[2]) * (k1[2] - k2[2]));




		//if (q > MaxQ)
		//{
		//	continue;
		//}

		q = q / dqPerVox;


		if (InterpolMode == 0) //nearest neighbor interpolation
		{
			long ValConv = 0;
			ValConv = (long)(Val*Multiplicator);

			unsigned int sc;
			sc = (unsigned int)(floor(q + 0.5)) + MaR_ScanAdd;

			atomic_add(&(CQ[sc]), ValConv);
		}
		if (InterpolMode == 1) //linear interpolation
		{
			unsigned int sc1, sc2;

			sc1 = (unsigned int)(floor(q)) + MaR_ScanAdd;
			sc2 = sc1 + 1;

			double Sep = q - (floor(q)); //separator

			long ValConv1 = 0;
			long ValConv2 = 0;
			ValConv1 = (long)(Val*(1 - Sep)*Multiplicator);
			ValConv2 = (long)(Val*(Sep)*Multiplicator);

			atomic_add(&(CQ[sc1]), ValConv1);
			atomic_add(&(CQ[sc2]), ValConv2);
		}




	}//while ((i = (i + 1) % N) != n);
}