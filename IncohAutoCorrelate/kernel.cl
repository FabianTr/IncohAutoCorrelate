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
	unsigned int VoxperdQ = (unsigned int)Params[1];
	unsigned int MeshSizeAB = (unsigned int)Params[2];
	unsigned int MeshSizeC = (unsigned int)Params[3];

	int Aind = (int)Params[4];
	int Bind = (int)Params[5];
	int Cind = (int)Params[6];

	unsigned int InterpolMode = (unsigned int)Params[7];

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
		printf("Vox per dQ: %d\n", VoxperdQ);
		printf("Interpolation Mode: %d\n", InterpolMode);
		printf("\n");
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

	//float INT_ind = IntensityData[ind];

	for (int i = 0; i < DetSize; i++) //Loop over all Pixel //< DetSize
	{
		double Val = (double)(IntensityData[ind] * IntensityData[i]);

		if (Val < 1e-37f) //no entry shortcut
		{
			continue;
		}

		unsigned long ValConv = 0;
		ValConv = (unsigned long)(Val*Multiplicator);
		
		k2[0] = KMap[0 + 3 * i];
		k2[1] = KMap[1 + 3 * i];
		k2[2] = KMap[2 + 3 * i];
		q1[0] = k1[0] - k2[0];
		q1[1] = k1[1] - k2[1];
		q1[2] = k1[2] - k2[2];

		if (sqrt(q1[0] * q1[0] + q1[1] * q1[1] + q1[2] * q1[2]) > MaxQ)
		{
			continue;
		}

		q1[0] = q1[0] * VoxperdQ;
		q1[1] = q1[1] * VoxperdQ;
		q1[2] = q1[2] * VoxperdQ;



		//Map to Mesh
		int fs, ms, ss;//fast-scan, medium-scan, slow-scan
		if (InterpolMode == 0) //((Nearest Neighbor))
		{
			fs = (int)(round(q1[Aind]) + MeshCenterAB);
			ms = (int)(round(q1[Bind]) + MeshCenterAB);
			ss = (int)(round(q1[Cind]) + MeshCenterC);

			atomic_add(&(CQsmall[fs + ms * MeshSizeAB + ss * MeshSizeAB * MeshSizeAB]), ValConv);

			//if (ss >= MeshSizeC || ms >= MeshSizeAB || fs >= MeshSizeAB)//Check scans and display overflows.
			//{
			//	printf("ME: scans:%d, %d, %d   q: %f, %f, %f\n", ss, ms, fs, q1[0], q1[1], q1[2]);
			//}
		}
		if (InterpolMode == 1) //(Linear)
		{

			double Sep_fs = 1 - (q1[Aind] - floor(q1[Aind]));
			double Sep_ms = 1 - (q1[Bind] - floor(q1[Bind]));
			double Sep_ss = 1 - (q1[Cind] - floor(q1[Cind]));

			fs = (int)(floor(q1[Aind]) + MeshCenterAB);
			ms = (int)(floor(q1[Bind]) + MeshCenterAB);
			ss = (int)(floor(q1[Cind]) + MeshCenterC);

			unsigned long Val_out;
			//fff  (fs: floor, ms: floor, ss: floor)
			double Val_fff = Val * Multiplicator  * (Sep_fs*Sep_ms*Sep_ss);
			Val_out = (unsigned long)Val_fff;
			atomic_add(&(CQsmall[fs + ms * MeshSizeAB + ss * MeshSizeAB * MeshSizeAB]), Val_out);
			//fft (fs: floor, ms: floor, ss: top)
			double Val_fft = Val * Multiplicator *(Sep_fs*Sep_ms*(1 - Sep_ss));
			Val_out = (unsigned long)Val_fft;
			atomic_add(&(CQsmall[(fs + 0) + (ms + 0) * MeshSizeAB + (ss + 1) * MeshSizeAB * MeshSizeAB]), Val_out);
			//tff
			double Val_tff = Val * Multiplicator  * ((1 - Sep_fs)*Sep_ms*Sep_ss);
			Val_out = (unsigned long)Val_tff;
			atomic_add(&(CQsmall[(fs + 1) + (ms + 0) * MeshSizeAB + (ss + 0) * MeshSizeAB * MeshSizeAB]), Val_out);
			//tft
			double Val_tft = Val * Multiplicator  * ((1 - Sep_fs)*Sep_ms*(1 - Sep_ss));
			Val_out = (unsigned long)Val_tft;
			atomic_add(&(CQsmall[(fs + 1) + (ms + 0) * MeshSizeAB + (ss + 1) * MeshSizeAB * MeshSizeAB]), Val_out);
			//ftf
			double Val_ftf = Val * Multiplicator * ((Sep_fs) * (1 - Sep_ms) * (Sep_ss));
			Val_out = (unsigned long)Val_ftf;
			atomic_add(&(CQsmall[(fs + 0) + (ms + 1) * MeshSizeAB + (ss + 0) * MeshSizeAB * MeshSizeAB]), Val_out);
			//ftt
			double Val_ftt = Val * Multiplicator *((Sep_fs) * (1 - Sep_ms) * (1 - Sep_ss));
			Val_out = (unsigned long)Val_ftt;
			atomic_add(&(CQsmall[(fs + 0) + (ms + 1) * MeshSizeAB + (ss + 1) * MeshSizeAB * MeshSizeAB]), Val_out);
			//ttf
			double Val_ttf = Val * Multiplicator *  ((1 - Sep_fs) * (1 - Sep_ms) * (Sep_ss));
			Val_out = (unsigned long)Val_ttf;
			atomic_add(&(CQsmall[(fs + 1) + (ms + 1) * MeshSizeAB + (ss + 0) * MeshSizeAB * MeshSizeAB]), Val_out);
			//ttt
			double Val_ttt = Val * Multiplicator * ((1 - Sep_fs) * (1 - Sep_ms) * (1 - Sep_ss));
			Val_out = (unsigned long)Val_ttt;
			atomic_add(&(CQsmall[(fs + 1) + (ms + 1) * MeshSizeAB + (ss + 1) * MeshSizeAB * MeshSizeAB]), Val_out);

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

	unsigned int Aind = (unsigned int)Params[3];
	unsigned int Bind = (unsigned int)Params[4];
	unsigned int Cind = (unsigned int)Params[5];

	int InterpolMode = (int)Params[6];

	double Multiplicator = Params[7];

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

		printf("Params[6]: %d\n", Params[6]);
		printf("InterPolMode: %d\n", InterpolMode);

		//for (int j = 0; j < ListSize; j++)
		//{
		//	printf("M&W: %f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n", RW[j*10+0], RW[j * 10 + 1], RW[j * 10 + 2], RW[j * 10 + 3], RW[j * 10 + 4], RW[j * 10 + 5], RW[j * 10 + 6], RW[j * 10 + 7], RW[j * 10 + 8], RW[j * 10 + 9]);
		//}
	}
	//END

	float q_in[3];
	q_in[0] = 0;
	q_in[0] = 1;
	q_in[0] = 2;
	int ss_in = 0, ms_in = 0, fs_in = 0;
	//retrive scans from index
	ss_in = ( ind / (smallMeshSize_AB*smallMeshSize_AB));
	ms_in = ((ind - (ss_in * smallMeshSize_AB*smallMeshSize_AB)) / (smallMeshSize_AB));
	fs_in = ind - (ss_in * smallMeshSize_AB*smallMeshSize_AB) - (ms_in * smallMeshSize_AB);
	//retrive q vectros (in voxels) fromm small C(q) mesh
	q_in[Aind] = fs_in - smallMeshCenterAB;
	q_in[Bind] = ms_in - smallMeshCenterAB;
	q_in[Cind] = ss_in - smallMeshCenterC;

	double Val_in = smallMesh[ind];


	if (Val_in < 1e-37f)
	{
		return;
	}

	for (unsigned int i = 0; i < ListSize; i++)
	{
	
		float q_out[3];
		//Rotate Vector
		q_out[0] = RW[10 * i + 0] * q_in[0] + RW[10 * i + 1] * q_in[1] + RW[10 * i + 2] * q_in[2];
		q_out[1] = RW[10 * i + 3] * q_in[0] + RW[10 * i + 4] * q_in[1] + RW[10 * i + 5] * q_in[2];
		q_out[2] = RW[10 * i + 6] * q_in[0] + RW[10 * i + 7] * q_in[1] + RW[10 * i + 8] * q_in[2];


		Val_in = smallMesh[ind];



		if (InterpolMode == 0) //Nearest Neighbour
		{
			Val_in = Val_in  * (double)RW[10 * i + 9] * Multiplicator;
			unsigned long Val_out = (unsigned long)Val_in;
			//get new scans
			int fs = 0, ms = 0, ss = 0;
			fs = (int)(round(q_out[0] ) + smallMeshCenterAB);
			ms = (int)(round(q_out[1] ) + smallMeshCenterAB);
			ss = (int)(round(q_out[2] ) + smallMeshCenterAB);//ss = (unsigned int)(floor(q_out[2] + 0.5);

			atomic_add(&(CQ[fs + ms * smallMeshSize_AB + ss * smallMeshSize_AB * smallMeshSize_AB]), Val_out);


			//if (smallMesh[ind] != 0)
			//{
			//	printf("%d: ValIn: %f; Weight: %f; ValOut %d\n",i, smallMesh[ind], RW[10 * i + 9], Val_out);
			//}
		}
		if (InterpolMode == 1) //Linear
		{
			double Sep_fs = 1 - (q_out[0] - floor(q_out[0]));
			double Sep_ms = 1 - (q_out[1] - floor(q_out[1]));
			double Sep_ss = 1 - (q_out[2] - floor(q_out[2]));

			int fs = 0, ms = 0, ss = 0; //floor scans
			fs = (int)(floor(q_out[0]) + smallMeshCenterAB);
			ms = (int)(floor(q_out[1]) + smallMeshCenterAB);
			ss = (int)(floor(q_out[2]) + smallMeshCenterAB);

			unsigned long Val_out;
			//fff  (fs: floor, ms: floor, ss: floor)
			double Val_fff = Val_in * Multiplicator * (double)RW[10 * i + 9] * (Sep_fs*Sep_ms*Sep_ss);
			Val_out = (unsigned long)Val_fff;
			atomic_add(&(CQ[fs + ms * smallMeshSize_AB + ss * smallMeshSize_AB * smallMeshSize_AB]), Val_out);
			//fft (fs: floor, ms: floor, ss: top)
			double Val_fft = Val_in * Multiplicator * (double)RW[10 * i + 9] * (Sep_fs*Sep_ms*(1 - Sep_ss));
			Val_out = (unsigned long)Val_fft;
			atomic_add(&(CQ[(fs + 0) + (ms + 0) * smallMeshSize_AB + (ss + 1) * smallMeshSize_AB * smallMeshSize_AB]), Val_out);
			//tff
			double Val_tff = Val_in * Multiplicator * (double)RW[10 * i + 9] * ((1 - Sep_fs)*Sep_ms*Sep_ss);
			Val_out = (unsigned long)Val_tff;
			atomic_add(&(CQ[(fs + 1) + (ms + 0) * smallMeshSize_AB + (ss + 0) * smallMeshSize_AB * smallMeshSize_AB]), Val_out);
			//tft
			double Val_tft = Val_in * Multiplicator * (double)RW[10 * i + 9] * ((1 - Sep_fs)*Sep_ms*(1 - Sep_ss));
			Val_out = (unsigned long)Val_tft;
			atomic_add(&(CQ[(fs + 1) + (ms + 0) * smallMeshSize_AB + (ss + 1) * smallMeshSize_AB * smallMeshSize_AB]), Val_out);
			//ftf
			double Val_ftf = Val_in * Multiplicator * (double)RW[10 * i + 9] * ((Sep_fs) * (1 - Sep_ms) * (Sep_ss));
			Val_out = (unsigned long)Val_ftf;
			atomic_add(&(CQ[(fs + 0) + (ms + 1) * smallMeshSize_AB + (ss + 0) * smallMeshSize_AB * smallMeshSize_AB]), Val_out);
			//ftt
			double Val_ftt = Val_in * Multiplicator * (double)RW[10 * i + 9] * ((Sep_fs) * (1 - Sep_ms) * (1 - Sep_ss));
			Val_out = (unsigned long)Val_ftt;
			atomic_add(&(CQ[(fs + 0) + (ms + 1) * smallMeshSize_AB + (ss + 1) * smallMeshSize_AB * smallMeshSize_AB]), Val_out);
			//ttf
			double Val_ttf = Val_in * Multiplicator * (double)RW[10 * i + 9] * ((1 - Sep_fs) * (1 - Sep_ms) * (Sep_ss));
			Val_out = (unsigned long)Val_ttf;
			atomic_add(&(CQ[(fs + 1) + (ms + 1) * smallMeshSize_AB + (ss + 0) * smallMeshSize_AB * smallMeshSize_AB]), Val_out);
			//ttt
			double Val_ttt = Val_in * Multiplicator * (double)RW[10 * i + 9] * ((1 - Sep_fs) * (1 - Sep_ms) * (1 - Sep_ss));
			Val_out = (unsigned long)Val_ttt;
			atomic_add(&(CQ[(fs + 1) + (ms + 1) * smallMeshSize_AB + (ss + 1) * smallMeshSize_AB * smallMeshSize_AB]), Val_out);
		}

	}
}




inline void AddQToPositiveSS(__global unsigned long * AC, int MeshSize, int MeshCenter, int fs, int ms, int ss, unsigned long val, bool QisZero)
{
	if (ss > 0) //positive half of slow scan dimension
	{
		atomic_add(&(AC[fs + ms * MeshSize + ss * MeshSize * MeshSize]), val);
	}
	else
	{
		if (ss == 0)
		{
			atomic_add(&(AC[fs + ms * MeshSize + ss * MeshSize * MeshSize]), val);
		}

		if (!QisZero)
		{
			//Mirrowed Entry
			fs = 2 * MeshCenter - fs;
			ms = 2 * MeshCenter - ms;
			ss = 0 - ss;

			atomic_add(&(AC[fs + ms * MeshSize + ss * MeshSize * MeshSize]), val);
		}
	}
}

void AddRotatedQToMesh(__global unsigned long * AC, double Multiplicator, int MeshSize, int MeshCenter, double val, float q0, float q1, float q2, int InterpolMode_lvl2,bool QisZero)
{
	float q[3];
	q[0] = q0;
	q[1] = q1;
	q[2] = q2;
	if (InterpolMode_lvl2 == 0) //nearest neighbour
	{
		unsigned long Val = (unsigned long)(val * Multiplicator);

		int fs, ms, ss;
		fs = (int)(MeshCenter + round(q[0]));
		ms = (int)(MeshCenter + round(q[1]));
		ss = (int)(round(q[2]));
		AddQToPositiveSS(AC, MeshSize, MeshCenter, fs, ms, ss, Val, QisZero);
	}

	if (InterpolMode_lvl2 == 1) //linear
	{
		double Sep_fs = 1.0 - (q[0] - floor(q[0]));
		double Sep_ms = 1.0 - (q[1] - floor(q[1]));
		double Sep_ss = 1.0 - (q[2] - floor(q[2]));

		int fs, ms, ss;
		ss = (int)(floor(q[2]));
		double t_Val = val;
		unsigned long Val;



		fs = (int)(MeshCenter + floor(q[0]));
		ms = (int)(MeshCenter + floor(q[1]));
		ss = (int)(floor(q[2]));
		//fff
		Val = (unsigned long)(t_Val * Multiplicator *((Sep_fs) * (Sep_ms)  * (Sep_ss)));
		AddQToPositiveSS(AC, MeshSize, MeshCenter, fs, ms, ss, Val, QisZero);
		//atomic_add(&(AC[fs + ms * MeshSize + ss * MeshSize * MeshSize]), Val);

		//tff
		Val = (unsigned long)(t_Val * Multiplicator *((1 - Sep_fs) * (Sep_ms)  * (Sep_ss)));
		AddQToPositiveSS(AC, MeshSize, MeshCenter, fs + 1, ms, ss, Val, QisZero);
		//atomic_add(&(AC[(fs + 1) + (ms + 0) * MeshSize + (ss + 0) * MeshSize * MeshSize]), Val);

		//fft
		Val = (unsigned long)(t_Val * Multiplicator *((Sep_fs) * (Sep_ms)  * (1 - Sep_ss)));
		AddQToPositiveSS(AC, MeshSize, MeshCenter, fs, ms, ss + 1, Val, QisZero);
		//atomic_add(&(AC[(fs + 0) + (ms + 0) * MeshSize + (ss + 1) * MeshSize * MeshSize]), Val);

		//tft
		Val = (unsigned long)(t_Val * Multiplicator *((1 - Sep_fs) * (Sep_ms)  * (1 - Sep_ss)));
		AddQToPositiveSS(AC, MeshSize, MeshCenter, fs + 1, ms, ss + 1, Val, QisZero);
		//atomic_add(&(AC[(fs + 1) + (ms + 0) * MeshSize + (ss + 1) * MeshSize * MeshSize]), Val);

		//ftf
		Val = (unsigned long)(t_Val * Multiplicator *((Sep_fs) * (1 - Sep_ms)  * (Sep_ss)));
		AddQToPositiveSS(AC, MeshSize, MeshCenter, fs, ms + 1, ss, Val, QisZero);
		//atomic_add(&(AC[(fs + 0) + (ms + 1) * MeshSize + (ss + 0) * MeshSize * MeshSize]), Val);

		//ttf
		Val = (unsigned long)(t_Val * Multiplicator *((1 - Sep_fs) * (1 - Sep_ms)  * (Sep_ss)));
		AddQToPositiveSS(AC, MeshSize, MeshCenter, fs + 1, ms + 1, ss, Val, QisZero);
		//atomic_add(&(AC[(fs + 1) + (ms + 1) * MeshSize + (ss + 0) * MeshSize * MeshSize]), Val);

		//ftt
		Val = (unsigned long)(t_Val * Multiplicator *((Sep_fs) * (1 - Sep_ms)  * (1 - Sep_ss)));
		AddQToPositiveSS(AC, MeshSize, MeshCenter, fs, ms + 1, ss + 1, Val, QisZero);
		//atomic_add(&(AC[(fs + 0) + (ms + 1) * MeshSize + (ss + 1) * MeshSize * MeshSize]), Val);

		//ttt
		Val = (unsigned long)(t_Val * Multiplicator *((1 - Sep_fs) * (1 - Sep_ms)  * (1 - Sep_ss)));
		AddQToPositiveSS(AC, MeshSize, MeshCenter, fs + 1, ms + 1, ss + 1, Val, QisZero);
		//atomic_add(&(AC[(fs + 1) + (ms + 1) * MeshSize + (ss + 1) * MeshSize * MeshSize]), Val);
		// ###########################################################	
	}
}

//Autocorrelates sparse (dense) HitList (analog to CPU implementation)
__kernel void Autocor_sparseHL(__global const float *SparseHitList,
	__global const double *Params,
	__global const float *RotMatrix,
	__global unsigned long *AC)
{
	unsigned int ind = get_global_id(0);


	unsigned int ListSize = (unsigned int)Params[0]; //Entrys in sparse HitList
	unsigned int dVdq = (unsigned int)Params[1]; // dq/dV

	int MeshSize = (unsigned int)Params[2]; //Only cube Meshes allowed here
	int MeshCenter = (MeshSize - 1) / 2;

	float MaxQ = (float)Params[4];
	int DoubleMapping = (int)Params[5]; //if 1, maps two times (before and after rotation)
	double Multiplicator = Params[6]; //for conversion float -> int

	int InterpolMode_lvl1 = (int)Params[7];
	int InterpolMode_lvl2 = (int)Params[8];

	//Debug Bullshit
	//if (ind == 0)//ind == 0
	//{
	//	printf("\n\nKernel (AC uw) is alive\n");
	//	printf("MeshShape A=B=C=: %d\n", MeshSize);
	//	printf("MeshCenter: %d\n", MeshCenter);
	//	printf("dVdq: %d\n", dVdq);
	//	printf("DoubleMapping: %d\n", DoubleMapping);
	//	printf("Multiplicator: %f\n", Multiplicator);
	//	printf("ListSize: %d\n\n", ListSize);
	//}

	//obtain k-vector and value given by kernel index
	float k1[3];
	k1[0] = SparseHitList[4 * ind + 0];
	k1[1] = SparseHitList[4 * ind + 1];
	k1[2] = SparseHitList[4 * ind + 2];
	double f_Val = (double)SparseHitList[4 * ind + 3];

	for (unsigned int i = ind; i < ListSize; i++)
	{
		float q[3];
		q[0] = k1[0] - SparseHitList[4 * i + 0];
		q[1] = k1[1] - SparseHitList[4 * i + 1];
		q[2] = k1[2] - SparseHitList[4 * i + 2];

		if (i == ind) //Q is exact zero >> QisZero = true
		{
			AddRotatedQToMesh(AC, Multiplicator, MeshSize, MeshCenter, f_Val * SparseHitList[4 * i + 3], 0, 0, 0, InterpolMode_lvl2, true);
			continue;
		}
		//Q can never be exact zero from here on >> QisZero = false

		//Check if inside of range
		if (sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2]) > MaxQ) //check if q is in range
		{
			continue;
		}

		//convert q to units of voxel
		q[0] = q[0] * dVdq;
		q[1] = q[1] * dVdq;
		q[2] = q[2] * dVdq;
		

		if(DoubleMapping == 1)
		{ 
			if (InterpolMode_lvl1 == 0) //1. lvl interpol: Nearest Neighbor
			{
				//map to mesh             
				int fs_l, ms_l, ss_l;
				fs_l = (int)(round(q[0] ) + MeshCenter);
				ms_l = (int)(round(q[1] ) + MeshCenter);
				ss_l = (int)(round(q[2] ) + MeshCenter);
				//printf("t_scan = [%d, %d, %d]\n", fs_l, ms_l, ss_l);
				//convert back to q
				q[0] = (float)(fs_l - MeshCenter);
				q[1] = (float)(ms_l - MeshCenter);
				q[2] = (float)(ss_l - MeshCenter);

				//Rotation
				float t_q[3];
				t_q[0] = q[0] * RotMatrix[0] + q[1] * RotMatrix[1] + q[2] * RotMatrix[2];
				t_q[1] = q[0] * RotMatrix[3] + q[1] * RotMatrix[4] + q[2] * RotMatrix[5];
				t_q[2] = q[0] * RotMatrix[6] + q[1] * RotMatrix[7] + q[2] * RotMatrix[8];

				//map to mesh
				AddRotatedQToMesh(AC, Multiplicator, MeshSize, MeshCenter, f_Val * SparseHitList[4 * i + 3], t_q[0], t_q[1], t_q[2], InterpolMode_lvl2, false);
			}
			if (InterpolMode_lvl1 == 1) //1. lvl interpol: Linear
			{
				double Sep_fs = 1.0 - (q[0] - floor(q[0]));
				double Sep_ms = 1.0 - (q[1] - floor(q[1]));
				double Sep_ss = 1.0 - (q[2] - floor(q[2]));

				int fs, ms, ss;
				fs = (int)(MeshCenter + floor(q[0]));
				ms = (int)(MeshCenter + floor(q[1]));
				ss = (int)(MeshCenter + floor(q[2]));



				float BaseVal = f_Val * SparseHitList[4 * i + 3];
				float t_val = 0.0;

				float t_q[3];
				//fff
				t_val = BaseVal * ((Sep_fs) * (Sep_ms)  * (Sep_ss));
				q[0] = (float)(fs - MeshCenter);
				q[1] = (float)(ms - MeshCenter);
				q[2] = (float)(ss - MeshCenter);
				t_q[0] = q[0] * RotMatrix[0] + q[1] * RotMatrix[1] + q[2] * RotMatrix[2];
				t_q[1] = q[0] * RotMatrix[3] + q[1] * RotMatrix[4] + q[2] * RotMatrix[5];
				t_q[2] = q[0] * RotMatrix[6] + q[1] * RotMatrix[7] + q[2] * RotMatrix[8];
				AddRotatedQToMesh(AC, Multiplicator, MeshSize, MeshCenter, t_val, t_q[0], t_q[1], t_q[2], InterpolMode_lvl2, false);

				//tff
				t_val = BaseVal * ((1 - Sep_fs) * (Sep_ms)  * (Sep_ss));
				q[0] = (float)((fs + 1) - MeshCenter);
				q[1] = (float)((ms + 0) - MeshCenter);
				q[2] = (float)((ss + 0) - MeshCenter);
				t_q[0] = q[0] * RotMatrix[0] + q[1] * RotMatrix[1] + q[2] * RotMatrix[2];
				t_q[1] = q[0] * RotMatrix[3] + q[1] * RotMatrix[4] + q[2] * RotMatrix[5];
				t_q[2] = q[0] * RotMatrix[6] + q[1] * RotMatrix[7] + q[2] * RotMatrix[8];
				AddRotatedQToMesh(AC, Multiplicator, MeshSize, MeshCenter, t_val, t_q[0], t_q[1], t_q[2], InterpolMode_lvl2, false);

				//fft
				t_val = BaseVal * ((Sep_fs) * (Sep_ms)  * (1 - Sep_ss));
				q[0] = (float)((fs + 0) - MeshCenter);
				q[1] = (float)((ms + 0) - MeshCenter);
				q[2] = (float)((ss + 1) - MeshCenter);
				t_q[0] = q[0] * RotMatrix[0] + q[1] * RotMatrix[1] + q[2] * RotMatrix[2];
				t_q[1] = q[0] * RotMatrix[3] + q[1] * RotMatrix[4] + q[2] * RotMatrix[5];
				t_q[2] = q[0] * RotMatrix[6] + q[1] * RotMatrix[7] + q[2] * RotMatrix[8];
				AddRotatedQToMesh(AC, Multiplicator, MeshSize, MeshCenter, t_val, t_q[0], t_q[1], t_q[2], InterpolMode_lvl2, false);

				//tft
				t_val = BaseVal * ((1 - Sep_fs) * (Sep_ms)  * (1 - Sep_ss));
				q[0] = (float)((fs + 1) - MeshCenter);
				q[1] = (float)((ms + 0) - MeshCenter);
				q[2] = (float)((ss + 1) - MeshCenter);
				t_q[0] = q[0] * RotMatrix[0] + q[1] * RotMatrix[1] + q[2] * RotMatrix[2];
				t_q[1] = q[0] * RotMatrix[3] + q[1] * RotMatrix[4] + q[2] * RotMatrix[5];
				t_q[2] = q[0] * RotMatrix[6] + q[1] * RotMatrix[7] + q[2] * RotMatrix[8];
				AddRotatedQToMesh(AC, Multiplicator, MeshSize, MeshCenter, t_val, t_q[0], t_q[1], t_q[2], InterpolMode_lvl2, false);

				//ftf
				t_val = BaseVal * ((Sep_fs) * (1 - Sep_ms)  * (Sep_ss));
				q[0] = (float)((fs + 0) - MeshCenter);
				q[1] = (float)((ms + 1) - MeshCenter);
				q[2] = (float)((ss + 0) - MeshCenter);
				t_q[0] = q[0] * RotMatrix[0] + q[1] * RotMatrix[1] + q[2] * RotMatrix[2];
				t_q[1] = q[0] * RotMatrix[3] + q[1] * RotMatrix[4] + q[2] * RotMatrix[5];
				t_q[2] = q[0] * RotMatrix[6] + q[1] * RotMatrix[7] + q[2] * RotMatrix[8];
				AddRotatedQToMesh(AC, Multiplicator, MeshSize, MeshCenter, t_val, t_q[0], t_q[1], t_q[2], InterpolMode_lvl2, false);

				//ttf
				t_val = BaseVal * ((1 - Sep_fs) * (1 - Sep_ms)  * (Sep_ss));
				q[0] = (float)((fs + 1) - MeshCenter);
				q[1] = (float)((ms + 1) - MeshCenter);
				q[2] = (float)((ss + 0) - MeshCenter);
				t_q[0] = q[0] * RotMatrix[0] + q[1] * RotMatrix[1] + q[2] * RotMatrix[2];
				t_q[1] = q[0] * RotMatrix[3] + q[1] * RotMatrix[4] + q[2] * RotMatrix[5];
				t_q[2] = q[0] * RotMatrix[6] + q[1] * RotMatrix[7] + q[2] * RotMatrix[8];
				AddRotatedQToMesh(AC, Multiplicator, MeshSize, MeshCenter, t_val, t_q[0], t_q[1], t_q[2], InterpolMode_lvl2, false);

				//ftt
				t_val = BaseVal * ((Sep_fs) * (1 - Sep_ms)  * (1 - Sep_ss));
				q[0] = (float)((fs + 0) - MeshCenter);
				q[1] = (float)((ms + 1) - MeshCenter);
				q[2] = (float)((ss + 1) - MeshCenter);
				t_q[0] = q[0] * RotMatrix[0] + q[1] * RotMatrix[1] + q[2] * RotMatrix[2];
				t_q[1] = q[0] * RotMatrix[3] + q[1] * RotMatrix[4] + q[2] * RotMatrix[5];
				t_q[2] = q[0] * RotMatrix[6] + q[1] * RotMatrix[7] + q[2] * RotMatrix[8];
				AddRotatedQToMesh(AC, Multiplicator, MeshSize, MeshCenter, t_val, t_q[0], t_q[1], t_q[2], InterpolMode_lvl2, false);

				//ttt
				t_val = BaseVal * ((1 - Sep_fs) * (1 - Sep_ms)  * (1 - Sep_ss));
				q[0] = (float)((fs + 1) - MeshCenter);
				q[1] = (float)((ms + 1) - MeshCenter);
				q[2] = (float)((ss + 1) - MeshCenter);
				t_q[0] = q[0] * RotMatrix[0] + q[1] * RotMatrix[1] + q[2] * RotMatrix[2];
				t_q[1] = q[0] * RotMatrix[3] + q[1] * RotMatrix[4] + q[2] * RotMatrix[5];
				t_q[2] = q[0] * RotMatrix[6] + q[1] * RotMatrix[7] + q[2] * RotMatrix[8];
				AddRotatedQToMesh(AC, Multiplicator, MeshSize, MeshCenter, t_val, t_q[0], t_q[1], t_q[2], InterpolMode_lvl2, false);
			}
		}
		else
		{
			//Rotation
			float t_q[3];
			t_q[0] = q[0] * RotMatrix[0] + q[1] * RotMatrix[1] + q[2] * RotMatrix[2];
			t_q[1] = q[0] * RotMatrix[3] + q[1] * RotMatrix[4] + q[2] * RotMatrix[5];
			t_q[2] = q[0] * RotMatrix[6] + q[1] * RotMatrix[7] + q[2] * RotMatrix[8];

			//map to mesh
			AddRotatedQToMesh(AC, Multiplicator, MeshSize, MeshCenter, f_Val * SparseHitList[4 * i + 3], t_q[0], t_q[1], t_q[2], InterpolMode_lvl2, false);
		}
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
		printf("dq/dx: %f\n", dqPerVox);
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


		if (q > MaxQ)
		{
			continue;
		}

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

			if ((unsigned int)(floor(q)) >= 1201)
			{
				printf("sc1: %d\n", (unsigned int)(floor(q)));
			}

			sc1 = (unsigned int)(floor(q)) + MaR_ScanAdd;
			sc2 = sc1 + 1;


			double Sep = q - (floor(q)); //separator

			long ValConv1 = 0;
			long ValConv2 = 0;
			ValConv1 = (long)(Val * (1 - Sep) * Multiplicator);
			ValConv2 = (long)(Val * (Sep) * Multiplicator);

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
				Curr_Pos[0] = PixelCentPos[0] + (v * v_Step[0]) + (u * u_Step[0]);
				Curr_Pos[1] = PixelCentPos[1] + (v * v_Step[1]) + (u * u_Step[1]);
				Curr_Pos[2] = PixelCentPos[2] + (v * v_Step[2]) + (u * u_Step[2]);
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


//create AngularAveraged AC for sparse (dense) HitList
__kernel void AutoCorr_sparseHL_AAV(__global const float *HitList,
	__global const double *Params,
	__global unsigned long *AC)
{
	unsigned int ind = get_global_id(0);
	unsigned int HLSize = (unsigned int)Params[0];
	unsigned int ArraySize = (unsigned int)Params[1]; //vec Size
	float MaxQ = (float)Params[2];
	float dqdx = (float)Params[3];
	double Multiplicator = Params[4];
	unsigned int MapAndReduce = (unsigned int)Params[5];
	int InterpolMode = (int)Params[6];

	unsigned int MaR_ScanAdd = ((unsigned int)((ind*MapAndReduce) / (HLSize))) * ArraySize;

	////Debug Bullshit
	//if (ind == 0)//ind == 0
	//{
	//	//printf("Kernel is alive\n");
	//	//printf("HitList Size: %d\n", HLSize);
	//	//printf("Array Size: %d\n", ArraySize);
	//	//printf("MaxQ: %f\n", MaxQ);
	//	//printf("dqdx: %f\n", dqdx);
	//	//printf("Multiplicator: %f\n", Multiplicator);
	//	//printf("MapAndReduce: %d\n", MapAndReduce);
	//	//printf("InterpolMode: %d\n", InterpolMode);
	//}
	////END

	//Get thread constant k1 and f_Val
	float k1[3];
	float k2[3];
	k1[0] = HitList[4 * ind + 0];
	k1[1] = HitList[4 * ind + 1];
	k1[2] = HitList[4 * ind + 2];
	double INT_ind = (double)HitList[4 * ind + 3];

	for (int i = 0; i < HLSize; i++) //Loop over all Pixel //< DetSize
	{
			
		k2[0] = HitList[4 * i + 0];
		k2[1] = HitList[4 * i + 1];
		k2[2] = HitList[4 * i + 2];

		float q = sqrt((k1[0] - k2[0]) * (k1[0] - k2[0])
					 + (k1[1] - k2[1]) * (k1[1] - k2[1])
					 + (k1[2] - k2[2]) * (k1[2] - k2[2]));

		double Val = INT_ind * (double)HitList[4 * i + 3];

		if (q  > MaxQ) //check if q is in boundaries
		{
			continue;
		}
		//scale to pixel
		q = q / dqdx;

		if (InterpolMode == 0) //nearest neighbor interpolation
		{
			unsigned long ValConv = 0;
			ValConv = (unsigned long)(Val * Multiplicator);

			unsigned int sc;
			sc = (unsigned int)(floor(q + 0.5)) + MaR_ScanAdd;

			atomic_add(&(AC[sc]), ValConv);
		}
		if (InterpolMode == 1) //linear interpolation
		{
			unsigned int sc1, sc2;

			sc1 = (unsigned int)(floor(q)) + MaR_ScanAdd;
			sc2 = sc1 + 1;

			double Sep = q - (floor(q)); //separator

			unsigned long ValConv1 = 0;
			unsigned long ValConv2 = 0;

			ValConv1 = (unsigned long)(Val * (1 - Sep) * Multiplicator);
			ValConv2 = (unsigned long)(Val * Sep * Multiplicator);

			atomic_add(&(AC[sc1]), ValConv1);
			atomic_add(&(AC[sc2]), ValConv2);
		}

	}

}