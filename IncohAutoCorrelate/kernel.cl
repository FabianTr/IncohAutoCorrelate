#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable



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



__kernel void AutoCorr_CQ(__global const float *IntensityData,
	__global const float *KMap,
	__global const float *Rotations_and_Weights,
	__global const double *Params,
	__global double *CQ)
{
	unsigned int ind = get_global_id(0);
	//double Params[5];
	//Params[0] = DetSize; //Numer of pixels (size[0]*size[1])
	//Params[1] = deltaV; //dq per Voxel
	//Params[2] = MeshSize; //  a  (V = a*a*(a+1)/2)
	//Params[3] = NumberOfEvents; // how many events, for roataion and weight loop
	//Params[4] = InterpolationMode;

	unsigned int DetSize = (unsigned int)Params[0];
	float dqPerVox = (float)Params[1];
	unsigned int MeshSize = (unsigned int)Params[2];
	unsigned int NumEvents = (unsigned int)Params[3];
	int InterpolMode = (unsigned int)Params[4];


	////Debug Bullshit
	//if (ind == 0)//ind == 0
	//{
	//	printf("Kernel is alive\n");
	//	printf("Detector Size: %d\n", DetSize);
	//	printf("Number of Events: %d\n", NumEvents);
	//	printf("Interpolation mode: %d\n", InterpolMode);
	//	printf("Mesh Size: %d\n", MeshSize);
	//	printf("dq per Vox: %f\n", dqPerVox);
	//}
	////END
		//local Variables
	float k1[3];
	float k2[3];
	float q_RAW[3];

	k1[0] = KMap[0 + 3 * ind];
	k1[1] = KMap[1 + 3 * ind];
	k1[2] = KMap[2 + 3 * ind];

	float q[3];


	for (unsigned int i = 0; i < DetSize; ++i) //Loop over all Pixel
	{
		//if (i == ind) //exclude zeroth peak
		//{
		//	continue;
		//}

		float Val_RAW = IntensityData[ind] * IntensityData[i];
		k2[0] = KMap[0 + 3 * i];
		k2[1] = KMap[1 + 3 * i];
		k2[2] = KMap[2 + 3 * i];
		q_RAW[0] = k1[0] - k2[0];
		q_RAW[1] = k1[1] - k2[1];
		q_RAW[2] = k1[2] - k2[2];

		int MeshCenter = (MeshSize - 1) / 2;


		for (unsigned int j = 0; j < NumEvents; ++j)//Loop over all Events
		{
			//rotate
			q[0] = q_RAW[0] * Rotations_and_Weights[0 + 10 * j] + q_RAW[1] * Rotations_and_Weights[1 + 10 * j] + q_RAW[2] * Rotations_and_Weights[2 + 10 * j];
			q[1] = q_RAW[0] * Rotations_and_Weights[3 + 10 * j] + q_RAW[1] * Rotations_and_Weights[4 + 10 * j] + q_RAW[2] * Rotations_and_Weights[5 + 10 * j];
			q[2] = q_RAW[0] * Rotations_and_Weights[6 + 10 * j] + q_RAW[1] * Rotations_and_Weights[7 + 10 * j] + q_RAW[2] * Rotations_and_Weights[8 + 10 * j];
			// HALF MESH:
			// Meshsize: [-fs/2, fs/2][-ms/2, ms/2],[0, ss/2] needed to save memory (1000x1000x500*8Byte ~= 4GB)
			if (q[2] < 0) //needs to be mirrowed in postprocessing on the CPU!
			{
				continue;
			}
			//weight (by square of mean intensity)
			float Val = Val_RAW * Rotations_and_Weights[9 + 10 * j] * Rotations_and_Weights[9 + 10 * j];

			//resize (for Mapping)
			q[0] = q[0] / dqPerVox;
			q[1] = q[1] / dqPerVox;
			q[2] = q[2] / dqPerVox;

			//Map to Mesh

			if (InterpolMode == 0) //nearest Neighbor
			{
				unsigned int fs, ms, ss;//fast-scan-, medium-scan-, slow-scan- floor
				fs = (unsigned int)floor(q[0] + 0.5) + MeshCenter;
				ms = (unsigned int)floor(q[1] + 0.5) + MeshCenter;
				ss = (unsigned int)floor(q[2] + 0.5);

				atomic_add_float(&(CQ[fs + ms * MeshSize + ss * MeshSize*MeshSize]), Val);
			}
			if (InterpolMode == 1) //linear
			{
				unsigned int fsf, msf, ssf;//fast-scan-, medium-scan-, slow-scan- floor
				fsf = (unsigned int)floor(q[0]) + MeshCenter;
				msf = (unsigned int)floor(q[1]) + MeshCenter;
				ssf = (unsigned int)floor(q[2]);
				float SepF, SepM, SepS; //according Seperators
				SepF = q[0] - floor(q[0]);
				SepM = q[1] - floor(q[1]);
				SepS = q[2] - floor(q[2]);

				//add to mesh (8 entries each)
				atomic_add_float(&(CQ[fsf + msf * MeshSize + ssf * MeshSize*MeshSize]), Val * (1 - SepF)*(1 - SepM)*(1 - SepS));              //A + 0
				atomic_add_float(&(CQ[(fsf + 1) + (msf + 0) * MeshSize + (ssf + 0) * MeshSize*MeshSize]), Val * (SepF)*(1 - SepM)*(1 - SepS));//ssf + 1
				atomic_add_float(&(CQ[(fsf + 0) + (msf + 1) * MeshSize + (ssf + 0) * MeshSize*MeshSize]), Val * (1 - SepF)*(SepM)*(1 - SepS));//msf + 1
				atomic_add_float(&(CQ[(fsf + 0) + (msf + 0) * MeshSize + (ssf + 1) * MeshSize*MeshSize]), Val * (1 - SepF)*(1 - SepM)*(SepS));//ssf + 1
				atomic_add_float(&(CQ[(fsf + 1) + (msf + 1) * MeshSize + (ssf + 0) * MeshSize*MeshSize]), Val * (SepF)*(SepM)*(1 - SepS));    //ffs + 1 ; msf + 1
				atomic_add_float(&(CQ[(fsf + 0) + (msf + 1) * MeshSize + (ssf + 1) * MeshSize*MeshSize]), Val * (1 - SepF)*(SepM)*(SepS));    //msf + 1 ; ssf + 1
				atomic_add_float(&(CQ[(fsf + 1) + (msf + 0) * MeshSize + (ssf + 1) * MeshSize*MeshSize]), Val * (SepF)*(1 - SepM)*(SepS));    //ffs + 1 ; ssf + 1
				atomic_add_float(&(CQ[(fsf + 1) + (msf + 1) * MeshSize + (ssf + 1) * MeshSize*MeshSize]), Val * (SepF)*(SepM)*(SepS));        // A + 1
			}



		}
	}
}




__kernel void AutoCorr_CQ_small(__global const float *IntensityData,
	__global const float *KMap,
	__global const double *Params,
	__global long *CQsmall)
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
	long Multiplicator = (long)Params[9];


	////Debug Bullshit
	//if (ind == 0)//ind == 0
	//{
	//	printf("Kernel is alive\n");
	//	printf("Detector Size: %d\n", DetSize);
	//	printf("Interpolation mode: %d\n", InterpolMode);
	//	printf("Mesh Size AB: %d\n", MeshSizeAB);
	//	printf("Mesh Size C: %d\n", MeshSizeC);
	//	printf("dq per Vox: %f\n", dqPerVox);
	//	printf("Max q: %f\n", MaxQ);
	//}
	////END


	//local Variables

	float k1[3];
	float k2[3];

	k1[0] = KMap[0 + 3 * ind];
	k1[1] = KMap[1 + 3 * ind];
	k1[2] = KMap[2 + 3 * ind];

	float q1[3];
	float q2[3];

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

		long ValConv = 0;
		ValConv = (long)(Val*Multiplicator);

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

		//q2[0] = q2[0] / dqPerVox;
		//q2[1] = q2[1] / dqPerVox;
		//q2[2] = q2[2] / dqPerVox;



		//Map to Mesh
		unsigned int fs, ms, ss;//fast-scan-, medium-scan-, slow-scan
		fs = (unsigned int) (floor(q1[Aind] + 0.5) + MeshCenterAB);
		ms = (unsigned int) (floor(q1[Bind] + 0.5) + MeshCenterAB);
		ss = (unsigned int) (floor(q1[Cind] + 0.5) + MeshCenterC);
		atomic_add(&(CQsmall[fs + ms * MeshSizeAB + ss * MeshSizeAB * MeshSizeAB]), ValConv);


		//CQsmall[fs + ms * MeshSizeAB + ss * MeshSizeAB*MeshSizeAB] += Val;

		//if (fs + ms * MeshSizeAB + ss * MeshSizeAB * MeshSizeAB > MeshSizeAB*MeshSizeAB*MeshSizeC)
		//{
		//	printf("q: %f, %f, %f ; Q = %f \n", q1[0], q1[1], q1[2], sqrt(q1[0] * q1[0] + q1[1] * q1[1] + q1[2] * q1[2]));
		//	printf("scans: %d, %d, %d \n",fs,ms,ss);
		//}

		//atomic_add_float(&(CQsmall[fs + ms * MeshSizeAB + ss * MeshSizeAB * MeshSizeAB]), Val);
		



	}//while ((i = (i + 1) % N) != n);
}















//OldStuff
__kernel void AutoCorr_CQ_small_Double(__global const float *IntensityData,
	__global const float *KMap,
	__global const double *Params,
	__global double *CQsmall)
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

	unsigned int DetSize = (unsigned int)Params[0];
	float dqPerVox = (float)Params[1];
	unsigned int MeshSizeAB = (unsigned int)Params[2];
	unsigned int MeshSizeC = (unsigned int)Params[3];

	int Aind = (int)Params[4];
	int Bind = (int)Params[5];
	int Cind = (int)Params[6];

	int InterpolMode = (unsigned int)Params[7];

	float MaxQ = (float)Params[8];


	////Debug Bullshit
	//if (ind == 0)//ind == 0
	//{
	//	printf("Kernel is alive\n");
	//	printf("Detector Size: %d\n", DetSize);
	//	printf("Interpolation mode: %d\n", InterpolMode);
	//	printf("Mesh Size AB: %d\n", MeshSizeAB);
	//	printf("Mesh Size C: %d\n", MeshSizeC);
	//	printf("dq per Vox: %f\n", dqPerVox);
	//	printf("Max q: %f\n", MaxQ);
	//}
	////END


	//local Variables

	float k1[3];
	float k2[3];

	k1[0] = KMap[0 + 3 * ind];
	k1[1] = KMap[1 + 3 * ind];
	k1[2] = KMap[2 + 3 * ind];

	float q1[3];
	float q2[3];

	int MeshCenterAB = (MeshSizeAB - 1) / 2;
	int MeshCenterC = (MeshSizeC - 1) / 2;

	float INT_ind = IntensityData[ind];





	//int N = 10;
	//int n = (ind*777)%N ;
	//int i = n;
	//do{


	for (int i = ind; i < DetSize; i++) //Loop over all Pixel //< DetSize
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
		q1[0] = (k1[0] - k2[0]);
		q1[1] = (k1[1] - k2[1]);
		q1[2] = (k1[2] - k2[2]);

		q2[0] = (k2[0] - k1[0]);
		q2[1] = (k2[1] - k1[1]);
		q2[2] = (k2[2] - k1[2]);

		if (sqrt(q1[0] * q1[0] + q1[1] * q1[1] + q1[2] * q1[2]) > MaxQ)
		{
			continue;
		}

		q1[0] = q1[0] / dqPerVox;
		q1[1] = q1[1] / dqPerVox;
		q1[2] = q1[2] / dqPerVox;

		q2[0] = q2[0] / dqPerVox;
		q2[1] = q2[1] / dqPerVox;
		q2[2] = q2[2] / dqPerVox;



		//Map to Mesh
		unsigned int fs, ms, ss;//fast-scan-, medium-scan-, slow-scan
		fs = (unsigned int)floor(q1[Aind] + 0.5) + MeshCenterAB;
		ms = (unsigned int)floor(q1[Bind] + 0.5) + MeshCenterAB;
		ss = (unsigned int)floor(q1[Cind] + 0.5) + MeshCenterC;
		//CQsmall[fs + ms * MeshSizeAB + ss * MeshSizeAB*MeshSizeAB] += Val;

		//if (fs + ms * MeshSizeAB + ss * MeshSizeAB * MeshSizeAB > MeshSizeAB*MeshSizeAB*MeshSizeC)
		//{
		//	printf("q: %f, %f, %f ; Q = %f \n", q1[0], q1[1], q1[2], sqrt(q1[0] * q1[0] + q1[1] * q1[1] + q1[2] * q1[2]));
		//	printf("scans: %d, %d, %d \n",fs,ms,ss);
		//}

		atomic_add_float(&(CQsmall[fs + ms * MeshSizeAB + ss * MeshSizeAB * MeshSizeAB]), Val);

		//mirrowed entry
		fs = (unsigned int)floor(q2[Aind] + 0.5) + MeshCenterAB;
		ms = (unsigned int)floor(q2[Bind] + 0.5) + MeshCenterAB;
		ss = (unsigned int)floor(q2[Cind] + 0.5) + MeshCenterC;
		//CQsmall[fs + ms * MeshSizeAB + ss * MeshSizeAB*MeshSizeAB] += Val;


		//if (fs + ms * MeshSizeAB + ss * MeshSizeAB * MeshSizeAB > MeshSizeAB*MeshSizeAB*MeshSizeC)
		//{
		//	printf("q: %f, %f, %f ; Q = %f \n", q2[0], q2[1], q2[2], sqrt(q2[0] * q2[0] + q2[1] * q2[1] + q2[2] * q2[2]));
		//	printf("scans: %d, %d, %d \n", fs, ms, ss);
		//}

		atomic_add_float(&(CQsmall[fs + ms * MeshSizeAB + ss * MeshSizeAB * MeshSizeAB]), Val);
		//	printf("%d  ", fs + ms * MeshSizeAB + ss * MeshSizeAB * MeshSizeAB);


	}//while ((i = (i + 1) % N) != n);
}