#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

__constant double PI = 3.1415926535897932;



inline void atomic_add_float(__global double *source, const double value) 
{
	// Atomic add of floats or double
	// Adapted from:
	// https://streamcomputing.eu/blog/2016-02-09/atomic-operations-for-floats-in-opencl-improved/
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

inline double d_abs(double x)
{
	if (x < 0)
	{
		x = -x;
	}
	return x;
}

inline double d_min(double a, double b)
{
	if (a < b)
	{
		return a;
	}
	else
	{
		return b;
	}
}


__kernel void AutoCorr_CQ(__global const unsigned int *IntensityData,
	__global const float *KMap,
	__global const float *Rotations_and_Weights,
	__global const double *Params,
	__global double *CQ)
{
	int ind = get_global_id(0);
	//double Params[5];
	//Params[0] = DetSize; //Numer of pixels (size[0]*size[1])
	//Params[1] = deltaV; //Voxel Unit
	//Params[2] = MeshSize; // [a, a, (a+1)/2]
	//Params[3] = NumberOfEvents; // how many events, for roataion and weight loop
	//Params[4] = InterpolationMode;



}









//double VoxelSize = Params[1];
//int max_PerpVox = (int)Params[2];
//int max_ZVox = (int)Params[3];

//int DetSize = (int)Params[0];
//double DetDist = Params[5];

////calculate Detector coordinates of current pixel (ind)
//int x_ind_us = ind % DetSize;
//int y_ind_us = ind / DetSize;
//double shift = ((double)DetSize - 1.) / 2.;
//double x_ind = (double)x_ind_us - shift;
//double y_ind = (double)y_ind_us - shift;
//double z_ind = sqrt(x_ind*x_ind + y_ind*y_ind + DetDist*DetDist) - DetDist;

//double norm_0 = 1. / sqrt(x_ind*x_ind + y_ind*y_ind + DetDist*DetDist);
//x_ind = x_ind * norm_0;
//y_ind = y_ind * norm_0;
//z_ind = z_ind * norm_0;
//
//int V_perp_shift = (max_PerpVox - 1) / 2;
//int V_z_shift = (max_ZVox - 1) / 2;

////printf("Params: %e \t %e \t %e \t %e \t %e \t %e \t \n", Params[0], Params[1], Params[2], Params[3], Params[4], Params[5]);
////printf("x_ind_us: %i y_ind_us: %i x_ind: %e y_ind: %e z_ind %e \n", x_ind_us, y_ind_us, x_ind, y_ind, z_ind);
////printf("shift: %i V_perp_shift: %i V_z_shift: %e\n", shift, V_perp_shift, V_z_shift);

//if (Params[4] ==0 )//no Interpolation (nearest neighbor)
//{

//	for (int i_ind = ind; i_ind < DetSize*DetSize; ++i_ind)//for (int i_y = 0; i_y < DetSize; ++i_y)// DetSize
//	{
//		//for (int i_x = 0; i_x < DetSize; ++i_x)
//		//{
//		int i_x = i_ind % DetSize;
//		int i_y = i_ind / DetSize;
//			if (x_ind_us == x_ind && y_ind_us == y_ind)
//				continue;

//			double ACorrVal = Intensity[ind]*Intensity[i_x + DetSize * i_y];//Autocorrellation value
//																			//calculate position
//			double x = (double)i_x - shift;
//			double y = (double)i_y - shift;
//			double z = sqrt(x*x + y*y + DetDist*DetDist) - DetDist;

//			double norm = 1. / sqrt(x*x + y*y + DetDist*DetDist);
//			x = x * norm;
//			y = y * norm;
//			z = z * norm;

//			double dx = x - x_ind;
//			double dy = y - y_ind;
//			double dz = z - z_ind;

//			
//			int V_x = round((dx / VoxelSize)) + V_perp_shift;
//			int V_y = round((dy / VoxelSize)) + V_perp_shift;
//			int V_z = round((dz / VoxelSize)) + V_z_shift;

//			

//			if((V_x > max_PerpVox) || V_x < 0|| (V_y > max_PerpVox) || V_y < 0|| (V_z > max_ZVox) || V_z < 0)
//			{
//			//	printf("dx: %i dy: %i dz: %i max perp: %i max z: %i\n", V_x, V_y, V_z, max_PerpVox, max_ZVox);
//				continue;
//			}
//			

//			atomic_add_float(&(AutoCorr[V_x + V_y*max_PerpVox + V_z*max_PerpVox*max_PerpVox]), ACorrVal);
//			atomic_add_float(&(AutoCorrWeight[V_x + V_y*max_PerpVox + V_z*max_PerpVox*max_PerpVox]), 1.);
//			
//		//}
//	}
//}
//else if (Params[4]==1)//linear interpolation
//{
//	for (int i_ind = ind; i_ind < DetSize*DetSize; ++i_ind)//for (int i_y = 0; i_y < DetSize; ++i_y)// DetSize
//	{
//		//for (int i_x = 0; i_x < DetSize; ++i_x)
//		//{
//		int i_x = i_ind % DetSize;
//		int i_y = i_ind / DetSize;
//			if (x_ind_us == x_ind && y_ind_us == y_ind)
//				continue;

//			double ACorrVal = Intensity[ind] * Intensity[i_x + DetSize * i_y];//Autocorrellation value
//																			//calculate position

//		


//			double x = (double)i_x - shift;
//			double y = (double)i_y - shift;
//			double z = sqrt(x*x + y*y + DetDist*DetDist) - DetDist;

//			double norm = 1. / sqrt(x*x + y*y + DetDist*DetDist);
//			x = x * norm;
//			y = y * norm;
//			z = z * norm;

//			double dx = x - x_ind;
//			double dy = y - y_ind;
//			double dz = z - z_ind;



//			int V_x = floor((dx / VoxelSize)) + V_perp_shift;
//			int V_y = floor((dy / VoxelSize)) + V_perp_shift;
//			int V_z = floor((dz / VoxelSize)) + V_z_shift;

//			if ((V_x>max_PerpVox) || V_x <0 || (V_y > max_PerpVox) || V_y < 0 || (V_z > max_ZVox) || V_z<0)
//			{
//				continue;
//			}

//			double r_x = (dx / VoxelSize) - floor((dx / VoxelSize));
//			double r_y = (dy / VoxelSize) - floor((dy / VoxelSize));
//			double r_z = (dz / VoxelSize) - floor((dz / VoxelSize));

//			double a[8];
//			a[0] = (1 - r_x) * (1 - r_y) * (1 - r_z);
//			a[1] = r_x * (1 - r_y) * (1 - r_z);
//			a[2] = (1 - r_x) * r_y * (1 - r_z);
//			a[3] = (1 - r_x) * (1 - r_y) * r_z;
//			a[4] = r_x * r_y * (1 - r_z);
//			a[5] = r_x * (1 - r_y) * r_z;
//			a[6] = (1 - r_x) * r_y * r_z;
//			a[7] = r_x * r_y * r_z;


//			//(0,0,0)
//			      atomic_add_float(&(AutoCorr[V_x + V_y*max_PerpVox + V_z*max_PerpVox*max_PerpVox]), ACorrVal * a[0]);
//			atomic_add_float(&(AutoCorrWeight[V_x + V_y*max_PerpVox + V_z*max_PerpVox*max_PerpVox]), 1. * a[0]);
//			//(1,0,0)
//			      atomic_add_float(&(AutoCorr[(V_x + 1) + V_y*max_PerpVox + V_z*max_PerpVox*max_PerpVox]), ACorrVal * a[1]);
//			atomic_add_float(&(AutoCorrWeight[(V_x + 1) + V_y*max_PerpVox + V_z*max_PerpVox*max_PerpVox]), 1. * a[1]);
//			//(0,1,0)
//			      atomic_add_float(&(AutoCorr[V_x + (V_y + 1)*max_PerpVox + V_z*max_PerpVox*max_PerpVox]), ACorrVal * a[2]);
//			atomic_add_float(&(AutoCorrWeight[V_x + (V_y + 1)*max_PerpVox + V_z*max_PerpVox*max_PerpVox]), 1. * a[2]);
//			//(0,0,1)
//			      atomic_add_float(&(AutoCorr[V_x + V_y*max_PerpVox + (V_z + 1)*max_PerpVox*max_PerpVox]), ACorrVal * a[3]);
//			atomic_add_float(&(AutoCorrWeight[V_x + V_y*max_PerpVox + (V_z + 1)*max_PerpVox*max_PerpVox]), 1. * a[3]);
//			//(1,1,0)
//			      atomic_add_float(&(AutoCorr[(V_x + 1) + (V_y + 1)*max_PerpVox + V_z*max_PerpVox*max_PerpVox]), ACorrVal * a[4]);
//			atomic_add_float(&(AutoCorrWeight[(V_x + 1) + (V_y + 1)*max_PerpVox + V_z*max_PerpVox*max_PerpVox]), 1. * a[4]);
//			//(1,0,1)
//			      atomic_add_float(&(AutoCorr[(V_x + 1) + V_y*max_PerpVox + (V_z + 1)*max_PerpVox*max_PerpVox]), ACorrVal * a[5]);
//			atomic_add_float(&(AutoCorrWeight[(V_x + 1) + V_y*max_PerpVox + (V_z + 1)*max_PerpVox*max_PerpVox]), 1. * a[5]);
//			//(0,1,1)
//			      atomic_add_float(&(AutoCorr[V_x + (V_y + 1)*max_PerpVox + (V_z + 1)*max_PerpVox*max_PerpVox]), ACorrVal * a[6]);
//			atomic_add_float(&(AutoCorrWeight[V_x + (V_y + 1)*max_PerpVox + (V_z + 1)*max_PerpVox*max_PerpVox]), 1. * a[6]);
//			//(1,1,1)
//			      atomic_add_float(&(AutoCorr[(V_x + 1) + (V_y + 1)*max_PerpVox + (V_z + 1)*max_PerpVox*max_PerpVox]), ACorrVal * a[7]);
//			atomic_add_float(&(AutoCorrWeight[(V_x + 1) + (V_y + 1)*max_PerpVox + (V_z + 1)*max_PerpVox*max_PerpVox]), 1. * a[7]);

//			
//			//DEBUG


//			//DEBUG ENDE
//		//}
//	}
//}