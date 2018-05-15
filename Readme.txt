*****

TODO:
	* check the format of the pixlemap and get it compatible with the stored intensity by cheetah !!!!!!!! 
	* develop some kind of external scripting to operate the programm with different modes and settings
	* profile and improve C(q) creation 


*****
Class structure:

Settings:
	Here all settings/options are stored which might be used by other classes.
	It also contains methods to load and parse config files

Detector:
	This class contains methods to load and store Pixel-maps, Pixel-masks and Intensity data.
	Also the main processing is done here (Autocorrelation kernels and C(q)-map creation).

ACMesh:
	This classes main purpose is to store correlated data in voxels and also store additional information such as voxel size and dimension mapping.
	For (sparse) kernels, ACMesh (as Bigmesh) accepts vectors with q values to be added to the Mesh.


*****
	C(q)-creation Profiling at Tesla K20 (per exposure):
		-LinInterpol: ~ 54min
		-NearesNeigh: ~ 12min 








	==================
	SCRATCH
	==================
			//if (InterpolMode == 0) //nearest Neighbor
		//{


			//if (ind == 0)
		//{
		//	printf("%d, %d, %d/n", fs, ms, ss);
		//}

		//}
		//if (InterpolMode == 1) //linear
		//{
		//	//unsigned int fsf, msf, ssf;//fast-scan-, medium-scan-, slow-scan- floor
		//	//fsf = (unsigned int)floor(q[0]) + MeshCenter;
		//	//msf = (unsigned int)floor(q[1]) + MeshCenter;
		//	//ssf = (unsigned int)floor(q[2]);
		//	//float SepF, SepM, SepS; //according Seperators
		//	//SepF = q[0] - floor(q[0]);
		//	//SepM = q[1] - floor(q[1]);
		//	//SepS = q[2] - floor(q[2]);

		//	////add to mesh (8 entries each)
		//	//atomic_add_float(&(CQ[fsf + msf * MeshSize + ssf * MeshSize*MeshSize]), Val * (1 - SepF)*(1 - SepM)*(1 - SepS));              //A + 0
		//	//atomic_add_float(&(CQ[(fsf + 1) + (msf + 0) * MeshSize + (ssf + 0) * MeshSize*MeshSize]), Val * (SepF)*(1 - SepM)*(1 - SepS));//ssf + 1
		//	//atomic_add_float(&(CQ[(fsf + 0) + (msf + 1) * MeshSize + (ssf + 0) * MeshSize*MeshSize]), Val * (1 - SepF)*(SepM)*(1 - SepS));//msf + 1
		//	//atomic_add_float(&(CQ[(fsf + 0) + (msf + 0) * MeshSize + (ssf + 1) * MeshSize*MeshSize]), Val * (1 - SepF)*(1 - SepM)*(SepS));//ssf + 1
		//	//atomic_add_float(&(CQ[(fsf + 1) + (msf + 1) * MeshSize + (ssf + 0) * MeshSize*MeshSize]), Val * (SepF)*(SepM)*(1 - SepS));    //ffs + 1 ; msf + 1
		//	//atomic_add_float(&(CQ[(fsf + 0) + (msf + 1) * MeshSize + (ssf + 1) * MeshSize*MeshSize]), Val * (1 - SepF)*(SepM)*(SepS));    //msf + 1 ; ssf + 1
		//	//atomic_add_float(&(CQ[(fsf + 1) + (msf + 0) * MeshSize + (ssf + 1) * MeshSize*MeshSize]), Val * (SepF)*(1 - SepM)*(SepS));    //ffs + 1 ; ssf + 1
		//	//atomic_add_float(&(CQ[(fsf + 1) + (msf + 1) * MeshSize + (ssf + 1) * MeshSize*MeshSize]), Val * (SepF)*(SepM)*(SepS));        // A + 1
		//	printf("TOIMPLEMENT");
		//}
















		===================

		Options.MReference << 6.227, 0, 0,	0, 8.066, 0,	0, 0, 11.1;