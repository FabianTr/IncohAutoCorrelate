*****

TODO:
	* check the format of the pixlemap and get it compatible with the stored intensity by cheetah
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









	==================
	SCRATCH
	==================
