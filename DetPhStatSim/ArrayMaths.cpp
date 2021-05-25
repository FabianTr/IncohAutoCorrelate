#include "ArrayMaths.h"

void ArrayMaths::Convolve2DFast(fftw_complex* Array, fftw_complex* Buffer, double * Output, std::array<size_t, 2> ArraySize, const fftw_complex* FTKernel, fftw_plan & fft2P, fftw_plan & ifft2P)
{
	if (ArraySize[0] % 2 != 0 || ArraySize[1] % 2 != 0)
	{
		std::cout << "Use even sized Detectors please..." << std::endl;
		throw;
	}



	//Fourier transform input
	fftw_execute_dft(fft2P, Array, Buffer);

	//multiply with (FT)Kernel
	for (size_t i = 0; i < (ArraySize[0] * ArraySize[1]); i++)
	{
		//multiplication of complex numbers
		double a = Buffer[i][0];
		double b = Buffer[i][1];
		double x = FTKernel[i][0];
		double y = FTKernel[i][1];

		Buffer[i][0] = a * x - b * y;
		Buffer[i][1] = a * y + b * x;
	}

	//inverse Fourier transform product
	fftw_execute_dft(ifft2P, Buffer, Array);
	for (int i = 0; i < (ArraySize[0] * ArraySize[1]); i++)
	{
		Array[i][0] /= (double)(ArraySize[0] * ArraySize[1]);
		Array[i][1] /= (double)(ArraySize[0] * ArraySize[1]);
	}

	//in principal there should be no imag contribution
	double * tmp = new double[ArraySize[0] * ArraySize[1]];
	for (size_t i = 0; i < (ArraySize[0] * ArraySize[1]); i++)
	{
		tmp[i] = (double)std::sqrt(Array[i][0] * Array[i][0] + Array[i][1] * Array[i][1]);
	}
	//fft shift
	fftshift2D(tmp, Output, ArraySize[0], ArraySize[1]);
	delete[] tmp;
}