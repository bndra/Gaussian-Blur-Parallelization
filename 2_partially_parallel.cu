#include<iostream>
#include<math.h>
#include<stdint.h>
#include<stdlib.h>
#define N 16
#define M 16

/*
each thread of this function handles one of the 25 convolution procedures
necessary for a 5x5 box filter(aka kernel). 25 pixels from the original 
image must each be matched to one of the 25 values from the box filter, 
and it must be done through use of their individual threadId.
*/
__global__
void convolve(uint8_t input[N][M], 
			  int *numer, 
			  int* denom, 
			  int* kernel, 
			  int i, 
			  int j)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	
	//location in box kernel to be convolved by this thread
	int kpos = tx + ty * 5;
	
	//pixel location in the input matrix, (x, y)
	int x = i + tx - 2;
	int y = j + ty - 2;
	
	/*
	We now know which location from the kernel matches which pixel 
	from the image, but before we continue we must account for 
	the bounds of the input matrix. Depending on the pixel being
	sampled from the original image at (i, j), we may not be able 
	to make use of the entire kernel. Some threads may try to 
	access out of bounds when (i, j) lies close to the border. In
	this case we only use the threads that lie within the bounds 
	of the image. Our image is of size NxM so:
		0 <= x < N
		0 <= y < M
	*/
	if (x>=0 && y>=0 && x<N && y<M)	
	{
		/*
		The convolution procedure is to average the pixel values 
		from the original image with some being weighted more than
		others. 25 pixels in the original image are weighted by 
		a factor equal to its corresponding value in the kernel.
		Then, all these weighted values are accumulated and divided
		by the total weight of the kernel. It would be pointless
		for each and every thread to perform the division (as it
		would be exactly the same every time), so we only go as 
		far as accumulating the weighted values and kernel values 
		in global memory. atomicAdd prevents the accumulation from
		writing over itself.
		*/
		int weightedVal = kernel[kpos] * int(input[x][y]);
		int kVal = kernel[kpos];
		
		atomicAdd(numer, weightedVal);
		atomicAdd(denom, kVal);
	}
}

void gauss(uint8_t input[N][M], uint8_t output[N][M])
{
	/*
	First I declare and allocate global space for our box filter.
	I will be using a Gaussian filter, which is a bell curve 
	with greater values in the middle. Using this filter for 
	such a convolution is called a gaussian blur and has several
	applications; I am familiar with it from scaling images and 
	feature extraction algorithms such as SIFT. Gaussian filters
	of different sizes and distributions may be employed here, 
	and generating them would be a significant upgrade over my 
	hardcoding of the standard 5x5 gaussian filter.
	*/
	int* kernel;
	cudaMallocManaged(&kernel, sizeof(int) * 25);
	int dummy[25] = { 1, 4, 7, 4, 1,
			  4,16,26,16, 4,
			  7,26,41,26, 7,
			  4,16,26,16, 4,
			  1, 4, 7, 4, 1 };
	for (int i=0; i<25; i++)
		kernel[i] = dummy[i];
	
	//accumulators which our convolve function requires
	int *numer;
	int *denom;
	cudaMallocManaged(&numer, sizeof(int));
	cudaMallocManaged(&denom, sizeof(int));
	
	/*
	Before I can call convolve I must define the dimensions of the
	block. A block is a collection of threads to be run together in 
	parallel, and I have decided each block will handle the gaussian
	of each pixel. That means we need 25 threads per block, which 
	can be arranged in a 5x5 formation to better align with the 5x5
	kernel.
	*/
	dim3 blockSize(5,5);
	
	/*
	(i, j) represents the coordinates of the pixel we're performing
	a gaussian blur on. the following nested loops iterate through 
	every pixel of the input image matrix.
	*/
	for (int j = 0; j<N; j++)
	{
		for (int i = 0; i<M; i++)
		{
			//explained in convolution procedure
			*numer = 0;
			*denom = 0;
			convolve<<<1,blockSize>>>(input, numer, denom, kernel, i, j);
			cudaDeviceSynchronize();
			
			//could this be parallelized as well? is it worth it?
			output[i][j] = uint8_t((*numer) / (*denom));
		}
	}
	cudaFree(kernel);
	cudaFree(numer);
	cudaFree(denom);
}

/*
print function for the values of a matrix of unsigned 8 bit ints,
otherwise known as the data values of a greyscale image.
*/
void print(uint8_t image[N][M])
{
	for (int i=0; i<N; i++)
	{
		for (int j=0; j<M; j++)
		{
			std::cout<< int(image[i][j]) << ",\t";
		}
		std::cout<< "\n";
	}
}


int main()
{
	srand(NULL);
	uint8_t *image, blur[N][M];
	cudaMallocManaged(&image, N*M*sizeof(uint8_t));
	for (int i = 0; i<N; i++)
		for (int j = 0; j<M; j++)
			reinterpret_cast<uint8_t (*)[M]>(image)[i][j] = rand()% 256;
	
	/*
	cudaMallocManaged has certain limitations when it comes to 2D arrays
	so image has been allocated as a 1D array and then cast to a 2D.
	blur doesn't need to be allocated to global mem (doesn't run on device
	code), so it's declared locally as a 2D array and passed as such.
	*/
	print(reinterpret_cast<uint8_t (*)[M]>(image));
	gauss(reinterpret_cast<uint8_t (*)[M]>(image), blur);
	std::cout<<"\n";
	print(blur);
	
	cudaFree(image);
	cudaFree(blur);
	
	return 0;
	
}
