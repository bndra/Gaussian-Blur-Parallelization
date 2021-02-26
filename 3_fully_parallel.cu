#include<iostream>
#include<math.h>
#include<stdint.h>
#include<stdlib.h>
#define N 16
#define M 16

__device__
void convolve(uint8_t input[N][M], 
			  int* numer, 
			  int* denom, 
			  int* kernel, 
			  int i, 
			  int j)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	
	int kpos = tx + ty * 5;
	
	int x = i + tx - 2;
	int y = j + ty - 2;
	
	
	if (x>=0 && y>=0 && x<N && y<M)	
	{
		int weightedVal = kernel[kpos] * int(input[x][y]);
		int kVal = kernel[kpos];
		
		atomicAdd(numer, weightedVal);
		atomicAdd(denom, kVal);
	}
}

__global__
void gauss(uint8_t input[N][M], uint8_t output[N][M], int* kernel)
{
	int j = blockIdx.y;
	int i = blockIdx.x;
	
	__shared__ int numer;
	__shared__ int denom;
	
	numer = 0;
	denom = 0;
	
	__syncthreads();
	
	convolve(input, &numer, &denom, kernel, i, j);
	
	if(threadIdx.x==0 && threadIdx.y==0)
	{
		output[i][j] = uint8_t((numer) / (denom));
	}
}

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
	
	uint8_t *image, *blur;
	cudaMallocManaged(&image, N*M*sizeof(uint8_t));
	cudaMallocManaged(&blur, N*M*sizeof(uint8_t));
	
	for (int i = 0; i<N; i++)
		for (int j = 0; j<M; j++)
			reinterpret_cast<uint8_t (*)[M]>(image)[i][j] = rand()% 256;
	
	int* kernel;
	cudaMallocManaged(&kernel, sizeof(int) * 25);
	
	int dummy[25] = { 1, 4, 7, 4, 1,
			  4,16,26,16, 4,
			  7,26,41,26, 7,
			  4,16,26,16, 4,
			  1, 4, 7, 4, 1 };
					  
	for (int i=0; i<25; i++)
		kernel[i] = dummy[i];
		
	dim3 blockSize(5, 5);
	dim3 gridSize(N, M);
	
	print(reinterpret_cast<uint8_t (*)[M]>(image));
	
	gauss<<<gridSize, blockSize>>>(reinterpret_cast<uint8_t (*)[M]>(image), 
					reinterpret_cast<uint8_t (*)[M]>(blur), 
					kernel);
	
	cudaDeviceSynchronize();
	
	std::cout<<"\n";
	print(reinterpret_cast<uint8_t (*)[M]>(blur));
	
	cudaFree(image);
	cudaFree(blur);
	cudaFree(kernel);
	
	return 0;
}
