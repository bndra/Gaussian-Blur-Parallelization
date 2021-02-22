#include<iostream>
#include<math.h>
#include<stdint.h>
#include<stdlib.h>
#define N 16
#define M 16

__global__
void convolve(uint8_t input[N][M], uint8_t *val, int i, int j)
{
	int kernel[25] = { 1, 4, 7, 4, 1,
			   4,16,26,16, 4,
			   7,26,41,26, 7,
			   4,16,26,16, 4,
			   1, 4, 7, 4, 1 };
	int k_pos = 0;
	int weight = 0;
	int denom = 0;
	for (int y = j-2; y<j+3; y++)
	{
		if (y>=0 && y<M)
		{
			for (int x = i-2; x<i+3; x++)
			{
				if (x>=0 && x<N)
				{
					//printf("(%d,%d)\nkpos = %d\n",x,y,k_pos);
					int k = kernel[k_pos];
					weight += k * int(input[x][y]);
					denom += k;
				}
				k_pos++;
			}
		}
		else
			k_pos+=5;
	}
	*val = uint8_t(weight/denom);
}

void gauss(uint8_t input[N][M], uint8_t output[N][M])
{
	uint8_t *val;
	cudaMallocManaged(&val, sizeof(uint8_t));
	for (int i = 0; i<N; i++)
	{
		for (int j = 0; j<M; j++)
		{
			convolve<<<1,1>>>(input, val, i, j);
			cudaDeviceSynchronize();
			output[i][j] = *val;
		}
	}
	cudaFree(val);
}

void print(uint8_t image[N][M])
{
	for (int i=0; i<N; i++)
	{
		for (int j=0; j<M; j++)
			std::cout<< int(image[i][j]) << ",\t";
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
	
	print(reinterpret_cast<uint8_t (*)[M]>(image));
	gauss(reinterpret_cast<uint8_t (*)[M]>(image), blur);
	std::cout<<"\n";
	print(blur);
	
	cudaFree(image);
	cudaFree(blur);
	
	return 0;
	
}
