#include <stdio.h>
#include "gputimer.h"

const int N= 1024; // matrix size is NxN
const int K= 32; // tile size is KxK

void fill_matrix(float *mat)
{
for(int j=0; j < N * N; j++)
mat[j] = (float) j;
}

__global__ void transpose_serial(float in[], float out[])
{
for(int j=0; j < N; j++)
for(int i=0; i < N; i++)
out[j + i*N] = in[i + j*N]; // out(j,i) = in(i,j)
}

__global__ void transpose_parallel_per_row(float in[], float out[])
{
int i = threadIdx.x;
for(int j=0; j < N; j++)
out[j + i*N] = in[i + j*N]; // out(j,i) = in(i,j)
}

__global__ void transpose_parallel_per_element(float in[], float out[])
{
int i = blockIdx.x * K + threadIdx.x;
int j = blockIdx.y * K + threadIdx.y;
out[j + i*N] = in[i + j*N]; // out(j,i) = in(i,j)
}

__global__ void transpose_parallel_per_element_tiled16(float in[], float out[])
{
int in_corner_i = blockIdx.x * 16, in_corner_j = blockIdx.y * 16;
int out_corner_i = blockIdx.y * 16, out_corner_j = blockIdx.x * 16;
int x = threadIdx.x, y = threadIdx.y;
__shared__ float tile[16][16];
// coalesced read from global mem, TRANSPOSED write into shared mem:
tile[y][x] = in[(in_corner_i + x) + (in_corner_j + y)*N];
__syncthreads();
// read from shared mem, coalesced write to global mem:
out[(out_corner_i + x) + (out_corner_j + y)*N] = tile[x][y];
}

int main(int argc, char **argv)
{
int numbytes = N * N * sizeof(float);
float *in = (float *) malloc(numbytes);
float *out = (float *) malloc(numbytes);
fill_matrix(in);
float *d_in, *d_out;
cudaMalloc(&d_in, numbytes);
cudaMalloc(&d_out, numbytes);
cudaMemcpy(d_in, in, numbytes, cudaMemcpyHostToDevice);
GpuTimer timer;

timer.Start();
transpose_serial<<<1,1>>>(d_in, d_out);
timer.Stop();
printf("Transpose_serial: %g ms.\n",timer.Elapsed());

timer.Start();
transpose_parallel_per_row<<<1,N>>>(d_in, d_out);
timer.Stop();
printf("Transpose_per_row: %g ms.\n",timer.Elapsed());

dim3 blocks(N/K,N/K); // blocks per grid
dim3 threads(K,K); // threads per block
timer.Start();
transpose_parallel_per_element<<<blocks,threads>>>(d_in, d_out);
timer.Stop();
printf("Transpose_per_element: %g ms.\n",timer.Elapsed());

dim3 blocks16x16(N/16,N/16); // blocks per grid
dim3 threads16x16(16,16); // threads per block
timer.Start();
transpose_parallel_per_element_tiled16<<<blocks16x16,threads16x16>>>(d_in, d_out);
timer.Stop();
printf("Transpose_tiled: %g ms.\n",timer.Elapsed());
return 0;
}
