//find transpose of a matrix

#include <stdio.h>
#include<stdlib.h>
const int N= 1024; // matrix size is NxN

void print_matrix(float *mat)
{
for(int j=0; j < N; j++)
{
for(int i=0; i < N; i++) { printf("%4.4g ", mat[i + j*N]); }
printf("\n");
}
}

// fill a matrix with sequential numbers in the range 0..N-1
void fill_matrix(float *mat)
{
for(int j=0; j < N * N; j++)
mat[j] = (float) j;
}

void transpose_CPU(float in[], float out[])
{
for(int j=0; j < N; j++)
for(int i=0; i < N; i++)
out[j + i*N] = in[i + j*N]; // out(j,i) = in(i,j)
}

int main(int argc, char **argv)
{
int numbytes = N * N * sizeof(float);
float *in = (float *) malloc(numbytes);
float *out = (float *) malloc(numbytes);
fill_matrix(in);
//print_matrix(in);
transpose_CPU(in, out);
//print_matrix(out);
return 0;
}



