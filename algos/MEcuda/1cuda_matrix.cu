#include "stdio.h"

__global__
void matrixMulNaive( float* C, float* A, float* B, int wA, int wB )
{
// Block Index
int bx = blockIdx.x;
int by = blockIdx.y;
// Thread Index
int tx = threadIdx.x;
int ty = threadIdx.y;
// Accumulate Row i of A and Column j of B
int i = by * blockDim.y + ty;
int j = bx * blockDim.x + tx;
float accu = 0.0;
for( int k=0; k <wA; k++ ) {
accu = accu + A[ i * wA + k ] * B[ k * wB + j ];
}
// Write the block sub-matrix to device memory;
// each thread writes one element
C[ i * wB + j ] = accu;
}