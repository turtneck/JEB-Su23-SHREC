#include "stdio.h"

__global__
void hello( void )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Which one am I?
    printf( "Hello world from thread %d!\n", i ); // What do I do?
}

int main( void ) {
    printf( "Running Kernel A\n" );
    hello<<<1,1>>>( );
    cudaDeviceSynchronize( );

    printf( "\n\nRunning Kernel B\n" );
    hello<<<1,32>>>( );
    cudaDeviceSynchronize( );

    printf( "\n\nRunning Kernel C\n" );
    hello<<<8,32>>>( );
    cudaDeviceSynchronize( );

    return 0;
}