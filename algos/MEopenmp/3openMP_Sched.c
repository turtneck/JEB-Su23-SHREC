#include <omp.h>
#include <stddef.h>
#define SIZE 100000000

int main( int argc, char **argv ) {
    srand( time( NULL ) ); // Seed Random
    int i = 0; // Loop Iterator

    // Initialize Vector Memory Spaces
    double *mainVector = malloc( SIZE * sizeof(double) );
    double *divVector = malloc( SIZE * sizeof(double) );
    double *solVector = malloc( SIZE * sizeof(double) );

    // Fill Vectors
    for( i = 0; i < SIZE; i++ ) {
    mainVector[i] = ( rand( ) % 10000000 ) * 0.01;
    divVector[i] = ( rand( ) % 1000 ) * 0.01;
    solVector[i] = 0;
    }

    // Perform Processing
    double start = omp_get_wtime( );
    #pragma omp parallel for num_threads( 4 ) \
    shared( mainVector, divVector, solVector ) \
    private( i ) schedule( static )
    for( i = 0; i < SIZE; i++ )
        {solVector[i] = mainVector[i] / divVector[i];}

    double end = omp_get_wtime( );
    double solTime = end - start;
    printf( "Complete - %lf Seconds\n", solTime );
    return 0;
}
