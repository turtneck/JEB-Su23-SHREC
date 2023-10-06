#include <omp.h>
int main( int argc, char **argv ) {
    int *a = malloc( 25 * sizeof( int ) );
    int i = 0; // Loop Iterator
    int n = 25; // Number of Iterations 
    int localSum = 0; // Private Local Sum 
    int totalSum = 0; // Shared Total Sum

    // Fill Array with Values 1 to 25 
    for( i = 0; i < n; i++ ) {a[i] = i + 1;} 
    #pragma omp parallel \
    shared( n, a, totalSum ) \
    private( localSum ) 
    {
        localSum = 0; 
        #pragma omp for 
        for( i = 0; i < n; i++ ) {localSum += a[i];}
        #pragma omp atomic
        totalSum += localSum;
    } 
    printf( "Total sum at end is %d.\n", totalSum );
    return 0;
}
