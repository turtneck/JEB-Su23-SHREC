#include <omp.h>

int main( int argc, char **argv ) { 
    int i = 0; // Loop Iterator
    int n = 12; // Number of Iterations
    #pragma omp parallel shared( n ) private( i )
    {
        #pragma omp for nowait
        for( i = 0; i < n; i++ )
        {
            printf( "Thread %d of %d - Iteration %d\n",
                omp_get_thread_num( ),
                omp_get_max_threads( ), i );
        }
        #pragma omp for nowait
        for( i = 0; i < n; i++ )
        {
            printf( "Thread %d of %d - Iteration %d\n",
                omp_get_thread_num( ),
                omp_get_max_threads( ), i );
        }
    }
    return 0;
}
