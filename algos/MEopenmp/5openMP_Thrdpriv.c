#include <omp.h>
int num = 0;
int main( int argc, char **argv ) { 
    // Disable Dynamic Threads
    omp_set_dynamic( 0 );

    // Set Num as ThreadPrivate
    #pragma omp threadprivate( num )

    int thread = 0; // Thread Number
    num = 6; // Change Primary Thread Num Value
    printf( "First Parallel Region\n" );
    #pragma omp parallel private( thread ) copyin( num )
    {
        thread = omp_get_thread_num( );
        num = thread * thread + num;
        printf( " Thread %d - Value %d\n", thread, num );
    }

    printf( "Primary Thread\n" );
    printf( "Second Parallel Region\n" );
    #pragma omp parallel
    {
        thread = omp_get_thread_num( );
        printf( " Thread %d - Value Remains %d\n",
        thread, num );
    }
    return 0;
}
