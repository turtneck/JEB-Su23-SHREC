#include <omp.h>
int main( int argc, char **argv ) { 
    omp_set_nested( 1 ); // Enable Nested Parallelism
    omp_set_dynamic( 0 ); // Disable Dynamic Threads
    
    // Outer Level Parallel Region - 2 Threads
    #pragma omp parallel num_threads( 2 )
    {
        printf( "Outer Level - See this twice.\n" );
        // Inner Level Parallel Region - 2 Threads Each
        #pragma omp parallel num_threads( 2 )
        {printf( "Inner Level - See this four times!\n" );}
    }
    return 0;
}
