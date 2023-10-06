#include <omp.h>
int main( int argc, char* argv[ ] ) { 
    int data = 0; // Data Read by Thread 
    int flag = 0; // Flag to Release Thread
    #pragma omp parallel num_threads( 2 )
    {
        if( omp_get_thread_num( ) == 0 )
        {
            data = 42;
            // Flush data to thread one, order the write.
            #pragma omp flush( flag, data )
            flag = 1;
            // Ensure thread 1 sees the change.
            #pragma omp flush( flag )
        }
        
        else if( omp_get_thread_num( ) == 1 )
            {
                // Loop until we see the update to the flag.
                #pragma omp flush( flag, data )
                while( flag < 1 ) {
                #pragma omp flush( flag, data )
                }
                // Values of flag and data are undefined.
                printf( "A Flag = %d Data = %d\n", flag, data );
                #pragma omp flush(flag, data)
                // Data will be 42, flag still undefined.
                printf( "B Flag = %d Data = %d\n", flag, data );
            }
    }
}
