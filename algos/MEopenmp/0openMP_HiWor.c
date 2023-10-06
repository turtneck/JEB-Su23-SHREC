#include <omp.h>

int main( int argc, char *argv[ ] ) {
    int num_threads = 0; // Number of Threads
    int thread_id = 0; // ID Number of Running Thread

    // Fork the threads, giving each one private copy of:
    #pragma omp parallel private( num_threads, thread_id )
    {
        // Get the thread number
        thread_id = omp_get_thread_num( );
        printf( "Hello World from Thread %d\n", thread_id );

        // Have primary print total number of threads used.
        if( thread_id == 0 )
        {
            num_threads = omp_get_num_threads( );
            printf( "Number of Threads = %d\n", num_threads );
        }
    } // All of the threads rejoin the primary thread.
    return 0;
}
