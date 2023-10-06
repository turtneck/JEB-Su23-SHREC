#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <time.h>

int main(){
    printf("\nopenMP0\n-----------\n");
    system("/home/vboxuser/src/algos/MEopenmp/openMP0");
    printf("\nopenMP1\n-----------\n");
    system("/home/vboxuser/src/algos/MEopenmp/openMP1");
    printf("\nopenMP2\n-----------\n");
    system("/home/vboxuser/src/algos/MEopenmp/openMP2");
    //printf("\nopenMP3\n-----------\n");
    //system("/home/vboxuser/src/algos/MEopenmp/openMP3");
    printf("\nopenMP4\n-----------\n");
    system("/home/vboxuser/src/algos/MEopenmp/openMP4");
    printf("\nopenMP5\n-----------\n");
    system("/home/vboxuser/src/algos/MEopenmp/openMP5");
    printf("\nopenMP6\n-----------\n");
    system("/home/vboxuser/src/algos/MEopenmp/openMP6");
    printf("\nopenMP7\n-----------\n");
    system("/home/vboxuser/src/algos/MEopenmp/openMP7");

    //log
    FILE *AlgoLog;
    AlgoLog = fopen("/home/vboxuser/src/algos/intermed2a.txt","a");
    if (AlgoLog == NULL){printf("CANT ACCESS");}  //check can open file
    clock_t t1=clock();
    double t2 = time(NULL)+(((double)t1)/CLOCKS_PER_SEC);
    fprintf(AlgoLog, "exeAE:\t%f\n",t2);

    return 0;
}