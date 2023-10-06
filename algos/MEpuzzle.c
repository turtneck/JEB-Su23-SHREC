#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <time.h>

int main(){
    system("/home/vboxuser/src/algos/MEpuzzle/test-solver");

    //log
    FILE *AlgoLog;
    AlgoLog = fopen("/home/vboxuser/src/algos/intermed2a.txt","a");
    if (AlgoLog == NULL){printf("CANT ACCESS");}  //check can open file
    clock_t t1=clock();
    double t2 = time(NULL)+(((double)t1)/CLOCKS_PER_SEC);
    fprintf(AlgoLog, "exeAE:\t%f\n",t2);

    return 0;
}