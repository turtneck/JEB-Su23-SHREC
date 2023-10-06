#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <time.h>

int main(){
    printf("watch this...\n");
    sleep(1);
    printf("done\n");

    FILE *AlgoLog;

    //line3
    //fgets(temp,100,f);
    //AlgoLog = fopen(temp,"a");
    AlgoLog = fopen("/home/vboxuser/src/algos/intermed2a.txt","a");
    if (AlgoLog == NULL){printf("CANT ACCESS");}  //check can open file

    //print time
    clock_t t1=clock();
    double t2 = time(NULL)+(((double)t1)/CLOCKS_PER_SEC);
    fprintf(AlgoLog, "exeAE:\t%f\n",t2);

    return 0;
}