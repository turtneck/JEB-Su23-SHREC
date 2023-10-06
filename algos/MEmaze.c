#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <time.h>

int main(){
    printf("Solving...\n");
    system("/home/vboxuser/src/algos/MEmaze/pathfinder /home/vboxuser/src/algos/MEmaze/tests/maze00.png /home/vboxuser/src/algos/MEmaze/tests/maze.png");
    
    //check file exists (solve)
    if (access("/home/vboxuser/src/algos/MEmaze/tests/maze.png",F_OK) ==0)
    {printf("Solved!\n");}
    
    printf("Removing file...\n");
    remove("/home/vboxuser/src/algos/MEmaze/tests/maze.png");


    //log
    FILE *AlgoLog;
    AlgoLog = fopen("/home/vboxuser/src/algos/intermed2a.txt","a");
    if (AlgoLog == NULL){printf("CANT ACCESS");}  //check can open file
    clock_t t1=clock();
    double t2 = time(NULL)+(((double)t1)/CLOCKS_PER_SEC);
    fprintf(AlgoLog, "exeAE:\t%f\n",t2);

    return 0;
}