#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

#define BUZZ 100

int main(){//int argc, char*argv[]){
    /*
    printf("%s\n",argv[1]);
    if (argv[1][0] == '0'){system("/home/vboxuser/src/algos/waitc");}
    else{printf("fail!\n");}
    */

    
    FILE *f = fopen("/home/vboxuser/src/algos/intermed.txt", "r");
    char ftxt[BUZZ];
    char temp[BUZZ];
    int AlgoCycles;

    if (f == NULL){printf("CANT ACCESS");}
    fgets(ftxt,BUZZ,f);
    printf("%s\n",ftxt);

    fgets(temp,100,f);
    AlgoCycles = atoi(temp);
    printf("Algo Cyc:\t%d\n",AlgoCycles);

    //system(ftxt);



    //---
    fgets(temp,100,f);
    FILE *AlgoLog = fopen(temp,"a");
    if (f == NULL){printf("CANT ACCESS");}  //check can open file

    fprintf(AlgoLog,"lol");



    fclose(f);
    return 0;
}