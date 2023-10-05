// ======================================================================
// \title  HelloWorld.cpp
// \author vboxuser
// \brief  cpp file for HelloWorld component implementation class
// ======================================================================


#include <MyComponents/HelloWorld/HelloWorld.hpp>
#include <FpConfig.hpp>

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

namespace MyComponents {

  // ----------------------------------------------------------------------
  // Construction, initialization, and destruction
  // ----------------------------------------------------------------------

  //INJECTIONS - init ==========================================================
  HelloWorld :: HelloWorld(const char *const compName) : HelloWorldComponentBase(compName)
  {
    m_greetingCount = 0;

    printf("*** < HelloWorld INIT >\n");  // home/vboxuser/src/fprime/MyProject/MyDeployment/logs/__date ran__/MyDeployment.log


    //dynamically open algo
    FILE *f = fopen("/home/vboxuser/src/algos/intermed.txt", "r");
    if (f == NULL){printf("CANT ACCESS");}  //check can open file

    //line1
    fgets(Algo,100,f);
    Algo[strlen(Algo)-1] = '\0'; //removing last character '\n'
    printf("*** < Algo Dir:\t%s >\n",Algo);

    //line2
    char temp[100];
    fgets(temp,100,f);
    AlgoCycles = atoi(temp); //allowed ticks
    printf("*** < Algo Cyc:\t%d >\n",AlgoCycles);

    //line3
    //fgets(temp,100,f);
    //AlgoLog = fopen(temp,"a");
    AlgoLog = fopen("/home/vboxuser/src/algos/intermed3.txt","a");
    if (f == NULL){printf("CANT ACCESS");}  //check can open file


    //print time
    clock_t t1=clock();
    double t2 = time(NULL)+(((double)t1)/CLOCKS_PER_SEC);
    fprintf(AlgoLog, "INIT :\t%f\n",t2);
    

    //END
    AlgoCount = 0;  //ticker per times algo run
    fclose(f);
  }

  HelloWorld :: ~HelloWorld()
  {}

  // ----------------------------------------------------------------------
  // Command handler implementations
  // ----------------------------------------------------------------------

  void HelloWorld ::
    SAY_HELLO_cmdHandler(
        const FwOpcodeType opCode,
        const U32 cmdSeq,
        const Fw::CmdStringArg& greeting
    )
  {
    // Copy the command string input into an event string for the Hello event
    Fw::LogStringArg eventGreeting(greeting.toChar());
    // Emit the Hello event with the copied string
    this->log_ACTIVITY_HI_Hello(eventGreeting);
    
    this->tlmWrite_GreetingCount(++this->m_greetingCount);
    
    // Tell the fprime command system that we have completed the processing of the supplied command with OK status
    this->cmdResponse_out(opCode, cmdSeq, Fw::CmdResponse::OK);

    //INJECTIONS - cycle ==========================================================
    printf("*** < is this thing on? >\n");
    
    clock_t t1=clock();
    double t2 = time(NULL)+(((double)t1)/CLOCKS_PER_SEC);

    fprintf(AlgoLog, "exeS%d:\t%f\n",AlgoCount,t2);
    printf("%d\n",AlgoCount);
    system(Algo);
    
    t1=clock();
    t2 = time(NULL)+(((double)t1)/CLOCKS_PER_SEC);
    fprintf(AlgoLog, "exeE%d:\t%f\n",AlgoCount,t2);
    AlgoCount++;


    //if (AlgoCount >= AlgoCycles){fclose(AlgoLog);exit(0);}
    //do interaction (use count,exit) with selenium not in here

    /*
    clock_t t=clock();
    system(Algo); //run prog
    t=clock()-t;
    fprintf(AlgoLog, "exec%d:\t%f\n",AlgoCount,((double)t)/CLOCKS_PER_SEC);
    AlgoCount++;
    */

  }

} // end namespace MyComponents
