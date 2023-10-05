/************************************************************************
 * NASA Docket No. GSC-18,719-1, and identified as “core Flight System: Bootes”
 *
 * Copyright (c) 2020 United States Government as represented by the
 * Administrator of the National Aeronautics and Space Administration.
 * All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may
 * not use this file except in compliance with the License. You may obtain
 * a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ************************************************************************/

/**
 * \file
 *   This file contains the source code for the Sample App.
 */

/*
** Include Files:
*/
#include "my_app_events.h"
#include "my_app_version.h"
#include "my_app.h"
#include "my_app_table.h"

/* The sample_lib module provides the SAMPLE_LIB_Function() prototype */
#include <string.h>
#include <time.h>
#include "sample_lib.h"

//MEEEEE
//#include <process.h>

/*
** global data
*/
MY_APP_Data_t MY_APP_Data;

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *  * *  * * * * **/
/*                                                                            */
/* Application entry point and main process loop                              */
/*                                                                            */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *  * *  * * * * **/
void MY_APP_Main(void)
{
    int32            status;
    CFE_SB_Buffer_t *SBBufPtr;

    /*
    ** Create the first Performance Log entry
    */
    CFE_ES_PerfLogEntry(MY_APP_PERF_ID);

    /*
    ** Perform application specific initialization
    ** If the Initialization fails, set the RunStatus to
    ** CFE_ES_RunStatus_APP_ERROR and the App will not enter the RunLoop
    */
    status = MY_APP_Init();
    if (status != CFE_SUCCESS)
    {
        MY_APP_Data.RunStatus = CFE_ES_RunStatus_APP_ERROR;
    }

    /*
    ** SAMPLE Runloop
    */
    while (CFE_ES_RunLoop(&MY_APP_Data.RunStatus) == true)
    {
        /*
        ** Performance Log Exit Stamp
        */
        CFE_ES_PerfLogExit(MY_APP_PERF_ID);

        /* Pend on receipt of command packet */
        status = CFE_SB_ReceiveBuffer(&SBBufPtr, MY_APP_Data.CommandPipe, CFE_SB_PEND_FOREVER);

        /*
        ** Performance Log Entry Stamp
        */

        //INJECTIONS==========================================================
        CFE_ES_PerfLogEntry(MY_APP_PERF_ID);
        //printf("* < RUN MA: Ogres have layers >\n");
        if (MY_APP_Data.AlgoCount >= MY_APP_Data.AlgoCycles)
        {fclose(MY_APP_Data.AlgoLog);exit(0);}

        if (status == CFE_SUCCESS)
        {
            printf("* < SUCCESS MA >\n");
            //run Algo

            //clock_t t=clock();
            clock_t t1=clock();
            double t2 = time(NULL)+(((double)t1)/CLOCKS_PER_SEC);

            //fprintf(MY_APP_Data.AlgoLog, "exeS%d:\t%ld\n",MY_APP_Data.AlgoCount,time(NULL));
            fprintf(MY_APP_Data.AlgoLog, "exeS%d:\t%f\n",MY_APP_Data.AlgoCount,t2);
            printf("%d\n",MY_APP_Data.AlgoCount);

            system(MY_APP_Data.Algo);
            
            t1=clock();
            t2 = time(NULL)+(((double)t1)/CLOCKS_PER_SEC);
            fprintf(MY_APP_Data.AlgoLog, "exeE%d:\t%f\n",MY_APP_Data.AlgoCount,t2);
            //fprintf(MY_APP_Data.AlgoLog, "exec%d:\t%f\n",MY_APP_Data.AlgoCount,((double)t)/CLOCKS_PER_SEC);
            MY_APP_Data.AlgoCount++;
            
            //-------
            MY_APP_ProcessCommandPacket(SBBufPtr);
        }
        else
        {
            printf("** < FAIL MA >\n");

            CFE_EVS_SendEvent(MY_APP_PIPE_ERR_EID, CFE_EVS_EventType_ERROR,
                              "My APP: SB Pipe Read Error, App Will Exit");

            MY_APP_Data.RunStatus = CFE_ES_RunStatus_APP_ERROR;
        }
    }

    /*
    ** Performance Log Exit Stamp
    */
    CFE_ES_PerfLogExit(MY_APP_PERF_ID);

    CFE_ES_ExitApp(MY_APP_Data.RunStatus);
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *  */
/*                                                                            */
/* Initialization                                                             */
/*                                                                            */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * **/
int32 MY_APP_Init(void)
{
    int32 status;

    MY_APP_Data.RunStatus = CFE_ES_RunStatus_APP_RUN;

    /*
    ** Initialize app command execution counters
    */
    MY_APP_Data.CmdCounter = 0;
    MY_APP_Data.ErrCounter = 0;

    /*
    ** Initialize app configuration data
    */
    MY_APP_Data.PipeDepth = MY_APP_PIPE_DEPTH;

    strncpy(MY_APP_Data.PipeName, "MY_APP_CMD_PIPE", sizeof(MY_APP_Data.PipeName));
    MY_APP_Data.PipeName[sizeof(MY_APP_Data.PipeName) - 1] = 0;

    /*
    ** Register the events
    */
    status = CFE_EVS_Register(NULL, 0, CFE_EVS_EventFilter_BINARY);
    if (status != CFE_SUCCESS)
    {
        CFE_ES_WriteToSysLog("Sample App: Error Registering Events, RC = 0x%08lX\n", (unsigned long)status);
        return status;
    }

    /*
    ** Initialize housekeeping packet (clear user data area).
    */
    CFE_MSG_Init(CFE_MSG_PTR(MY_APP_Data.HkTlm.TelemetryHeader), CFE_SB_ValueToMsgId(MY_APP_HK_TLM_MID),
                 sizeof(MY_APP_Data.HkTlm));

    /*
    ** Create Software Bus message pipe.
    */
    status = CFE_SB_CreatePipe(&MY_APP_Data.CommandPipe, MY_APP_Data.PipeDepth, MY_APP_Data.PipeName);
    if (status != CFE_SUCCESS)
    {
        CFE_ES_WriteToSysLog("Sample App: Error creating pipe, RC = 0x%08lX\n", (unsigned long)status);
        return status;
    }

    /*
    ** Subscribe to Housekeeping request commands
    */
    status = CFE_SB_Subscribe(CFE_SB_ValueToMsgId(MY_APP_SEND_HK_MID), MY_APP_Data.CommandPipe);
    if (status != CFE_SUCCESS)
    {
        CFE_ES_WriteToSysLog("Sample App: Error Subscribing to HK request, RC = 0x%08lX\n", (unsigned long)status);
        return status;
    }

    /*
    ** Subscribe to ground command packets
    */
    status = CFE_SB_Subscribe(CFE_SB_ValueToMsgId(MY_APP_CMD_MID), MY_APP_Data.CommandPipe);
    if (status != CFE_SUCCESS)
    {
        CFE_ES_WriteToSysLog("Sample App: Error Subscribing to Command, RC = 0x%08lX\n", (unsigned long)status);

        return status;
    }

    /*
    ** Register Table(s)
    */
    status = CFE_TBL_Register(&MY_APP_Data.TblHandles[0], "SampleAppTable", sizeof(MY_APP_Table_t),
                              CFE_TBL_OPT_DEFAULT, MY_APP_TblValidationFunc);
    if (status != CFE_SUCCESS)
    {
        CFE_ES_WriteToSysLog("Sample App: Error Registering Table, RC = 0x%08lX\n", (unsigned long)status);

        return status;
    }
    else
    {
        status = CFE_TBL_Load(MY_APP_Data.TblHandles[0], CFE_TBL_SRC_FILE, MY_APP_TABLE_FILE);
    }

    CFE_EVS_SendEvent(MY_APP_STARTUP_INF_EID, CFE_EVS_EventType_INFORMATION, "SAMPLE App Initialized.%s",
                      MY_APP_VERSION_STRING);


    //INJECTIONS==========================================================
    printf("** < INIT MA >\n");


    //dynamically open algo
    FILE *f = fopen("/home/vboxuser/src/algos/intermed.txt", "r");
    if (f == NULL){printf("CANT ACCESS");}  //check can open file

    //line1
    fgets(MY_APP_Data.Algo,100,f);
    MY_APP_Data.Algo[strlen(MY_APP_Data.Algo)-1] = '\0'; //removing last character '\n'
    printf("*** < Algo Dir:\t%s >\n",MY_APP_Data.Algo);

    //line2
    char temp[100];
    fgets(temp,100,f);
    MY_APP_Data.AlgoCycles = atoi(temp);
    printf("*** < Algo Cyc:\t%d >\n",MY_APP_Data.AlgoCycles);

    //line3
    //fgets(temp,100,f);
    //MY_APP_Data.AlgoLog = fopen(temp,"a");
    MY_APP_Data.AlgoLog = fopen("/home/vboxuser/src/algos/intermed2.txt","w");
    if (f == NULL){printf("CANT ACCESS");}  //check can open file


    //print time
    clock_t t1=clock();
    double t2 = time(NULL)+(((double)t1)/CLOCKS_PER_SEC);
    fprintf(MY_APP_Data.AlgoLog, "INIT :\t%f\n",t2);
    

    //END
    MY_APP_Data.AlgoCount=0;
    fclose(f);
    //-------
    return CFE_SUCCESS;
}

/* * * * * * * * * * * * a* * * * * * * * * * * * * * * * * * * * * * * * * * **/
/*                                                                            */
/*  Purpose:                                                                  */
/*     This routine will process any packet that is received on the SAMPLE    */
/*     command pipe.                                                          */
/*                                                                            */
/* * * * * * * * * * * * * * * * * * * * * * * *  * * * * * * *  * *  * * * * */
void MY_APP_ProcessCommandPacket(CFE_SB_Buffer_t *SBBufPtr)
{
    CFE_SB_MsgId_t MsgId = CFE_SB_INVALID_MSG_ID;

    CFE_MSG_GetMsgId(&SBBufPtr->Msg, &MsgId);

    switch (CFE_SB_MsgIdToValue(MsgId))
    {
        case MY_APP_CMD_MID:
            MY_APP_ProcessGroundCommand(SBBufPtr);
            break;

        case MY_APP_SEND_HK_MID:
            MY_APP_ReportHousekeeping((CFE_MSG_CommandHeader_t *)SBBufPtr);
            break;

        default:
            CFE_EVS_SendEvent(MY_APP_INVALID_MSGID_ERR_EID, CFE_EVS_EventType_ERROR,
                              "SAMPLE: invalid command packet,MID = 0x%x", (unsigned int)CFE_SB_MsgIdToValue(MsgId));
            break;
    }
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * **/
/*                                                                            */
/* SAMPLE ground commands                                                     */
/*                                                                            */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * **/
void MY_APP_ProcessGroundCommand(CFE_SB_Buffer_t *SBBufPtr)
{
    CFE_MSG_FcnCode_t CommandCode = 0;

    CFE_MSG_GetFcnCode(&SBBufPtr->Msg, &CommandCode);

    /*
    ** Process "known" SAMPLE app ground commands
    */
    switch (CommandCode)
    {
        case MY_APP_NOOP_CC:
            if (MY_APP_VerifyCmdLength(&SBBufPtr->Msg, sizeof(MY_APP_NoopCmd_t)))
            {
                MY_APP_Noop((MY_APP_NoopCmd_t *)SBBufPtr);
            }

            break;

        case MY_APP_RESET_COUNTERS_CC:
            if (MY_APP_VerifyCmdLength(&SBBufPtr->Msg, sizeof(MY_APP_ResetCountersCmd_t)))
            {
                MY_APP_ResetCounters((MY_APP_ResetCountersCmd_t *)SBBufPtr);
            }

            break;

        case MY_APP_PROCESS_CC:
            if (MY_APP_VerifyCmdLength(&SBBufPtr->Msg, sizeof(MY_APP_ProcessCmd_t)))
            {
                MY_APP_Process((MY_APP_ProcessCmd_t *)SBBufPtr);
            }

            break;

        /* default case already found during FC vs length test */
        default:
            CFE_EVS_SendEvent(MY_APP_COMMAND_ERR_EID, CFE_EVS_EventType_ERROR,
                              "Invalid ground command code: CC = %d", CommandCode);
            break;
    }
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * **/
/*                                                                            */
/*  Purpose:                                                                  */
/*         This function is triggered in response to a task telemetry request */
/*         from the housekeeping task. This function will gather the Apps     */
/*         telemetry, packetize it and send it to the housekeeping task via   */
/*         the software bus                                                   */
/* * * * * * * * * * * * * * * * * * * * * * * *  * * * * * * *  * *  * * * * */
int32 MY_APP_ReportHousekeeping(const CFE_MSG_CommandHeader_t *Msg)
{
    int i;

    /*
    ** Get command execution counters...
    */
    MY_APP_Data.HkTlm.Payload.CommandErrorCounter = MY_APP_Data.ErrCounter;
    MY_APP_Data.HkTlm.Payload.CommandCounter      = MY_APP_Data.CmdCounter;

    /*
    ** Send housekeeping telemetry packet...
    */
    CFE_SB_TimeStampMsg(CFE_MSG_PTR(MY_APP_Data.HkTlm.TelemetryHeader));
    CFE_SB_TransmitMsg(CFE_MSG_PTR(MY_APP_Data.HkTlm.TelemetryHeader), true);

    /*
    ** Manage any pending table loads, validations, etc.
    */
    for (i = 0; i < MY_APP_NUMBER_OF_TABLES; i++)
    {
        CFE_TBL_Manage(MY_APP_Data.TblHandles[i]);
    }

    return CFE_SUCCESS;
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * **/
/*                                                                            */
/* SAMPLE NOOP commands                                                       */
/*                                                                            */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * **/
int32 MY_APP_Noop(const MY_APP_NoopCmd_t *Msg)
{
    MY_APP_Data.CmdCounter++;

    CFE_EVS_SendEvent(MY_APP_COMMANDNOP_INF_EID, CFE_EVS_EventType_INFORMATION, "SAMPLE: NOOP command %s",
                      MY_APP_VERSION);

    return CFE_SUCCESS;
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * **/
/*                                                                            */
/*  Purpose:                                                                  */
/*         This function resets all the global counter variables that are     */
/*         part of the task telemetry.                                        */
/*                                                                            */
/* * * * * * * * * * * * * * * * * * * * * * * *  * * * * * * *  * *  * * * * */
int32 MY_APP_ResetCounters(const MY_APP_ResetCountersCmd_t *Msg)
{
    MY_APP_Data.CmdCounter = 0;
    MY_APP_Data.ErrCounter = 0;

    CFE_EVS_SendEvent(MY_APP_COMMANDRST_INF_EID, CFE_EVS_EventType_INFORMATION, "SAMPLE: RESET command");

    return CFE_SUCCESS;
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * **/
/*                                                                            */
/*  Purpose:                                                                  */
/*         This function Process Ground Station Command                       */
/*                                                                            */
/* * * * * * * * * * * * * * * * * * * * * * * *  * * * * * * *  * *  * * * * */
int32 MY_APP_Process(const MY_APP_ProcessCmd_t *Msg)
{
    int32               status;
    MY_APP_Table_t *TblPtr;
    const char *        TableName = "MY_APP.SampleAppTable";

    /* Sample Use of Table */

    status = CFE_TBL_GetAddress((void *)&TblPtr, MY_APP_Data.TblHandles[0]);

    if (status < CFE_SUCCESS)
    {
        CFE_ES_WriteToSysLog("Sample App: Fail to get table address: 0x%08lx", (unsigned long)status);
        return status;
    }

    CFE_ES_WriteToSysLog("Sample App: Table Value 1: %d  Value 2: %d", TblPtr->Int1, TblPtr->Int2);

    MY_APP_GetCrc(TableName);

    status = CFE_TBL_ReleaseAddress(MY_APP_Data.TblHandles[0]);
    if (status != CFE_SUCCESS)
    {
        CFE_ES_WriteToSysLog("Sample App: Fail to release table address: 0x%08lx", (unsigned long)status);
        return status;
    }

    /* Invoke a function provided by MY_APP_LIB */
    SAMPLE_LIB_Function();

    return CFE_SUCCESS;
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * **/
/*                                                                            */
/* Verify command packet length                                               */
/*                                                                            */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * **/
bool MY_APP_VerifyCmdLength(CFE_MSG_Message_t *MsgPtr, size_t ExpectedLength)
{
    bool              result       = true;
    size_t            ActualLength = 0;
    CFE_SB_MsgId_t    MsgId        = CFE_SB_INVALID_MSG_ID;
    CFE_MSG_FcnCode_t FcnCode      = 0;

    CFE_MSG_GetSize(MsgPtr, &ActualLength);

    /*
    ** Verify the command packet length.
    */
    if (ExpectedLength != ActualLength)
    {
        CFE_MSG_GetMsgId(MsgPtr, &MsgId);
        CFE_MSG_GetFcnCode(MsgPtr, &FcnCode);

        CFE_EVS_SendEvent(MY_APP_LEN_ERR_EID, CFE_EVS_EventType_ERROR,
                          "Invalid Msg length: ID = 0x%X,  CC = %u, Len = %u, Expected = %u",
                          (unsigned int)CFE_SB_MsgIdToValue(MsgId), (unsigned int)FcnCode, (unsigned int)ActualLength,
                          (unsigned int)ExpectedLength);

        result = false;

        MY_APP_Data.ErrCounter++;
    }

    return result;
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                 */
/* Verify contents of First Table buffer contents                  */
/*                                                                 */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
int32 MY_APP_TblValidationFunc(void *TblData)
{
    int32               ReturnCode = CFE_SUCCESS;
    MY_APP_Table_t *TblDataPtr = (MY_APP_Table_t *)TblData;

    /*
    ** Sample Table Validation
    */
    if (TblDataPtr->Int1 > MY_APP_TBL_ELEMENT_1_MAX)
    {
        /* First element is out of range, return an appropriate error code */
        ReturnCode = MY_APP_TABLE_OUT_OF_RANGE_ERR_CODE;
    }

    return ReturnCode;
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                 */
/* Output CRC                                                      */
/*                                                                 */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
void MY_APP_GetCrc(const char *TableName)
{
    int32          status;
    uint32         Crc;
    CFE_TBL_Info_t TblInfoPtr;

    status = CFE_TBL_GetInfo(&TblInfoPtr, TableName);
    if (status != CFE_SUCCESS)
    {
        CFE_ES_WriteToSysLog("Sample App: Error Getting Table Info");
    }
    else
    {
        Crc = TblInfoPtr.Crc;
        CFE_ES_WriteToSysLog("Sample App: CRC: 0x%08lX\n\n", (unsigned long)Crc);
    }
}
