import sys
import time
import datetime
import os
from threading import Thread
import keyboard
import statistics

def logtime():
    return str(time.time()).split(".")[0]

def avg(arr):
    t=0
    for i in arr: t=t+i
    return t/len(arr)

def fmetric(arr,algo,type):
    # -- split up data: arr
    c_met=[]
    sx_arr=[];e_arr=[];ex_arr=[];t_arr=arr[2:-1]
    #print("met_arr split:",t_arr)
    for x in range(int(len(t_arr)/3)):
        sx_arr.append(t_arr[(3*x)])
        e_arr.append(t_arr[(3*x)+1])
        ex_arr.append(t_arr[(3*x)+2])
    #print("sx:",sx_arr);print("e: ",e_arr);print("ex:",ex_arr)

    # ----- cFS metrics
    cycl = int(len(sx_arr))
    #print("cycl",cycl)
    #ExecAvg
    '''t=0
    for x in range(cycl): t=t+(e_arr[x]-sx_arr[x])
    c_met.append(t/cycl)'''
    #ExecCall
    t=0
    for x in range(cycl): t=t+(ex_arr[x]-sx_arr[x])
    c_met.append(t/cycl)
    #ThreadCall
    '''t=0
    for x in range(cycl-1): t=t+(sx_arr[x+1]-ex_arr[x])
    c_met.append(t/(cycl-1))'''
    #Std dev
    temp=[]
    for x in range(cycl): temp.append(ex_arr[x]-sx_arr[x])
    c_met.append( statistics.stdev(temp) )


    # ------------------------------------
    #t_str = "INIT,ExecAvg,ExecCall,ThreadCall:\t"+str(round(arr[1]-arr[0] ,6))+", "
    t_str = "INIT,ExecCall,StdExecCall:\t"+str(round(arr[1]-arr[0] ,6))+", "
    for x in range(len(c_met)):
        t_str=t_str+str(round(c_met[x],6))
        if x<len(c_met)-1: t_str=t_str+", "
    print(t_str)

    with open("/home/vboxuser/src/logs/"+algo+".log", 'a') as f:
        if type == 1:
            f.write(str(cycl)+'\n')
            f.write("cFS\t"+t_str+'\n')
        else:
            f.write("FPR\t"+t_str+'\n')
    if type==1:
        with open("/home/vboxuser/src/logs/cFS_init_"+algo+".log", 'a') as f:
            f.write(str(round(arr[1]-arr[0] ,6))+'\n')
    else:
        with open("/home/vboxuser/src/logs/FPR_init_"+algo+".log", 'a') as f:
            f.write(str(round(arr[1]-arr[0] ,6))+'\n')





'''def execAVG(a1,a2):
def execCall(a1,a2):
def ThreadCall(a1,a2):'''
#----
def run_cFS(t):
    print("running cFS...\n--------------------------")
    close_f = open('algos/intermed2a.txt', 'w')
    close_f.close()

    log = open("/home/vboxuser/src/logs/"+t+"/cFS.log", 'w')

    #log.write("Start:\t"+str(datetime.datetime.now())+"\n")
    #log.write("Start:\t"+logtime()+"\n")
    log.write("Start:\t"+str(time.time())+"\n")

    res = os.system("./cfsbd.sh")

    # read from intermediate file for cfs logs cause of linux permissions blehhhh
    arr1=[];arr2=[]
    with open('algos/intermed2.txt', 'r') as f:
        for item in f:  arr1.append(item)
    with open('algos/intermed2a.txt', 'r') as f:
        for item in f:  arr2.append(item)

    #print(arr1);print(arr2)

    log.write(arr1[0])  #init
    for x in range(len(arr2)):
        #print(x,":",(2*x)+1,(1+x)*2)
        log.write(arr1[(2*x)+1])
        log.write(arr2[x])
        log.write(arr1[(1+x)*2])


    #log.write("End  :\t"+str(datetime.datetime.now())+"\n")
    #log.write("End  :\t"+logtime())
    log.write("End  :\t"+str(time.time())+"\n")

    print("-----------okie-----------")

def run_FPRIME(t):
    print("\n\n\n\n\n\nrunning FPRIME...\n--------------------------")
    close_f = open('algos/intermed2a.txt', 'w')
    close_f.close()
    close_f = open('algos/intermed3.txt', 'w')
    close_f.close()

    log = open("/home/vboxuser/src/logs/"+t+"/FPRIME.log", 'w')

    #log.write("Start:\t"+str(datetime.datetime.now())+"\n")
    #log.write("Start:\t"+logtime()+"\n")
    log.write("Start:\t"+str(time.time())+"\n")

    res = os.system("./fpbd.sh")

    # read from intermediate file for cfs logs cause of linux permissions blehhhh
    arr1=[];arr2=[]
    with open('algos/intermed3.txt', 'r') as f:
        for item in f:  arr1.append(item)
    with open('algos/intermed2a.txt', 'r') as f:
        for item in f:  arr2.append(item)

    #print(arr1);print(arr2)

    log.write(arr1[0])  #init
    for x in range(len(arr2)):
        #print(x,":",(2*x)+1,(1+x)*2)
        log.write(arr1[(2*x)+1])
        log.write(arr2[x])
        log.write(arr1[(1+x)*2])


    #log.write("End  :\t"+str(datetime.datetime.now())+"\n")
    #log.write("End  :\t"+logtime())
    log.write("End  :\t"+str(time.time())+"\n")

    print("-----------okie-----------")


if __name__ == '__main__':
    print(sys.argv)
    arr = ["waitc","MEmazec","MEpuzzlec","MEsortc","MEopenmp_c","MEcuda_c"]
    print(
"""=============================
Which Prog?
1: wait
2: maze         (serial, ece302)
3: puzzle       (serial, ece302)
4: BinTreeSort  (serial, ece302)
5: OpenMP Suite (parallel, SHREC)
6: Cuda         (parallel, SHREC)
=============================""")

    inp1 = int(sys.argv[1])
    inp2 = "0"#input("Cycles?: ")

    #----
    print("building c files...")
    res = os.system("./cbuild.sh")

    #----
    print("making logs...")
    t=str(datetime.datetime.now())[:-7]
    t=t.replace(" ","_")
    print(t)
    os.makedirs("/home/vboxuser/src/logs/"+t)
    c_file = open("/home/vboxuser/src/logs/"+t+"/cFS.log", 'w')
    f_file = open("/home/vboxuser/src/logs/"+t+"/FPRIME.log", 'w')

    #----
    print("making intermediary file...")
    with open('algos/intermed.txt', 'w') as f:
        f.write('/home/vboxuser/src/algos/'+arr[int(inp1)-1]+'\n')
        f.write(inp2+'\n')
        #f.write("/home/vboxuser/src/logs/"+t+"/cFS.log")



    #===============================================================================
    #run-----
    run_cFS(t)
    #stall = input("press to continue")
    run_FPRIME(t)


    #===============================================================================


    #----
    print("analysis...")
    c_arr=[];f_arr=[]
    with open("/home/vboxuser/src/logs/"+t+"/cFS.log", 'r') as f:
        for item in f:  c_arr.append(float(item.split(":\t")[1][:-1]))
    with open("/home/vboxuser/src/logs/"+t+"/FPRIME.log", 'r') as f:
        for item in f:  f_arr.append(float(item.split(":\t")[1][:-1]))
    



    # ----------------------------------------------------------------------
    print("=============================")
    print("Algo:",arr[int(inp1)-1])
    print("cycl:",inp2)

    print("CFS: INIT\t"+str(round(c_arr[1]-c_arr[0],6)))
    print("FPR: INIT\t"+str(round(f_arr[1]-f_arr[0],6)))

    with open("/home/vboxuser/src/logs/inits/cFS_init_"+arr[int(inp1)-1]+".log", 'a') as f:
        f.write(str(round(c_arr[1]-c_arr[0] ,6))+'\n')
    with open("/home/vboxuser/src/logs/inits/FPR_init_"+arr[int(inp1)-1]+".log", 'a') as f:
        f.write(str(round(f_arr[1]-f_arr[0] ,6))+'\n')
