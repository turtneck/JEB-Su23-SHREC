import sys
import time
import os
#from threading import Thread
import threading
import keyboard


#----------------------------
def t_func1():
    res = os.system("sudo python3 run_inits.py "+inp)
def t_func2():
    time.sleep(20)
    keyboard.press_and_release('ctrl+c')

inp = input("#:")
print(sys.argv)

x1 = threading.Thread(target=t_func1, args=[])
x2 = threading.Thread(target=t_func2, args=[])
x1.start();x2.start()
x1.join();x2.join()