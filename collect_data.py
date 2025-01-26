# collect_data.py

import numpy as np
from grabscreen import grab_screen
import cv2
import time
from getkeys import key_check
import os


w =  [1,0,0,0,0,0]
wa = [0,1,0,0,0,0]
wd = [0,0,1,0,0,0]
a_key =  [0,0,0,1,0,0]
d_key =  [0,0,0,0,1,0]
s_key =  [0,0,0,0,0,1]

def keys_to_output(keys):
    '''
    Convert keys to a ...multi-hot... array
     0  1  2  3  4   5   6   
    [W, A, S, D, WA, WD, NOKEY] boolean values. 
    '''
    output = [0, 0, 0]

    if 'W' in keys and 'A' in keys:
        output = wa
    elif 'W' in keys and 'D' in keys:
        output = wd
    elif 'W' in keys:
        output = w
    elif 'A' in keys:  # key 'A'
        output = a_key
    elif 'D' in keys:  # key 'D'
        output = d_key
    elif 'S' in keys:  # key 'S'
        output = s_key  
    return output

file_name = 'training_data.npy'

if os.path.isfile(file_name):
    print('File exists, loading previous data!')
    training_data = list(np.load(file_name, allow_pickle=True))
else:
    print('File does not exist, starting fresh!')
    training_data = []


def main():

    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)


    paused = False
    while(True):

        if not paused:
            # 800x600 windowed mode
            screen = grab_screen(region=(0,40,800,600))
            last_time = time.time()
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            screen = cv2.resize(screen, (600,500))
            # resize to something a bit more acceptable for a CNN
            keys = key_check()
            output = keys_to_output(keys)
            training_data.append([screen,output])
            
            if len(training_data) % 500 == 0:
                print(len(training_data))
                
                if len(training_data) == 5000:
                    np.save(file_name, training_data, allow_pickle=True)
                    for i in range(25):
                        print('DONE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    break

        keys = key_check()
        if 'T' in keys:
            if paused:
                paused = False
                print('unpaused!')
                time.sleep(1)
            else:
                print('Pausing!')
                paused = True
                time.sleep(1)


main()
