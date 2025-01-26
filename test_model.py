# test_model.py

import numpy as np
from grabscreen import grab_screen
import cv2
import time
from directkeys import PressKey,ReleaseKey, W, A, S, D
from alexnet import alexnet
from getkeys import key_check

import random

WIDTH = 600
HEIGHT = 500
LR = 1e-3
EPOCHS = 10
MODEL_NAME = 'pygta5-car-fast-{}-{}-{}-epochs-300K-data.model'.format(LR, 'alexnetv2',EPOCHS)

t_time = 1

def straight():
##    if random.randrange(4) == 2:
##        ReleaseKey(W)
##    else:
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    time.sleep(t_time)
    ReleaseKey(W)

def left():
    PressKey(A)
    ReleaseKey(W)
    ReleaseKey(D)
    #ReleaseKey(A)
    time.sleep(t_time)
    ReleaseKey(A)

def right():
    PressKey(D)
    #ReleaseKey(W)
    #ReleaseKey(D)
    time.sleep(t_time)
    ReleaseKey(D)
def right():
    PressKey(D)
    #ReleaseKey(W)
    #ReleaseKey(D)
    time.sleep(t_time)
    ReleaseKey(D)

def WA():
    PressKey(W)
    PressKey(A)
    time.sleep(t_time)
    ReleaseKey(A)

def WD():
    PressKey(W)
    PressKey(D)
    time.sleep(t_time)
    ReleaseKey(D)

def s():
    PressKey(S)
    time.sleep(t_time)
    ReleaseKey(S)

model = alexnet(WIDTH, HEIGHT, LR)
model.load(MODEL_NAME)

def main():
    last_time = time.time()
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    paused = False
    while(True):
        
        if not paused:
            # 800x600 windowed mode
            #screen =  np.array(ImageGrab.grab(bbox=(0,40,800,640)))
            screen = grab_screen(region=(0,40,800,640))
            print('loop took {} seconds'.format(time.time()-last_time))
            last_time = time.time()
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            screen = cv2.resize(screen, (600,500))
            print("Acelerar    |     WA     |     A")
            prediction = model.predict([screen.reshape(600,500,1)])[0]
            moves = [0] * len(prediction)  # Inicializar com zeros
            moves[np.argmax(prediction)] = 1  # Configurar o índice do maior valor como 1

            print(moves, prediction)
            if moves == [1,0,0,0,0,0]:
                straight()
            elif moves == [0,1,0,0,0,0]:
                WA()
            elif moves == [0,0,1,0,0,0]:
                WD()
            elif moves == [0,0,0,1,0,0]:
                left()
            elif moves == [0,0,0,0,1,0]:
                right()
            elif moves == [0,0,0,0,0,1]:
                s()
            else:
                print("nothing")
           

        keys = key_check()

        # p pauses game and can get annoying.
        if 'T' in keys:
            if paused:
                paused = False
                time.sleep(1)
            else:
                paused = True
                ReleaseKey(A)
                ReleaseKey(W)
                ReleaseKey(D)
                time.sleep(1)

main()       
