#train_model.py

import numpy as np
from alexnet import alexnet

WIDTH = 600
HEIGHT = 500
LR = 1e-3
EPOCHS = 10
MODEL_NAME = 'pygta5-car-fast-{}-{}-{}-epochs-300K-data.model'.format(LR, 'alexnetv2',EPOCHS)

model = alexnet(WIDTH, HEIGHT, LR)

train_data = np.load('training_data.npy', allow_pickle=True)

train = train_data[:-1]
test = train_data[-1:]

X = np.array([i[0] for i in train]).reshape(-1, WIDTH, HEIGHT, 1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1, WIDTH, HEIGHT, 1)
test_y = [i[1] for i in test]

print(f"Training data size: {len(train)}")
print(f"Testing data size: {len(test)}")

model.fit({'input': X}, {'targets': Y}, n_epoch=1, validation_set=({'input': test_x}, {'targets': test_y}),
          snapshot_step=500, show_metric=True, run_id=MODEL_NAME, batch_size=32)


model.save(MODEL_NAME)

# tensorboard --logdir=foo:C:/Users/Singularity/Documents/Car/log
