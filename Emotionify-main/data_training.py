import os

import numpy as np

os.environ.setdefault("KERAS_BACKEND", "jax")

from keras.layers import Dense, Input
from keras.models import Model
from keras.utils import to_categorical

is_init = False
size = -1

label = []
dictionary = {}
c = 0

for file_name in os.listdir():
    if file_name.split(".")[-1] == "npy" and file_name.split(".")[0] != "labels":
        if not is_init:
            is_init = True
            X = np.load(file_name)
            size = X.shape[0]
            y = np.array([file_name.split('.')[0]] * size).reshape(-1, 1)
        else:
            X = np.concatenate((X, np.load(file_name)))
            y = np.concatenate((y, np.array([file_name.split('.')[0]] * size).reshape(-1, 1)))

        label.append(file_name.split('.')[0])
        dictionary[file_name.split('.')[0]] = c
        c += 1

for index in range(y.shape[0]):
    y[index, 0] = dictionary[y[index, 0]]
y = np.array(y, dtype="int32")
y = to_categorical(y)

X_new = X.copy()
y_new = y.copy()
counter = 0

cnt = np.arange(X.shape[0])
np.random.shuffle(cnt)

for index in cnt:
    X_new[counter] = X[index]
    y_new[counter] = y[index]
    counter += 1

ip = Input(shape=(X.shape[1],))

m = Dense(512, activation="relu")(ip)
m = Dense(256, activation="relu")(m)
op = Dense(y.shape[1], activation="softmax")(m)

model = Model(inputs=ip, outputs=op)
model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["acc"])
model.fit(X, y, epochs=50)
model.save("model.h5")
np.save("labels.npy", np.array(label))
