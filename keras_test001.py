from keras import Input
from keras import backend as K
from keras.engine import Model
from keras.layers import Lambda

import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

a = Input(shape=(2,))
b = Input(shape=(2,))

def minus(inputs):
    x, y = inputs
    return K.mean(x - y, axis=1)
    # return K.mean(x - y)

m = Lambda(minus, name='minus')([a, b])
model = Model(inputs=[a, b], outputs=[m])

v0 = np.array([1,2])
v1 = np.array([3,4])
v2 = np.array([5,6])
print(model.predict([v0.reshape(1,2), v1.reshape(1,2)]))
print(model.predict([v0.reshape(1,2), v2.reshape(1,2)]))
print(model.predict([np.array([v0, v0]), np.array([v1, v2])]))