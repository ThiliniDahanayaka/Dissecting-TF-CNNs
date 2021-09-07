import os

from configobj import ConfigObj
from keras.models import model_from_json
import keras.backend as K
from keras.optimizers import Adam
from keras.models import Model
from utility_DC import LoadDataNoDefCW
import matplotlib.pyplot as plt
import numpy as np
import pickle

def load_model(model_path):
    # load json and create model
    json_file = open(model_path + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_path + ".h5")
    return loaded_model


def create_model(layer_name):
    model = load_model("/home/sec-user/thilini/1.PAPER_FINAL/DC_model_training/models/Video_CNN_model1")
    optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0) # Optimizer
    metrics = ['accuracy']
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=metrics)

    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)
    return intermediate_layer_model

x, _, _, _, _, _ = LoadDataNoDefCW()

ind = np.arange(x.shape[0])
np.random.shuffle(ind)
x = x[ind[0:30]]
x = x.astype('float32')
x = x[:, :,np.newaxis]

layer_name = 'conv3'
n_filters = 128

model = create_model(layer_name)

out = model.predict(x)
out = np.mean(out, axis=0)
out = np.transpose(out)

with open('DC_L3_filter_out.txt', 'wb') as file:
    pickle.dump(out, file)

'''
font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 28,
        }
plt.rcParams["figure.figsize"] = (22, 13)

plt.pcolor(out, cmap='gist_heat_r')
c = plt.colorbar()
c.ax.tick_params(labelsize=28)

plt.xlabel('Activation position', fontdict=font, labelpad=15)
plt.ylabel('Filter index', fontdict=font)
plt.title('Layer 2', fontdict=font)
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)
plt.savefig('DC_layer2.png')
'''
