import os

from configobj import ConfigObj
from keras.models import model_from_json
import keras.backend as K
from keras.optimizers import Adamax
from keras.models import Model
from utility_AWF import LoadDataNoDefCW
import matplotlib.pyplot as plt
import matplotlib.colors as colors
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
    model = load_model("/home/sec-user/thilini/1.PAPER_FINAL/DF_CNN_model_training/DF_model_on_AWFdataset/models/DF_onAWF_data_model1")
    optimizer = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    metrics = ['accuracy']
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=metrics)

    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)
    return intermediate_layer_model
    
x, _, _, _, _, _ = LoadDataNoDefCW()

ind = np.arange(x.shape[0])
np.random.shuffle(ind)
x = x[ind[0:500]]
x = x.astype('float32')
plt.plot(np.mean(x, axis=0))
plt.savefig('input.png')
plt.close()
x = x[:, :,np.newaxis]

layer_name = 'block1_adv_act1'


model = create_model(layer_name)

out = model.predict(x)
out = np.mean(out, axis=0)
out = np.transpose(out)

with open('AWF_L1_filter_out.txt', 'wb') as file:
    pickle.dump(out, file)

'''
font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 28,
        }
plt.rcParams["figure.figsize"] = (22, 13)

pcm = plt.pcolor(out, cmap='gist_heat_r', vmax=np.max(out), vmin=np.min(out))
c = plt.colorbar(pcm)
c.ax.tick_params(labelsize=28)

plt.xlabel('Activation position', fontdict=font, labelpad=15)
plt.ylabel('Filter index', fontdict=font)
plt.title('Layer 1', fontdict=font)
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)
plt.savefig('AWF_layer1.png')
'''
