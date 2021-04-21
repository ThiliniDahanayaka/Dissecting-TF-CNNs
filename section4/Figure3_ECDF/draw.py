import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = "";

from keras.optimizers import Adam
from keras.models import model_from_json
import numpy as np
import pickle
import os
import matplotlib as matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#%matplotlib inline

plt.rc('font', serif='Arial')
plt.rc('text', usetex='false')
plt.rcParams.update({'font.size': 14})



#normalize between 1 and -1
def normalize(data):
    a = -1
    b = 1
    min = np.min(data)
    max = np.max(data)

    data = a + ((data - min)/(max-min))*(b-a)
    return data


# Create function for ECDF
def ecdf(data):
    xaxis=np.sort(data)  # sort the data
    yaxis=np.arange(1,len(data)+1)/len(data)  # create percentages for y axis from 1% to 100%
    return xaxis, yaxis


def load_model(model_path):
    # load json and create model
    json_file = open(model_path + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_path + ".h5")
    return loaded_model

def create_model(model_path):
    model = load_model(model_path)
    optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
    metrics = ['accuracy']
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=metrics)

    return model

def getWeights(model, layer_name):
    all_weights = {}

    for layer in model.layers:
        w = layer.get_weights()
        n = layer.name
        all_weights[n] = w

    filter_weights = all_weights[layer_name]
    return filter_weights
"/home/sec-user/thilini/WWW-19_backup/Section_4/Figure3_ECDF/raw_weights/archive/DF_onDF_data_model5_ecdf_.pkl"

#font = 60
#plt.rcParams["figure.figsize"] = (22, 13)
plt.grid(b=True, which='major', axis='y', linestyle='-', linewidth=0.5)

model_names1 = ['Video_CNN_model1', 'Video_CNN_model2', 'Video_CNN_model3', 'Video_CNN_model4', 'Video_CNN_model5', ]

model_names2 = ['DF_onDF_data_model1', 'DF_onDF_data_model2', 'DF_onDF_data_model3', 'DF_onDF_data_model4', 'DF_onDF_data_model5' ]

model_names3 = ['DF_onAWF_data_model1', 'DF_onAWF_data_model2', 'DF_onAWF_data_model3', 'DF_onAWF_data_model4', 'DF_onAWF_data_model5' ]

fs_l = 25
fs_t = 30
f = 20

#----------------------------------- To plot ecdf's saved in archives-----------------------------------------------------------------
for model_path in model_names1:
    with open(os.path.join("raw_weights/archive", (model_path+"_ecdf_.pkl")), 'rb') as file:
        h = pickle.load(file)
    x = h[0]
    y = h[1]
    plt.scatter(x, y,alpha=0.5)

plt.legend(['Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5'], fancybox=True, fontsize=fs_l, framealpha=0, handletextpad=0.05)
plt.xlabel("Weight (un-normalized)", labelpad=15, fontsize=fs_t)
plt.ylabel('Percentage of weights', labelpad=15, fontsize=fs_t)
plt.xticks(fontsize=f)
plt.yticks(fontsize=f)
print('before save')
plt.savefig("ECDF_of_DC_raw.png", bbox_inches='tight')

for model_path in model_names2:
    with open(os.path.join("raw_weights/archive", (model_path+"_ecdf_.pkl")), 'rb') as file:
        h = pickle.load(file)
    x = h[0]
    y = h[1]
    plt.scatter(x, y,alpha=0.5)

plt.legend(['Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5'], fancybox=True, fontsize=fs_l, framealpha=0, handletextpad=0.1)
plt.xlabel("Weight (un-normalized)", labelpad=15, fontsize=fs_t)
plt.ylabel('Percentage of weights', labelpad=15, fontsize=fs_t)
plt.xticks(fontsize=f)
plt.yticks(fontsize=f)
print('before save')
plt.savefig("ECDF_of_DF_raw.png", bbox_inches='tight')

for model_path in model_names3:
    with open(os.path.join("raw_weights/archive", (model_path+"_ecdf_.pkl")), 'rb') as file:
        h = pickle.load(file)
    x = h[0]
    y = h[1]
    plt.scatter(x, y,alpha=0.5)

plt.legend(['Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5'], fancybox=True, fontsize=fs_l, framealpha=0, handletextpad=0.1)
plt.xlabel("Weight (un-normalized)", labelpad=15, fontsize=fs_t)
plt.ylabel('Percentage of weights', labelpad=15, fontsize=fs_t)
plt.xticks(fontsize=f)
plt.yticks(fontsize=f)
print('before save')
plt.savefig("ECDF_of_AWF_raw.png", bbox_inches='tight')
