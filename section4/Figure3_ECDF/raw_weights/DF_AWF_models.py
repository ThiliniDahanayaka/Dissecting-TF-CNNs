import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = "";

from keras.optimizers import Adamax
from keras.models import model_from_json
import numpy as np
import pickle
import matplotlib.pyplot as plt

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
    optimizer = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
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


font = 60
plt.rcParams["figure.figsize"] = (22, 13)
plt.grid(b=True, which='major', axis='y', linestyle='-', linewidth=0.5)

model_names = ['DF_onAWF_data_model1', 'DF_onAWF_data_model2', 'DF_onAWF_data_model3', 'DF_onAWF_data_model4', 'DF_onAWF_data_model5' ]
# model_names = ['DF_onDF_data_model1', 'DF_onDF_data_model2', 'DF_onDF_data_model3', 'DF_onDF_data_model4', 'DF_onDF_data_model5' ]

#DF_on_AWFdataset_model4
#DF_onDF_data_model

#----------------------------------- To generate and plot ecdf's -----------------------------------------------------------------
for model_path in model_names:
    model = create_model(os.path.join("/home/sec-user/thilini/1.PAPER_FINAL/DF_CNN_model_training/DF_model_on_AWFdataset/models/", model_path))
    # model = create_model(os.path.join("/home/sec-user/thilini/1.PAPER_FINAL/DF_CNN_model_training/DF_model/models/",model_path))
    print(model.summary())
    conv_layers = ['block1_conv1', 'block1_conv2', 'block2_conv1', 'block2_conv2', 'block3_conv1', 'block3_conv2',
                   'block4_conv1', 'block4_conv2']
    Dense_layers = ['fc1', 'fc2', 'fc3']
    
    # list1 = np.zeros((2139935,))
    list1 = np.zeros((2193800,))
    i = 0
    for layer_name in conv_layers:
        weights = getWeights(model, layer_name)
        weights = weights[0].reshape(-1)
        list1[i:i+len(weights)] = weights
        i = i+len(weights)

    for layer_name in Dense_layers:
        weights = getWeights(model, layer_name)
        weights = weights[0].reshape(-1)
        list1[i:i + len(weights)] = weights
        i = i + len(weights)

    x, y = ecdf(list1)
    res = []
    res.append(x)
    res.append(y)

    with open('archive/'+model_path+'_ecdf_.pkl', 'wb') as file:
        pickle.dump(res, file)

    plt.scatter(x, y)
plt.legend(['Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5'], fontsize=font, loc='upper left')
plt.xlabel("Value of weight un-normalized", fontsize=font, labelpad=15)
plt.ylabel('Percentage of weights', fontsize=font)
plt.xticks(fontsize=font)
plt.yticks(fontsize=font)
#plt.title('ECDF plots of weights of DF models trained on AWF dataset', fontdict=font)
plt.savefig("archive/ECDF_of_df_on_AWFdataset_raw.png", bbox_inches='tight')


'''
#----------------------------------- To plot ecdf's saved in archies-----------------------------------------------------------------
for model_path in model_names:
    with open(os.path.join("/home/sec-user/thilini/1.PAPER_FINAL/ECDFs_of_all_weights/raw_weights/archive/", (model_path+"_ecdf_.pkl")), 'rb') as file:
        h = pickle.load(file)
    x = h[0]
    y = h[1]

    plt.scatter(x, y)
plt.legend(['Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5'], fontsize=28, loc='lower right')
plt.xlabel("Value of weight", fontdict=font)
plt.ylabel('Percentage of weights', fontdict=font)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
#plt.title('ECDF plots of raw weights of DF models trained on DF dataset', fontdict=font)
plt.savefig("archive/ECDF_of_df_on_AWFdataset_raw_weights.png")
'''
