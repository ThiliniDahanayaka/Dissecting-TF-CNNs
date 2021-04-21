import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = "0";

from keras.optimizers import Adam
from keras.models import model_from_json
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

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


font = 60
plt.rcParams["figure.figsize"] = (22, 13)
plt.grid(b=True, which='major', axis='y', linestyle='-', linewidth=0.5)

model_names = ['Video_CNN_model1', 'Video_CNN_model2', 'Video_CNN_model3',
               'Video_CNN_model4', 'Video_CNN_model5', ]

'''
#----------------------------------- To generate and plot ecdf's -----------------------------------------------------------------
for model_path in model_names:
    model = create_model(os.path.join("/home/sec-user/thilini/1.PAPER_FINAL/DC_model_training/models/",model_path))
    conv_layers = ['conv1', 'conv2',  'conv3']
    Dense_layers = ['fc1', 'fc_before_softmax']

    list1 = np.zeros((187690,))
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
    list1 = normalize(list1)
    x, y = ecdf(list1)
    res = []
    res.append(x)
    res.append(y)

    with open('archive/'+model_path+'_ecdf_.pkl', 'wb') as file:
        pickle.dump(res, file)

    plt.scatter(x, y)
    print(model_path)
    
plt.legend(['Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5'])
plt.xlabel("Value of weight", fontdict=font)
plt.ylabel('Percentage of weights', fontdict=font)
plt.title('ECDF plots of normalized weights of DC models', fontdict=font)
plt.savefig("archive/ECDF_of_dc.png")
print('done')
'''

#----------------------------------- To plot ecdf's saved in archies-----------------------------------------------------------------
for model_path in model_names:
    with open(os.path.join("/home/sec-user/thilini/1.PAPER_FINAL/ECDFs_of_all_weights/normalized_weights/archive/", (model_path+"_ecdf_.pkl")), 'rb') as file:
        h = pickle.load(file)
    x = h[0]
    y = h[1]

    plt.scatter(x, y)
plt.legend(['Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5'], fontsize=40, loc='lower right'
)
plt.xlabel("Value of weight normalized", fontsize=font, labelpad=15)
plt.ylabel('Percentage of weights', fontsize=font)
plt.xticks(np.arange(-1, 1.5, 0.5), ['-1', '-0.5', '0', '0.5', '1'], fontsize=font)
plt.yticks(fontsize=font)
#plt.title('ECDF plots of weights of DC models', fontdict=font)
plt.savefig("archive/ECDF_of_dc_norm.png", bbox_inches='tight')
