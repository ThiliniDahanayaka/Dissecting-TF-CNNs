'''
Edit Model path and layer name
'''


from keras.models import model_from_json
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import numpy as np

plt.rc('font', serif='Arial')
plt.rc('text', usetex='false')
plt.rcParams.update({'font.size': 14})
plt.rcParams["figure.figsize"] = (12, 6)

def load_model(model_path):
    # load json and create model
    json_file = open(model_path + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_path + ".h5")
    return loaded_model


def create_model():
    model = load_model("/home/sec-user/thilini/1.PAPER_FINAL/DC_model_training/models/Video_CNN_model1")
    optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
    metrics = ['accuracy']
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=metrics)

    # # intermediate_layer_model = Model(inputs=model.input,
    #                                  outputs=model.get_layer(layer_name).output)
    return model


def getSingleOutput(model, layer_name):
    all_weights = {}

    for layer in model.layers:
        w = layer.get_weights()
        n = layer.name
        all_weights[n] = w

    temp = all_weights[layer_name]
    filter_weights = temp[0]
    return filter_weights

all_neg = 0
all_pos = 0

model = create_model()

layer_names = ['conv1', 'conv2', 'conv3']

for layer_name in layer_names:
    out = getSingleOutput(model, layer_name)

    for f in range(out.shape[2]):
	    for subf in range(out.shape[1]):
		    temp = out[:, subf, f]
		    if (temp > 0).all():
			    all_pos = all_pos +1
		    elif (temp < 0).all():
			    all_neg = all_neg +1 

print("all negs:{}, all pos:{}".format(all_neg, all_pos))


