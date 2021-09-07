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


model = create_model()
out = getSingleOutput(model, 'conv3')
out = out[:, 0, :]

# ------------------------------custom --------------------------------
#max = np.max(np.abs(np.copy(out)))
#li = []
#temp = np.linspace(0, max, 5)
#for i in range(4, 0, -1):
#    li.append(-temp[i])

#for i in range(0, 5, 1):
#    li.append(temp[i])
# cmap = clr.ListedColormap(["darkgoldenrod", "gold", "khaki", "lightgoldenrodyellow", "mistyrose", "lightpink", "lightcoral", "red"])
#cmap = clr.ListedColormap(["darkblue", "royalblue", "lightsteelblue", "lightblue", "mistyrose", "lightpink", "lightcoral", "darkred"])
#norm = clr.BoundaryNorm(li, 8)

#plt.pcolor(out, cmap=cmap, norm=norm)
# plt.pcolor(out, cmap='BrBG')
#c = plt.colorbar()
#c.ax.tick_params(labelsize=50)

pcm = plt.pcolor(out, cmap=plt.get_cmap('seismic', 6), vmax=np.max(out), vmin=np.min(out))
c = plt.colorbar(pcm)
c.ax.tick_params(labelsize=14)

plt.xlabel('Filter Index')
plt.ylabel('Filter Weights')
#plt.title('Distribution of Conv layer 3 weights of DC model', fontdict=font)
plt.show()
plt.savefig('DC_layer13.pdf', bbox_inches='tight')
plt.close()


# #-------------------sequential___----------------------
# cmap = plt.get_cmap('hot', 8)
# mat = plt.matshow(out, cmap=cmap, vmin = np.min(out)-.5, vmax = np.max(out)+.5)
# e = np.arange(np.min(out),np.max(out)+1,(np.max(out)+1 -np.min(out))/10)
# cax = plt.colorbar(mat, ticks=np.arange(np.min(out),np.max(out),(np.max(out) -np.min(out))/10))
# plt.show()
