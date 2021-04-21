import matplotlib.pyplot as plt
import numpy as np
from utility import LoadDataNoDefCW

# #DC dataset
# def get_classwise_data(x, y):
#     temp = np.concatenate((x, y[:, np.newaxis]), axis=1)
#     temp = temp[temp[:, 500].argsort()]
#     return temp
#
# def get_data(sorted, class_label):
#     traces_per_class = 253
#     x = sorted[class_label*traces_per_class:(class_label+1)*traces_per_class]
#
#     return x
#
#
# x1, y = LoadDataNoDefCW()
# sorted = get_classwise_data(x1, y)
#
# for i in range(0, 1):
#     x = get_data(sorted[:, 0:500], i)
#     plt.bar(np.arange(500), x[5], color='orange')
#     x = np.mean(x, axis=0)
#     # plt.plot(np.arange(696, 1500), np.zeros((804,)), color='orange')
#     plt.rcParams["figure.figsize"] = (22, 13)
#     plt.plot(x, color='blue')
#     plt.legend(['Mean input trace of class 0', 'Sample input trace of class 0'], loc='upper right', fontsize=28)
#     plt.xticks(fontsize=28)
#     plt.yticks(fontsize=28)
#     plt.xlabel('Time step', fontsize=28, labelpad=15)
#     plt.ylabel('Number of uplink packets', fontsize=28)
#     #plt.title("Class_"+str(i)+"_mean_input")
#     plt.show()
#     #plt.savefig("rf\DF_Dataset_class_"+str(i)+"_mean_input.png")
#     plt.close()

#DF dataset
def get_classwise_data(x, y):
    temp = np.concatenate((x, y[:, np.newaxis]), axis=1)
    temp = temp[temp[:, 1500].argsort()]
    return temp

def get_data(sorted, class_label):
    traces_per_class = 800
    x = sorted[class_label*traces_per_class:(class_label+1)*traces_per_class]

    return x


x1, y = LoadDataNoDefCW()
sorted = get_classwise_data(x1, y)

for i in range(55, 56):
    x = get_data(sorted[:, 0:1500], i)
    plt.bar(np.arange(1500), x[5], color='orange')
    x = np.mean(x, axis=0)
    plt.rcParams["figure.figsize"] = (22, 13)
    plt.plot(x, color='blue')
    plt.legend(['Mean input trace of class 55', 'Sample input trace of class 55'], loc='upper right', fontsize=28)
    plt.plot(np.arange(696, 1500), np.zeros((804,)), color='orange')
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.xlabel('Packet Number', fontsize=28, labelpad=15)
    plt.ylabel('Packet direction', fontsize=28)
    #plt.title("Class_"+str(i)+"_mean_input")
    plt.show()
    #plt.savefig("rf\DF_Dataset_class_"+str(i)+"_mean_input.png")
    plt.close()


# #AWF dataset
# def get_classwise_data(x, y):
#     temp = np.concatenate((x, y[:, np.newaxis]), axis=1)
#     temp = temp[temp[:, 1500].argsort()]
#     return temp
#
# def get_data(sorted, class_label):
#     traces_per_class = 1750
#     x = sorted[class_label*traces_per_class:(class_label+1)*traces_per_class]
#
#     return x
#
#
# x1, y = LoadDataNoDefCW()
# sorted = get_classwise_data(x1, y)
#
# for i in range(175, 176):
#     x = get_data(sorted[:, 0:1500], i)
#     plt.bar(np.arange(1500), x[5], color='orange')
#     x = np.mean(x, axis=0)
#     plt.rcParams["figure.figsize"] = (22, 13)
#     plt.plot(x, color='blue')
#     plt.legend(['Mean input trace of website sharepoint.com', 'Sample input trace of website sharepoint.com'], loc='upper right', fontsize=28)
#     plt.xticks(fontsize=28)
#     plt.yticks(fontsize=28)
#     plt.xlabel('Packet number', fontsize=28, labelpad=15)
#     plt.ylabel('Packet direction', fontsize=28)
#     #plt.title("Class_"+str(i)+"_mean_input")
#     plt.show()
#     #plt.savefig("rf\DF_Dataset_class_"+str(i)+"_mean_input.png")
#     plt.close()
