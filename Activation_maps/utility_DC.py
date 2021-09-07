import pickle
import numpy as np

# Load data for non-defended dataset for CW setting
def LoadDataNoDefCW():

    print ("Loading non-defended dataset for closed-world scenario")
    # Point to the directory storing data
    dataset_dir = "/home/sec-user/thilini/1.PAPER_FINAL/DC_model_training/Dataset/"
    # X represents a sequence of traffic directions
    # y represents a sequence of corresponding label (website's label)

    # Load training data
    with open(dataset_dir + 'video_X_train.pkl', 'rb') as handle:
        X_train = np.array(pickle.load(handle))
        X_train = X_train[:, 0:1500]
    with open(dataset_dir + 'video_y_train.pkl', 'rb') as handle:
        y_train = np.array(pickle.load(handle))

    # Load validation data
    with open(dataset_dir + 'video_X_valid.pkl', 'rb') as handle:
        X_valid = np.array(pickle.load(handle))
        X_valid = X_valid[:, 0:1500]
    with open(dataset_dir + 'video_y_valid.pkl', 'rb') as handle:
        y_valid = np.array(pickle.load(handle))

    # Load testing data
    with open(dataset_dir + 'video_X_test.pkl', 'rb') as handle:
        X_test = np.array(pickle.load(handle))
        X_test = X_test[:, 0:1500]
    with open(dataset_dir + 'video_y_test.pkl', 'rb') as handle:
        y_test = np.array(pickle.load(handle))

    print ("Data dimensions:")
    print ("X: Training data's shape : ", X_train.shape)
    print ("y: Training data's shape : ", y_train.shape)
    print ("X: Validation data's shape : ", X_valid.shape)
    print ("y: Validation data's shape : ", y_valid.shape)
    print ("X: Testing data's shape : ", X_test.shape)
    print ("y: Testing data's shape : ", y_test.shape)

    return X_train, y_train, X_valid, y_valid, X_test, y_test