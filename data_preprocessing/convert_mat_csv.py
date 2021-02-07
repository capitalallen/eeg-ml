import mat73
import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import r2_score
# from sklearn.metrics import mean_squared_error
# from IPython.display import display
# from sklearn.metrics import accuracy_score
# from sklearn.utils import shuffle
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import GridSearchCV
# from xgboost import XGBClassifier

# convert male or female mat to numpy 
# f or m 
def convert_mat_np(gender=None):
    m_file = "../raw_data/Emotrans1_girl_data_preprocessed_42.mat"
    f_file = "../raw_data/Emotrans1_Boy_data_preprocessed_42.mat"
    # m_file = "../raw_data/Emotrans1_Boy_data_preprocessed_33.mat"
    # f_file = "../raw_data/Emotrans1_girl_data_preprocessed_33.mat"
    if gender == "f":
        data_dict_female = mat73.loadmat(f_file, use_attrdict=True)
        return np.array(data_dict_female["All_Feature"])
    elif gender == 'm':
        data_dict_female = mat73.loadmat(m_file, use_attrdict=True)
        return np.array(data_dict_female["All_Feature"])  
    else: 
        raise Exception("gender not defined")

def combine_male_female():
    m = convert_mat_np("m")
    f = convert_mat_np("f")
    return np.concatenate((m,f),axis=0)

#
# for netural to netural and negative to netural 
# select positives and negatives from 4*4 grid
# type neg or net 
def get_pos_or_neg(data,position,t=None):
    arr = []
    label = []
    label_type = -1
    if t == 'neg':
        label_type = 0
    elif t == 'net':
        label_type = 1
    else:
        print("Wrong type - get_pos_or_neg()")
        return 
    for i in range(data.shape[0]):
        temp = []
        for j in position:
            for k in data[i][j[0]][j[1]]:
                temp.append(k)
        temp = np.array(temp)
        if label_type == 1:
            label.append(np.ones((temp.shape[0],1)))
        else:
            label.append(np.zeros((temp.shape[0],1)))
        arr.append(temp)
    return np.array(arr), np.array(label)

#
# for netural to negative 
# select positives and negatives from 4*4 grid
def get_difference(data,position,type=None):
    arr = []
    label = []
    label_type = -1
    if type == 'neg':
        label_type = 0
    elif type == 'net':
        label_type = 1
    else:
        print("Wrong type - get_pos_or_neg()")
        return 
    for i in range(data.shape[0]):
        diff = (data[i][position[0][0]][position[0][1]]+data[i][position[1][0]][position[1][1]])/2 - (data[i][position[3][0]][position[3][1]]+data[i][position[2][0]][position[2][1]])/2
        if label_type == 1:
            label.append(np.ones((diff.shape[0],1)))
        else:
            label.append(np.zeros((diff.shape[0],1)))
        arr.append(diff)
    return np.array(arr), np.array(label)


# select frequenceis and (0-4s -> 0 or 0.5-4.5s -> 1)
# output: x 
def choose_freq(data=None,freq = None, sec=None):
    if freq:
        data = np.delete(data,freq,axis=3)
    if sec == 0:
        for i in range(data.shape[0]):
            data[i] = np.delete(data[i], 1, axis=3)
        return data
    elif sec == 1:
        for i in range(data.shape[0]):
            data[i] = np.delete(data[i], 0, axis=3) 
        return data
    else:
        print('sec not specified')
        return data 

# reshape to 128*8*1 = 1024
# return x 
def squeeze_feature_size(data):
    # print(data.shape)
    # print(data.shape)
    # size = data.shape
    # # print(size)
    # data = data.reshape(size[0],size[1],size[2]*size[3]*size[4])
    # return data
    # print(data.shape)
    for i in range(data.shape[0]):
        size = data[i].shape
        data[i] = data[i].reshape(size[0], size[1]*size[2]*size[3])
    print(data.shape)
    return data

# combine net and neg 
def combine_net_neg(netX,netY,negX,negY,type=False):
    # iterate through each person concat x and y respectively
    if type:
        netX =  np.concatenate((netX,negX),axis=0)
        netY = np.concatenate((netY,negY),axis=0)
    else:
        for i in range(netX.size):
            netX[i] = np.concatenate((netX[i],negX[i]),axis=0)
            netY[i] = np.concatenate((netY[i],negY[i]),axis=0)
    return netX,netY

def add_labels(data,label,type=False):
    # loop through data size 
    # add one dimension to data 
    # add label value to that new dimension 
    if type:
        return np.concatenate((data,label),axis=2)
    else:
        for i in range(data.shape[0]):
            data[i] = np.concatenate((data[i],label[i]),axis=1)
        return np.array(data) 

# store to csv 
# input name data 
# output True or false 
# file name: type + person # + csv 
def store_to_csv(folder,data):
    folder += "" if folder.endswith("/") else "/"
    for i in range(data.shape[0]):
        curr_path = folder + str(i)+".csv"  
        np.savetxt(curr_path, data[i], delimiter=",")

# get label and delete last value 
# data from csv. delete 1025 and append that as label 
# output: data and label 
def get_data_label(data):
    label=[]
    for i in range(data.shape[0]):
        label.append(data[i][:,1024])
        data[i] = np.delete(data[i],1024,axis=1)
    return data, np.array(label)

def test():
    print("success")

