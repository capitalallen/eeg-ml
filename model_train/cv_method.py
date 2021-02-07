from numpy.lib.polynomial import roots
import Load_csv_data as lcd 
import train 
from sklearn.metrics import accuracy_score
import numpy as np
import store_to_json as stj 
def get_x_all(data):
    x = data[0]
    # data.size
    for i in range(1,2):
        x = np.concatenate((x,data[i]),axis=0)
    return x

def get_y_all(data):
    y = data[0]
    for i in range(1,2):
        y = np.concatenate((y,data[i]),axis=0)
    return y  
def get_x_train_test(data,index):
    x = data[0]
    x_test = data[index]
    x_train = np.delete(data,index,axis=0)
    return x_train,x_test

def get_y_train_test(data,index):
    y = data[0]
    y_test = data[index]
    y_train = np.delete(data,index,axis=0)
    return y_train.astype(int),y_test.astype(int)

# def get_x_train_test(index,data):
#     train,test = None,None
#     train = data[0] if index !=0 else data[1]
#     for i in range(data.size):
#         if i == index:
#             test = data[i]
#         else:
#             train = np.concatenate((train,data[i]),axis=0)
#     return train,test

# def get_y_train_test(index,data):
#     train,test = None,None
#     train = data[0] if index !=0 else data[1]
#     for i in range(data.size):
#         if i == index:
#             test = data[i]
#         else:
#             train = np.concatenate((train,data[i]),axis=0)
#     return train,test

def with_in(folder,type="rf"):
    # load data 
    get_data = lcd.Load_data() 
    x,y = get_data.get_all_data(folder)
    cv_size = x.size
    # init class
    train_model = train.Train(folder)
    train_model.perform_grid_search(get_x_all(x),get_y_all(y),type)
    # perform grid search
    accs = []
    for i in range(cv_size):
        y_tests, y_preds = [],[]
        for j in range(x[i].shape[0]):
            # define train and test set
            x_train,x_test = get_x_train_test(x[i],j)
            y_train,y_test = get_x_train_test(y[i],j)
            # train 
            temp1,temp2 = train_model.ex_train_within(type,x_train,y_train,x_test,y_test)
            y_tests.append(temp1)
            y_preds.append(temp2)
            # get test set accuracy 
            # append to accs
        accs.append(accuracy_score(y_tests,y_preds))
        # accuracy average 
        # append to json file 
        avg = sum(accs)/len(accs)
        stj.update_results(folder+str(i),[avg,accs])

def run_train_removed(folder,type,remove_type):
    # load data 
    get_data = lcd.Load_data() 
    indexs = []
    if remove_type=="male":
        indexs = [11,18]
    elif remove_type=="female":
        indexs = [13]
    elif remove_type=="mf":
        indexs = [11,18,31]
    x,y = get_data.get_all_data(folder,indexs)
    cv_size = x.size
    # init class
    train_model = train.Train(folder)
    train_model.perform_grid_search(get_x_all(x),get_y_all(y),type)
    accs = []
    for i in range(cv_size):
        y_tests, y_preds = [],[]
        for j in range(x[i].shape[0]):
            # define train and test set
            x_train,x_test = get_x_train_test(x[i],j)
            y_train,y_test = get_x_train_test(y[i],j)
            # train 
            temp1,temp2 = train_model.ex_train_within(type,x_train,y_train,x_test,y_test)
            y_tests.append(temp1)
            y_preds.append(temp2)
            # get test set accuracy 
            # append to accs
        accs.append(accuracy_score(y_tests,y_preds))
        # accuracy average 
        # append to json file 
        avg = sum(accs)/len(accs)
        stj.update_results(folder+"removed"+str(i),[avg,accs])

root_file = "../data/42/"
exp_type = ['neg_vs_net/','neg_and_net/','neg_to_net/']
# for net and neg 
net_and_neg = ['neg/','net/']
gender = ["combined"]
sec = ["_0/"]
model_type = ['rf']
# folder += "neg_and_net/"
for e in exp_type:
    folder = root_file+e
    for n in net_and_neg:
        for i in gender:
            for j in sec:
                for m in model_type:
                    curr_folder = folder + n+i+j
                    # run_train(curr_folder,m)
                    with_in(curr_folder,m)
                    run_train_removed(curr_folder,m,"mf")
