import Load_csv_data as lcd 
import train 
import numpy as np

def get_x_train_test(data,index):
    x = data[0]
    x_test = data[index]
    x_train = np.delete(data,index,axis=0)
    return x_train,x_test

def get_y_all(data):
    y = data[0]
    for i in range(1,data.size):
        y = np.concatenate((y,data[i]),axis=0)
    return y  

def get_x_train_test(index,data):
    train,test = None,None
    train = data[0] if index !=0 else data[1]
    for i in range(data.size):
        if i == index:
            test = data[i]
        else:
            train = np.concatenate((train,data[i]),axis=0)
    return train,test

def get_y_train_test(index,data):
    train,test = None,None
    train = data[0] if index !=0 else data[1]
    for i in range(data.size):
        if i == index:
            test = data[i]
        else:
            train = np.concatenate((train,data[i]),axis=0)
    return train,test

def with_in(folder,type):
    # load data 
    get_data = lcd.Load_data() 
    x,y = get_data.get_all_data(folder)
    cv_size = x.size
    # init class
    train_model = train.Train(folder)
    # train_model.perform_grid_search(get_x_all(x),get_y_all(y),type)
    # perform grid search
    for i in range(cv_size):
        accs = []
        for j in range(x[i].shape[0]):
            # define train and test set
            x_train,x_test = get_x_train_test(x[i],j)
            print(x_train.shape)
            print(x_test.shape)
            return
            # train 
            # get test set accuracy 
            # append to accs
        # accuracy average 
        # append to json file 
    # train_model.perform_grid_search(get_x_all(x),get_y_all(y),type)
    # for i in range(cv):
    #     # type,cv_num,x_train,y_train,x_test,y_test
    #     x_train,x_test = get_x_train_test(i,x)
    #     y_train,y_test = get_y_train_test(i,y)
    #     # type,cv_num,x_train,y_train,x_test,y_test
    #     train_model.ex_train(type,str(i),x_train,y_train,x_test,y_test)
folder = "../data/"
exp_type = ['neg_and_net/']
# for net and neg 
net_and_neg = ['neg/','net/']
gender = ["combined"]
sec = ["_0/"]
model_type = ['rf']
folder += "neg_and_net/"
for n in net_and_neg:
    for i in gender:
        for j in sec:
            for m in model_type:
                curr_folder = folder + n+i+j
                # run_train(curr_folder,m)
                with_in(curr_folder,m)
