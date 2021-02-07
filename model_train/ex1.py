import Load_csv_data as lcd 
import train 
import numpy as np

def get_x_all(data):
    x = data[0]
    for i in range(1,data.shape[0]):
        x = np.concatenate((x,data[i]),axis=0)
    return x

def get_y_all(data):
    y = data[0]
    for i in range(1,data.shape[0]):
        y = np.concatenate((y,data[i]),axis=0)
    return y  

def get_x_train_test(index,data):
    train,test = None,None
    train = data[0] if index !=0 else data[1]
    for i in range(data.shape[0]):
        if i == index:
            test = data[i]
        else:
            train = np.concatenate((train,data[i]),axis=0)
    return train,test

def get_y_train_test(index,data):
    train,test = None,None
    train = data[0] if index !=0 else data[1]
    for i in range(data.shape[0]):
        if i == index:
            test = data[i]
        else:
            train = np.concatenate((train,data[i]),axis=0)
    return train,test

def run_train(folder,type):
    # load data 
    get_data = lcd.Load_data() 
    x,y = get_data.get_all_data(folder)
    cv = x.shape[0]
    # init class
    train_model = train.Train(folder)
    
    # perform grid search
    
    train_model.perform_grid_search(get_x_all(x),get_y_all(y),type)
    for i in range(cv):
        print("cv:",i)
        # type,cv_num,x_train,y_train,x_test,y_test
        x_train,x_test = get_x_train_test(i,x)
        y_train,y_test = get_y_train_test(i,y)
        # type,cv_num,x_train,y_train,x_test,y_test
        train_model.ex_train(type,str(i),x_train,y_train,x_test,y_test,"to")
def main():
    folder = "../data/42/"
    # exp_type = ['neg_vs_net/','neg_and_net/','neg_to_net/']
    # # for net and neg 
    # net_and_neg = ['neg/','net/']
    # gender = ["male",'female']
    # sec = ["_0/","_1/"]
    # model_type = ['rf','boost']
    # folder += "neg_and_net/"
    # for net and neg 
    net_and_neg = ['neg/','net/']
    gender = ["male",'female',"combined"]
    sec = ["_0/"]
    model_type = ['rf']
    folder += "neg_to_net/"
    # for n in net_and_neg:
    for i in gender:
        for j in sec:
            for m in model_type:
                curr_folder = folder+i+j
                print(curr_folder)
                run_train(curr_folder,m)
if __name__ == "__main__":
    main()
