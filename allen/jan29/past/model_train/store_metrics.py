import pymongo
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt 
import numpy as np 
import os
def changeName(name):
    folder ="../results/"
    name = name.replace("/",",")
    name = name.replace(".","")
    path = folder+name+"/"
    print(path)
    create_folder(path)
    return path

def get_accuracy(y_true,y_pred):
    return accuracy_score(y_true, y_pred)

def save_accuracy(path,accs):
    a = np.asarray(accs)
    np.savetxt(path, a, delimiter=",")

def save_acc_graph(path,accs):
    plt.scatter(range(len(accs)),accs)
    plt.savefig(path)
    plt.close()
def save_coef_graph(path,coefs):
    plt.bar(range(len(coefs)),coefs)
    plt.savefig(path) 
    plt.close()
def get_index(keys):
    keys = list(keys)
    keys.remove('_id')
    keys.remove('name')
    return keys
def create_folder(file_path):
    if not os.path.exists(file_path):
        os.mkdir(file_path)

def get_all_results():
    client = pymongo.MongoClient(
        "mongodb+srv://capitalallen:allen123@cluster0.bc8gt.mongodb.net/<dbname>?retryWrites=true&w=majority")
    db = client.get_database('eeg')
    records = db.results
    for x in records.find():
        # init var 
        path = ""
        indexs = None
        acc = []
        # get name 
        # get path 
        path = changeName(x['name'])
        # get all indexs from key 
        indexs = get_index(x.keys())
        # loop through index 
        for i in indexs:
            # cal and store accuracy 
            # get_accuracy(temp['0']['raw_data']['y_test'],temp['0']['raw_data']['y_test_pred'])
            # store coef graph
            # plt.bar(range(len(temp['0']['coefs'])),temp['0']['coefs'])
            acc.append(get_accuracy(x[str(i)]['raw_data']['y_test'],x[str(i)]['raw_data']['y_test_pred']))
            save_coef_graph(path+"coef"+str(i)+".png",x[str(i)]['coefs'])
        # store acc to csv and image 
        save_accuracy(path+"acc_chart"+".csv",acc)
        save_acc_graph(path+"acc_graph"+".png",acc)
get_all_results()