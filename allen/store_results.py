import json 
import numpy as np 
import matplotlib.pyplot as plt 
from pathlib import Path
# store accuracy 
# def write_to_json(file_name,all_acc,avg_acc):
#     data = {} 
#     data['name'] = file_name 
#     data['all_acc'] = all_acc
#     data['avg_acc'] = avg_acc
#     with open(file_name, 'w') as outfile:
#         json.dump(data, outfile)

# store accuracy and coefs
def write_to_json(file_name,avg_acc,coefs=None):
    data = {} 
    data['name'] = file_name 
    data['avg_acc'] = avg_acc
    # data['coefs'] = coefs
    with open(file_name, 'w') as outfile:
        json.dump(data, outfile)

# read_accuracy 
def read_json(file_name):
    return json.load(file_name)

"""
input: folder, coefs_np_arr
change to abs value 
store only 0 and 1 to adjust 
store avg of all people to avg 
plot 
store sorted and top 20 to json file 
"""
def write_to_json_graph(file_name,avg_acc,coefs):
    data = {} 
    data['top_20'] = avg_acc
    data['sorted'] = coefs
    with open(file_name, 'w') as outfile:
        json.dump(data, outfile)
def store_graph_sorted_results(folder,arr,threhold=0.0001):
    Path(folder).mkdir(parents=True, exist_ok=True)
    folder += "/"
    arr = np.absolute(arr)
    adjust = arr >= threhold
    adjust = adjust.astype(int)
    adjust = np.sum(adjust,axis=0)
    x = [i for i in range(adjust.shape[1])]
    
    avg = np.average(arr,axis=0)

    plt.bar(x, adjust[0])
    plt.xlabel('2048 features')
    plt.ylabel('count of 0s or 1s of all people')
    plt.savefig(folder+"count.png")
    plt.close()

    plt.bar(x, avg[0])
    plt.xlabel('2048 features')
    plt.ylabel('coef of average of all people')
    plt.savefig(folder+"results.png")
    plt.close()
    
    temp = [] 
    for i in range(len(x)):
        temp.append([x[i],avg[0][i]])
    temp.sort(key=lambda x:x[1],reverse=True)
    write_to_json_graph(folder+"sorted_results.json",temp[:20],temp)