from inspect import indentsize
from types import prepare_class
from preprocessing import ex_preprocessing, ex_preprocessing_stress
from preprocessing import shuffle_and_group, get_preprocessed_combined_net_neg,choose_one_stree_group
from cv import Train
from store_results import write_to_json
from multiprocessing import Process
from multiprocessing import Pool
import numpy as np 
import os 
"""
leave one out random 3 group 
"""
def ex_train(x,y,path):
    train = Train(x,y,0.1)
    accs,_ = train.leav_one_train() 
    write_to_json(path,accs)
def ex():
    folder = "./results/sleep_condition/"
    if not os.path.exists(folder):
        os.makedirs(folder)
    # define low and high array 
    id_good = [0,1,5,17,20,21,23,24,28,37,38]
    id_bad = [2,3,4,7,9,12,14,18,27,29,32] 
    # get combined x & y
    x,y = get_preprocessed_combined_net_neg()
    # get selected x and selected y 
    # train leaveone 
    # multiprocessing two trains  
    x_l, y_l = choose_one_stree_group(x,y,id_good)
    x_h, y_h = choose_one_stree_group(x,y,id_bad)
    p = Process(target=ex_train, args=(x_l,y_l,f"{folder}good.json"))
    p.start()

    p1 = Process(target=ex_train, args=(x_h,y_h,f"{folder}bad.json"))
    p1.start()     

def test():
    print("test")
if __name__ == "__main__":
    ex()
