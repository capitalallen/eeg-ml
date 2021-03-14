from inspect import indentsize
from types import prepare_class
from preprocessing import get_preprocessed_male_female
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
    accs,_ = train.leave_one_train() 
    write_to_json(path,accs)
def ex():
    folder = "./results/male_female_order/"
    if not os.path.exists(folder):
        os.makedirs(folder)

    xm,ym = get_preprocessed_male_female("m")
    xf,yf = get_preprocessed_male_female("f")
    
    p = Process(target=ex_train, args=(xm,ym,f"{folder}male.json"))
    p.start()

    p1 = Process(target=ex_train, args=(xf,yf,f"{folder}female.json"))
    p1.start()      

def test():
    print("test")
if __name__ == "__main__":
    ex()
