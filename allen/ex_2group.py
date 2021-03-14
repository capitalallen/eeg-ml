from inspect import indentsize
from types import prepare_class
from preprocessing import ex_preprocessing, ex_preprocessing_stress
from preprocessing import shuffle_and_group, get_preprocessed_combined_net_neg,choose_one_stree_group
from cv import Train
from store_results import write_to_json
from multiprocessing import Process
from multiprocessing import Pool
import numpy as np 
"""
leave one out random 3 group 
"""
def ex_stress_groups():
    folder = "./results/stree_2_group/"
    # define low and high array 
    id_l = [1, 6, 9, 10, 11, 12, 15, 16, 18, 19, 21, 23, 24, 26, 29, 30, 31, 32, 33, 39]
    id_h = [0, 2, 3, 4, 5, 7, 8, 13, 14, 17, 20, 22, 25, 27, 28, 34, 35, 36, 37, 38] 
    # get combined x & y
    x,y = get_preprocessed_combined_net_neg()
    # get selected x and selected y 
    # train leaveone 
    # multiprocessing two trains  
    x_l, y_l = choose_one_stree_group(x,y,id_l)
    x_h, y_h = choose_one_stree_group(x,y,id_h)
    trainl = Train(x_l,y_l,0.1)
    trainh = Train(x_h,y_h,0.1)
    accsl,_= trainl.leav_one_train()
    accsh,_= trainl.leav_one_train()
    write_to_json(f"{folder}group_low_.json",accsl)      
    write_to_json(f"{folder}group_high_.json",accsh)      

def test():
    print("test")
if __name__ == "__main__":
    ex_stress_groups()
