from inspect import indentsize
from types import prepare_class
from preprocessing import ex_preprocessing, ex_preprocessing_stress
from preprocessing import shuffle_and_group, get_preprocessed_combined_net_neg,choose_one_stree_group
from cv import Train
from store_results import write_to_json,write_to_json_random
from multiprocessing import Process
from multiprocessing import Pool
import numpy as np 
def ex_within(i):
    # within, with bad data 
    folder = "./results/"
    x,y = ex_preprocessing(0)
    file = "within_logistic"
    # for i in [0.1,50,100,1000]: # ,50,100,1000
    train = Train(x,y,i)
    avgs,coefs = train.within_train()
    write_to_json(folder+file+str(i)+".json",avgs,coefs)
    # remove bad data: boy: 4, 12; girls: 1,2
    # x,y = ex_preprocessing(1)
    # train = Train(x,y)
    # all_acc,avg_acc = train.within_train()
    # write_to_json(file+"within_without_bad.json",all_acc,avg_acc)

# def ex_within():
#     # within, with bad data 
#     file = "./results/"
#     index = [1,2,3]
#     x,y = ex_preprocessing_sepecial(0)
#     print(x.shape)
#     return 
    # train = Train(x,y)
    # index = 0
    # all_acc,avg_acc = train.within_train_special(4,index)
    # write_to_json(file+"within_with_bad.json",all_acc,avg_acc)
    # remove bad data: boy: 4, 12; girls: 1,2
    # x,y = ex_preprocessing(1)
    # train = Train(x,y)
    # all_acc,avg_acc = train.within_train()
    # write_to_json(file+"within_without_bad.json",all_acc,avg_acc)

def ex_leave_one(i):
    file = "./results/"
    remove_index = [] 
    x,y = ex_preprocessing(0)
    # for i in [0.1,50,100,1000]:
    train = Train(x,y,i)
    # # train.update_param(i)
    avg_acc, coefs= train.leav_one_train()
    write_to_json(file+"leaveone_46_raw_" +str(i)+".json",avg_acc,coefs)
    # remove bad data: boy: 4, 12; girls: 1,2
    # x,y = ex_preprocessing(1,index=[4,12,24,25])
    # train = Train(x,y)
    # all_acc = train.leav_one_train()
    # write_to_json(file+"leavone_without_bad.json",all_acc,[])

def ex_stress_leavone(i):
    file = "./results/"
    stress_levels = ['l','m','h'] # ,'m','h'
    train_type = "order_46_raw_stress_" # if i == 0 else "without_bad_"
    for j in stress_levels:
        print(train_type,j)
        x,y = ex_preprocessing_stress(0,j)
        train = Train(x,y,i)
        accs,coefs= train.leav_one_train()
        write_to_json(file+"leave_one_"+train_type+j+str(i)+".json",accs,coefs)

    # remove bad data: boy: 4, 12; girls: 1,2
    # x,y = ex_preprocessing_stress(1)
    # train = Train(x,y)
    # all_acc = train.leav_one_train()
    # write_to_json(file+"stress_leavone_without_bad.json",all_acc,[])
"""
leave one out random 3 group 
"""
def ex_stress_random():
    folder = "./results/random_stress/"
    # define 40 element array 
    ids = [i for i in range(40)]
    # get combined x & y
    x,y = get_preprocessed_combined_net_neg()
    # do 100 trial 
    for trail in range(100):
        # get randomly splitted group 
        groups = shuffle_and_group(ids)
        # iterate 3 group
        for num in range(len(groups)):
            # train each group 
            # store accuracy trail#_group#
            x_selected, y_selected = choose_one_stree_group(x,y,groups[num])
            train = Train(x_selected,y_selected,0.1)
            accs,_= train.leav_one_train()
            write_to_json_random(f"{folder}trailNum{str(trail)}_groupNum{str(num)}.json",accs,groups[num])   
def test():
    print("test")
if __name__ == "__main__":
    # p = Process(target=ex_within)
    # p.start()
    # p.join()

    # p1 = Process(target=ex_leave_one)
    # p1.start()
    # p1.join()
    # ex_within()
    # ex_within()
    # ex_stress_leavone()
    # 0.1,50,100,1000
    ex_stress_random()
    # p = Process(target=ex_leave_one, args=(50,))
    # p.start()

    # p1 = Process(target=ex_stress_leavone, args=(50,))
    # p1.start()

    # p2 = Process(target=ex_stress_leavone, args=(100,))
    # p2.start()

    # p3 = Process(target=ex_stress_leavone, args=(1000,))
    # p3.start()
    #ex_stress_leavone() 
    # ex_leave_one_33()
    # ex_leave_one_stress_33()
    # alphas = [0.1] # [0.1, 50, 100,1000]
    # with Pool(5) as p:
    #     p.map(ex_stress_leavone, alphas)
