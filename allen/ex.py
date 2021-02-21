from types import prepare_class
from preprocessing import ex_preprocessing, ex_preprocessing_stress
from cv import Train
from store_results import write_to_json
from multiprocessing import Process
def ex_within():
    # within, with bad data 
    file = "./results/"
    # x,y = ex_preprocessing(0)
    # train = Train(x,y)
    # all_acc,avg_acc = train.within_train()
    # write_to_json(file+"within_with_bad.json",all_acc,avg_acc)
    # remove bad data: boy: 4, 12; girls: 1,2
    x,y = ex_preprocessing(1)
    train = Train(x,y)
    all_acc,avg_acc = train.within_train()
    write_to_json(file+"within_without_bad.json",all_acc,avg_acc)

def ex_leave_one():
    file = "./results/"
    x,y = ex_preprocessing(0)
    train = Train(x,y)
    all_acc= train.leav_one_train()
    write_to_json(file+"leaveone_with_bad.json",all_acc,[])
    # remove bad data: boy: 4, 12; girls: 1,2
    # x,y = ex_preprocessing(1,index=[4,12,24,25])
    # train = Train(x,y)
    # all_acc = train.leav_one_train()
    # write_to_json(file+"leavone_without_bad.json",all_acc,[])

def ex_stress_leavone():
    file = "./results/"
    x,y = ex_preprocessing_stress(0)
    train = Train(x,y)
    all_acc= train.leav_one_train()
    write_to_json(file+"stress_leaveone_with_bad.json",all_acc,[])

    # remove bad data: boy: 4, 12; girls: 1,2
    # x,y = ex_preprocessing_stress(1)
    # train = Train(x,y)
    # all_acc = train.leav_one_train()
    # write_to_json(file+"stress_leavone_without_bad.json",all_acc,[])

def test():
    print("test")
if __name__ == "__main__":
    # p = Process(target=ex_within)
    # p.start()
    # p.join()

    # p1 = Process(target=ex_leave_one)
    # p1.start()
    # p1.join()

    ex_stress_leavone()   
    
