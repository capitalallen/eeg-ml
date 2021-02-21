import Neg_vs_net as nsn 
import Neg_to_net as ntn
import Neg_and_net as nan
def main():
    # temp = nsn.Neg_vs_net()
    # for i in [1]:
    #     temp.store_male(sec=i)
    #     temp.store_female(sec=i)
    #     temp.store_mf(sec=i)
    # temp = ntn.Neg_and_net()
    # for i in [0,1]:
    #     for j in ['net','neg']:
    #         # temp.store_male(sec=i,net_or_neg=j)
    #         temp.store_female(sec=i,net_or_neg=j)
    #         # temp.store_mf(sec=i,net_or_neg=j)      

    temp = ntn.Neg_to_net()
    file = "../data/42/"
    i = 0
    temp.store_male(file,sec=i)
    temp.store_female(file,sec=i)
    temp.store_mf(file,sec=i)      
if __name__ == "__main__":
    main()

