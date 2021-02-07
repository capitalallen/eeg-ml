from types import prepare_class
import convert_mat_csv as cmc 
from os import mkdir
from os import path
# neg to net => 1
# net to neg => 0
class Neg_and_net:
    def __init__(self):
        self.data = None 
        self.net_position_c = [[0,0],[0,1]]
        self.net_position_d = [[1,2],[1,3]] 
        self.neg_position_c = [[0,2],[0,3]]
        self.neg_position_d = [[1,0],[1,1]] 
        self.exp_type = "neg_and_net"

    def store_male(self,folder = "../data/",sec=0,net_or_neg=None):
        # create folder for experiment type 
        folder  = folder+self.exp_type+"/"
        self.create_folder(folder)
        # net to net or neg to neg 
        if net_or_neg == "net":
            self.preprocess("m",sec,0)
        else:
            self.preprocess("m",sec,1)

        # create folder 
        folder += net_or_neg + "/"
        self.create_folder(folder)
        # creat folder for gender and second 
        folder += "male" +"_"+str(sec)+"/"
        self.create_folder(folder)
        cmc.store_to_csv(folder,self.data)
    def create_folder(self,file_path):
        if not path.exists(file_path):
            mkdir(file_path)

    def store_female(self,folder = "../data/",sec=0,net_or_neg=None):
        folder  = folder+self.exp_type+"/"
        self.create_folder(folder)
        
        if net_or_neg == "net":
            self.preprocess("f",sec,0)
        else:
            self.preprocess("f",sec,1)
        folder += net_or_neg + "/"
        self.create_folder(folder)    

        folder += "female" +"_"+str(sec)+"/"
        self.create_folder(folder)

        cmc.store_to_csv(folder,self.data)

    def store_mf(self,folder = "../data/",sec=0,net_or_neg=None):
        folder  = folder+self.exp_type+"/"
        self.create_folder(folder)

        if net_or_neg == "net":
            self.preprocess("mf",sec,0)
        else:
            self.preprocess("mf",sec,1)
        folder += net_or_neg + "/"
        self.create_folder(folder)  

        folder += "combined"+"_"+str(sec)+"/"
        self.create_folder(folder)
        cmc.store_to_csv(folder,self.data)
    # preprocessing - general pipelines
    # 0: netural c - 1 vs netural d - 0
    # 1: negative c - 1 vs negative d - 0  
    def preprocess(self,gender,sec,type):
        # get np from mat 
        raw_data = None
        if gender == "m":
            raw_data = cmc.convert_mat_np("m")
        elif gender == "f":
            raw_data = cmc.convert_mat_np("f")
        elif gender == "mf":
            raw_data = cmc.combine_male_female()

        # get pos and neg
        netx,negx = None,None 
        if type==0:
            netx,nety = cmc.get_pos_or_neg(raw_data,self.net_position_c,"net")
            negx,negy = cmc.get_pos_or_neg(raw_data,self.net_position_d,"neg")
        else:
            netx,nety = cmc.get_pos_or_neg(raw_data,self.neg_position_c,"net")
            negx,negy = cmc.get_pos_or_neg(raw_data,self.neg_position_d,"neg")            
        # choose freqency 
        if sec == 0:
            netx = cmc.choose_freq(netx,None,0)
            negx = cmc.choose_freq(negx,None,0)
        elif sec == 1:
            netx = cmc.choose_freq(netx,None,1)
            negx = cmc.choose_freq(negx,None,1)
        else:
            raise Exception("second not defined")
    
        # squeeze_feature
        netx = cmc.squeeze_feature_size(netx)
        negx = cmc.squeeze_feature_size(negx)

        # combine_net_neg 
        x,y = cmc.combine_net_neg(netx,nety,negx,negy)
        
        # add label 
        self.data = cmc.add_labels(x,y)
    
    def test(self):
        print("success")
