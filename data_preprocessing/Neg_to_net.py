from types import prepare_class
import convert_mat_csv as cmc 
from os import mkdir
from os import path
class Neg_to_net:
    def __init__(self):
        self.data = None 
        self.net_position = [[0,0],[0,1],[1,2],[1,3],[2,0],[2,1],[2,2],[2,3]]
        self.neg_position = [[0,2],[0,3],[1,0],[1,1],[3,0],[3,1],[3,2],[3,3]] 
        self.exp_type = "neg_to_net"

    def store_male(self,folder = "../data/",sec=0):
        folder  = folder+self.exp_type+"/"
        self.create_folder(folder)

        self.preprocess("m",sec)
        
        folder += "male" +"_"+str(sec)+"/"
        self.create_folder(folder)
        cmc.store_to_csv(folder,self.data)

    def store_female(self,folder = "../data/",sec=0):
        folder  = folder+self.exp_type+"/"
        self.create_folder(folder)

        self.preprocess("f",sec)
        
        folder += "female" +"_"+str(sec)+"/"
        self.create_folder(folder)
        cmc.store_to_csv(folder,self.data)

    def store_mf(self,folder = "../data/",sec=0):
        folder  = folder+self.exp_type+"/"
        self.create_folder(folder)

        self.preprocess("mf",sec)

        folder += "combined"+"_"+str(sec)+"/"
        self.create_folder(folder)
        cmc.store_to_csv(folder,self.data)

    def create_folder(self,file_path):
        if not path.exists(file_path):
            mkdir(file_path)
    # preprocessing - general pipelines 
    def preprocess(self,gender,sec):
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
        netx,nety = cmc.get_difference(raw_data,self.net_position,"net")
        negx,negy = cmc.get_difference(raw_data,self.neg_position,"neg")

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
