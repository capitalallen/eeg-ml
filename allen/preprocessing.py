import mat73
import numpy as np
import matplotlib.pyplot as plt

class Data_prepare:
    def __inti__(self):
        pass 
    def convert_mat_np(self,gender=None):
        m_file = "./raw_data/Emotrans1_girl_data_preprocessed_42.mat"
        f_file = "./raw_data/Emotrans1_Boy_data_preprocessed_42.mat"
        # m_file = "../raw_data/Emotrans1_Boy_data_preprocessed_33.mat"
        # f_file = "../raw_data/Emotrans1_girl_data_preprocessed_33.mat"
        if gender == "f":
            data_dict_female = mat73.loadmat(f_file, use_attrdict=True)
            return np.array(data_dict_female["All_Feature"])
        elif gender == 'm':
            data_dict_female = mat73.loadmat(m_file, use_attrdict=True)
            return np.array(data_dict_female["All_Feature"])  
        else: 
            raise Exception("gender not defined")

    def combine_male_female(self):
        m = self.convert_mat_np("m")
        f = self.convert_mat_np("f")
        return np.concatenate((m,f),axis=0)

    # input: data and index of the removed person 
    # output: data without removed person
    def remove_person(self,df,index):
        if not index:
            return df 
        new_df = []
        for i in range(df.shape[0]):
            if i in index:
                continue 
            new_df.append(df[i])
        return np.array(new_df)

    # type: 0 --> average 3+4 and 1+2; 1--> use only 3 and 2  
    def get_pos_or_neg(self,data,position,type=0):
        arr = []
        index = 0
        for i in range(data.shape[0]):
    #         print(data[i].shape)
            if type == 0:
                diff = (data[i][position[0][0]][position[0][1]]+data[i][position[1][0]][position[1][1]])/2 - (data[i][position[3][0]][position[3][1]]+data[i][position[2][0]][position[2][1]])/2
            elif type == 1:
                diff = (data[i][position[1][0]][position[1][1]] - data[i][position[2][0]][position[2][1]])/2
            arr.append(diff)
        return np.array(arr)

    # select frequenceis and (0-4s -> 0 or 0.5-4.5s -> 1)
    # output: x 
    def choose_freq(self,data=None,freq = None, sec=None):
        if freq:
            data = np.delete(data,freq,axis=3)
        if sec == 0:
            for i in range(data.shape[0]):
                data[i] = np.delete(data[i], 1, axis=3)
            return data
        elif sec == 1:
            for i in range(data.shape[0]):
                data[i] = np.delete(data[i], 0, axis=3) 
            return data
        else:
            print('sec not specified')
            return data 

    # reshape to 128*8*2 = 2048
    # return x 
    def squeeze_feature_size(self,data):
        for d in range(data.shape[0]):
    #         for i in range(df[d].shape[0]):
            size = data[d].shape
            data[d] = data[d].reshape(size[0], size[1]*size[2]*size[3])
        return data

    # generate labels for x 
    # input: data and label_type (0:zeros, 1: ones )
    def generate_labels(self,df,type=None):
        arr = []
        label = []
        label_type = -1
        if type == 0:
            label_type = 0
        elif type == 1:
            label_type = 1
        else:
            print("Wrong type - get_pos_or_neg()")
            return 
        for d in range(df.shape[0]):
    #         for i in range(df[d].shape[0]):
            size = df[d].shape
    #         print(size)
            if label_type == 1:
                label.append(np.ones((size[0],1)))
            else:
                label.append(np.zeros((size[0],1)))
        return np.array(label)

    # combine net and neg 
    def combine_net_neg(self,netX,netY,negX,negY):
        for i in range(netX.shape[0]):
            netX[i] = np.concatenate((netX[i],negX[i]),axis=0)
            netY[i] = np.concatenate((netY[i],negY[i]),axis=0)
        return netX,netY
    # for stress classification 
    def combine_cases(sef,data):
        new_df = []
        for i in range(data.shape[0]):
            xs = None
            init = True
            for j in range(data[i].shape[0]):
                for k in range(data[i][j].shape[0]):
                    if init:
                        xs = data[i][j][k]
                    else:
                        xs = np.append(xs,data[i][j][k],axis=0)
                    init = False 
            new_df.append(xs)
        return np.array(new_df)
        
    def generate_label_stress(self,x):
        labels = []
        size = 40
        classes = [2, 0, 2, 1, 2, 1, 0, 1, 2, 0, 0, 1, 0, 2, 2, 1, 0, 2, 1, 0, 2, 2, 2, 0, 1, 2, 0, 2, 1, 0, 1, 0, 0, 0, 1, 2, 2, 1, 2, 0]
        for i in range(size):
            labels.append(np.full((x[i].shape[0]),classes[i]))
        return np.array(labels)
# type
    # 0 for not removing bad data 
    # 1 for removing bad data
# index: index of bad data
def ex_preprocessing(type=None,index=None):
    index = [11,18,36]
    dp = Data_prepare() 
    pos = [[0,3],[0,2],[0,1],[0,0]]
    neg = [[1,3],[1,2],[1,1],[1,0]]
    if type == 0:
        df = dp.combine_male_female()
        pos_df = dp.get_pos_or_neg(df,pos)
        neg_df = dp.get_pos_or_neg(df,neg)
        pos_sequeezed = dp.squeeze_feature_size(pos_df)
        neg_sequeezed = dp.squeeze_feature_size(neg_df)
        pos_labels = dp.generate_labels(pos_sequeezed,1)
        neg_labels = dp.generate_labels(neg_sequeezed,0)
        # netX,netY,negX,negY
        # return x and y
        return dp.combine_net_neg(pos_sequeezed,pos_labels,neg_sequeezed,neg_labels)
    elif type ==1:
        df = dp.combine_male_female()
        df = dp.remove_person(df,index)
        pos_df = dp.get_pos_or_neg(df,pos)
        neg_df = dp.get_pos_or_neg(df,neg)
        pos_sequeezed = dp.squeeze_feature_size(pos_df)
        neg_sequeezed = dp.squeeze_feature_size(neg_df)
        pos_labels = dp.generate_labels(pos_sequeezed,1)
        neg_labels = dp.generate_labels(neg_sequeezed,0)
        # netX,netY,negX,negY
        # return x and y
        return dp.combine_net_neg(pos_sequeezed,pos_labels,neg_sequeezed,neg_labels)

def ex_preprocessing_stress(type = None, index=[40,41]):
    dp = Data_prepare() 
    pos = [[0,3],[0,2],[0,1],[0,0]]
    neg = [[1,3],[1,2],[1,1],[1,0]]
    if type == 0:
        index=[40,41]
        df = dp.combine_male_female()
        df = dp.remove_person(df,index)
        pos_df = dp.get_pos_or_neg(df,pos)
        neg_df = dp.get_pos_or_neg(df,neg)
        pos_sequeezed = dp.squeeze_feature_size(pos_df)
        neg_sequeezed = dp.squeeze_feature_size(neg_df)
        pos_labels = dp.generate_labels(pos_sequeezed,1)
        neg_labels = dp.generate_labels(neg_sequeezed,0)
        x,_ = dp.combine_net_neg(pos_sequeezed,pos_labels,neg_sequeezed,neg_labels)
        labels = dp.generate_label_stress(x)
        # netX,netY,negX,negY
        # return x and y
        return x,labels
    elif type == 1:
        index=[11,18,36,40,41]
        df = dp.combine_male_female()
        df = dp.remove_person(df,index)
        pos_df = dp.get_pos_or_neg(df,pos)
        neg_df = dp.get_pos_or_neg(df,neg)
        pos_sequeezed = dp.squeeze_feature_size(pos_df)
        neg_sequeezed = dp.squeeze_feature_size(neg_df)
        pos_labels = dp.generate_labels(pos_sequeezed,1)
        neg_labels = dp.generate_labels(neg_sequeezed,0)
        x,_ = dp.combine_net_neg(pos_sequeezed,pos_labels,neg_sequeezed,neg_labels)
        labels = dp.generate_label_stress(x)
        # netX,netY,negX,negY
        # return x and y
        return x,labels  