import numpy as np 
import os 
class Load_data:
    def __init__(self):
        pass 

    def files_in_dir(self,folder):
        ordered_list = []
        file_list = os.listdir(folder)
        for i in range(len(file_list)):
            ordered_list.append(folder +str(i)+".csv")
        return ordered_list
    def remove_person(self,index,file_list):
        del file_list[index]
        return file_list
    def convert_csv_np(self,file_path):
        return np.genfromtxt(file_path, delimiter=',')

    def get_data_label(self,data):
        label = data[:,1024]
        data = np.delete(data,1024,axis=1)
        return data, label
    def get_all_data(self,folder):
        file_list = self.files_in_dir(folder)
        x,y = [],[]
        for i in file_list:
            data = self.convert_csv_np(i)
            temp_x,temp_y = self.get_data_label(data)
            x.append(temp_x)
            y.append(temp_y)
        return np.asarray(x),np.asarray(y)

    def get_perfect_data(self,folder,index):
        file_list = self.files_in_dir(folder)
        file_list = self.remove_person(index,folder)
        x,y = [],[]
        for i in file_list:
            data = self.convert_csv_np(i)
            temp_x,temp_y = self.get_data_label(data)
            x.append(temp_x)
            y.append(temp_y)
        return np.asarray(x),np.asarray(y)