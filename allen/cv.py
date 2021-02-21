import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

class Train:
    def __init__(self,x,y):
        self.x = x 
        self.y = y
        self.model = RandomForestClassifier(n_estimators=150) 

    # get x train and test set for witin CV 
    def get_x_train_test(self,data,index):
        x_test = data[index]
        x_train = np.delete(data,index,axis=0)
        return x_train,x_test.reshape(1,-1)

    # get y train and test set for witin CV 
    def get_y_train_test(self,data,index):
        y_test = data[index]
        y_train = np.delete(data,index,axis=0)
        return y_train.astype(int).ravel(),y_test.astype(int)
    model = RandomForestClassifier(n_estimators=50)

    def model_train(self,x_train,x_test,y_train,y_test,model):
        model.fit(x_train,y_train)
        y_pred = model.predict(x_test)
        return y_test, y_pred
        
    def within_train(self):
        # with_in
        all_accuracy = []
        # train a model 
        # return y_test, y_pred 
        avg_accuracy = []
        for d in range(self.x.shape[0]):
            # training set and test set
            accuracy =[]
            print("within # person:",d)
            for i in range(self.x[d].shape[0]):
                x_train,x_test = self.get_x_train_test(self.x[d],i)
                y_train,y_test = self.get_y_train_test(self.y[d],i)
                y_test,y_pred = self.model_train(x_train,x_test,y_train,y_test,self.model)
                accuracy.append(accuracy_score(y_test,y_pred))
                # print(x_train.shape,x_test.shape)
                # print(y_train.shape,y_test.shape)
            avg_accuracy.append(sum(accuracy)/len(accuracy))
            all_accuracy.append(accuracy)
        return all_accuracy,avg_accuracy
    
    # get x and y training set for leave one out 
    def get_x_y_train(self,x,y):
        new_x = x[0]
        new_y = y[0]
        for i in range(1,len(x)):
            # print(new_x.shape)
            new_x=np.append(new_x,x[i],axis=0)
            new_y=np.append(new_y,y[i],axis=0)
        print(new_x.shape)
        return new_x,new_y
    def leav_one_train(self):
        avg_accuracy=[]
        all_accuracy=[]
        for d in range(self.x.shape[0]):
            print("leave one out # person:",d)
            if d == 0:
                x_train,y_train = self.get_x_y_train(self.x[d+1:],self.y[d+1:])
                x_test,y_test =self.x[d],self.y[d]
            elif d == self.x.shape[0]-1:
                x_train,y_train = self.get_x_y_train(self.x[:d],self.y[:d])
                x_test,y_test = self.x[d],self.y[d]
            else:
                x_train,y_train = self.get_x_y_train(np.append(self.x[:d],self.x[d+1:],axis=0),np.append(self.y[:d],self.y[d+1:],axis=0))
                x_test,y_test = self.x[d],self.y[d]
            # print(x_train.shape,x_test.shape)
            # print(y_train.shape,y_test.shape)
            # return
            y_test,y_pred=self.model_train(x_train,x_test,y_train,y_test,self.model)
            accuracy = accuracy_score(y_test,y_pred)
            print(accuracy)
            all_accuracy.append(accuracy)
        return all_accuracy