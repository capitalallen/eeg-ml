import numpy as np 
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import store_results as sr 
class Train:
    def __init__(self,name):
        self.rf_parameters = {'max_depth': [5, 10,None],
                                'n_estimators': [100, 256],
                                'min_impurity_decrease':[0.025,0.0]}
        self.boost_parameters = {'max_depth': [3, 5, 7],
                                'n_estimators': [100, 256],
                                "learning_rate":[0.1,0.05],
                                "gamma":[0,0.0001,0.01]}
        self.rf_model = RandomForestClassifier(criterion='entropy', # How to train the trees. Also supports entropy.           
                                        min_samples_split=2, # Minimum samples to create a split.
                                        min_samples_leaf=0.001, # Minimum samples in a leaf. Accepts fractions for %. This is 0.1% of sample.
                                        min_weight_fraction_leaf=0.0, # Same as above, but uses the class weights.
                                        max_features='auto', # Maximum number of features per split (not tree!) by default is sqrt(vars)
                                        max_leaf_nodes=None, # Maximum number of nodes.
                                        oob_score=True,  # If report accuracy with non-selected cases.
                                        n_jobs=-1, # Parallel processing. Set to -1 for all cores. Watch your RAM!!
                                        verbose=1, # If to give info during training. Set to 0 for silent training.
                                        warm_start=False, # If train over previously trained tree.
                                        class_weight='balanced',
                                        bootstrap=True,
                                            )
        self.boosting = XGBClassifier(verbosity=1,                  # If to show more errors or not.
                                    objective='binary:logistic',  # Type of target variable.
                                    booster='gbtree',             # What to boost. Trees in this case.
                                    n_jobs=2,                     # Parallel jobs to run. Set your processor number.
                                    subsample=0.5,                  # Subsample ratio. Can set lower
                                    colsample_bytree=1,           # Subsample ratio of columns when constructing each tree.
                                    colsample_bylevel=1,          # Subsample ratio of columns when constructing each level. 0.33 is similar to random forest.
                                    colsample_bynode=1,           # Subsample ratio of columns when constructing each split.
                                    reg_alpha=1,                  # Regularizer for first fit. alpha = 1, lambda = 0 is LASSO.
                                    reg_lambda=0,                 # Regularizer for first fit.
                                    scale_pos_weight=1,           # Balancing of positive and negative weights.
                                    base_score=0.5,               # Global bias. Set to average of the target rate.
                                    missing=None                  # How are nulls encoded?
                                    )
        self.params = None
        self.name = name
    def perform_grid_search(self,x,y,type):
        if type=="rf":
            model = GridSearchCV(self.rf_model, self.rf_parameters, scoring='roc_auc', n_jobs=-1)
        else:
            model = GridSearchCV(self.boosting, self.boost_parameters, scoring='roc_auc', n_jobs=-1)
        model.fit(x,y)
        # clf.cv_results_['mean_test_score'],
        self.params = model.best_params_


    def model_training(self,type,x,y):
        if type=="rf":
            model = RandomForestClassifier(max_depth=self.params['max_depth'],
                                            n_estimators=self.params['n_estimators'],
                                            min_impurity_decrease=self.params['min_impurity_decrease'],
                                            criterion='entropy', # How to train the trees. Also supports entropy.           
                                            min_samples_split=2, # Minimum samples to create a split.
                                            min_samples_leaf=0.001, # Minimum samples in a leaf. Accepts fractions for %. This is 0.1% of sample.
                                            min_weight_fraction_leaf=0.0, # Same as above, but uses the class weights.
                                            max_features='auto', # Maximum number of features per split (not tree!) by default is sqrt(vars)
                                            max_leaf_nodes=None, # Maximum number of nodes.
                                            oob_score=True,  # If report accuracy with non-selected cases.
                                            n_jobs=-1, # Parallel processing. Set to -1 for all cores. Watch your RAM!!
                                            verbose=1, # If to give info during training. Set to 0 for silent training.
                                            warm_start=False, # If train over previously trained tree.
                                            class_weight='balanced',
                                            bootstrap=True,
                                            )
        else:
            model= XGBClassifier(max_depth=self.params['max_depth'],
                            n_estimators=self.params['n_estimators'],
                            learning_rate=self.params['learning_rate'],
                            gamma=self.params['gamma'],
                            verbosity=1,                  # If to show more errors or not.
                            objective='binary:logistic',  # Type of target variable.
                            booster='gbtree',             # What to boost. Trees in this case.
                            n_jobs=2,                     # Parallel jobs to run. Set your processor number.
                            subsample=0.5,                  # Subsample ratio. Can set lower
                            colsample_bytree=1,           # Subsample ratio of columns when constructing each tree.
                            colsample_bylevel=1,          # Subsample ratio of columns when constructing each level. 0.33 is similar to random forest.
                            colsample_bynode=1,           # Subsample ratio of columns when constructing each split.
                            reg_alpha=1,                  # Regularizer for first fit. alpha = 1, lambda = 0 is LASSO.
                            reg_lambda=0,                 # Regularizer for first fit.
                            scale_pos_weight=1,           # Balancing of positive and negative weights.
                            base_score=0.5,               # Global bias. Set to average of the target rate.
                            missing=None                  # How are nulls encoded?
                            )
        model.fit(x,y)
        return model

    def get_coefs(self,model):
        return model.feature_importances_

    def get_predicts(self,x_train,x_test,model):
        # print(x_train.shape)
        # print(x_test.shape)
        print(x_train.shape)
        print(x_test.shape)
        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)
        return y_train_pred,y_test_pred
    
    def ex_train(self,type,cv_num,x_train,y_train,x_test,y_test,record_name="results"):
        if type =="rf":
            model = self.model_training(type,x_train,y_train)
        else:
            model = self.model_training(type,x_train,y_train)
        coefs = self.get_coefs(model)
        y_train_pred,y_test_pred = self.get_predicts(x_train,x_test,model)
        sr.store_one_cv(self.name+type,str(cv_num),coefs,y_train,y_train_pred,y_test,y_test_pred,record_name)
    
    def ex_train_within(self,type,x_train,y_train,x_test,y_test):
        if type =="rf":
            model = self.model_training(type,x_train,y_train)
        else:
            model = self.model_training(type,x_train,y_train)
        coefs = self.get_coefs(model)
        y_train_pred,y_test_pred = self.get_predicts(x_train,x_test,model)
        return (y_test,y_test_pred)
    def ex_train_removed(self,type,cv_num,x_train,y_train,x_test,y_test,record_name="results"):
        if type =="rf":
            model = self.model_training(type,x_train,y_train)
        else:
            model = self.model_training(type,x_train,y_train)
        coefs = self.get_coefs(model)
        y_train_pred,y_test_pred = self.get_predicts(x_train,x_test,model)
        sr.store_one_cv(self.name+type+"/removed",str(cv_num),coefs,y_train,y_train_pred,y_test,y_test_pred,record_name)