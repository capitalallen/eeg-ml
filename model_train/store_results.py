import pymongo

def store_one_cv(name,cv,coefs,y_train,y_train_pred,y_test,y_test_pred):
        client = pymongo.MongoClient(
            "mongodb+srv://capitalallen:allen123@cluster0.bc8gt.mongodb.net/<dbname>?retryWrites=true&w=majority")
        db = client.get_database('eeg')
        records = db.results
        data = {}
        data['name'] = name
        data[cv] = {}
        data[cv]['raw_data'] = {}
        data[cv]['coefs'] = coefs.tolist()
        data[cv]['y_train'] = y_train.tolist()
        data[cv]['y_train_pred'] = y_train_pred.tolist()
        data[cv]['y_test'] = y_test.tolist()
        data[cv]['y_test_pred'] = y_test_pred.tolist()
        print(type(coefs))
        print(type(y_train))
        print(type(y_train_pred))
        print(type(y_test))
        print(type(y_test_pred))
        data[cv]['metrics'] = {}
        records.insert_one(data)
