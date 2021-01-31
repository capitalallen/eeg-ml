import pymongo

def store_one_cv(name,cv,coefs,y_train,y_train_pred,y_test,y_test_pred):
    client = pymongo.MongoClient(
        "mongodb+srv://capitalallen:allen123@cluster0.bc8gt.mongodb.net/<dbname>?retryWrites=true&w=majority")
    db = client.get_database('eeg')
    records = db.results
    id = records.find_one({'name':name})

    data = {}
    data[cv] = {}
    data[cv]['raw_data'] = {}
    data[cv]['coefs'] = coefs.tolist()
    data[cv]['raw_data']['y_train'] = y_train.tolist()
    data[cv]['raw_data']['y_train_pred'] = y_train_pred.tolist()
    data[cv]['raw_data']['y_test'] = y_test.tolist()
    data[cv]['raw_data']['y_test_pred'] = y_test_pred.tolist()
    data[cv]['metrics'] = {}

    if not id:
        data['name'] = name
        records.insert_one(data)
    else:
        id = id.get("_id")
        records.update({"_id":id},{"$set":data})
