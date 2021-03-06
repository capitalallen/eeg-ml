import json 

# store accuracy 
# def write_to_json(file_name,all_acc,avg_acc):
#     data = {} 
#     data['name'] = file_name 
#     data['all_acc'] = all_acc
#     data['avg_acc'] = avg_acc
#     with open(file_name, 'w') as outfile:
#         json.dump(data, outfile)

# store accuracy and coefs
def write_to_json(file_name,avg_acc,coefs):
    data = {} 
    data['name'] = file_name 
    data['avg_acc'] = avg_acc
    data['coefs'] = coefs
    with open(file_name, 'w') as outfile:
        json.dump(data, outfile)

# read_accuracy 
def read_json(file_name):
    return json.load(file_name)