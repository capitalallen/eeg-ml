import json
def get_results(file = "within.json"):
    data = None
    with open(file, "r") as jsonFile:
        data = json.load(jsonFile)
    return data

def update_results(key,val,file="within.json"):
    data = get_results()
    data[key] = val 
    with open(file, "w") as jsonFile:
        json.dump(data, jsonFile)
    return True
print(get_results())