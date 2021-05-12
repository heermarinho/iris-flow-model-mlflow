import requests 
from datetime import datetime as dt 

url = "http://0.0.0.0:5001/invocations"

data = [[7.2,3.8,5.8,1.6]]

data = {
    "columns":['sepal_length','sepal_width','petal_length','petal_width'],
    "data":data
}

def check_label(pred):
    #  0 (setosa), 1 (versicolor), and 2 (virginica).
    if pred == 0:
        return pred,"setosa"
    elif pred == 1:
        return pred, 'versicolor'
    else:
        return pred, 'virginica'
    
    
header = {'Content-Type':'application/json'}
r = requests.post(url,json=data,headers=header).json()
print(dt.now(),"result api: ", check_label(r[0]),type(r))