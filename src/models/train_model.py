import mlflow
import math
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Iris script train")
    parser.add_argument("--test_size", type=float, default=0.3, help="passar test size por favor")
    parser.add_argument("--random_state",type=int, default=42, help="passe um int random_state"    )
    return parser.parse_args()
            
mlflow.set_tracking_uri("http://ec2-54-91-136-29.compute-1.amazonaws.com:3431")
mlflow.set_experiment("MODELO-IRIS-FLOWER")

df = pd.read_csv("https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv")
df['species'] = df['species'].map({'setosa': 0, 'virginica': 1, 'versicolor': 2})

X = df.drop('species', axis = 1)
y = df['species'].copy()

# CLF 
def main():   
     args = parse_args()
     X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size=args.test_size,random_state=args.random_state)

     with mlflow.start_run():
            mlflow.sklearn.autolog()
            model = RandomForestClassifier()
            model.fit(X_train, Y_train)
           
            data = [[1.1, 1.0, 2.0, 1.6]]
            pred = model.predict(X_test)
            # metrica 
            acc = accuracy_score(pred,Y_test)
            mlflow.log_metric("acc",acc)
            print('accuracy is',acc)
            
if __name__ == "__main__":
    main()