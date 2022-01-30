import joblib
import train
import re
import gensim as gs
import pandas as pd

data_path = "2500282966.csv"



def test():
    w2v = joblib.load("w2vmodel.pkl")
    trainn = joblib.load("train.pkl")
    threshold = joblib.load("threshold.pkl")

    data =pd.read_csv(data_path,error_bad_lines=False)

    train._preprocess(data)
    #print(data)



    print(train.infer(w2v,trainn,"DHCPACKfromxidxcaccad",data,threshold))

if __name__ == "__main__":
    test()
