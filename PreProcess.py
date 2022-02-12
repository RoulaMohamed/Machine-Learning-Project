import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import svm
import os
from datetime import datetime
from sklearn.model_selection import train_test_split

class PreProcess:
    def __init__(self):
        self.data = pd.read_csv('AppleStore_training_classification.csv')

        # pre proccessing
        # self.data.dropna(how='any', inplace=True)

        #self.data['cont_rating'] = [i.replace('+', '') for i in self.data['cont_rating']]
        #self.data['cont_rating'] = self.data['cont_rating'].astype(int)

    def prepTrainDataset(self):
        self.data = self.data.drop_duplicates()

        self.data.dropna(how='all', axis='columns', inplace=True)
        self.data.dropna(how='all', axis='rows', inplace=True)

        self.data.drop(columns=['id', 'track_name', 'currency', 'vpp_lic', 'ver'], inplace=True)
        #self.data.drop( columns = ['sup_devices.num', 'ipadSc_urls.num', 'lang.num'], inplace = True )

        #self.data["sup_devices.num"]
        supDev = self.data["sup_devices.num"].mode(dropna=True)
        print(supDev[0])
        self.data["sup_devices.num"].fillna(supDev[0], inplace=True)

        ipadSc = self.data["ipadSc_urls.num"].mode(dropna=True)
        print(ipadSc[0])
        self.data["ipadSc_urls.num"].fillna(ipadSc[0], inplace=True)

        langNum = self.data["lang.num"].mode(dropna=True)
        print(langNum[0])
        self.data["lang.num"].fillna(langNum[0], inplace=True)




        # filling empty cells with avgs
        self.data["size_bytes"].fillna(self.data["size_bytes"].sum(skipna=True) / len(self.data["size_bytes"]), inplace=True)
        self.data["size_bytes"] = self.data["size_bytes"].map(lambda x: round(x / (1024 * 1024), 2)) # conv to megabyte

        self.data["price"].fillna(self.data["price"].sum(skipna=True) / len(self.data["price"]), inplace=True)

        self.data["rating_count_tot"].fillna(self.data["rating_count_tot"].sum(skipna=True) / len(self.data["rating_count_tot"]), inplace=True)
        self.data["rating_count_ver"].fillna(self.data["rating_count_ver"].sum(skipna=True) / len(self.data["rating_count_ver"]), inplace=True)

        #self.data['cont_rating'] = [i.replace('+', '') for i in self.data['cont_rating']]
        #self.data['cont_rating'] = self.data['cont_rating'].astype(int)
        #self.data["cont_rating"] = self.data["cont_rating"].map(lambda x: x.replace(" ", ""))
        #self.data["cont_rating"] = self.data["cont_rating"].map(lambda x: float(x[:-1]))
        #self.data["cont_rating"] = [ (i.replace('+', ''))  for i in enumerate(self.data['cont_rating'])]
        #print(self.data['cont_rating'])
        self.data['cont_rating'] = (self.data['cont_rating'].str.replace('+', ''))
        self.data['cont_rating'] = self.data['cont_rating'].astype(float)
        #print(self.data['cont_rating'])
        self.data["cont_rating"].fillna(self.data["cont_rating"].sum(skipna=True) / len(self.data["cont_rating"]), inplace=True)

        modeGenre = self.data['prime_genre'].mode(dropna=True)
        self.data["prime_genre"].fillna( modeGenre , inplace=True)
        #self.data["prime_genre"] = self.data["prime_genre"].map(lambda x: x.lower())
        self.data['prime_genre'] = (self.data['prime_genre'].str.replace(' ', ''))
        # print(self.data['prime_genre'].mode(dropna=True))
        #self.data["prime_genre"] = self.data["prime_genre"].map(lambda x: x.replace(" ", ""))

        # local one hot encoding
        distinict = set()
        for i in self.data["prime_genre"]:
            if i == None or i == '0' or i == '':
                continue
            distinict.add(i)
            #print(i)
        #print(distinict)
        for i in distinict:
            genreList = []
            bool = 0
            for j in self.data["prime_genre"]:
                if (i == j): bool = 1
                genreList.append(bool)
                bool = 0
            self.data.insert(self.data.shape[1], i, genreList, True)

        self.data.drop(columns=["prime_genre"], inplace=True)
        # drop columns with empty headings
        self.data.drop( self.data.columns[9], axis =1 , inplace=True)
        self.data.dropna(axis='columns', how='all')

        #print("passed")
        H = 'High'; I = 'Intermediate'; L = 'Low'
        self.data["rate"] = self.data["rate"].map(lambda x: 2 if x == H else 1 if x == I else 0 if x == L else 2)


        # Normalizing values
        columns = ['size_bytes', 'price', 'rating_count_tot', 'rating_count_ver', 'cont_rating','sup_devices.num', 'ipadSc_urls.num', 'lang.num']
        for col in columns:
            val = preprocessing.MinMaxScaler(feature_range=(0, 1))
            val.fit(self.data[col].values.reshape(-1, 1))
            self.data[col] = val.transform(self.data[col].values.reshape(-1, 1))


        self.data.to_csv("TrainingDataset.csv", encoding='utf-8', index=False)

        Y_label = self.data['rate']
        X_data = self.data.drop(columns=["rate"], inplace=False)

        return X_data, Y_label



def main():
    pre = PreProcess()
    pre.prepTrainDataset()

if __name__ == "__main__":
    main()







