import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier,OneVsOneClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.svm import SVC , LinearSVC
import os
from PreProcess import PreProcess
from sklearn.decomposition import PCA
import pickle
from sklearn.preprocessing import StandardScaler
import timeit
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

class Classify:

    def __init__(self):
        return

    #def __init__(self, X_train, y_train, X_test, y_test):
    #    self.call_Models(X_train, y_train, X_test, y_test)

    #def __init__(self, X_test, y_test):
    #    self.call_Saved_Models(X_test, y_test)

    def OneVsOnelinear(self, X_train, y_train, X_test, y_test):
        C = 8
        #print("my y: ")
        #print(y_test.shape)
        #print("my x: ")
        #print(X_test.shape)
        start_tm = timeit.default_timer()
        model = OneVsOneClassifier(SVC(kernel='linear', C=C)).fit(X_train, y_train)
        stop_tm = timeit.default_timer()

        start_tm2 = timeit.default_timer()
        accuracy = model.score(X_test, y_test)
        stop_tm2 = timeit.default_timer()

        acc = str(accuracy * 100)
        train_tm = str(stop_tm - start_tm);
        test_tm = str(stop_tm2 - start_tm2)
        print('One VS One SVM accuracy: ' + acc, ", Time Trainig: " + train_tm, ", Time Testing: " + test_tm)

        filename = 'OneVsOnelinear.pkl'
        pickle.dump(model, open("models/" + filename, 'wb'))

    def OneVsOne_LinearSVC(self, X_train, y_train, X_test, y_test):
        C = 7
        start_tm = timeit.default_timer()
        model1 = OneVsOneClassifier(LinearSVC(C=C)).fit(X_train, y_train)
        stop_tm = timeit.default_timer()

        start_tm2 = timeit.default_timer()
        accuracy_1 = model1.score(X_test, y_test)
        stop_tm2 = timeit.default_timer()

        acc = str(accuracy_1 * 100)
        train_tm = str(stop_tm - start_tm);
        test_tm = str(stop_tm2 - start_tm2)
        print('One VS One SVC accuracy: ' + acc, ", Time Trainig: " + train_tm, ", Time Testing: " + test_tm)

        filename = 'OneVsOne_LinearSVC.pkl'
        pickle.dump(model1, open("models/" + filename, 'wb'))

    def OneVsRestlinear(self, X_train, y_train, X_test, y_test):
        C = 1
        start_tm = timeit.default_timer()
        model = OneVsRestClassifier(SVC(kernel='linear', C=C)).fit(X_train, y_train)
        stop_tm = timeit.default_timer()

        start_tm2 = timeit.default_timer()
        accuracy = model.score(X_test, y_test)
        stop_tm2 = timeit.default_timer()

        acc = str(accuracy * 100)
        train_tm = str(stop_tm - start_tm);
        test_tm = str(stop_tm2 - start_tm2)
        print('One VS One SVC accuracy: ' + acc, ", Time Trainig: " + train_tm, ", Time Testing: " + test_tm)

        filename = 'OneVsRestlinear.pkl'
        pickle.dump(model, open("models/" + filename, 'wb'))

    def OneVsRest_LinearSVC(self, X_train, y_train, X_test, y_test):
        C = 2 # --------------------------- 1
        start = timeit.default_timer()
        model = OneVsRestClassifier(LinearSVC(C=C)).fit(X_train, y_train)
        stop = timeit.default_timer()

        # lin_svc = svm_model_linear_ovr.predict(X_test)
        start2 = timeit.default_timer()
        accuracy = model.score(X_test, y_test)
        stop2 = timeit.default_timer()

        print('One VS Rest SVM accuracy :> Kernel|Linear:' + str(accuracy * 100),
              " Time Trainig : " + str(stop - start), " Time Testing : " + str(stop2 - start2))

        filename = 'OneVsRest_LinearSVC.pkl'


    def OneVsOne_rbf(self, X_train, y_train, X_test, y_test):
        C = .000001
        # svm_model_linear_ovr = OneVsRestClassifier(SVC(kernel='rbf', gamma=0.4, C=1).fit(X_train, y_train))
        start = timeit.default_timer()
        model = OneVsOneClassifier(SVC(kernel='rbf', gamma=0.4, C=C)).fit(X_train, y_train)
        stop = timeit.default_timer()

        start2 = timeit.default_timer()
        accuracy = model.score(X_test, y_test)
        stop2 = timeit.default_timer()
        print('One VS One SVM accuracy Kernel == rbf Gaussian : ' + str(accuracy * 100),
              " Time Trainig : " + str(stop - start), " Time Testing : " + str(stop2 - start2))

        filename = 'OneVsOne_rbf.pkl'


    def OneVsRest_rbf(self, X_train, y_train, X_test, y_test):
        C = 1
        # svm_model_linear_ovr = OneVsRestClassifier(SVC(kernel='rbf', gamma=0.4, C=1).fit(X_train, y_train))
        start = timeit.default_timer()
        svm_model_linear_ovr = OneVsRestClassifier(SVC(kernel='rbf', gamma=0.4, C=C)).fit(X_train, y_train)
        stop = timeit.default_timer()

        start2 = timeit.default_timer()
        accuracy = svm_model_linear_ovr.score(X_test, y_test)
        stop2 = timeit.default_timer()

        print('One VS Rest SVM accuracy Kernel == rbf Gaussian : ' + str(accuracy * 100),
              " Time Trainig : " + str(stop - start), " Time Testing : " + str(stop2 - start2))
        filename = 'OneVsRest_rbf.pkl'


    def OneVsOne_ploy(self, X_train, y_train, X_test, y_test):
        C = 1
        start = timeit.default_timer()
        model = OneVsOneClassifier(SVC(kernel='poly', degree=2, C=C)).fit(X_train, y_train)
        stop = timeit.default_timer()

        # poly_svc = svm_model_linear_ovr.predict(X_test)
        start2 = timeit.default_timer()
        accuracy = model.score(X_test, y_test)
        stop2 = timeit.default_timer()

        print('One VS One SVM accuracy Kernel == Poly: ' + str(accuracy * 100), " Time Trainig : " + str(stop - start),
              " Time Testing : " + str(stop2 - start2))
        filename = 'OneVsOne_ploy.pkl'


    def OneVsRest_ploy(self, X_train, y_train, X_test, y_test):
        C = 1
        start = timeit.default_timer()
        model = OneVsRestClassifier(SVC(kernel='poly', degree=3, C=C)).fit(X_train, y_train)
        stop = timeit.default_timer()

        # poly_svc = svm_model_linear_ovr.predict(X_test)
        start2 = timeit.default_timer()
        accuracy = model.score(X_test, y_test)
        stop2 = timeit.default_timer()

        print('One VS Rest SVM accuracy Kernel == Poly: ' + str(accuracy * 100), " Time Trainig : " + str(stop - start),
              " Time Testing : " + str(stop2 - start2))
        filename = 'OneVsRest_ploy.pkl'




    def adaBoost(self, X_train, y_train, X_test, y_test):
        mx_dpth = 15
        ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=mx_dpth), algorithm="SAMME.R", n_estimators=20)

        start_tm = timeit.default_timer()
        ada.fit(X_train, y_train)
        stop_tm = timeit.default_timer()

        start_tm2 = timeit.default_timer()
        y_pred = ada.predict(X_test)
        accuracy = np.mean(y_pred == y_test)
        stop_tm2 = timeit.default_timer()

        acc = str(accuracy * 100)
        train_tm = str(stop_tm - start_tm);
        test_tm = str(stop_tm2 - start_tm2)
        print("Adaboost Accuracy:  " + acc, ", Time Trainig: " + train_tm, ", Time Testing: " + test_tm)

        filename = 'adaBoost.pkl'
        pickle.dump(ada, open("models/" + filename, 'wb'))

    def decisionTree(self, X_train, y_train, X_test, y_test):
        dec_tree = tree.DecisionTreeClassifier(max_depth=15)
        dec_tree.fit(X_train, y_train)

        start_tm = timeit.default_timer()
        y_pred = dec_tree.predict(X_test)
        stop_tm = timeit.default_timer()

        start_tm2 = timeit.default_timer()
        accuracy = np.mean(y_pred == y_test)
        stop_tm2 = timeit.default_timer()

        acc = str(accuracy * 100)
        train_tm = str(stop_tm - start_tm);
        test_tm = str(stop_tm2 - start_tm2)
        print("Decision Tree Accuracy: " + acc, ", Time Trainig: " + train_tm, ", Time Trainig: " + test_tm)

        filename = 'decisionTree.pkl'
        pickle.dump(dec_tree, open("models/" + filename, 'wb'))

    def KNN(self, X_train, y_train, X_test, y_test):
        start_tm = timeit.default_timer()
        knn = KNeighborsClassifier(n_neighbors=12) # 11
        knn.fit(X_train, y_train)
        stop_tm = timeit.default_timer()

        start_tm2 = timeit.default_timer()
        pred_i = knn.predict(X_test)
        err = (np.mean(pred_i != y_test))
        accuracy = (np.mean(pred_i == y_test))
        stop_tm2 = timeit.default_timer()

        acc = str(accuracy * 100)
        train_tm = str(stop_tm - start_tm);
        test_tm = str(stop_tm2 - start_tm2)
        print("KNN Accuracy: " + acc, ", Time Trainig: " + train_tm, ", Time Trainig: " + test_tm)

        filename = 'KNN.pkl'
        pickle.dump(knn, open("models/" + filename, 'wb'))

    def custom_train_test_split(self, X, Y, test_sz):
        H = 2; I = 1; L = 0 # H = 2; I = 1; L = 0
        L_data   = X[Y == L];   L_label = X[Y == L]
        I_data   = X[Y == I];   I_label = X[Y == I]
        H_data   = X[Y == H];   H_label = X[Y == H]

        H_X_train, H_X_test, H_Y_train, H_Y_test = train_test_split(H_data, H_label, test_size=test_sz, shuffle=True)
        I_X_train, I_X_test, I_Y_train, I_Y_test = train_test_split(I_data, I_label, test_size=test_sz, shuffle=True)
        L_X_train, L_X_test, L_Y_train, L_Y_test = train_test_split(L_data, L_label, test_size=test_sz, shuffle=True)

        X_train = np.concatenate((H_X_train, I_X_train, L_X_train))
        y_train = np.concatenate((H_Y_train, I_Y_train, L_Y_train))
        X_test  = np.concatenate((H_X_test, I_X_test, L_X_test))
        y_test  = np.concatenate((H_Y_test, I_Y_test, L_Y_test))

        return X_train, X_test, y_train, y_test

    def getData(self):
        if (os.path.exists("TrainingDataset.csv")):
            data = pd.read_csv("TrainingDataset.csv")
            Y = data['rate']  ;  X = data.drop(columns=["rate"], inplace=False)
            #print("---")
            #print(Y)
            return X, Y
        else:
            preProcess = PreProcess()
            X, Y = preProcess.prepTrainDataset()
            #print(Y)
        return X, Y

    def RunModels(self, savedLbl):
        X, Y = self.getData()
        #X_train, X_test, y_train, y_test = self.custom_train_test_split(X, Y, test_sz= 0.2)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, shuffle=True)

        #print(y_train)
        #print("###")
        if savedLbl == 0:
            self.call_Models(X_train, X_test, y_train, y_test)
        else:
            self.call_Saved_Models(X_test, y_test)

    def call_Models(self, X_train, X_test,y_train,  y_test):
        #pca = PCA(n_components=20)
        #pca.fit(X_train)
        #x = pca.transform(X_train)
        #X_train = x
        ## ratio= pca.explained_variance_ratio_
        #xtest = pca.transform(X_test)
        #X_test = xtest

        self.OneVsOnelinear(X_train, y_train, X_test, y_test)
        self.OneVsOne_LinearSVC(X_train, y_train, X_test, y_test)
        self.OneVsOne_rbf(X_train, y_train, X_test, y_test)
        self.OneVsOne_ploy(X_train, y_train, X_test, y_test)
        self.OneVsRest_LinearSVC(X_train, y_train, X_test, y_test)
        self.OneVsRest_ploy(X_train, y_train, X_test, y_test)
        self.OneVsRest_rbf(X_train, y_train, X_test, y_test)
        self.OneVsRestlinear(X_train, y_train, X_test, y_test)
        self.adaBoost(X_train, y_train, X_test, y_test)
        self.decisionTree(X_train, y_train, X_test, y_test)
        self.KNN(X_train, y_train, X_test, y_test)

    def call_Saved_Models(self, X_test, Y_test):
        # ---------------OneVsOneLinear:
        loaded_model = pickle.load(open("models/OneVsOnelinear.pkl", 'rb'))

        start_tm = timeit.default_timer()
        res = loaded_model.score(X_test, Y_test)
        stop_tm = timeit.default_timer()

        test_tm =  str(stop_tm - start_tm)
        acc =  str(res * 100)
        print(acc, ", Time Testing : " + test_tm)
        y_pred = loaded_model.predict(self.X_test)
        print(confusion_matrix(self.Y_test, y_pred))

        # ----------------OneVsOneLinearSVC:
        loaded_model2 = pickle.load(open("models/OneVsOne_LinearSVC.pkl", 'rb'))

        start_tm2 = timeit.default_timer()
        res2 = loaded_model.score(self.X_test, self.Y_test)
        stop_tm2 = timeit.default_timer()

        test_tm2 = str(stop_tm2 - start_tm2)
        acc2 = str(res2 * 100)
        print(acc2,", Time Testing : " + test_tm2)
        y_pred2 = loaded_model.predict(self.X_test)
        print(confusion_matrix(self.Y_test, y_pred2))

        # ----------------Adaboost:
        loaded_model3 = pickle.load(open("models/adaBoost.pkl", 'rb'))

        start_tm3 = timeit.default_timer()
        res3 = loaded_model.score(self.X_test, self.Y_test)
        stop_tm3 = timeit.default_timer()

        test_tm3 = str(stop_tm3 - start_tm3)
        acc3 = str(res3 * 100)
        print(acc3,", Time Testing: " + test_tm3)
        y_pred3 = loaded_model.predict(self.X_test)
        print(confusion_matrix(self.Y_test, y_pred3))

        # ------------------Decision Tree:
        loaded_model4 = pickle.load(open("models/decisionTree.pkl", 'rb'))

        start_tm4 = timeit.default_timer()
        res4 = loaded_model.score(self.X_test, self.Y_test)
        stop_tm4 = timeit.default_timer()

        test_tm4 = str(stop_tm4 - start_tm4)
        acc4 = str(res4 * 100)
        print(acc4,", Time Testing : " + test_tm4)
        y_pred4 = loaded_model.predict(self.X_test)
        print(confusion_matrix(self.Y_test, y_pred4))

        # -------------------KNN:
        loaded_model5 = pickle.load(open("models/KNN.pkl", 'rb'))

        start_tm5 = timeit.default_timer()
        res5 = loaded_model.score(self.X_test, self.Y_test)
        stop_tm5 = timeit.default_timer()

        test_tm5 = str(stop_tm5 - start_tm5)
        acc5 = str(res5 * 100)
        print(acc5,", Time Testing : " + test_tm5)
        y_pred5 = loaded_model.predict(self.X_test)
        print(confusion_matrix(self.Y_test, y_pred5))

def main():
    # for training models:
    cls = Classify()
    cls.RunModels(savedLbl=0)


if __name__ == "__main__":
    main()











