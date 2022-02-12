import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import PreProcess
from Classify import Classify


def main():
    # for training models:
    cls = Classify()
    cls.RunModels(savedLbl=0)


if __name__ == "__main__":
    main()