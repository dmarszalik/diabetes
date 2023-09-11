import pandas as pd
import numpy as np
from sklearn.base import ClassifierMixin, TransformerMixin
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from typing import Union
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import GridSearchCV
import sklearn




class Dataset:
    def __init__(self, path: str = "datasets/diabetes.csv") -> None:
        """
        Init dataset, read_csv from the given path.

        :param path: path to csv, defaults to "datasets/diabetes.csv"
        :type path: str, optional
        """
        self.ds = pd.read_csv(path)

        label_encoder = LabelEncoder()
        for column in self.ds.columns:
            if self.ds[column].dtype == 'object':  # Sprawdź, czy kolumna ma typ 'object' (czyli jest kategoryczna)
                self.ds[column] = label_encoder.fit_transform(self.ds[column])

    def print_info(self) -> None:
        """Print some informations about dataset."""
        print(
            self.ds.head(5), '/n',
            self.ds.info(), '/n',
            self.ds.describe(), '/n',
            self.ds.columns
            )

    def get_splitted_data(
        self,
        test_size: float = 0.2,
        normalizer: TransformerMixin = None,
        discretizer: TransformerMixin = None,
        verbose: bool = False,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Get splitted dataset for training and testing.

        :param test_size: test/train ratio, defaults to 0.2
        :type test_size: float, optional
        :param normalizer: normalization method, defaults to None
        :type normalizer: TransformerMixin, optional
        :param discretizer: discretization method, defaults to None
        :type discretizer: TransformerMixin, optional
        :param verbose: verbose or not?, defaults to False
        :type verbose: bool, optional
        <i class="fas fa-return"></i> X_train, X_test, y_train, y_test
        <i class="fas fa-rtype"></i> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]
        """
        self.x_train, self.x_test, self.y_train,  self.y_test = train_test_split(self.ds.iloc[:, 0:-1], self.ds.iloc[:, -1], test_size=test_size, stratify=self.ds.iloc[:, -1])

        if verbose:
                print("X_train", self.x_train.shape)
                print("y_train", self.y_train.shape)
                print("X_test", self.x_test.shape)
                print("y_test", self.y_test.shape)

        return (self.x_train, self.x_test, self.y_train,  self.y_test)

    def perform_eda(self) -> None:
        """Shows some visualization."""

        self.numeric_columns = self.ds.select_dtypes(include=['number'])

        self.correlation_matrix = self.numeric_columns.corr()
        # print(self.correlation_matrix)
        # fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))
        fig, ax = plt.subplots(figsize=(10, 8))
        # sns.heatmap(self.correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5, ax=ax[0,0])
        sns.heatmap(self.correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
        plt.title("Macierz Korelacji")

        # sns.histplot(data=self.ds, ax=ax[1,0])
        plt.show()

    
    def test_classifier(
            self,
            classifier: ClassifierMixin,
            scoring: str = 'precision'
            ): # -> dict[str, Union[ClassifierMixin, float]]:
         x_train, x_test, y_train, y_test = self.get_splitted_data(verbose=True)

         classifier.fit(x_train, y_train)
         y_pred = classifier.predict(x_test)

        #  cross_val_accuracy = cross_val_score(classifier, x_train, y_train, cv=5, scoring='accuracy')
         accuracy = accuracy_score(y_test, y_pred)
         recall = recall_score(y_test, y_pred)
         precision = precision_score(y_test, y_pred)
         f1 = f1_score(y_test, y_pred)
         print(
              f'Accuracy: {accuracy} /n/n',
            #   f'Cross_val_accuracy: {cross_val_accuracy} /n/n',
              f'Recall: {recall} /n/n',
              f'Precision: {precision} /n/n',
              f'f1 score: {f1}'
              )

    def choose_best_clasifier(self, clasifiers: list[ClassifierMixin], scoring: str = 'precision'): #-> ClassifierMixin:
        x_train, x_test, y_train, y_test = self.get_splitted_data()
        for clasifier in clasifiers:
            #  print(clasifier)
            if isinstance(clasifier,LogisticRegression):
                param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2']}
                grid_search = GridSearchCV(clasifier, param_grid, cv=5, scoring=scoring)
                grid_search.fit(x_train, y_train)
                # best_model = grid_search.best_estimator_
                print(f'Results for {clasifier}:  /n {grid_search.cv_results_}')


 

ds = Dataset()

logistic_regression = LogisticRegression()
# naive_bayes_classifier = GaussianNB()
# svm_classifier = SVC()
# rf_classifier = RandomForestClassifier()
# grb_classifier = GradientBoostingClassifier()


ds.choose_best_clasifier([logistic_regression])


