from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler


# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')

    class Normalize(BaseEstimator, TransformerMixin):
    def __init__(self, columns, ignore_columns):
        self.columns = columns
        self.ignore_columns = ignore_columns
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data = X.copy()
        ct = ColumnTransformer(
            [("norml2", Normalizer(norm='l2'), data.drop(self.ignore_columns, axis=1).columns)],
            remainder='passthrough'
        )
        data[data.columns] = ct.fit_transform(X=data)
        return data

class standardize(BaseEstimator, TransformerMixin):
    def __init__(self, columns, ignore_columns):
        self.columns = columns
        self.ignore_columns = ignore_columns
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data = X.copy()
        ct = ColumnTransformer(
            [("std", StandardScaler(), data.drop(self.ignore_columns, axis=1).columns)],
            remainder='passthrough'
        )
        data[data.columns] = ct.fit_transform(X=data)
        return data

class MinMax(BaseEstimator, TransformerMixin):
    def __init__(self, columns, ignore_columns):
        self.columns = columns
        self.ignore_columns = ignore_columns
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data = X.copy()
        ct = ColumnTransformer(
            [("minmax", MinMaxScaler(), data.drop(self.ignore_columns, axis=1).columns)],
            remainder='passthrough'
        )
        data[data.columns] = ct.fit_transform(X=data)
        return data
