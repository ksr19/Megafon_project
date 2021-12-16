import pandas as pd
import datetime as dt
import pickle
import dill
import sys
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score, confusion_matrix, \
    log_loss, f1_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer


def transform_features(df):
    df['buy_time'] = df['buy_time'].apply(lambda date: dt.datetime.fromtimestamp(date))
    df = df.set_index('id')
    df.sort_index(axis=0, inplace=True)
    return df


def transform_test(df):
    df['buy_time'] = df['buy_time'].apply(lambda date: dt.datetime.fromtimestamp(date))
    df['offer_day'] = df['buy_time'].dt.day
    df['offer_weekday'] = df['buy_time'].dt.dayofweek
    df['offer_hour'] = df['buy_time'].dt.hour
    return df


class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)

        try:
            return X[self.columns]
        except KeyError:
            cols_error = list(set(self.columns) - set(X.columns))
            raise KeyError("DataFrame не содердит следующие колонки: %s" % cols_error)


def output_df(df1, df2):
    df2 = df2.reset_index()
    df = df1.merge(df2, on=['id', 'vas_id', 'buy_time'], how='left')
    return df


if __name__ == '__main__':
    assert len(sys.argv) > 2, 'Необходимо передать тестовый датасет и датасет с ' \
                              'пользовательскими профилями в качестве аргументов'

    print('Считываем датасеты')
    initial_test_df = pd.read_csv(sys.argv[1]).drop(columns=['Unnamed: 0'])
    features_df = pd.read_csv(sys.argv[2], sep='\t').drop(columns=['Unnamed: 0'])

    print('Преобразуем датасеты')
    features_df = transform_features(features_df)

    sorted_test_df = initial_test_df.set_index('id')
    sorted_test_df.sort_index(axis=0, inplace=True)
    buy_times = sorted_test_df['buy_time']
    test_df = transform_test(sorted_test_df)

    print('Объединяем датасеты')
    # Объединяем датасеты
    df = pd.merge_asof(test_df, features_df, left_index=True, right_index=True,
                       by='buy_time', direction='nearest')

    print('Предсказание')
    model = dill.load(open('model.dill', 'rb'))
    target_pred_proba = model.predict_proba(df)[:, 1]

    print('Сохранение')
    test_df['target'] = target_pred_proba
    test_df['buy_time'] = buy_times.values
    test_df.drop(['offer_day', 'offer_weekday', 'offer_hour'], axis=1, inplace=True)
    df = output_df(initial_test_df, test_df)
    df.to_csv('answers_test.csv', index=False)
