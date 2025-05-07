import pandas as pd
import numpy as np

def add_new_features(df):
    # Yeni öznitelikler oluşturuluyor
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['Age'] = df['YrSold'] - df['YearBuilt']
    df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
    df['TotalBath'] = df['FullBath'] + (0.5 * df['HalfBath']) + df['BsmtFullBath'] + (0.5 * df['BsmtHalfBath'])
    df['TotalPorchSF'] = df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch']
    df['IsRemodeled'] = (df['YearRemodAdd'] != df['YearBuilt']).astype(int)
    return df

def load_and_preprocess_data():
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')

    # Çok eksik verili kolonları sil
    drop_cols = ['Alley', 'PoolQC', 'Fence', 'MiscFeature']
    train.drop(columns=drop_cols, inplace=True, errors='ignore')
    test.drop(columns=drop_cols, inplace=True, errors='ignore')

    # Sayısal kolonları ortalama ile doldur
    for col in ['LotFrontage', 'MasVnrArea', 'GarageYrBlt']:
        if col in train.columns:
            train[col] = train[col].fillna(train[col].mean())
        if col in test.columns:
            test[col] = test[col].fillna(test[col].mean())

    # Kategorik kolonları 'None' ile doldur
    fill_none_cols = ['MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
                      'BsmtFinType1', 'BsmtFinType2', 'GarageType', 
                      'GarageFinish', 'GarageQual', 'GarageCond']
    for col in fill_none_cols:
        if col in train.columns:
            train[col] = train[col].fillna('None')
        if col in test.columns:
            test[col] = test[col].fillna('None')

    # En sık değerle doldurma
    if 'Electrical' in train.columns:
        train['Electrical'] = train['Electrical'].fillna(train['Electrical'].mode()[0])

    # Yeni öznitelikleri ekle
    train = add_new_features(train)
    test = add_new_features(test)

    # One-Hot Encoding
    train = pd.get_dummies(train)
    test = pd.get_dummies(test)

    # Kolonları hizala
    train, test = train.align(test, join='left', axis=1, fill_value=0)

    return train, test
