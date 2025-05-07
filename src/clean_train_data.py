import pandas as pd

# 1. Veriyi oku
df = pd.read_csv('data/train.csv')

# 2. Çok eksik olan kolonları sil
df.drop(columns=['Alley', 'PoolQC', 'Fence', 'MiscFeature'], inplace=True, errors='ignore')

# 3. Sayısal verileri ortalama ile doldur
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mean())

# 4. Kategorik verileri mod (en sık) ile doldur
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])

# 5. Temizlenmiş veriyi kaydet
df.to_csv('data/clean_train.csv', index=False)

print("✅ Temizlik tamamlandı. Yeni dosya: data/clean_train.csv")
