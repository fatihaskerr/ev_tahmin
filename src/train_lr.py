import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression

# Veriyi yükle
train = pd.read_csv('data/clean_train.csv')
test = pd.read_csv('data/clean_test.csv')

# Özellikler ve hedef değişkeni ayır
X = train.drop(columns=['SalePrice'])
y = train['SalePrice']

# Kategorik veriyi sayısala çevir
X = pd.get_dummies(X)
test = pd.get_dummies(test)

# Kolonları hizala
X, test = X.align(test, join='left', axis=1, fill_value=0)

# Veriyi ölçeklendir
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test)  # Test verisinde aynı scaler'ı uygula

# K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmse_list = []
mae_list = []
r2_list = []

for train_index, val_index in kf.split(X_scaled):
    X_train, X_val = X_scaled[train_index], X_scaled[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # Modeli oluştur ve eğit
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Tahminler
    y_pred = model.predict(X_val)
    
    # Hata hesaplamaları
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    rmse_list.append(rmse)
    mae_list.append(mae)
    r2_list.append(r2)

# Sonuçları yazdır
print("📊 Lineer Regresyon Performansı (K-Fold Cross Validation):")
print(f"RMSE: {np.mean(rmse_list)}")
print(f"MAE: {np.mean(mae_list)}")
print(f"R2 Score: {np.mean(r2_list)}")

# Tahminleri kaydet
final_model = LinearRegression()
final_model.fit(X_scaled, y)
predictions = final_model.predict(test_scaled)

submission = pd.DataFrame({'Id': range(1461, 1461 + len(predictions)), 'SalePrice': predictions})
submission.to_csv('outputs/submission_lr_cv.csv', index=False)
print("✅ Tahminler outputs/submission_lr_cv.csv dosyasına kaydedildi.")
