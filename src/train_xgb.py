import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

# Veriyi yükle
train = pd.read_csv('data/clean_train.csv')
test = pd.read_csv('data/clean_test.csv')

# Özellikler ve hedef değişkeni ayır
X = train.drop(columns=['SalePrice', 'Id'])
y = train['SalePrice']
test_ids = test['Id']
X_test = test.drop(columns=['Id'])

# Kategorik veriyi sayısala çevir
X = pd.get_dummies(X)
X_test = pd.get_dummies(X_test)

# Kolonları hizala
X, X_test = X.align(X_test, join='left', axis=1, fill_value=0)

# Veriyi ölçeklendir
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmse_list = []
mae_list = []
r2_list = []

for train_index, val_index in kf.split(X_scaled):
    X_train, X_val = X_scaled[train_index], X_scaled[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # Modeli oluştur ve eğit
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
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
print("📊 XGBoost Performansı (K-Fold Cross Validation):")
print(f"RMSE: {np.mean(rmse_list)}")
print(f"MAE: {np.mean(mae_list)}")
print(f"R2 Score: {np.mean(r2_list)}")

# Son olarak modelin tamamını tüm veri ile eğit
final_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
final_model.fit(X_scaled, y)

# Test verisinde tahmin yap
test_preds = final_model.predict(X_test_scaled)

# Sonuçları submission dosyasına kaydet
submission = pd.DataFrame({'Id': test_ids, 'SalePrice': test_preds})
submission.to_csv('outputs/submission_xgb_cv.csv', index=False)
print("✅ Tahminler outputs/submission_xgb_cv.csv dosyasına kaydedildi.")
