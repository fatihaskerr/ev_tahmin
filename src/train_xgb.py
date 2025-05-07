import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

# 1. Temiz verileri yükle
train = pd.read_csv('data/clean_train.csv')
test = pd.read_csv('data/clean_test.csv')

# 2. Hedef ve ID'yi ayır
X = train.drop(['SalePrice', 'Id'], axis=1)
y = train['SalePrice']
test_ids = test['Id']
X_test = test.drop(['Id'], axis=1)

# 3. Kategorik verileri One-Hot Encoding ile dönüştür
X = pd.get_dummies(X)
X_test = pd.get_dummies(X_test)

# 4. Eğitim ve test verisini hizala
X, X_test = X.align(X_test, join='left', axis=1, fill_value=0)

# 5. Verileri ölçeklendir
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# 6. Eğitim/test ayır
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 7. XGBoost modelini oluştur ve eğit
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
model.fit(X_train, y_train)

# 8. Doğrulama setinde tahmin ve değerlendirme
y_pred = model.predict(X_val)
print("📊 XGBoost Performansı:")
print("RMSE:", np.sqrt(mean_squared_error(y_val, y_pred)))
print("MAE:", mean_absolute_error(y_val, y_pred))
print("R2 Score:", r2_score(y_val, y_pred))

# 9. Gerçek test verisinde tahmin yap
test_preds = model.predict(X_test_scaled)

# 10. Sonuçları submission dosyasına yaz
submission = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': test_preds
})

submission.to_csv('outputs/submission_xgb.csv', index=False)
print("✅ Tahminler outputs/submission_xgb.csv dosyasına kaydedildi.")
