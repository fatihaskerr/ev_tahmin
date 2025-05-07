import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras import layers, models

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
test_scaled = scaler.transform(test)

# Eğitim ve doğrulama seti
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Yapay sinir ağı modelini tanımla
model = models.Sequential([
    layers.Input(shape=(X_train.shape[1],)),  # Giriş katmanı
    layers.Dense(128, activation='relu'),  # İlk gizli katman
    layers.Dense(64, activation='relu'),   # İkinci gizli katman
    layers.Dense(1)  # Çıktı katmanı
])

# Modeli derle
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Modeli eğit
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), verbose=1)

# Modeli değerlendir
y_pred = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print("📊 Yapay Sinir Ağı Performansı:")
print("RMSE:", rmse)
print("MAE:", mae)
print("R2 Score:", r2)

# Tahminleri kaydet
predictions = model.predict(test_scaled)
submission = pd.DataFrame({'Id': range(1461, 1461 + len(predictions)), 'SalePrice': predictions.flatten()})
submission.to_csv('outputs/submission_ann.csv', index=False)
print("✅ Tahminler outputs/submission_ann.csv dosyasına kaydedildi.")
