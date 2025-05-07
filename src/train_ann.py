import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Verileri oku
data_path = 'data/clean_train.csv'
test_path = 'data/clean_test.csv'
train = pd.read_csv(data_path)
test = pd.read_csv(test_path)

# Test verisinde hedef sütunu (SalePrice) kaldır
test = test.drop(columns=['SalePrice'], errors='ignore')

# Kategorik verileri sayısala çevir
train = pd.get_dummies(train)
test = pd.get_dummies(test)

# Kolonları hizala
train, test = train.align(test, join='left', axis=1, fill_value=0)

# Özellikleri ve hedef değişkeni ayır
X = train.drop(columns=['SalePrice'])
y = train['SalePrice']

# Ölçeklendirme
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test)  # Test verisinde aynı scaler'ı uygula

# Eğitim ve doğrulama seti
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Yapay sinir ağı modeli tanımla
model = models.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Modeli eğit
history = model.fit(X_train, y_train, epochs=100, batch_size=32,
                    validation_data=(X_val, y_val), verbose=1)

# Tahmin ve değerlendirme
y_pred = model.predict(X_val)
rmse = mean_squared_error(y_val, y_pred, squared=False)
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print("\n📊 Yapay Sinir Ağı Performansı:")
print("RMSE:", rmse)
print("MAE:", mae)
print("R2 Score:", r2)

# Tahminleri dosyaya yaz
predictions = model.predict(test_scaled)
submission = pd.DataFrame({'Id': range(1461, 1461 + len(predictions)), 'SalePrice': predictions.flatten()})
os.makedirs('outputs', exist_ok=True)
submission.to_csv('outputs/submission_ann.csv', index=False)
print("✅ Tahminler outputs/submission_ann.csv dosyasına kaydedildi.")

# Eğitim kaybı grafiği
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('ANN Eğitim Grafiği')
plt.legend()
os.makedirs('outputs', exist_ok=True)
plt.savefig('outputs/ann_training_loss.png')
plt.show()
