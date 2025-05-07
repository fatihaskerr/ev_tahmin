from preprocess import load_and_preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import pandas as pd
import os

# Gerekiyorsa klasörleri oluştur
os.makedirs('models', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# Veriyi yükle ve işle
train, test = load_and_preprocess_data()

# Özellik ve hedef değişkenleri ayır
X = train.drop(['SalePrice', 'Id'], axis=1)
y = train['SalePrice']

# Ölçeklendir
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Eğitim/Doğrulama ayır
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Yapay sinir ağı modeli
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Eğitimi başlat
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val))

# Modeli kaydet
model.save('models/ann_model.h5')

# Eğitim kaybı grafiği
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig('outputs/training_loss.png')
plt.show()

# Tahmin üret
test_ids = test['Id']
# 👇 'Id' ve 'SalePrice' varsa ikisini birden çıkar
test = test.drop(columns=['Id'], errors='ignore')
test = test.drop(columns=['SalePrice'], errors='ignore')
test_scaled = scaler.transform(test)

predictions = model.predict(test_scaled).ravel()

submission = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': predictions
})

# 🔽 Şu üç satırı mutlaka ekle
print(submission.head())  # Tahminler ekranda gözüksün
submission.to_csv('outputs/submission.csv', index=False)
print("✅ Tahminler 'outputs/submission.csv' dosyasına başarıyla kaydedildi.")



