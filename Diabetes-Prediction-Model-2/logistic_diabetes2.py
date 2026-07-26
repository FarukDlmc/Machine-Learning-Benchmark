import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from datetime import datetime
import os

# 1. VERİ YÜKLEME (CDC Veri Seti)
print("Veri seti yükleniyor, lütfen bekleyin...")
full_dataset = pd.read_csv('data/diabetes_binary_health_indicators_BRFSS2015.csv')

print(f"Toplam veri sayısı: {len(full_dataset)}")
print("Rastgele 25.000 kişi seçiliyor...")

# SVM kodundaki mantığın aynısı
dataset = full_dataset.sample(n=25000)

# CDC Verisinde Target en baştadır (0. index)
y = dataset.iloc[:, 0].values      
X = dataset.iloc[:, 1:].values     

# 2. EĞİTİM VE TEST AYRIMI
# random_state=0: SVM ile adil karşılaştırma için
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# 3. ÖLÇEKLENDİRME
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print("\n--- LOJİSTİK REGRESYON MODELİ EĞİTİLİYOR ---\n")

# 4. MODEL KURMA
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# 5. TAHMİN VE SKORLAR
y_pred = classifier.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"Lojistik Regresyon Doğruluğu: %{acc * 100:.2f}")
print("Confusion Matrix:\n", cm)

print("\n--- DETAYLI KARNE (Classification Report) ---")
# Bakalım Lojistik Regresyon hastaları bulabilecek mi?
print(classification_report(y_test, y_pred, zero_division=0))

# 6. GÖRSELLEŞTİRME
print("\nGrafik çiziliyor...")
plt.figure(figsize=(8, 6))

# Mor Tema (Purples)
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', cbar=False, annot_kws={"size": 16})

plt.title(f'CDC Lojistik Regresyon\nAccuracy: %{acc*100:.1f}', fontsize=15)
plt.xlabel('Tahmin Edilen', fontsize=12)
plt.ylabel('Gerçek Durum', fontsize=12)
plt.xticks([0.5, 1.5], ['Sağlıklı', 'Şeker Hastası'])
plt.yticks([0.5, 1.5], ['Sağlıklı', 'Şeker Hastası'])

# Kaydet
zaman_damgasi = datetime.now().strftime("%Y%m%d_%H%M%S")
dosya_ismi = f"cdc_logistic_sonuc_{zaman_damgasi}.png"
output_path = os.path.join("outputs", dosya_ismi)
os.makedirs("outputs", exist_ok=True)

plt.savefig(output_path)
print(f"Grafik kaydedildi: {output_path}")
plt.show()