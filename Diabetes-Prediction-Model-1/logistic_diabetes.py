import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from datetime import datetime
import os

# 1. VERİ YÜKLEME VE İŞLEME
dataset = pd.read_csv('data/diabetes.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# DİKKAT: SVM ile adil karşılaştırma için random_state=0 AYNI OLMALI
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print("\n--- LOJİSTİK REGRESYON MODELİ EĞİTİLİYOR ---\n")

# 2. MODEL KURMA (Logistic Regression)
# random_state=0: Her seferinde aynı sonucu versin
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# 3. TAHMİN VE SKOR
y_pred = classifier.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"Lojistik Regresyon Doğruluğu: %{acc * 100:.2f}")
print("Confusion Matrix:\n", cm)

# 4. GÖRSELLEŞTİRME (Tekli Heatmap)
print("\nGrafik çiziliyor...")
plt.figure(figsize=(8, 6))

# Mor tonları (Purples) kullanalım, karışmasın
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', cbar=False, annot_kws={"size": 16})

plt.title(f'Lojistik Regresyon\nAccuracy: %{acc*100:.1f}', fontsize=15)
plt.xlabel('Tahmin Edilen', fontsize=12)
plt.ylabel('Gerçek Durum', fontsize=12)
plt.xticks([0.5, 1.5], ['Sağlıklı', 'Şeker Hastası'])
plt.yticks([0.5, 1.5], ['Sağlıklı', 'Şeker Hastası'])

# Kaydet
zaman_damgasi = datetime.now().strftime("%Y%m%d_%H%M%S")
dosya_ismi = f"logistic_diabetes_sonuc_{zaman_damgasi}.png"
output_path = os.path.join("outputs", dosya_ismi)
os.makedirs("outputs", exist_ok=True)

plt.savefig(output_path)
print(f"Grafik kaydedildi: {output_path}")
plt.show()