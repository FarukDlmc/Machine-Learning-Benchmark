import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from datetime import datetime
import os

# 1. VERİ YÜKLEME
dataset = pd.read_csv('data/Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values

# 2. SPLIT (SVM ile aynı şartlarda)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# 3. SCALING
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print("\n--- LOJİSTİK REGRESYON MODELİ EĞİTİLİYOR ---\n")

# 4. MODEL (Logistic Regression)
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)
print("Model Eğitimi Tamamlandı!")

# 5. TAHMİN
y_pred = classifier.predict(X_test)

# 6. SONUÇLAR
cm = confusion_matrix(y_test, y_pred)
print("\nKarmaşıklık Matrisi:")
print(cm)

acc = accuracy_score(y_test, y_pred)
print(f"\nDoğruluk Oranı: %{acc * 100:.2f}")

# 7. GÖRSELLEŞTİRME 
print("Grafik çiziliyor...")
plt.figure(figsize=(10, 6))

X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j, edgecolors='black')

plt.title('Lojistik Regresyon (Test Seti)')
plt.xlabel('Yaş (Ölçeklenmiş)')
plt.ylabel('Maaş (Ölçeklenmiş)')
plt.legend()

# Kaydet
zaman_damgasi = datetime.now().strftime("%Y%m%d_%H%M%S")
dosya_ismi = f"logistic_social_sonuc_{zaman_damgasi}.png"
output_path = os.path.join("outputs", dosya_ismi)
os.makedirs("outputs", exist_ok=True) 
plt.savefig(output_path)

print(f"Grafik kaydedildi: {output_path}")
plt.show()