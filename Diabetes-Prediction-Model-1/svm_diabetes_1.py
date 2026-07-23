import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from datetime import datetime
import os

print("--- MODEL 1: PIMA INDIANS DIABETES VERİ SETİ ---")
print("Veri seti yükleniyor...")

# 1. Veri Yükleme (Sonuç kolonu sonda)
dataset = pd.read_csv('data/diabetes.csv')
X = dataset.iloc[:, :-1].values # İlk 8 kolon özellikler
y = dataset.iloc[:, -1].values  # Son kolon hedef (Şeker hastası mı?)

# 2. Veri Bölme ve Ölçeklendirme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train) 
X_test = sc.transform(X_test)

# 3. Model Eğitimi (Linear vs RBF)
print("\n--- MODEL KARŞILAŞTIRMASI BAŞLIYOR ---\n")

linear_classifier = SVC(kernel='linear', random_state=0)
linear_classifier.fit(X_train, y_train)
y_pred_linear = linear_classifier.predict(X_test)
acc_linear = accuracy_score(y_test, y_pred_linear)
cm_linear = confusion_matrix(y_test, y_pred_linear)

rbf_classifier = SVC(kernel='rbf', random_state=0)
rbf_classifier.fit(X_train, y_train)
y_pred_rbf = rbf_classifier.predict(X_test)
acc_rbf = accuracy_score(y_test, y_pred_rbf)
cm_rbf = confusion_matrix(y_test, y_pred_rbf)

print(f"1. Linear Kernel Doğruluğu: %{acc_linear * 100:.2f}")
print(f"2. RBF Kernel Doğruluğu:    %{acc_rbf * 100:.2f}")

# 4. Raporlama
print("\n--- DETAYLI KARNE (Classification Report) ---")
print("\nLINEAR MODEL KARNESİ:")
print(classification_report(y_test, y_pred_linear, zero_division=0))
print("\nRBF MODEL KARNESİ:")
print(classification_report(y_test, y_pred_rbf, zero_division=0)) 

# 5. Görselleştirme
print("\nGrafikler hazırlanıyor...")
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

sns.heatmap(cm_linear, annot=True, fmt='d', cmap='Blues', ax=ax[0], cbar=False, annot_kws={"size": 16})
ax[0].set_title(f'Linear Kernel\nAccuracy: %{acc_linear*100:.1f}', fontsize=14)
ax[0].set_xlabel('Tahmin Edilen (Predicted)', fontsize=12)
ax[0].set_ylabel('Gerçek Durum (Actual)', fontsize=12)
ax[0].set_xticklabels(['Sağlıklı', 'Şeker Hastası'])
ax[0].set_yticklabels(['Sağlıklı', 'Şeker Hastası'])

sns.heatmap(cm_rbf, annot=True, fmt='d', cmap='Oranges', ax=ax[1], cbar=False, annot_kws={"size": 16})
ax[1].set_title(f'RBF Kernel\nAccuracy: %{acc_rbf*100:.1f}', fontsize=14)
ax[1].set_xlabel('Tahmin Edilen (Predicted)', fontsize=12)
ax[1].set_ylabel('Gerçek Durum (Actual)', fontsize=12)
ax[1].set_xticklabels(['Sağlıklı', 'Şeker Hastası'])
ax[1].set_yticklabels(['Sağlıklı', 'Şeker Hastası'])

plt.tight_layout() 

os.makedirs("outputs", exist_ok=True)
zaman_damgasi = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = os.path.join("outputs", f"svm_diabetes_model1_{zaman_damgasi}.png")
plt.savefig(output_path)
print(f"Karşılaştırma tablosu kaydedildi: {output_path}")
plt.show()