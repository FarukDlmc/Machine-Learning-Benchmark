import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


dataset = pd.read_csv('data/Social_Network_Ads.csv')
#print(dataset.head()) dataseti doğru yüklemiş miyim kontrolü (ilk 5 satır)
X= dataset.iloc[:, [2, 3]].values #sadece yaş ve maaş bilgisini alıyoruz. (id ve cinsiyet gereksiz)
y= dataset.iloc[:,-1].values #çıktımız satın alıp almayacağı

from sklearn.preprocessing import StandardScaler #Standardizasyon işlemi için hazır kütüphane
from sklearn.model_selection import train_test_split #test verisini ayırmak için hazır kütüphane

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0) # %25'lik test verisi ayırıyoruz. 

sc = StandardScaler()

X_train = sc.fit_transform(X_train) 
#fit: yaş ve maaş bilgilerinin standart sapmasını ve ortalamasını hesaplar. [öğretmenin dersi anlatması]
#transform: standart sapma ve ortalama'yı kullanarak eğitim verisindeki tüm sayıları standartlaştırır. [ders kitabını özetlemesi]

X_test = sc.transform(X_test)
#transform: öğrendiği kuralları kullanarak test verisini dönüştürür. [öğrencinin sınavı çözmesi]
#fit yok: test verisinde fit yapsaydık öğrenci sınav sorularını önceden öğrenmiş olurdu. [kopya çekmeyi önlemek]

from sklearn.svm import SVC #sınıflandırma kütüphanesi

#Modeli oluşturduk
#classifier = SVC(kernel='linear', random_state=0) #random_state: her çalıştığında aynı sonucu vermesi için
classifier = SVC(kernel='rbf', random_state=0) #açısal kıvrımlı ayrım 

#Modeli eğitiyoruz
classifier.fit(X_train, y_train)
print("Model Eğitimi Tamamlandı!")

y_pred = classifier.predict(X_test) # X_test: sınav soruları,  y_pred: tahminler

from sklearn.metrics import confusion_matrix, accuracy_score #sonuçları değerlendirmek için hazır kütüphaneler.

cm = confusion_matrix(y_test, y_pred) #Karmaşıklık Matrisi
print("\nKarmaşıklık Matrisi (Confusion Matrix):")
print(cm)

acc = accuracy_score(y_test, y_pred) #Doğruluk Oranı
print(f"\nDoğruluk Oranı: %{acc * 100:.2f}")


# GÖRSELLEŞTİRME 
from matplotlib.colors import ListedColormap

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

plt.title('SVM (Test Seti Sonuçları)')
plt.xlabel('Yaş (Ölçeklenmiş)')
plt.ylabel('Maaş (Ölçeklenmiş)')
plt.legend()

# Kaydet
from datetime import datetime
import os
zaman_damgasi = datetime.now().strftime("%Y%m%d_%H%M%S")
dosya_ismi = f"svm_reklam_sonuc_{zaman_damgasi}.png"
output_path = os.path.join("outputs", dosya_ismi)
os.makedirs("outputs", exist_ok=True) 
plt.savefig(output_path)

print(f"Harika! Grafik şuraya kaydedildi: {output_path}")
plt.show() 