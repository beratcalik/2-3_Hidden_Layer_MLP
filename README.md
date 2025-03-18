# BankNote Authentication ile MLP Sınıflandırma Çalışması

## 1. Giriş
Bu çalışma, Kaggle’dan temin edilen **BankNote_Authentication** veri seti kullanılarak, yapay sinir ağları (MLP) ile sınıflandırma probleminin çözümünü amaçlamaktadır. Çalışmada, laboratuvar saatlerinde eğitilen model mimarisi temel alınarak:
- **2-Layer Model:** 1 gizli katman + 1 çıkış katmanından oluşan basit yapı,
- **3-Layer Model:** 2 gizli katman + 1 çıkış katmanından oluşan daha derin yapı,
olarak iki farklı mimari ile deneyler gerçekleştirilmiştir. Her iki modelde de gizli katman aktivasyon fonksiyonu olarak `tanh` ve `ReLU` kullanılarak model performansı karşılaştırılmıştır.

## 2. Yöntem
### 2.1 Veri Seti
- **Veri Seti:** BankNote_Authentication.csv  
- **Özellikler:** Variance, skewness, curtosis, entropy  
- **Etiket:** class (0 veya 1)

### 2.2 Model Mimarileri ve Deneysel Ayarlar
- **Model Yapıları:**  
  - *2-Layer Model:* Giriş katmanı, 1 gizli katman (nöron sayısı deneysel olarak seçilebilir) ve çıkış katmanı (sigmoid aktivasyon).
  - *3-Layer Model:* Giriş katmanı, 2 gizli katman (her ikisinde aynı nöron sayısı kullanılmıştır) ve çıkış katmanı (sigmoid aktivasyon).

- **Aktivasyon Fonksiyonları:**  
  - Gizli katmanlarda: `tanh` ve `ReLU`
  - Çıkış katmanında: `sigmoid`

- **Eğitim Parametreleri:**  
  - **Optimizasyon:** Stokastik Gradient İnişi (SGD)
  - **Kayıp Fonksiyonu:** Binary Cross-Entropy Loss
  - **Eğitim Adımı (n_steps):** Deneysel olarak belirlenen iterasyon sayısı
  - **Performans Metrikleri:** Accuracy, Precision, Recall, F1-Score ve Confusion Matrix

- **Ek Uygulamalar:**  
  - Seçilen en iyi model, aynı mimari ile *Scikit-learn* ve *PyTorch* kullanılarak yeniden uygulanmıştır.
  - PyTorch kısmında hem 3-Layer hem de 2-Layer modeller için ayrı kod örnekleri yer almaktadır.

### 2.3 Uygulama Adımları
1. **Veri Seti Hazırlığı:**  
   - Veri setinin okunması, özellik ve etiket ayrımının yapılması.
   - Train/Test bölünmesi ve (opsiyonel) ölçeklendirme işlemleri.

2. **Model Eğitimi:**  
   - Tanımlanan MLP sınıfı üzerinden ileri yayılım, geri yayılım ve parametre güncelleme işlemleri ile eğitim.
   - Farklı aktivasyon fonksiyonları ve katman yapılarına göre deneysel çalışmalar.

3. **Model Değerlendirme:**  
   - Eğitim sonrası test verileri üzerinden sınıflandırma sonuçlarının hesaplanması.
   - Confusion Matrix ve diğer metriklerin raporlanması.

4. **Alternatif Uygulamalar:**  
   - Scikit-learn MLPClassifier ile aynı mimari yeniden uygulanması.
   - PyTorch kullanılarak hem 2-Layer hem de 3-Layer modellerin oluşturulması ve eğitimi.

## 3. Sonuçlar
Deneysel çalışmalar sonucunda:
- **Model Performansı:** Farklı mimariler ve aktivasyon fonksiyonları, model doğruluğu ve eğitim adım sayısı açısından karşılaştırılmıştır.
- **En İyi Model Seçimi:** %90 doğruluk gibi kriterler göz önünde bulundurularak, eğitim adım sayısı ve kayıp değeri bazında en iyi model belirlenmiştir.
- **Karmaşıklık Matrisi ve Diğer Metrikler:** Tüm modeller için confusion matrix, accuracy, precision, recall ve F1-score hesaplanmıştır.

Detaylı deney sonuçları ve karşılaştırmalar, rapor dosyası (opsiyonel: `rapor.pdf`) içerisinde sunulabilir.

## 4. Tartışma
Çalışma, model mimarisi ve aktivasyon fonksiyonu seçimlerinin sınıflandırma performansı üzerinde belirgin etkileri olduğunu göstermiştir:
- **2-Layer Model:** Daha basit yapısı sayesinde daha hızlı eğitim süresi ve düşük eğitim adım sayısı ile tercih edilebilir. Ancak karmaşık veri yapılarında performans sınırlamaları görülebilir.
- **3-Layer Model:** Ek gizli katman sayesinde daha yüksek doğruluk elde edilebilir fakat eğitim süresi artabilir ve aşırı öğrenme (overfitting) riski oluşabilir.
- **Aktivasyon Fonksiyonları Karşılaştırması:** `tanh` ve `ReLU` fonksiyonlarının farklı durumlarda avantajları ve dezavantajları deneysel olarak gözlemlenmiştir.

Bu sonuçlar, ileri çalışmalar için model seçiminde ve hiperparametre ayarlarında yol gösterici niteliktedir.

## 5. Kurulum ve Kullanım
### 5.1 Gereksinimler
- Python 3.x  
- Gerekli kütüphaneler:
  - numpy
  - pandas
  - matplotlib
  - scikit-learn
  - torch

### 5.2 Kurulum
Aşağıdaki komutla gerekli paketleri yükleyebilirsiniz:
```bash
pip install numpy pandas matplotlib scikit-learn torch
