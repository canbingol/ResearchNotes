# Sinir Ağı Nedir?

Sinir ağları (Neural Networks - NN), insan beynindeki biyolojik nöronların yapısından ve işlevlerinden ilham alan hesaplama modelleridir. Beyin, milyarlarca nöronun birbiriyle bağlantılı çalışması sayesinde bilgiyi işler. Bu biyolojik mekanizma, yapay sinir ağlarının temelini oluşturur. 

![Biyolojik ve Yapay Nöronlar Karşılaştırması](https://github.com/canbingol/ResearchNotes/blob/main/images/neurons.png)

Bir yapay sinir ağı, düğümler veya yapay nöronlar olarak adlandırılan birimlerden oluşur. Bu nöronlar arasında, bağlantıların gücünü belirten ve öğrenme sürecinin temelini oluşturan ağırlıklar (weights) bulunur. Her yapay nöron, kendisine gelen girdilerin ağırlıkları ile çarpımını toplar, ardından bu toplamı bir aktivasyon fonksiyonu ile işler. Sonuçta, sistem karmaşık desenleri tanıyabilir veya belirli bir görevi yerine getirebilir.

Sinir ağları, ham verilerden doğrudan öğrenme ve karmaşık ilişkileri modelleme yetenekleri sayesinde geniş bir uygulama alanına sahiptir. Örneğin:

- Görsellerde nesneleri tanımak (**bilgisayarlı görü**),
- Metinlerden anlam çıkarmak (**doğal dil işleme**),
- Veri kümelerini sınıflandırmak veya desenleri belirlemek gibi pek çok görevi başarıyla yerine getirir.

## Bir Sinir Ağı Aşağıdaki Temel Bileşenlerden Oluşur:

## 1. İleri Yayılım (Forward Propagation)
İleri yayılım, sinir ağının aldığı girdileri işleyerek bir çıktı üretme sürecidir. Bu, sinir ağının tahmin yapma aşamasıdır. İşleyiş şu şekilde ilerler:
- **Parametreleri başlatma**: Modelin parametreleri (ağırlıklar, biaslar vb.) en başta rastgele atanır. Rastgele parametreler, modelin başlangıçta verilerden öğrenmeye başlaması için bir temel oluşturur. Eğer tüm parametreler aynı başlangıç değeri (örneğin 0) ile başlatılsaydı, tüm nöronlar aynı girdilere aynı şekilde tepki verirdi. Bu durumda modelin öğrenmesi mümkün olmazdı.
- **Girdiler işleme alınır**: Girdiler ağdaki nöronlar üzerinden geçerken, her bağlantıda farklı ağırlıklar ile çarpılır. Bu ağırlıklar, modelin tahminlerini nasıl şekillendirdiğini kontrol eder.
- **Bias eklenir**: Ağırlıklı toplamların üzerine bir bias değeri eklenir. Bias, modelin daha esnek hale gelmesini sağlar ve verilerin belirli bir yönde kaymasını kolaylaştırır.
- **Aktivasyon fonksiyonu uygulanır**: Hesaplanan toplam, doğrusal olmayan bir ilişkiyi öğrenebilmesi için bir aktivasyon fonksiyonundan geçirilir. Aktivasyon fonksiyonu, modelin yalnızca düz çizgi (doğrusal) ilişkileri değil, karmaşık desenleri de öğrenmesine olanak tanır. Örneğin, ReLU (Rectified Linear Unit) veya Sigmoid gibi fonksiyonlar sıklıkla kullanılır.

İlerleyen bölümde bu anlatılanların uygulaması yapılacak.

---

### 2. Geri Yayılım (Backward Propagation)
Geri yayılım, sinir ağının yaptığı hataları öğrenme ve bu hataları düzeltmek için ağdaki parametreleri (ağırlık ve bias) güncelleme sürecidir. Bu aşama modelin kendini geliştirmesini sağlar. İşleyiş şu şekildedir:
- **Hata hesaplanır**: Modelin ürettiği tahmin ile gerçek değer arasındaki fark bulunur. Bu fark, bir kayıp fonksiyonu (loss function) ile ölçülür. Örneğin, sınıflandırma problemlerinde “cross-entropy loss” veya regresyon problemlerinde “mean squared error” kullanılabilir.
- **Gradyanlar hesaplanır**: Hatanın kaynağı bulunur. Zincir kuralı ve türevler kullanılarak, her bir ağırlığın ve bias değerinin hataya ne kadar katkıda bulunduğu hesaplanır. Buna gradyan hesaplama denir.
- **Hata geriye yayılır**: Bu gradyanlar sinir ağı boyunca geriye doğru yayılır. Amaç, her bir bağlantının (parametrenin) modelin hatasına olan etkisini anlamaktır.

---

### 3. Parametre Güncelleme (Parameter Update)
Geri yayılımda hesaplanan hatalar ve gradyanlar, modelin ağırlıklarını ve biaslarını güncellemek için kullanılır. Bu, sinir ağının her tekrarda daha iyi hale gelmesini sağlayan önemli bir adımdır. İşleyiş şu şekildedir:
- **Gradyan bilgisi kullanılır**: Her bir ağırlık ve bias, geri yayılım sırasında hesaplanan gradyanlara göre güncellenir. Bu güncelleme, bir hiperparametre olan öğrenme oranı (learning rate) ile kontrol edilir. Öğrenme oranı, modelin ne kadar büyük adımlarla güncelleme yapacağını belirler.
- **Parametre optimizasyonu**: Gradyan bilgisi, ağırlıkların ve biasların hatayı azaltacak şekilde nasıl değiştirileceğine rehberlik eder. Model, bu adımda hatayı minimize etmek için kendini sürekli geliştirir.
- **Hatayı minimize etmek**: Güncelleme işlemleri sayesinde model, hatayı giderek azaltır. Bu, modelin bir sonraki iterasyonda girdilere daha doğru yanıtlar verebilmesini sağlar.

Bu üç adım, bir sinir ağının öğrenme sürecinin çekirdeğini oluşturur ve modelin girdileri istenen çıktılarla etkili bir şekilde eşleştirmesini sağlar. Eğitim sürecinde loss’umuz istenilen seviyeye düşene kadar bu döngü devam eder.


## İleri Yönlü Yayılım (Forward Propagation)

İleri yönlü yayılım, bir sinir ağının öğrenme sürecindeki en temel işlemdir ve modelin girdilerden çıktılara nasıl ulaştığını açıklar. Bu süreçte, ağın katmanlarından geçen veri, belirli matematiksel işlemlerle dönüştürülerek nihai bir çıktı elde edilir.

---

### Adım Adım İleri Yönlü Yayılım:

#### 1. Girdilerin Alınması:
- İlk olarak, ağın giriş katmanına verilen veriler alınır. Bu veriler, modelin analiz edeceği ham bilgilerden oluşur.

#### 2. Ağırlıklarla Çarpım:
- Her bir giriş, belirli bir ağırlık değeri ile çarpılır. Ağırlıklar, ağın öğrenme süreci sırasında optimize edilen parametrelerdir ve her girdinin önemini temsil eder.

#### 3. Toplama İşlemi:
- Çarpımların sonuçları toplanır ve her bir nöron için bias (*b*) adı verilen bir sabit değer eklenir. Bu işlem, her nöronun aktivasyon öncesindeki toplam girdisini hesaplar:
  

z = (w_1 * x_1) + (w_2 * x_2)  + b


#### 4. Aktivasyon Fonksiyonu:
- Hesaplanan toplam (*z*), bir aktivasyon fonksiyonuna (örneğin, Sigmoid, ReLU gibi) uygulanır. Aktivasyon fonksiyonu, modelin doğrusal olmayan ilişkileri öğrenmesine olanak tanır ve nöronun çıktı değerini belirler:

y = f(z)


#### 5. Sonraki Katmana Geçiş:
- Elde edilen çıktı değerleri, ağın bir sonraki katmanına giriş olarak aktarılır. Bu işlem, ağın tüm katmanları boyunca tekrarlanır ve nihai çıktıya ulaşılır.

#### 6. Çıktı Üretimi:
- Ağın son katmanından elde edilen sonuç, modelin tahmin ettiği çıktı değeridir. Bu değer:
  - Sınıflandırma problemleri için bir olasılık,
  - Regresyon problemleri için bir sürekli değer olabilir.


## Adım Adım İleri Yayılım Süreci

Bir sinir ağının ileri yayılım süreci, girdilerden başlayarak nihai tahmine kadar yapılan hesaplamaları kapsar. Bu süreci daha iyi anlamak için aşağıdaki adımları inceleyelim:

![Sinir Ağı Görselleştirme](https://github.com/canbingol/ResearchNotes/blob/main/images/zero_netwrok.png)

---

### Gizli Katman İçin İleri Yayılım

Modelimize verilen girdiler `[1, 2]` olup, bu değerler gizli katman nöronlarına ağırlıklar ve bias değerleri ile birlikte işlenir. Aşağıda ağırlıklar ve bias değerleri tablo olarak verilmiştir:

| **Nöron**           | **Ağırlık 1 (w1)** | **Ağırlık 2 (w2)** | **Bias (b)** |
|----------------------|--------------------|--------------------|--------------|
| Gizli Katman Nöron 1 | 0.5                | -0.1               | 0.2          |
| Gizli Katman Nöron 2 | 0.3                | 0.6                | 0.3          |

Hesaplama şu şekilde gerçekleşir:

1. **İlk Nöron (Z1):**
   Girdi değerleri, ağırlıklarla çarpılır ve bias eklenir:

   Z_1 = (1  0.5) + (2  -0.1) + 0.2 = 0.5
   

2. **İkinci Nöron (Z2):**
   Aynı işlem diğer nöron için yapılır:
   
   Z_2 = (1  0.3) + (2  0.6) + 0.3 = 1.8
   

Bu adımlar sonucunda gizli katman nöronlarının çıktıları:

Z_1 = 0.5,  Z_2 = 1.8

olarak hesaplanır.

Bu değerler, ağın bir sonraki katmanına girdi olarak aktarılır.

![Z Hesaplama Görselleştirme](https://github.com/canbingol/ResearchNotes/blob/main/images/calculate_Z.png)



## Çıktı Katmanı İçin Hesaplama

Gizli katmandan gelen \(Z_1\) ve \(Z_2\) değerleri, çıktı katmanındaki ağırlıklar ve bias kullanılarak işlenir. Aşağıda çıktı katmanı için ağırlıklar ve bias değerleri tablo halinde verilmiştir:

| **Nöron**           | **Ağırlık \(Z_1\)** | **Ağırlık \(Z_2\)** | **Bias (b)** |
|----------------------|---------------------|---------------------|--------------|
| Çıktı Katmanı        | -1.2               | 0.02                | 2.1          |

Hesaplama şu şekilde gerçekleşir:



y = (Z_1 * -1.2) + (Z_2 * 0.02) + 2.1


Burada \(Z_1\) ve \(Z_2\) değerlerini yerlerine koyarsak:


y = (0.5 * -1.2) + (1.8 * 0.02) + 2.1 = 1.536


Sonuç olarak, modelin tahmin ettiği çıktı değeri **1.536** olarak elde edilir. Burada elde edilen değer, modelimizin tahmin ettiği değerdir.

> **Not:** Bu hesaplamalarda işlemleri basit tutmak için **aktivasyon fonksiyonları** yok sayılmıştır.

![Çıktı Katmanı Hesaplama Görselleştirme](https://github.com/canbingol/ResearchNotes/blob/main/images/predict.png)


## Paralel Hesaplama ve Matris Operasyonları

Günümüz sinir ağlarında bu işlemler tek tek veya for döngüleri içinde yapılmaz. Bunun yerine daha verimli olan ve GPU'lar ile paralel olarak hesaplanabilen **matris işlemleri** kullanılır.

---

## Loss Hesaplama

Loss, modelimizin tahmini ile istediğimiz değer arasındaki sapmayı ifade eder. Amacımız, bu sapmayı (yani loss’u) en aza indirmektir.

Loss hesaplamak için birden fazla fonksiyon vardır ve bu fonksiyonlar, problemin türüne göre seçilir. Bu örnekte **MSE (Mean Squared Error)** yani **ortalama kare hatası** kullanacağız.

### MSE Loss Hesaplama

Diyelim ki, tahmin etmek istediğimiz değer **2**. Peki, MSE loss nasıl hesaplanır?


MSE =  (1/N) × Σ(Hedef - Tahmin)^2


MSE hesaplanırken:
1. Hedef değerler ile tahmin edilen değerler arasındaki fark hesaplanır.
2. Farkın karesi alınır.
3. Bu farkların kareleri toplanır ve toplam, tahmin sayısına bölünür.

#### Örnek Hesaplama:

1. Tahmin edilen çıktı ile hedef değer arasındaki fark hesaplanır:
   
    Fark = 2 - 1.536 = 0.464
   

2. Bu farkın karesi alınır:
   
  Fark^2 = (0.464)^2 = 0.215

4. Tek bir veri noktası için MSE, doğrudan bu değere eşit olur:

   MSE = 0.215
   

#### Birden Fazla Veri İçin MSE Hesaplama

Eğer model birden fazla veri noktasıyla çalışıyorsa, her bir veri için tahmin ve gerçek değer arasındaki farkın karesi hesaplanır ve bu değerlerin ortalaması alınır.

---

## İleri Yayılım için Basit Python Uygulaması

Aşağıda bir sinir ağında ileri yayılım ve MSE hesaplaması için basit bir Python kodu verilmiştir:

```python
#girdiler
x1 = 1
x2 = 2

# gizli katman 1. nöron ağırlıları
w11 = 0.5
w12 = -0.1

# gizli katman 2. nöron ağırlıları
w21 = 0.3
w22 = 0.6

# gizli katman biasları
b1 = 0.2
b2 = 0.3

# gizli katman çıktı hesaplama
z1 = (x1 * w11) + (x2 * w12) + b1
z2 = (x1 * w21) + (x2 * w22) + b2
print(f"z1: {z1} \nz2: {z2}")

# çıktı katmanı ağırlıkarı ve biası
w31 = -1.2
w32 = 0.02
b3 = 2.1

tahmin = (z1 * w31) + (z2 * w32) + b3
print("Tahmin:",tahmin)


# MSE loss hesaplama
hedef = 2
mse = (hedef - tahmin) ** 2 
mse = mse / 1 # burada toplam hedef sayısına bölünür
print("MSE:", mse)
```
### Çıktı

```çıktı
z1: 0.5 
z2: 1.8
Tahmin: 1.536
MSE: 0.215296
```
