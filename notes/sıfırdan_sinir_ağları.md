# Sinir Ağı Nedir?

Sinir ağları (Neural Networks - NN), insan beynindeki biyolojik nöronların yapısından ve işlevlerinden ilham alan hesaplama modelleridir. Beyin, milyarlarca nöronun birbiriyle bağlantılı çalışması sayesinde bilgiyi işler. Bu biyolojik mekanizma, yapay sinir ağlarının temelini oluşturur. 

![Biyolojik ve Yapay Nöronlar Karşılaştırması](https://github.com/canbingol/ResearchNotes/blob/main/images/neurons.png)

Bir yapay sinir ağı, düğümler veya yapay nöronlar olarak adlandırılan birimlerden oluşur. Bu nöronlar arasında, bağlantıların gücünü belirten ve öğrenme sürecinin temelini oluşturan ağırlıklar (**weights**) bulunur. Her yapay nöron, kendisine gelen girdilerin ağırlıkları ile çarpımını toplar, ardından bu toplamı bir aktivasyon fonksiyonu ile işler. Sonuçta, sistem karmaşık desenleri tanıyabilir veya belirli bir görevi yerine getirebilir.

Sinir ağları, ham verilerden doğrudan öğrenme ve karmaşık ilişkileri modelleme yetenekleri sayesinde geniş bir uygulama alanına sahiptir. Örneğin:

- Görsellerde nesneleri tanımak (**bilgisayarlı görü**),
- Metinlerden anlam çıkarmak (**doğal dil işleme**),
- Veri kümelerini sınıflandırmak veya desenleri belirlemek gibi pek çok görevi başarıyla yerine getirir.

---

## Adım Adım İleri Yayılım Süreci

Bir sinir ağının ileri yayılım süreci, girdilerden başlayarak nihai tahmine kadar yapılan hesaplamaları kapsar. Bu süreci daha iyi anlamak için aşağıdaki adımları inceleyelim:

![Sinir Ağı Görselleştirme](https://github.com/canbingol/ResearchNotes/blob/main/images/zero_netwrok.png)

---

### Gizli Katman İçin İleri Yayılım

Gizli katman için kullanılan ağırlıklar ve bias değerleri aşağıdaki tabloda verilmiştir:

| **Katman**  | **Ağırlıklar (Weights)** | **Bias (b)** |
|-------------|--------------------------|--------------|
| Gizli Katman Nöron 1 | w11 = 0.5, w12 = -0.1 | b1 = 0.2 |
| Gizli Katman Nöron 2 | w21 = 0.3, w22 = 0.6  | b2 = 0.3 |

Hesaplama şu şekilde gerçekleşir:

1. **İlk Nöron (Z1):**
   Girdi değerleri, ağırlıklarla çarpılır ve bias eklenir:
   Z1 = (x1 * w11) + (x2 * w12) + b1
   Z1 = (1 * 0.5) + (2 * -0.1) + 0.2 = 0.5

2. **İkinci Nöron (Z2):**
   Aynı işlem diğer nöron için yapılır:
   Z2 = (x1 * w21) + (x2 * w22) + b2
   Z2 = (1 * 0.3) + (2 * 0.6) + 0.3 = 1.8

Bu adımlar sonucunda gizli katman nöronlarının çıktıları:
Z1 = 0.5, Z2 = 1.8

Bu değerler, ağın bir sonraki katmanına girdi olarak aktarılır.

![Z Hesaplama Görselleştirme](https://github.com/canbingol/ResearchNotes/blob/main/images/calculate_Z.png)

---

## Çıktı Katmanı İçin Hesaplama

Çıktı katmanı için kullanılan ağırlıklar ve bias değerleri:

| **Ağırlıklar (Weights)** | **Bias (b)** |
|---------------------------|--------------|
| w31 = -1.2, w32 = 0.02    | b3 = 2.1     |

Gizli katmandan gelen Z1 ve Z2 değerleri, çıktı katmanındaki ağırlıklar ve bias kullanılarak işlenir. Hesaplama şu şekildedir:

y = (Z1 * w31) + (Z2 * w32) + b3

Burada Z1 ve Z2 değerlerini yerlerine koyarsak:

y = (0.5 * -1.2) + (1.8 * 0.02) + 2.1 = 1.536

Sonuç olarak, modelin tahmin ettiği çıktı değeri **1.536** olarak elde edilir. Burada elde edilen değer, modelimizin tahmin ettiği değerdir.

> **Not:** Bu hesaplamalarda işlemleri basit tutmak için **aktivasyon fonksiyonları** yok sayılmıştır.

![Çıktı Katmanı Hesaplama Görselleştirme](https://github.com/canbingol/ResearchNotes/blob/main/images/predict.png)

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
print(f"z1: {z1} \\nz2: {z2}")

# çıktı katmanı ağırlıkarı ve biası
w31 = -1.2
w32 = 0.02
b3 = 2.1

tahmin = (z1 * w31) + (z2 * w32) + b3
print("Tahmin:", tahmin)

# MSE loss hesaplama
hedef = 2
mse = (hedef - tahmin) ** 2 
mse = mse / 1 # burada toplam hedef sayısına bölünür
print("MSE:", mse)
### Çıktı

```python
z1: 0.5 
z2: 1.8
Tahmin: 1.536
MSE: 0.215296
