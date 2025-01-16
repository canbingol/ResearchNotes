
Doğal Dil İşleme (NLP) alanında dil modelleri, herhangi bir dilde cümlelerin veya kelimelerin hangi sırayla geldiğini tahmin etmek için kullanılır. Örneğin, bir cümlede “Ben bir elma…” diye başladığımızda, dil modeli bundan sonraki tokenlerin ne olabileceğini olasılıksal olarak tahmin etmeye çalışır.

Bu yazıda, dil modelinin en basit örneklerinden biri olan bigram modeli ile karakter tabanlı bir dil modeli geliştireceğiz. Bigram modeli, yalnızca bir öncekikelimeye bakarak “bir sonraki” kelimeyi tahmin etmeye çalışır. Yani, adından da anlaşılacağı gibi, her seferinde iki kelime arasındaki ilişki (bigram) dikkate alınır.

Bu yazıda paylaşacaklarım, [Andrej Karpathy](https://www.youtube.com/@AndrejKarpathy)'nin YouTube’da yayınladığı “Neural Networks: Zero to Hero” serisinden öğrendiklerime dayanmaktadır. Derin öğrenme ve yapay zeka konularında daha derinlemesine bilgi edinmek isteyenler için bu seriyi mutlaka izlemenizi tavsiye ederim.


# 1. Bigram Modeli Nedir?
Bigram modeli, bir metindeki tokenlerin sırasını inceleyerek, her token için bir önceki tokene bağlı bir olasılık belirler. Bunu formül olarak şu şekilde gösterebiliriz:

![formula](https://github.com/canbingol/ResearchNotes/blob/main/images/formula.jpg)

Metnin bütününü ifade eden bir dizi kelimenin (örneğin, “Merhaba dünya nasılsınız”) olasılığı, birbirini izleyen bu ikililerin (bigramların) olasılıklarının çarpımıyla hesaplanır. Model, “Merhaba”dan sonra en sık “dünya” geliyorsa, bu ikilinin olasılığı yüksek olarak kaydedilir.

Kullanılan temel varsayım ise, bir kelimenin yalnızca hemen önceki kelimeye bağlı olmasıdır. Bu, doğal dilin çok daha karmaşık ilişkileri olduğunu göz ardı etse de, teorik olarak dil modelleme mantığını kavramak için iyi bir başlangıçtır.

# 2. Kodlama
## 2.1 importlar ve veriyi yükleme

```python
# !pip install torch
import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings # uyarıları kapatmak için 
warnings.filterwarnings("ignore")
```
PyTorch, TensorFlow ile birlikte derin öğrenme alanında en çok kullanılan kütüphanelerden biridir. Veri işleme, veri üzerinde değişiklik yapma ve derin öğrenme modelleri geliştirme konularında oldukça kullanışlıdır. Özellikle yeni başlayanlar için anlaşılır ve esnek bir yapı sunar.

Derin öğrenmeye PyTorch ile başlamak isteyenler için harika bir eğitim öneriyorum:

[pytroch eğitimi](https://youtu.be/V_xro1bcAuA?si=m7aVxGE-UtzAjskh)

## 2.2 Veriye İlk Bakış:
Veri seti olarak küçük bir Shakespeare veriseti kullanacağız.

```python
with open("more.txt", "r", encoding="utf-8") as f:
    text = f.read()
text[:100]
```
çıktı :
```
"\nThe top in a world by susphoring grace.\n\nLUCIO:\nWe muse hath resistes him so sovere: son't his othe"
```

Dil modelimiz karakter tabanlı olacağı için, modeli eğitmeden önce metindeki tüm benzersiz karakterleri belirlememiz gerekir. Bu sayede modelin hangi karakterlerle çalışacağını ve toplamda kaç farklı karakter üzerinde işlem yapacağını netleştirmiş oluruz. Bu işlem sonucunda elde edilen karakterler, modelin öğrenme sürecinde temel yapı taşlarını oluşturur. Ayrıca, bu karakterlerin sayısı modelin sözlük boyutunu belirler


```python
chars = sorted(set(text))
vocab_size = len(chars)
print("".join(chars))
print(f'chars len is {len(chars)}')
```

Metindeki tüm benzersiz karakterleri belirlemek için, önce tekrar edenleri ayıklayıp ardından alfabetik olarak sıralıyoruz. Bu sayede modelin çalışacağı karakter kümesi düzenli ve tekrarsız bir şekilde hazırlanmış oluyor.

çıktı:
```
 !',-.:;?ABCDEFGHIKLMNOPQRSTUVWYabcdefghijklmnopqrstuvwxyz
chars len is 59
```

# 2.3 Tokenizer:
Tokenizer, modele vereceğimiz verileri modelin anlayabileceği bir formata dönüştüren bir araçtır. Metin verisini alır ve bunu modelin işleyebileceği sayısal bir formata çevirir. Bu sayede model, kelimeler veya karakterler arasındaki ilişkileri daha kolay öğrenebilir.

Gelişmiş modellerde BPE (Byte Pair Encoding) veya SentencePiece gibi karmaşık tokenizer yöntemleri kullanılır. Ancak biz daha basit bir model eğiteceğimiz için, aynı şekilde basit bir tokenizer tercih edeceğiz. Bu amaçla Python’daki sözlük yapılarından faydalanacağız. Böylece karakterleri kolayca sayılara, sayıları da tekrar karakterlere çevirebileceğiz.
```
s2i = {s:i for i,s in enumerate(chars)}
i2s = {i:s for s,i in s2i.items()}
encode = lambda s: [s2i[c] for c in s]
decode = lambda l: "".join([i2s[i] for i in l])

print(encode("merhaba muhterem"))
ids = encode("merhabalar muhterem")
print(decode(ids)) 
```

- s2i (String to Integer): Her karaktere benzersiz bir sayı atar. Böylece karakterler sayılara dönüştürülebilir.
- i2s (Integer to String): Sayıları tekrar karakterlere çevirir. Böylece modelin çıktısı okunabilir hale gelir.
- encode: Verilen metni karakterlerden sayılara çevirir.
- decode: Sayılardan oluşan listeyi tekrar metne dönüştürür.

çıktı:
```
[45, 37, 50, 40, 33, 34, 33, 1, 45, 53, 40, 52, 37, 50, 37, 45]
merhabalar muhterem
```

örneğin burada 45 değeri “m” harfine karşılık gelmeli
```
m_ids = encode("m")
print(m_ids)
m = decode(m_ids)
print(m)
```
çıktı:

```
[45]
m
```

Şimdi, veri setimizin tamamını bir sonraki adım olan Dataloader için uygun hale getirelim. Bunun için tüm veriyi tokenlara dönüştüreceğiz. Ayrıca, oluşturduğumuz bu token listesini bir PyTorch tensörüne çevirerek modelin üzerinde kolayca işlem yapabileceği bir formata sokacağız. Bu sayede veriyi modelimize rahatlıkla aktarabilir ve verimli bir şekilde eğitim gerçekleştirebiliriz.

```
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:100])
```

çıktı:
```
torch.Size([10001]) torch.int64
tensor([ 0, 28, 40, 37,  1, 52, 47, 48,  1, 41, 46,  1, 33,  1, 55, 47, 50, 44,
        36,  1, 34, 57,  1, 51, 53, 51, 48, 40, 47, 50, 41, 46, 39,  1, 39, 50,
        33, 35, 37,  6,  0,  0, 20, 29, 12, 18, 23,  7,  0, 31, 37,  1, 45, 53,
        51, 37,  1, 40, 33, 52, 40,  1, 50, 37, 51, 41, 51, 52, 37, 51,  1, 40,
        41, 45,  1, 51, 47,  1, 51, 47, 54, 37, 50, 37,  7,  1, 51, 47, 46,  3,
        52,  1, 40, 41, 51,  1, 47, 52, 40, 37])
```

Modelin daha iyi öğrenebilmesi ve öğrendiklerini değerlendirebilmek için veri setimizi ikiye ayıracağız: eğitim (training) ve doğrulama (validation) setleri.

- Eğitim Seti: Modelin öğrenmesi için kullanılır. Bu verilerle model, karakterler arasındaki ilişkileri keşfeder.
- Doğrulama Seti: Modelin öğrendiklerini değerlendirmek için kullanılır. Eğitim sırasında modelin genelleme yeteneğini test etmeye yarar.
Bu ayrım, modelin aşırı ezberlemesini önler ve daha sağlam sonuçlar üretmesini sağlar.

```
n = int(len(data) * 0.9) # veri setinin %90'ını eğitim seti olarak alacağız
train_ds = data[:n]
val_ds = data[n:]
```

## 2.4 Dataloader:
İlk olarak block size kavramını anlamak önemli. Modelimizi, belirlediğimiz block size kadar karakteri tek seferde işleyebilecek şekilde ayarlayacağız.

Block size, modelin aynı anda görebileceği en uzun karakter dizisidir. Bu sınır, modelin hem ne kadar bilgi işleyeceğini hem de hesaplama yükünü doğrudan etkiler. Daha küçük bir block size, daha az hesaplama gerektirirken; daha büyük bir block size, modelin daha fazla bağlamı öğrenmesini sağlar.

Bu ayarlama, hesaplamaları optimize etmek ve eğitim sürecini daha verimli hale getirmek için yapılır.

```
block_size = 8
train_ds[:block_size+1]
```
çıktı:
```
tensor([ 0, 28, 40, 37,  1, 52, 47, 48,  1])

```

Modelin ön eğitimi sırasında, amacı bir sonraki tokeni (bizim durumumuzda bir sonraki karakteri) tahmin etmek olacak. Yani model, verilen bir karakter dizisini okuyacak ve sıradaki karakterin ne olacağını öğrenmeye çalışacak.

Bu yaklaşım, modelin karakterler arasındaki ilişkileri ve dilin doğal akışını anlamasını sağlar. Model, sık kullanılan karakter dizilimlerini öğrenerek anlamlı ve tutarlı yeni metinler üretebilir.

Bu yöntemle model, dilin kurallarını ezberlemek yerine, dilin yapısını öğrenir ve daha doğru tahminler yapar.

Ufak bir örnek

```
x = train_ds[:block_size]
y = train_ds[1:block_size+1]

for i in range(block_size):
    context = x[:i+1]
    target = y[i]
    print(f'when  input is {context} the target: {target}')
```

çıktı:
```
when  input is tensor([0]) the target: 28
when  input is tensor([ 0, 28]) the target: 40
when  input is tensor([ 0, 28, 40]) the target: 37
when  input is tensor([ 0, 28, 40, 37]) the target: 1
when  input is tensor([ 0, 28, 40, 37,  1]) the target: 52
when  input is tensor([ 0, 28, 40, 37,  1, 52]) the target: 47
when  input is tensor([ 0, 28, 40, 37,  1, 52, 47]) the target: 48
when  input is tensor([ 0, 28, 40, 37,  1, 52, 47, 48]) the target: 1
```

Burada görüldüğü gibi, her adımda modele hedef (target) olarak bir sonraki karakteri veriyoruz. Model, ilk başta küçük bir girdiyle başlıyor ve tahmin ettiği karakteri girdiye ekleyerek ilerliyor. Bu süreç, belirlediğimiz block size değerine ulaşana kadar devam ediyor.

Bu yöntem sayesinde model, önceki karakterlerden yola çıkarak sıradaki karakteri tahmin etmeyi öğrenir. Böylece karakterler arasındaki bağı ve dilin doğal akışını daha iyi kavrayarak tutarlı metinler üretir.

```
torch.manual_seed(1337)
context_size = 8 # Tahmin için maksimum bağlam uzunluğu
batch_size = 4 # İleri yayılımda birbirinden bağımsız şekilde paralel olarak işlenecek dizi/cümle sayısı

def get_batch(split):
    data = train_ds if split == 'train' else val_ds
    ix = torch.randint(len(data) - block_size, (batch_size,)) # 0 ile (len(data) - block_size) arasında  batch_size kadar random sayı üretme   
    x = torch.stack([data[i:i+block_size] for i in ix] ) # torch.stack her bir diziyi (örn [ 0, 28, 40]) alıp hepsini birleştiriyor 1x3 -> 4x3
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x,y
```

Bu kısımda, modelin eğitimi için eğitim (train) veya doğrulama (validation) verisinden rastgele örnekler seçerek batch’ler oluşturuyoruz.

- Batch size, modele aynı anda verilecek örnek sayısını belirler. Birden fazla örneği aynı anda işleyerek eğitim sürecini hızlandırır ve hesaplama verimliliğini artırır.
- ix, veri kümesinden rastgele başlangıç noktaları seçerek her batch için farklı ve çeşitlendirilmiş örnekler oluşturur.
- x, seçilen bu başlangıç noktalarından itibaren belirli uzunlukta (block size) karakter dizilerini giriş verisi olarak alır.
- y, her girdinin bir sonraki karakterini hedef olarak belirler. Böylece model, hangi karakterden sonra hangi karakterin geldiğini öğrenir.

![model_training](https://github.com/canbingol/ResearchNotes/blob/main/images/model_trainig.jpg)

Görselde de gösterildiği gibi, model her adımda yalnızca mevcut tokeni (bizim durumumuzda karakter) ve ondan önce gelenleri görebilir. Model, kendinden sonraki tokenleri göremez. Bu kısıtlama, modelin yalnızca mevcut bağlamı kullanarak bir sonraki tokeni tahmin etmesini sağlar.Biz de verimizi bu mantığa uygun şekilde hazırlayacağız

Bu yaklaşım, modelin doğru bir şekilde öğrenmesi için önemlidir çünkü model gelecekteki bilgileri görmeden tahmin yapmayı öğrenir. Bu şekilde dilin doğal akışını ve karakterler arasındaki ilişkileri daha gerçekçi bir şekilde kavrayabilir.

## 2.5 Model Yapısı:

```
class BigramLanguageModel(nn.Module):  # İkili Dil Modeli sınıfı (Bigram)

    def __init__(self, vocab_size):
        super().__init__()
        # Her bir token için gömme (embedding) tablosu oluşturuyoruz
        self.token_embd_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # Girdi indeksi için gömme tablosundan sonuçları alıyoruz
        logits = self.token_embd_table(idx)  # (B,T,C) -> Batch, Time, Classes
        if targets is None:
            loss = None  # Eğer hedef yoksa kayıp hesaplanmaz. Burası tahmin sırasında loss hesaplanamayacağı için hata almamak adına eklendi
        else:
            # Logits ve target boyutlarını ayarlıyoruz
            B, T, C = logits.shape
            logits = logits.view(B * T, C)  # (B*T,C) -> Her token için ayrı olasılık
            targets = targets.view(B * T)  # (B*T,) -> Hedeflerin düzleştirilmesi
            loss = F.cross_entropy(logits, targets)  # Çapraz entropi kaybını hesaplıyoruz
        return logits, loss  # Her zaman logits (tahminler) ve kayıp döner

    def generate(self, idx, max_new_tokens):
        # idx -> (B,T) -> Şu anki bağlamda token indeksleri
        for _ in range(max_new_tokens):
            # Tahmin yapıyoruz
            logits, loss = self(idx)
            # Sadece son zaman adımını alıyoruz
            logits = logits[:, -1, :]  # (B,C) -> Sadece son tokenin olasılıkları
            # Softmax uygulayarak olasılık dağılımını elde ediyoruz
            probs = F.softmax(logits, dim=-1)
            # Olasılık dağılımından bir token örnekliyoruz
            idx_next = torch.multinomial(probs, num_samples=1)
            # Yeni tokeni mevcut diziye ekliyoruz
            idx = torch.cat((idx, idx_next), dim=1)  # (B,T+1) -> Yeni token eklenir
        return idx  # Üretilen diziyi döndürür
```

Bu sınıf, Bigram Language Model (İkili Dil Modeli)’ni temsil eder. Modelin amacı, verilen bir karakter dizisine bakarak bir sonraki karakteri tahmin etmektir. Temel işleyişi şu şekildedir:

**1. Modelin Kurulumu (__init__ Metodu):**

- Modelin temel yapı taşı olan embedding (gömme) tablosu oluşturulur.
- Bu tablo, her karaktere özgü bir vektör temsili atar.
- Bu sayede model, karakterler arasındaki ilişkileri öğrenerek tahmin yapabilir.

**2. Tahmin ve Kayıp Hesaplama (forward Metodu):**

- Modelin tahmin üretme ve varsa hata (kayıp) hesaplama sürecidir.
- Girdi olarak verilen karakterler, embedding tablosundan geçirilerek tahminler (logits) üretilir.
- Eğer doğru cevaplar (targets) verilmişse, modelin tahminleriyle gerçek değerler arasındaki fark Cross Entropy Loss kullanılarak hesaplanır.

**Çıktıların Boyutları:**

- B (Batch Size): Aynı anda işlenen veri örneklerinin sayısı.
- T (Time Steps / Sequence Length): Girdi dizisinin uzunluğu (block size).
- C (Channels / Vocab Size): Tahmin edilen karakterlerin olasılık dağılımı, yani tüm karakter kümesi.
- logits (B, T, C): Modelin tahmin ettiği değerler.
- loss: Tahmin edilen değerlerle gerçek değerler arasındaki fark.

**3. Metin Üretme (generate Metodu):**

- Verilen bir başlangıç dizisinden başlayarak yeni karakterler üretir.
- Her adımda bir sonraki karakteri tahmin eder ve mevcut dizinin sonuna ekler.
- Bu işlem, belirlenen max_new_tokens sayısına ulaşana kadar devam eder.

**Nasıl Çalışır?**

- Model, mevcut diziyi işler ve son karakter üzerinden tahmin yapar.
- Tahmin edilen karakter, diziye eklenir.
- Bu süreç tekrarlanarak yeni bir metin oluşturulur.

## 2.6 Çıkarım
Modelimizi oluşturup ilk çıkarımlarımızı yapalım

```
m = BigramLanguageModel(vocab_size)
logits, loss = m(xb,yb)
print(logits.shape)
print(loss)
print(decode(m.generate(torch.zeros((1,1), dtype=torch.long), max_new_tokens=100)[0].tolist()))
```

çıktı:

```
torch.Size([32, 59])
tensor(4.4102, grad_fn=<NllLossBackward0>)

Dxi-HuCP
qmyfCpLath
S;merqmyLh
j?U?Yki,iU;oFtK!xFmEprRAv uK.Nsbnqq!BMI;ddiViPqeRk
S-dqB:qzMTy-gKAsqL
```

Modelin ağırlıkları ilk başlatıldığında rastgele değerlerle ayarlanır. Bu nedenle, model eğitimin başında dili henüz öğrenmediği için ürettiği metinler anlamsız ve rastgele olur.

## 2.7 Eğitim
Optimizer, modelin hata payını azaltmak için ağırlıkları güncelleyen algoritmadır. AdamW, momentum ve ağırlık çürümesini (weight decay) birleştirerek daha dengeli ve hızlı öğrenme sağlar.
```
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
```

Eğitim döngüsü

```
batc_size = 32
for steps in range(10000):
    # sample batch of data
    xb, yb = get_batch("train")
    # evalulate the loss 
    logits , loss = m(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())
```

Bu döngü, modelin eğitilmesini sağlayan temel süreçtir. Adım adım nasıl işlediğine bakalım:

**1. Veri Hazırlığı:**
- Her adımda get_batch("train") fonksiyonu ile eğitim verisinden rastgele 32 örnek alıyoruz (batch size = 32).
- Bu küçük veri grupları, modelin daha hızlı ve dengeli öğrenmesine yardımcı olur.

**2. Tahmin ve Kayıp Hesaplama:**

- Model bu verilerle tahmin yapıyor ve yaptığı tahminlerle gerçek değerleri karşılaştırarak kayıp (loss) değerini hesaplıyor.
- Bu kayıp değeri, modelin ne kadar hata yaptığını gösteriyor.

**3. Geri Yayılım (Backpropagation):**

- loss.backward() komutuyla modelin hatalarının nereden kaynaklandığını bulup öğrenmesi sağlanıyor.
**4. Ağırlıkların Güncellenmesi:**

- optimizer.step() ile modelin ağırlıkları hataları azaltacak şekilde güncelleniyor.
- optimizer.zero_grad() ise önceki adımlardan kalan hataları sıfırlıyor, böylece her adımda temiz bir başlangıç yapılıyor.
- 
Bu süreç 10.000 kez tekrarlanarak her adımda modelin biraz daha iyi tahminler yapmayı öğrenmesi hedeflaniyor

eğitim sonrasi modelden 2500 karakter uzunluğunda çıkarım
```
print(decode(m.generate(torch.zeros((1,1), dtype=torch.long), max_new_tokens=100)[0].tolist()))
```

sonuç:

```
Haig OUnowriencouss t t rd knd y
Haray d d-h omn'sun I t l ghre: f Homy'thart sse sho IRDdise mpblyL:

BzDRI way byofoday l'O hacoad, har hen ss I s.
Dd cofear, my shatlier s pan:
RHE:APYU
UGYO:
hapit
```

Beklediğiniz gibi bir çıktı alamadık, değil mi? Bunun temel nedeni, kullandığımız Bigram Modelinin yalnızca iki karakter arasındaki ilişkiye odaklanmasıdır. Bu sınırlı bakış açısı, modelin daha geniş bağlamları anlayamamasına ve dolayısıyla daha anlamlı metinler üretememesine neden olur. Bir sonraki yazımda, Transformer mimarisi ile nasıl daha tutarlı ve anlamlı cümleler üretebileceğimizi detaylı bir şekilde anlatacağım.

Bu projede, karakter tabanlı basit bir Bigram Dil Modeli geliştirdik ve eğittik. Model, verilen bir metindeki karakterler arasındaki geçişleri öğrenerek bir sonraki karakteri tahmin etmeye çalıştı. Öncelikle metindeki benzersiz karakterleri belirleyip sayısal değerlere dönüştürdük, ardından bu veriyi eğitim ve doğrulama setlerine ayırdık. Modelin karakterler arasındaki ilişkileri öğrenebilmesi için PyTorch’un embedding katmanını kullanarak karakterleri vektör temsillerine çevirdik. Eğitim sürecinde, modelin hatalarını azaltmak için AdamW optimizasyon algoritmasını kullandık ve düzenli aralıklarla modelin performansını izledik.

Aşağıda, tüm bu adımları bir araya getiren kapsamlı kodu görebilirsiniz. Bu kodda ekstra olarak modeli mevcut donanıma uygun şekilde (CPU veya GPU) eğitmek için gerekli cihaz seçimi yapılmıştır. Bu sayede model, varsa GPU üzerinde daha hızlı ve verimli bir şekilde eğitilebiliyor.

```
import torch
import torch.nn as nn
from torch.nn import functional as F

# Hiperparametreler
batch_size = 32  # Aynı anda işlenecek bağımsız dizi sayısı
block_size = 8   # Tahmin için maksimum bağlam uzunluğu
max_iters = 10000  # Eğitim döngüsü adım sayısı
eval_interval = 300  # Değerlendirme aralığı
learning_rate = 1e-2  # Öğrenme oranı
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Cihaz seçimi (GPU varsa kullanılır)
print("device is", device)
eval_iters = 200  # Değerlendirme için iterasyon sayısı

torch.manual_seed(1337)  # Rastgelelik için sabit seed değeri

# Veri yükleme
data_path = 'more.txt'
with open(data_path, 'r', encoding='utf-8') as f:
    text = f.read()

# Benzersiz karakterleri çıkarıyoruz ve karakter-sayı eşlemeleri yapıyoruz
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}  # Karakterden sayıya dönüşüm
itos = {i: ch for i, ch in enumerate(chars)}  # Sayıdan karaktere dönüşüm
encode = lambda s: [stoi[c] for c in s]  # Metni sayılara çevirir
decode = lambda l: ''.join([itos[i] for i in l])  # Sayıları metne çevirir

# Veri setini eğitim ve doğrulama olarak ikiye ayırıyoruz
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # %90 eğitim, %10 doğrulama
train_data = data[:n]
val_data = data[n:]

# Veri kümesinden rastgele batch oluşturuyoruz
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))  # Rastgele başlangıç indeksleri
    x = torch.stack([data[i:i + block_size] for i in ix])  # Girdi verisi
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])  # Hedef verisi (bir sonraki karakter)
    x, y = x.to(device), y.to(device)
    return x, y

# Kayıp (loss) hesaplama
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Basit Bigram Dil Modeli
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)  # Gömme (embedding) tablosu

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)  # Tahminler (B, T, C)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)  # Kayıp hesaplama
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]  # Son zaman adımına odaklanma
            probs = F.softmax(logits, dim=-1)  # Olasılık dağılımı
            idx_next = torch.multinomial(probs, num_samples=1)  # Rastgele örnekleme
            idx = torch.cat((idx, idx_next), dim=1)  # Yeni tahmini ekleme
        return idx

# Modeli başlatıyoruz
model = BigramLanguageModel(vocab_size)
m = model.to(device)

# Optimizasyon için AdamW kullanıyoruz
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Eğitim döngüsü
for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Modelden metin üretimi
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
```
