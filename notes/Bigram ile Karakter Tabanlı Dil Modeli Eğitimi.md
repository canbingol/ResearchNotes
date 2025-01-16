
Doğal Dil İşleme (NLP) alanında dil modelleri, herhangi bir dilde cümlelerin veya kelimelerin hangi sırayla geldiğini tahmin etmek için kullanılır. Örneğin, bir cümlede “Ben bir elma…” diye başladığımızda, dil modeli bundan sonraki tokenlerin ne olabileceğini olasılıksal olarak tahmin etmeye çalışır.

Bu yazıda, dil modelinin en basit örneklerinden biri olan bigram modeli ile karakter tabanlı bir dil modeli geliştireceğiz. Bigram modeli, yalnızca bir öncekikelimeye bakarak “bir sonraki” kelimeyi tahmin etmeye çalışır. Yani, adından da anlaşılacağı gibi, her seferinde iki kelime arasındaki ilişki (bigram) dikkate alınır.

Bu yazıda paylaşacaklarım, ![Andrej Karpathy](https://www.youtube.com/@AndrejKarpathy)’nin YouTube’da yayınladığı “Neural Networks: Zero to Hero” serisinden öğrendiklerime dayanmaktadır. Derin öğrenme ve yapay zeka konularında daha derinlemesine bilgi edinmek isteyenler için bu seriyi mutlaka izlemenizi tavsiye ederim.

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
















