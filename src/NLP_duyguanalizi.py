import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from bs4 import BeautifulSoup
import re
import nltk
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords

# Veriseti yükleme
df = pd.read_csv('./database/NLPlabeledData.tsv', delimiter="\t", quoting=3)

# Verilerin ilk 10 satırını yazdırma ve boyut kontrolü
print(df.head(10))
print("Toplam yorum sayısı:", len(df))
print("Yorum kolonundaki değerler:", len(df["review"]))

# Sentiment dağılımına bakma (Pozitif/Negatif)
print(df['sentiment'].value_counts())

# Stopwords indir
nltk.download('stopwords')

# İlk yorumu işleyelim
ilk_yorum = df.review[0]
print("İlk Yorum (Ham):", ilk_yorum)

# HTML etiketlerini temizleme
ilk_yorum = BeautifulSoup(ilk_yorum).get_text()
print("HTML etiketlerinden arındırılmış:", ilk_yorum)

# Noktalama işaretlerini ve sayıları temizleme
ilk_yorum = re.sub("[^a-zA-Z]", " ", ilk_yorum)
print("Temizlenmiş Yorum:", ilk_yorum)

# Yorumları küçük harfe çevirme ve listeye dönüştürme
ilk_yorum = ilk_yorum.lower().split()
print("Küçük harfe dönüştürülmüş ve kelimelere ayrılmış:", ilk_yorum)

# Stopwords'leri çıkarma
stop_words = set(stopwords.words('english'))
ilk_yorum = [x for x in ilk_yorum if x not in stop_words]
print("Stopwords'den arındırılmış:", ilk_yorum)

# Yorum işleme fonksiyonu
def process(review):
    review = BeautifulSoup(review).get_text()  # HTML taglerini kaldır
    review = re.sub("[^a-zA-Z]", " ", review)  # Noktalama işaretleri ve sayıları çıkar
    review = review.lower().split()  # Küçük harfe dönüştür ve split et
    swords = set(stopwords.words("english"))  # İngilizce stopwords'leri hazırla
    review = [x for x in review if x not in swords]  # Stopwords'leri çıkar
    return " ".join(review)

# Tüm yorumları işleme
df_yeni = []
for r in range(len(df["review"])):
    if (r + 1) % 1000 == 0:
        print(f"{r + 1} yorum işlendi")
    df_yeni.append(process(df["review"][r]))

# İlk 10 temizlenmiş yorumu yazdır
for i in range(10):
    print(f"Temizlenmiş Yorum {i+1}: {df_yeni[i]}")

# Özellik ve etiketleri ayırma
x = df_yeni
y = np.array(df["sentiment"])

# Eğitim ve test setlerine ayırma
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.30, random_state=42)

# Bag of Words modelini oluşturma
vectorizer = CountVectorizer(max_features=50000)
x_train = vectorizer.fit_transform(x_train).toarray()

# RandomForestClassifier modeli eğitme
rf = RandomForestClassifier(n_estimators=100)
rf.fit(x_train, y_train)

# Test setini Bag of Words ile dönüştürme ve tahmin yapma
test_x = vectorizer.transform(x_test).toarray()
test_predict = rf.predict(test_x)

# Doğruluk skorunu hesaplama
dogruluk = roc_auc_score(y_test, test_predict)
print("Doğruluk Oranı:", dogruluk)

# Yeni yorum analizi
def analyze_review(new_review):
    cleaned_review = process(new_review)  # Yorum temizleme
    test_vector = vectorizer.transform([cleaned_review]).toarray()  # Vektör haline getirme
    prediction = rf.predict(test_vector)  # Tahmin yapma
    return "Pozitif yorum" if prediction == 1 else "Negatif yorum"

# Örnek yeni yorum analizi
new_review = "I absolutely loved this movie. The acting was fantastic and the story was so engaging!"
print("Yorum Analizi:", analyze_review(new_review))
