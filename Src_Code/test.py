####Pengambilan Data
import csv
import snscrape.modules.twitter as sntwitter
import pandas as pd
# Creating list to append tweet data to
tweets_list2 = []
# Using TwitterSearchScraper to scrape data and append tweets to list
for i,tweet in enumerate(sntwitter.TwitterSearchScraper('@IndiHome since:2021-10-1 until:2021-11-1').get_items()):
    if i>500000:
        break
    tweets_list2.append([tweet.date, tweet.user.username, tweet.content])
    
# Creating a dataframe from the tweets list above
tweets_df2 = pd.DataFrame(tweets_list2, columns=['Tanggal', 'User ID', 'Data Teks'])
tweets_df2.to_csv('Stemmer_2\FInal\data_crawling11.csv')
tweets_df2

#
# EDA
# Exploratory Data Analysis adalah proses yang memungkinkan analyst memahami isi data yang digunakan, mulai dari distribusi, frekuensi, korelasi dan lainnya.

#Dalam proses ini pemahaman konteks data juga diperhatikan karena akan menjawab masalah - masalah dasar.
# 1. Import Libraries
#Import library yang akan digunakan
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
# 2. Load Dataset
#Load dataset hasil Crawling dengan menggunakan tweepy sebelumnya
# Load Dataset
data1 = pd.read_csv('Data Final\Data_4_Prepro_rem\Indihome.csv', sep=',')

## Dataset info

#Menampilkan banyak data dan Dtype tiap kolomnya.
# Info
for i in [data1]:
    i.info()
    print()
# # Merge Info
# data = pd.concat([data1,data2,data3])
# data.info()
# Melihat banyak Tweet perhari
data['Tanggal'] = pd.to_datetime(data['Tanggal'])
tph = data['Data Teks'].groupby(data['Tanggal'].dt.date).count()
frek = tph.values
h_index = {6:'Minggu',0:'Senin',1:'Selasa',2:'Rabu',3:'Kamis',4:'Jumat',5:"Sabtu"}
hari = [x.weekday() for x in tph.index]
hari = [h_index[x] for x in hari] 
for i in range(len(hari)):
    hari[i] = str(tph.index[i]) + f'\n{hari[i]}'
# Plotting Line
plt.figure(figsize = (10,10))
sns.lineplot(range(len(frek)), frek)
for i, v in enumerate(frek.tolist()):
    if i == 0 or i==2 or i ==4 or i == len(tph.values)-2:
        plt.text(i-.25, v - 1000, str(v),fontsize=11)
    elif i == 1 or i == 3 or i==6 or i == len(tph.values)-1:
        plt.text(i-.25, v + 400, str(v),fontsize=11)
    else :
        plt.text(i+.07, v, str(v),fontsize=11)
plt.title('Banyak Tweet per Hari',fontsize=20)
plt.xticks(range(len(tph.values)), hari, rotation=45)
plt.xlabel('Tanggal',fontsize=16)
plt.ylabel('Frekuensi',fontsize=16)
plt.show()
# Melihat banyak Tweet perjam
tpj = []
for i in range(1,len(tph.index)) :
    if i != len(tph.index)-1 :
        tpj.append(data['Tanggal'][(data['Tanggal'] >= str(tph.index[i])) & (data['Tanggal']<str(tph.index[i+1]))])
    else :
        tpj.append(data['Tanggal'][data['Tanggal']>=str(tph.index[i])])
tpj = [x.groupby(x.dt.hour).count() for x in tpj]
# Ploting Line
fig, axes = plt.subplots(nrows=2, ncols=4,figsize=(20,10))
for i in range(len(tpj)):
    sns.lineplot(tpj[i].index.tolist(),tpj[i].values,ax=axes[i//4,i%4])
    axes[i//4,i%4].set_title(f'{hari[i+1]}')
    axes[i//4,i%4].set(xlabel = 'Jam', ylabel = 'Frekuensi')
    plt.tight_layout()
#fig.suptitle('Banyak Tweet per Jam',fontsize=24)
plt.show()


#
# Sampling
import numpy as np
import pandas as pd

from datetime import datetime
# Import data
# Load Dataset
data1 = pd.read_csv('Data Final\Data_1_Raw\Indihome.csv')
data2 = pd.read_csv('Data Final\Data_1_Raw\indihomecare.csv')
# Concating data
data = pd.concat([data1, data2])
data.info()
# data = data.drop_duplicates('Data Teks')
# data.info()
BANYAK_SAMPLE_DATA = 8000
sample = data.sample(n=BANYAK_SAMPLE_DATA, random_state=42)
sample.info()
sample.to_csv('Data Final\Data_2_sampel\Indihome_all.csv')

# = Analisis Sentimen =
import os
import csv
import pandas as pd
import numpy as np
import string
import re
import nltk
from sklearn import pipeline
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

#Visualisasi
import matplotlib.pyplot as plt
import seaborn as sns
# = Load Data =
# ------ Data Text --------
# csv Data name
Data_sampel = ('Data\Data_2_sampel\Indihome_all.csv')
Data_rem = ('Data\Data_2_S_Remove\Rem_indihome.csv') #remove duppilcate
Data_prepro = ('Data\Data_4_prepro\Pre_Indihome.csv')
# Data_prepro_2 = ('Data\Data_4_Prepro_rem\Indihome.csv') 
Data_lex = ('Data\Data_5_Lex_sen\Lex_Indihome.csv')
#data1 = data input
data1 = pd.read_csv(Data_sampel, sep=(','))
data1.head()

# Remove Duplicate
# #sort by name or number
# data1.sort_values('Data Teks_Stopword', ascending=False)

# jumlah data awal
index = data1.index
jumlah = len(index)

print('Jumlah data awal:',jumlah)

#jumlah data yang sama
data_sama = data1.duplicated(subset= "Data Teks").sum()
print('Jumlah data yang sama:',data_sama)

#remove duplicate data
remove_data = data1.drop_duplicates(subset = 'Data Teks', keep = 'first', inplace = True)
index = data1.index
jumlah = len(index)

print('Jumlah data sekarang:',jumlah)
data1.to_csv(Data_rem)
# = Pre-Processing =
# = CaseFolding =
# ------ Proses CaseFolding --------
#------ Clean Text --------
#casefolding
data1['Data Teks_CaseFolding'] = data1['Data Teks'].str.lower()

def clean_text(tweet_clean):
  tweet_clean = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",str(tweet_clean)).split()) #mengghilangkan data teks (@, # dan link)
  tweet_clean = re.sub(r'\b\w{1,2}\b', '', tweet_clean) #menghilangkan 2 kata
  tweet_clean = re.sub('\s+',' ',tweet_clean)
  tweet_clean = re.sub(r"\d+", "", tweet_clean)
  #tweet_clean = removeDupWithOrder(tweet_clean) #mengghilangkan kata duplikat
  tweet_clean = tweet_clean.translate(str.maketrans("","",string.punctuation))# ini untuk apa 
  return tweet_clean
data1['Data Teks_CaseFolding'] = data1['Data Teks_CaseFolding'].apply(lambda x: clean_text(x))
data1.head()
# = Normalize =
alay_dict = pd.read_csv(r'Dict_Used\kamus_baku.csv', encoding='latin-1', header=None)
alay_dict = alay_dict.rename(columns={0: 'original', 
                                      1: 'replacement'})
alay_dict_map = dict(zip(alay_dict['original'], alay_dict['replacement']))
def normalize_alay(text):
    return ' '.join([alay_dict_map[word] if word in alay_dict_map else word for word in text.split(' ')])

data1['Data Teks_clean_alay'] = data1['Data Teks_CaseFolding'].apply(normalize_alay) 
data1.drop_duplicates(keep=False,inplace=True)
# = Tokenize =
# ------ Proses Tokenizing --------
def token(tweet_clean):
  tweet_clean = re.split('\W+', tweet_clean)
  return tweet_clean

data1['Data Teks_Token'] = data1['Data Teks_clean_alay'].apply(lambda x: token(x))
data1.head()
# = Stopwords =
nltk.download('stopwords')
from nltk.corpus import stopwords
# ----------------------- get stopword from NLTK stopword -------------------------------
# get stopword indonesia
list_stopwords = stopwords.words('indonesian')
print(len(list_stopwords))

# ---------------------------- manualy add stopword  ------------------------------------
# append additional stopword
list_stopwords.extend(['b',"yg", "dg", "rt", "dgn", "ny", "d", 'klo', 
                       'kalo', 'amp', 'biar', 'bikin', 'bilang', 
                       'gak', 'ga', 'krn', 'nya', 'nih', 'sih', 
                       'si', 'tau', 'tdk', 'tuh', 'utk', 'ya', 
                       'jd', 'jgn', 'sdh', 'aja', 'n', 't', 
                       'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt',
                       '&amp', 'yah', 'kak'])
len(list_stopwords)
# ----------------------- add stopword from txt file ------------------------------------
# read txt stopword using pandas
txt_stopword = pd.read_csv("Dict_Used\stopwordsID.csv", names= ["stopwords"], header = None)

# convert stopword string to list & append additional stopword
list_stopwords.extend(txt_stopword["stopwords"][0].split(' '))
len(list_stopwords)
# ---------------------------------------------------------------------------------------

# convert list to dictionary
list_stopwords = set(list_stopwords)


#remove stopword pada list token
def stopwords_removal(words):
    return [word for word in words if word not in list_stopwords]

data1['Data Teks_Stopword'] = data1['Data Teks_Token'].apply(lambda x: stopwords_removal(x)) 


print(data1['Data Teks_Stopword'])

# = Stemming =

stop_removal = data1['Data Teks_Stopword']

def fit_stopwords(tweet_clean):
    tweet_clean =np.array(tweet_clean)
    tweet_clean =' '.join(tweet_clean)
#    tweet_clean = tweet_clean.tosring()
#    tweet_clean = str(tweet_clean)
    return tweet_clean

data1['Data Teks_Stemming'] = data1['Data Teks_Stopword'].apply(lambda x: fit_stopwords(x))
data1.head(5)
# +Save Preprocessing
data1.dropna()
# data1['Data Teks_Stemming']=data1['Data Teks_Stemming'].str.replace(' ', '')
data1.drop_duplicates(subset='Data Teks_Stemming',
                      keep= 'first',inplace=True)
data1.to_csv(Data_prepro, index=False, header=True)
# data2 = pd.read_csv(Data_prepro, sep=',').dropna(subset=['Data Teks_Stemming'])
data1.head(5)
# jumlah data setelah prepro
index = data1.index
jumlah = len(index)

print('Jumlah data prepo:',jumlah)
#=============================================================================================================
# +++++++++++++VISUALISASI++++++++++++++++++++
# Menghitung 10 Besar Kata yang sering muncul (dari atas)
top_N = 20
a = data1['Data Teks_Stemming'].str.cat(sep=' ')
words = nltk.tokenize.word_tokenize(a)
word_dist = nltk.FreqDist(words)
rslt = pd.DataFrame(word_dist.most_common(top_N),
                    columns=['Kata', 'Banyak'])
print(rslt)
# Plotting Barplot
plt.figure(figsize = (10,10))
sns.barplot(x = rslt['Kata'],y = rslt['Banyak'])
for i, v in enumerate(rslt['Banyak'].tolist()):
    plt.text(i-len(str(v))/10-.05, v + 50, str(v),fontsize=10)
plt.title('Kata Dengan Bobot Terbesar',fontsize=20)
plt.xticks(rotation=90)
plt.xlabel('Kata',fontsize=16)
plt.ylabel('Frekuensi',fontsize=16)
plt.show()

# Source CODE
https://github.com/commitunuja/analisis-sentimen-naive-bayes-tf-idf/blob/master/.ipynb_checkpoints/Untitled-checkpoint.ipynb
# = Lexicon =
#sentimen Lexicon
#Determine sentiment polarity of tweets using indonesia sentiment lexicon (source : https://github.com/fajri91/InSet)

# Loads lexicon positive and negative data
lexicon_positive = dict()
import csv
with open('Dict_Used\lexicon_positive.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        lexicon_positive[row[0]] = int(row[1])

lexicon_negative = dict()
import csv
with open('Dict_Used\lexicon_negative.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        lexicon_negative[row[0]] = int(row[1])
        
# Function to determine sentiment polarity of tweets        
def sentiment_analysis_lexicon_indonesia(text):
    #for word in text:
    score = 0
    for word in text:
        if (word in lexicon_positive):
            score = score + lexicon_positive[word]
    for word in text:
        if (word in lexicon_negative):
            score = score + lexicon_negative[word]
    polarity=''
    if (score > 0):
        polarity = 'positif'
    elif (score < 0):
        polarity = 'negatif'
    else:
        polarity = 'netral'
    return score, polarity

results = data1['Data Teks_Token'].apply(sentiment_analysis_lexicon_indonesia)
results = list(zip(*results))
data1['jumlah_polarity'] = results[0]
data1['sentimen'] = results[1]
print(data1['sentimen'].value_counts())
# +Save Leksikon
data1.dropna()
data1.drop_duplicates(subset='Data Teks_Stemming',
                      keep= 'first',inplace=True)
data1.to_csv(Data_lex, index=False, header=True)
# data2 = pd.read_csv(Data_lex,usecols=['Data Teks_Stemming', 'jumlah_polarity', 'sentimen']).dropna()
# data2.drop_duplicates(subset=['Data Teks_Stemming'], keep= False, inplace= True)
data1.head(5)
# jumlah data setelah leksikon
index = data1.index
jumlah = len(index)

print('Jumlah data lex:',jumlah)
# = Visualiasi =
### Perbandingan Sentimen
# Plotting Pie
def pct_pie(pct, allvals):
    absolute = int(pct/100.*np.sum(allvals))
    return "{:.1f}%\n{:d}".format(pct, absolute)

sentimen = data1['Data Teks_Stopword'].groupby(data1['sentimen']).count().values

plt.figure(figsize = (8,8))
plt.pie(sentimen, explode=(0,0,0.1), labels=['Negatif', 'Netral','Positif'], shadow=True,
        autopct=lambda pct: pct_pie(pct, sentimen),startangle=90)
plt.title('Perbandingan Sentiment',fontsize=18)
plt.axis('equal')
plt.legend(fontsize=11)
plt.show()
# Melihat banyak Tweet perhari berdasarkan sentiment
data1['Tanggal'] = pd.to_datetime(data1['Tanggal'])
tph = data1['Data Teks_Stopword'].groupby([data1['Tanggal'].dt.date, data1['sentimen']]).count()
frek = tph.values

# To Data Frame
tanggal = [ i for i, j in tph.index.tolist() ]
senti = [ j for i, j in tph.index.tolist() ]
sent = pd.DataFrame({'Tanggal':tanggal,'sentiment':senti, 'Frekuensi':frek})
# Plotting line
plt.figure(figsize = (10,10))
sns.lineplot(x='Tanggal',y='Frekuensi',hue='sentiment',data=sent)
plt.title('Banyak Tweet per Hari',fontsize=20)
plt.xticks(rotation=45)
plt.xlabel('Tanggal',fontsize=16)
plt.ylabel('Frekuensi',fontsize=16)
plt.legend(['Negatif','Netral','Positif'])
plt.show()

#
# Pemodelan
#import lib
import csv
import pandas as pd
import numpy as np
import string
import re
import nltk

#
import matplotlib.pyplot as plt
import seaborn as sns
# Analisi Sentimen
### Source CODE dari
https://github.com/commitunuja/analisis-sentimen-naive-bayes-tf-idf/blob/master/.ipynb_checkpoints/Untitled-checkpoint.ipynb
%run S_4_Analisis_sentimen.ipynb
# Naive Bayes Classifier in Python KAGGLE
https://www.kaggle.com/prashant111/naive-bayes-classifier-in-python
datax = pd.read_csv('Data\Data_5_Lex_sen\Lex_Indihome.csv')
del datax['Unnamed: 0.1']
del datax['Unnamed: 0']
# data.rename( columns={'Unnamed: 0':'Indeks'}, inplace=True )
data = datax.copy()

### Prediction TF-IDF
# convert list formated string to list
import ast
import numpy as np
index = 0

def convert_text_list(texts):
    texts = ast.literal_eval(texts)
    return [text for text in texts]

data['Data Teks_List'] = data['Data Teks_Token'].apply(convert_text_list)


print(data['Data Teks_List'][index])

print('\ntype : ', type(data['Data Teks_List']))
def calc_TF(document):
    # Counts the number of times the word appears in review
    TF_dict = {}
    for term in document:
        if term in TF_dict:
            TF_dict[term] += 1
        else:
            TF_dict[term] = 1
    # Computes tf for each word
    for term in TF_dict:
        TF_dict[term] = TF_dict[term] / len(document)
    return TF_dict

data["TF_dict"] = data['Data Teks_List'].apply(calc_TF)

data["TF_dict"].head()
print('%20s' % "term", "\t", "TF\n")
for key in data["TF_dict"][index]:
    print('%20s' % key, "\t", data["TF_dict"][index][key])
def calc_DF(tfDict):
    count_DF = {}
    # Run through each document's tf dictionary and increment countDict's (term, doc) pair
    for document in tfDict:
        for term in document:
            if term in count_DF:
                count_DF[term] += 1
            else:
                count_DF[term] = 1
    return count_DF

DF = calc_DF(data["TF_dict"])
n_document = len(data)

def calc_IDF(__n_document, __DF):
    IDF_Dict = {}
    for term in __DF:
        IDF_Dict[term] = np.log(__n_document / (__DF[term] + 1))
    return IDF_Dict
  
#Stores the idf dictionary
IDF = calc_IDF(n_document, DF)
#calc TF-IDF
def calc_TF_IDF(TF):
    TF_IDF_Dict = {}
    #For each word in the review, we multiply its tf and its idf.
    for key in TF:
        TF_IDF_Dict[key] = TF[key] * IDF[key]
    return TF_IDF_Dict

#Stores the TF-IDF Series
data["TF-IDF_dict"] = data["TF_dict"].apply(calc_TF_IDF)
print('%20s' % "term", "\t", '%10s' % "TF", "\t", '%20s' % "TF-IDF\n")
for key in data["TF-IDF_dict"][index]:
    print('%20s' % key, "\t", data["TF_dict"][index][key] ,"\t" , data["TF-IDF_dict"][index][key])
# sort descending by value for DF dictionary 
sorted_DF = sorted(DF.items(), key=lambda kv: kv[1], reverse=True)[:50]

# Create a list of unique words from sorted dictionay `sorted_DF`
unique_term = [item[0] for item in sorted_DF]

def calc_TF_IDF_Vec(__TF_IDF_Dict):
    TF_IDF_vector = [0.0] * len(unique_term)

    # For each unique word, if it is in the review, store its TF-IDF value.
    for i, term in enumerate(unique_term):
        if term in __TF_IDF_Dict:
            TF_IDF_vector[i] = __TF_IDF_Dict[term]
    return TF_IDF_vector

data["TF_IDF_Vec"] = data["TF-IDF_dict"].apply(calc_TF_IDF_Vec)

print("print first row matrix TF_IDF_Vec Series\n")
print(data["TF_IDF_Vec"][0])

print("\nmatrix size : ", len(data["TF_IDF_Vec"][0]))
data.to_csv('Data\Data_8_Pemod\Mod_Indihome.csv')
data
# data
# NBC
https://github.com/rizkiamanullah/Analisis-Sentiment-Naive-Bayes/blob/main/rizkiamanullah-Analysis-Sentiment-NaiveB.ipynb

# ini penting (confusion matrix)
https://www.datacamp.com/community/tutorials/naive-bayes-scikit-learn
# copy dataframe
data_s = datax.copy()
data_s.head(5)
# menghapus dataframe netrel
data_s = data_s[data_s.sentimen != 'netral']
data_s.head(5)
# Conversi nilai
def convernilai(data):
    if data == 'positif':
        return 1
    elif data == 'negatif':
        return -1

data_s['label'] = data_s['sentimen'].apply(lambda x: convernilai(x))
data_s
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB

vectorizer = TfidfVectorizer(max_features=2500)
model_g = GaussianNB()

v_data= vectorizer.fit_transform(data_s['Data Teks_clean_alay'].values.astype('U'))
print(v_data)
v_data= vectorizer.fit_transform(data_s['Data Teks_clean_alay'].values.astype('U')).toarray()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(v_data, data_s['label'], test_size=0.2, random_state=0)
model_g.fit(X_train,y_train)

print('Banyak data train :',len(X_train))
print('Banyak data test  :',len(X_test))
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

y_preds = model_g.predict(X_test)

print(confusion_matrix(y_test,y_preds))
print(classification_report(y_test,y_preds))
print('nilai akurasinya adalah {:.2f}'.format((accuracy_score(y_test,y_preds)*100)))
# Pemodelan
import pandas as pd 
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,classification_report
#Data
data = pd.read_csv('Data\Data_5_Lex_sen\Lex_Indihome.csv', sep=(','))
data.head(5)
A=data['Data Teks_Stemming']
b=data['sentimen']

a_train, a_test, b_train, b_test = train_test_split(A,b,test_size = 0.20, random_state = 5)

print('Banyak data train :',len(a_train))
print('Banyak data test  :',len(a_test))

### Source CODE
# https://github.com/commitunuja/analisis-sentimen-naive-bayes-tf-idf/blob/master/klasifikasi_akurasi.ipynb
tvec_lxn=TfidfVectorizer()
clf1_lxn= KNeighborsClassifier()


#Pipeline
model_lxn = Pipeline([('vectorizer',tvec_lxn),
                 ('classifier',clf1_lxn)])


model_lxn.fit(a_test,b_test)
hasil_lxn = model_lxn.predict(a_test)
print(confusion_matrix(b_test, hasil_lxn))
print(classification_report(hasil_lxn, b_test))
print("Accuracy Score: {:.2f}".format((accuracy_score(hasil_lxn,b_test)*100)))
print("Precision Score: {:.2f}".format(precision_score(hasil_lxn, b_test, average='macro')))
print("Recall Score: {:.2f}".format(recall_score(hasil_lxn, b_test, average='macro')))


#
# Dimensi Kualitas
# import lib
import csv
import pandas as pd
import numpy as np
import string
import re
import nltk
# VISUALISASI
import matplotlib.pyplot as plt
import seaborn as sns
# input data
data = pd.read_csv('Data\Data_5_Lex_sen\Lex_Indihome.csv')

# mengubah tipe data
# data['Data Teks_Stopword'] = data["Data Teks_Stopword"].astype(str)
# menghilangkan data kosong
data['Data Teks_Stemming']=data['Data Teks_Stemming'].fillna("")
# Relevan
### Network Quality
datanq = data.copy()
labels = {'lambat': 'Network Quality', 
          'wifi': 'Network Quality', 
          'internet': 'Network Quality', 
          'rusak': 'Network Quality'} 

def matcher(k):
    x = (i for i in labels if i in k.split(' '))
    return ' | '.join(map(labels.get, x))

datanq['kata_kunci'] = datanq['Data Teks_Stemming'].map(matcher)
datanq = datanq.drop(datanq[datanq.kata_kunci == ''].index) #menghapus baris kosong
datanq.to_csv('Data\Data_7_Relev\IndiHome\Id_1_NQ.csv')
#### visualisasi
# Plotting Pie
def pct_pie(pct, allvals):
    absolute = int(pct/100.*np.sum(allvals))
    return "{:.1f}%\n{:d}".format(pct, absolute)

sentimen = datanq['Data Teks_Stopword'].groupby(datanq['sentimen']).count().values

plt.figure(figsize = (8,8))
plt.pie(sentimen, explode=(0,0,0.1), labels=['Negatif', 'Netral','Positif'], shadow=True,
        autopct=lambda pct: pct_pie(pct, sentimen),startangle=90)
plt.title('Perbandingan Sentiment',fontsize=18)
plt.axis('equal')
plt.legend(fontsize=11)
plt.show()
# Melihat banyak Tweet perhari berdasarkan sentiment
datanq['Tanggal'] = pd.to_datetime(datanq['Tanggal'])
tph = datanq['Data Teks_Stopword'].groupby([datanq['Tanggal'].dt.date, datanq['sentimen']]).count()
frek = tph.values

# To Data Frame
tanggal = [ i for i, j in tph.index.tolist() ]
senti = [ j for i, j in tph.index.tolist() ]
sent = pd.DataFrame({'Tanggal':tanggal,'sentiment':senti, 'Frekuensi':frek})
# Plotting line
plt.figure(figsize = (10,10))
sns.lineplot(x='Tanggal',y='Frekuensi',hue='sentiment',data=sent)
plt.title('Banyak Tweet per Hari',fontsize=20)
plt.xticks(rotation=45)
plt.xlabel('Tanggal',fontsize=16)
plt.ylabel('Frekuensi',fontsize=16)
plt.legend(['Negatif','Netral','Positif'])
plt.show()
### Customer Service
datacs = data.copy()
# labels = {'layanan' : 'Customer Service',
#           'nomor internet': 'Customer Service',
#           'nomor aktif': 'Customer Service', 
#           'mohon maaf': 'Customer Service', 
#           'dibantu respon': 'Customer Service',
#           'IndiHomeCare' : 'Customer Service'}
 
labels = {'IndiHomeCare' : 'Customer Service',
          'Telkomsel' : 'Customer Service'}

def matcher(k):
    x = (i for i in labels if i in k.split(' '))
    return ' | '.join(map(labels.get, x))

datacs['kata_kunci'] = datacs['User ID'].map(matcher)
# datacs['kata_kunci'] = datacs['Data Teks_Stemming'].map(matcher)
datacs = datacs.drop(datacs[datacs.kata_kunci == ''].index) #menghapus baris kosong
datacs.to_csv('Data\Data_7_Relev\IndiHome\Id_2_CS.csv')
#### visualisasi
# Plotting Pie
def pct_pie(pct, allvals):
    absolute = int(pct/100.*np.sum(allvals))
    return "{:.1f}%\n{:d}".format(pct, absolute)

sentimen = datacs['Data Teks_Stopword'].groupby(datacs['sentimen']).count().values

plt.figure(figsize = (8,8))
plt.pie(sentimen, explode=(0,0,0.1), labels=['Negatif', 'Netral','Positif'], shadow=True,
        autopct=lambda pct: pct_pie(pct, sentimen),startangle=90)
plt.title('Perbandingan Sentiment',fontsize=18)
plt.axis('equal')
plt.legend(fontsize=11)
plt.show()
# Melihat banyak Tweet perhari berdasarkan sentiment
datacs['Tanggal'] = pd.to_datetime(datacs['Tanggal'])
tph = datacs['Data Teks_Stopword'].groupby([datacs['Tanggal'].dt.date, datacs['sentimen']]).count()
frek = tph.values

# To Data Frame
tanggal = [ i for i, j in tph.index.tolist() ]
senti = [ j for i, j in tph.index.tolist() ]
sent = pd.DataFrame({'Tanggal':tanggal,'sentiment':senti, 'Frekuensi':frek})
# Plotting line
plt.figure(figsize = (10,10))
sns.lineplot(x='Tanggal',y='Frekuensi',hue='sentiment',data=sent)
plt.title('Banyak Tweet per Hari',fontsize=20)
plt.xticks(rotation=45)
plt.xlabel('Tanggal',fontsize=16)
plt.ylabel('Frekuensi',fontsize=16)
plt.legend(['Negatif','Netral','Positif'])
plt.show()
### Information Quality
dataiq = data.copy()
labels = {'semangat': 'Information Quality',
          'digital' : 'Information Quality',
          'muda' : 'Information Quality'} 

def matcher(k):
    x = (i for i in labels if i in k.split(' '))
    return ' | '.join(map(labels.get, x))

dataiq['kata_kunci'] = dataiq['Data Teks_Stemming'].map(matcher)

dataiq = dataiq.drop(dataiq[dataiq.kata_kunci == ''].index) #menghapus baris kosong
dataiq.to_csv('Data\Data_7_Relev\IndiHome\Id_3_IQ.csv')
#### visualisasi
# Plotting Pie
def pct_pie(pct, allvals):
    absolute = int(pct/100.*np.sum(allvals))
    return "{:.1f}%\n{:d}".format(pct, absolute)

sentimen = dataiq['Data Teks_Stopword'].groupby(dataiq['sentimen']).count().values

plt.figure(figsize = (8,8))
plt.pie(sentimen, explode=(0,0,0.1), labels=['Negatif', 'Netral','Positif'], shadow=True,
        autopct=lambda pct: pct_pie(pct, sentimen),startangle=90)
plt.title('Perbandingan Sentiment',fontsize=18)
plt.axis('equal')
plt.legend(fontsize=11)
plt.show()
# Melihat banyak Tweet perhari berdasarkan sentiment
dataiq['Tanggal'] = pd.to_datetime(dataiq['Tanggal'])
tph = dataiq['Data Teks_Stopword'].groupby([dataiq['Tanggal'].dt.date, dataiq['sentimen']]).count()
frek = tph.values

# To Data Frame
tanggal = [ i for i, j in tph.index.tolist() ]
senti = [ j for i, j in tph.index.tolist() ]
sent = pd.DataFrame({'Tanggal':tanggal,'sentiment':senti, 'Frekuensi':frek})
# Plotting line
plt.figure(figsize = (10,10))
sns.lineplot(x='Tanggal',y='Frekuensi',hue='sentiment',data=sent)
plt.title('Banyak Tweet per Hari',fontsize=20)
plt.xticks(rotation=45)
plt.xlabel('Tanggal',fontsize=16)
plt.ylabel('Frekuensi',fontsize=16)
plt.legend(['Negatif','Netral','Positif'])
plt.show()
### Security and Privacy
datasp = data.copy()
labels = {'via': 'Security dan Privacy',
          'pesan' : 'Security dan Privacy'} 

def matcher(k):
    x = (i for i in labels if i in k.split(' '))
    return ' | '.join(map(labels.get, x))

datasp['kata_kunci'] = datasp['Data Teks_Stemming'].map(matcher)
datasp = datasp.drop(datasp[datasp.kata_kunci == ''].index) #menghapus baris kosong
datasp.to_csv('Data\Data_7_Relev\IndiHome\Id_4_SP.csv')
#### visualisasi
# Plotting Pie
def pct_pie(pct, allvals):
    absolute = int(pct/100.*np.sum(allvals))
    return "{:.1f}%\n{:d}".format(pct, absolute)

sentimen = datasp['Data Teks_Stopword'].groupby(datasp['sentimen']).count().values

plt.figure(figsize = (8,8))
plt.pie(sentimen, explode=(0,0,0.1), labels=['Negatif', 'Netral','Positif'], shadow=True,
        autopct=lambda pct: pct_pie(pct, sentimen),startangle=90)
plt.title('Perbandingan Sentiment',fontsize=18)
plt.axis('equal')
plt.legend(fontsize=11)
plt.show()
# Melihat banyak Tweet perhari berdasarkan sentiment
datasp['Tanggal'] = pd.to_datetime(datasp['Tanggal'])
tph = datasp['Data Teks_Stopword'].groupby([datasp['Tanggal'].dt.date, datasp['sentimen']]).count()
frek = tph.values

# To Data Frame
tanggal = [ i for i, j in tph.index.tolist() ]
senti = [ j for i, j in tph.index.tolist() ]
sent = pd.DataFrame({'Tanggal':tanggal,'sentiment':senti, 'Frekuensi':frek})
# Plotting line
plt.figure(figsize = (10,10))
sns.lineplot(x='Tanggal',y='Frekuensi',hue='sentiment',data=sent)
plt.title('Banyak Tweet per Hari',fontsize=20)
plt.xticks(rotation=45)
plt.xlabel('Tanggal',fontsize=16)
plt.ylabel('Frekuensi',fontsize=16)
plt.legend(['Negatif','Netral','Positif'])
plt.show()