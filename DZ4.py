'''Для заданого корпусу поділити тексти на теми і визначити ключові слова для кожної теми

Виявилось, що articles.csv досить шумний/складний датасет

Тому якщо є бажання можна викорастати будь який інший публічний корпус

Просто у такому випадку зазначте, як його можна отримати. Або якщо датасет не великий можна просто файлом надіслати.'''
#pip install pandas scikit-learn nltk gensim


import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Завантаження датасету
newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))

# Попередня обробка даних
def preprocess(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(tokens)

processed_texts = [preprocess(text) for text in newsgroups_train.data]

# Токенізація та побудова словника
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
doc_term_matrix = vectorizer.fit_transform(processed_texts)

# Тренування LDA моделі
lda = LatentDirichletAllocation(n_components=10, random_state=42)
lda.fit(doc_term_matrix)

# Отримання ключових слів для кожної теми
def get_topic_keywords(model, feature_names, n_top_words):
    topic_keywords = []
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        topic_keywords.append(top_features)
    return topic_keywords

n_top_words = 10
feature_names = vectorizer.get_feature_names_out()
topic_keywords = get_topic_keywords(lda, feature_names, n_top_words)

# Виведення ключових слів для кожної теми
for idx, topic in enumerate(topic_keywords):
    print(f"Тема {idx+1}: {', '.join(topic)}")


''' Results:
Тема 1: db, armenian, armenians, turkish, turkey, israel, armenia, turks, killed, people
Тема 2: like, car, good, time, gun, know, think, use, people, make
Тема 3: people, god, think, know, say, like, said, believe, time, way
Тема 4: team, game, year, games, play, season, good, players, new, hockey
Тема 5: file, information, send, list, mail, email, anonymous, available, internet, address
Тема 6: drive, card, know, like, use, disk, new, thanks, hard, dos
Тема 7: space, nasa, launch, earth, satellite, university, data, orbit, shuttle, lunar
Тема 8: max, jesus, bhj, giz, wm, god, gk, law, bj, christ
Тема 9: use, windows, window, file, program, using, image, version, available, files
Тема 10: people, government, key, president, law, right, encryption, public, use, state '''
