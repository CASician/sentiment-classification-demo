from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score

# ----------------- Corpus preparation ------------------------------
import os

# Percorso alle cartelle
positive_path = "C:/Users/wewan/Desktop/Università/IA/mix20_rand700_tokens/tokens/pos"
negative_path = "C:/Users/wewan/Desktop/Università/IA/mix20_rand700_tokens/tokens/neg"

# Leggi i file nelle cartelle
positive_reviews = [open(os.path.join(positive_path, f), encoding="latin-1").read()
                    for f in os.listdir(positive_path)]
negative_reviews = [open(os.path.join(negative_path, f), encoding="latin-1").read()
                    for f in os.listdir(negative_path)]

# Combina il testo e le etichette
corpus = positive_reviews + negative_reviews
labels = [1] * len(positive_reviews) + [0] * len(negative_reviews)  # 1 = positivo, 0 = negativo


# Crea un'istanza di CountVectorizer
vectorizer = CountVectorizer(min_df=4)

X = vectorizer.fit_transform(corpus)


# ----------------------- Training ---------------------------------

model = MultinomialNB()

scores = cross_val_score(model, X, labels, cv=3)


# Stampa i risultati
print("Cross-validation scores:", scores)
print("Average accuracy:", scores.mean())
# print("Numero di feature:", len(vectorizer.get_feature_names_out()))



