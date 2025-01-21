from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
import nltk
from nltk import word_tokenize, pos_tag
import os

# ----------------- Corpus preparation ------------------------------

# Percorso alle cartelle
positive_path = "C:/Users/wewan/Desktop/Università/IA/mix20_rand700_tokens/tokens/pos"
negative_path = "C:/Users/wewan/Desktop/Università/IA/mix20_rand700_tokens/tokens/neg"

# Funzione per mantenere solo gli aggettivi
def keep_only_adjectives(text):
    tokens = word_tokenize(text)  # Tokenizza il testo
    tagged_tokens = pos_tag(tokens)  # Applica POS tagging
    adjectives = [word for word, tag in tagged_tokens if tag in {"JJ", "JJR", "JJS"}]
    return " ".join(adjectives)  # Ritorna solo gli aggettivi come testo

# Leggi i file nelle cartelle e filtra solo gli aggettivi
positive_reviews = [keep_only_adjectives(open(os.path.join(positive_path, f), encoding="latin-1").read())
                    for f in os.listdir(positive_path)]
negative_reviews = [keep_only_adjectives(open(os.path.join(negative_path, f), encoding="latin-1").read())
                    for f in os.listdir(negative_path)]

# Combina il testo e le etichette
corpus = positive_reviews + negative_reviews
labels = [1] * len(positive_reviews) + [0] * len(negative_reviews)  # 1 = positivo, 0 = negativo

# ----------------- Feature Extraction ------------------------------

# Crea una matrice BoW filtrata per min_df=4
vectorizer = CountVectorizer(min_df=4)
X = vectorizer.fit_transform(corpus)

# ----------------------- Training ---------------------------------

# Inizializza il modello Naive Bayes
#model = MultinomialNB()
model = Perceptron(max_iter=1000, tol=1e-3, random_state=42)

# Esegui la cross-validation
scores = cross_val_score(model, X, labels, cv=3)

# Stampa i risultati
print("Cross-validation scores:", scores)
print("Average accuracy:", scores.mean())
print("Numero di feature (solo aggettivi):", len(vectorizer.get_feature_names_out()))
