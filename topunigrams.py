from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
import os

def topunigrams():
    # ----------------- Corpus preparation ------------------------------
    
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

    # ----------------- Feature Extraction ------------------------------

    # Crea una matrice BoW con solo le 3228 parole più frequenti
    vectorizer = CountVectorizer(max_features=3228, binary=True)
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
    print("Numero di feature (unigrams più frequenti):", len(vectorizer.get_feature_names_out()))