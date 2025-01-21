from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from nltk import word_tokenize, pos_tag
import os

def unigramsPOS():
    # ----------------- Corpus preparation ------------------------------

    # Percorso alle cartelle
    positive_path = "C:/Users/wewan/Desktop/Università/IA/mix20_rand700_tokens/tokens/pos"
    negative_path = "C:/Users/wewan/Desktop/Università/IA/mix20_rand700_tokens/tokens/neg"

    # Funzione per aggiungere POS tags
    def add_pos_tags(text):
        tokens = word_tokenize(text)  # Tokenizza il testo
        tagged_tokens = pos_tag(tokens)  # Applica POS tagging
        return " ".join([f"{word}/{tag}" for word, tag in tagged_tokens])  # Combina parola e tag

    # Leggi i file nelle cartelle e applica POS tagging
    positive_reviews = [add_pos_tags(open(os.path.join(positive_path, f), encoding="latin-1").read())
                        for f in os.listdir(positive_path)]
    negative_reviews = [add_pos_tags(open(os.path.join(negative_path, f), encoding="latin-1").read())
                        for f in os.listdir(negative_path)]

    # Combina il testo e le etichette
    corpus = positive_reviews + negative_reviews
    labels = [1] * len(positive_reviews) + [0] * len(negative_reviews)  # 1 = positivo, 0 = negativo

    # ----------------- Feature Extraction ------------------------------

    # Crea una matrice con unigrams (POS tagged) e filtra con min_df=4
    vectorizer = CountVectorizer(ngram_range=(1, 1), min_df=4, binary=True)  # Solo unigrams
    X = vectorizer.fit_transform(corpus)

    # ----------------------- Training ---------------------------------

    # Inizializza il modello Naive Bayes
    model = MultinomialNB()
    # model = Perceptron(max_iter=1000, tol=1e-3, random_state=42)

    # Esegui la cross-validation
    scores = cross_val_score(model, X, labels, cv=3)

    # Stampa i risultati
    print("Cross-validation scores:", scores)
    print("Average accuracy:", scores.mean())
    print("Numero di feature selezionate:", len(vectorizer.get_feature_names_out()))