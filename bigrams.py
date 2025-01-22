from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
import os

def bigrams():
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

    # Crea una matrice con unigrams e bigrams
    vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=4, binary=True)  # Filtra direttamente gli unigrams con min_df=4
    X_full = vectorizer.fit_transform(corpus)

    # Ottieni i nomi delle feature
    feature_names = vectorizer.get_feature_names_out()

    # Calcola la frequenza totale di ciascuna feature
    feature_frequencies = X_full.sum(axis=0).A1  # Frequenze totali in una singola operazione

    # Indici dei bigrams e filtra quelli con frequenza ≥7
    bigram_indices = [i for i, feat in enumerate(feature_names) if len(feat.split()) == 2]
    valid_bigrams = [i for i in bigram_indices if feature_frequencies[i] >= 7]

    # Indici degli unigrams già filtrati con min_df=4
    unigram_indices = [i for i, feat in enumerate(feature_names) if len(feat.split()) == 1]

    # Combina unigrams e valid bigrams
    selected_indices = valid_bigrams
    X_filtered = X_full[:, selected_indices]

    # ----------------------- Training ---------------------------------

    # Inizializza i modelli
    model = MultinomialNB()
    #model = Perceptron(max_iter=1000, tol=1e-3, random_state=42)

    # Esegui la cross-validation
    scores = cross_val_score(model, X_filtered, labels, cv=3)

    # Stampa i risultati
    print("Only Bigrams")
    print(f"CV: {[f'{score:.3f}' for score in scores]}")  # 3 decimali
    print(f"Average accuracy: {scores.mean():.3f}" + "  NF:", len(selected_indices))  # 3 decimali
    print("-------------------------------------------")
