from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
import os

def position(model_wanted):
    # ----------------- Corpus preparation ------------------------------

    # Percorso alle cartelle
    positive_path = "C:/Users/wewan/Desktop/Università/IA/mix20_rand700_tokens/tokens/pos"
    negative_path = "C:/Users/wewan/Desktop/Università/IA/mix20_rand700_tokens/tokens/neg"

    # Funzione per aggiungere tag di posizione alle parole
    def tag_words_with_position(text):
        words = text.split()  # Tokenizza il testo in parole
        n = len(words)
        tagged_words = []
        for i, word in enumerate(words):
            if i < n / 4:
                tagged_words.append(f"{word}_first")  # Primo quartile
            elif i >= 3 * n / 4:
                tagged_words.append(f"{word}_last")  # Ultimo quartile
            else:
                tagged_words.append(f"{word}_middle")  # Metà centrale
        return " ".join(tagged_words)

    # Leggi i file nelle cartelle e aggiungi tag di posizione
    positive_reviews = [tag_words_with_position(open(os.path.join(positive_path, f), encoding="latin-1").read())
                        for f in os.listdir(positive_path)]
    negative_reviews = [tag_words_with_position(open(os.path.join(negative_path, f), encoding="latin-1").read())
                        for f in os.listdir(negative_path)]

    # Combina il testo e le etichette
    corpus = positive_reviews + negative_reviews
    labels = [1] * len(positive_reviews) + [0] * len(negative_reviews)  # 1 = positivo, 0 = negativo

    # ----------------- Feature Extraction ------------------------------

    # Crea una matrice BoW con le parole taggate
    vectorizer = CountVectorizer(binary=True, min_df=4)  # Puoi aggiungere max_features se necessario
    X = vectorizer.fit_transform(corpus)

    # ----------------------- Training ---------------------------------

    if model_wanted == 'NB':
        model = MultinomialNB()
    else:
        model = Perceptron(max_iter=1000, tol=1e-3, random_state=42)

    # Esegui la cross-validation
    scores = cross_val_score(model, X, labels, cv=3)

    # Stampa i risultati
    print("Position tagging using " + model_wanted)
    print(f"CV: {[f'{score:.3f}' for score in scores]}")  # 3 decimali
    print(f"Average accuracy: {scores.mean():.3f}" + "  NF:", len(vectorizer.get_feature_names_out()))  # 3 decimali
    print("-------------------------------------------")
