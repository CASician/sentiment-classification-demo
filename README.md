# sentiment-classification-demo
## A replica of Pang et al 2002 using Naive-Bayes and Perceptron

Using [SciKit-Learn library](https://scikit-learn.org/stable/index.html) and the [database](https://www.cs.cornell.edu/people/pabo/movie-review-data/) that is cited in the article itself, I
attempted to recreate the results published in the paper. 

### Files
Each file (except main) is autonomous and recreates the corpus of reviews, applies tags if needed, creates the BoW (Bag of Words)
and finally calculates the accuracy of the three-folder-cross-validation. The main file calls each one of this file,
allowing the user to choose between NB (Naive-Bayes) and P (Perceptron). 

Following the order as in Pang et al 2002 :
1. unigramsfreq : the BoW contains all the single words that appear more than four times across the entire corpus.
1. unigramspres : as unigramsfreq, but now the BoW matrix takes into account only if the word is present, not how many times.
1. unibi: unigrams and bigrams, is taking into account both single words and combination of two words.
1. bigrams: considering only bigrams
1. unigramsPOS: each unigram is tagged as a POS (part of speech) using [nltk library](https://www.nltk.org/) 
1. adjectives: using the POS tagging, the BoW now only contains adjectives
1. topunigrams: compares the results of using as many top unigrams as adjectives. 
1. position: the reviews are divided into three parts: start, middle and end. Each word receives a tag according to their position inside the review.


### Preparation
To run the programme, open the main file and modify line 11, where you choose which model you want to use: Naive Bayes or Perceptron.
Run and visualize the results.


### More details
For further reading, check the pdf. 