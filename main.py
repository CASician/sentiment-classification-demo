from unigramsfreq import unigrams_freq
from unigramspres import unigramspresence
from adjectives import adjectives
from bigrams import bigrams
from position import position
from topunigrams import topunigrams
from unibi import unigrams_bigrams
from unigramsPOS import unigramsPOS

# Choose between NP and P
model_wanted = 'P'

unigrams_freq(model_wanted)
unigramspresence(model_wanted)
unigrams_bigrams(model_wanted)
bigrams(model_wanted)
unigramsPOS(model_wanted)
adjectives(model_wanted)
topunigrams(model_wanted)
position(model_wanted)