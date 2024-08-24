
############################## Morfologik Tahlil ###############################

import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# NLTK kutubxonasidagi morfologik tahlil uchun dastlabki sozlamalar
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

# Matn
text = "Oʻzbekiston Respublikasi poytaxti Toshkent shahridir."

# So‘zlarni tokenizatsiya qilish (ajratish)
tokens = word_tokenize(text)

# Morfologik tahlil (so‘zlarni turini aniqlash)
tagged_tokens = pos_tag(tokens)

print("Tokenlar va morfologik teglar:")
for token, tag in tagged_tokens:
    print(f"{token}: {tag}")


############################## Sintaktik Tahlil ##################################

from nltk import CFG
from nltk.parse import ChartParser
from nltk.tree import Tree

# CFG grammatikani aniqlash
grammar = CFG.fromstring("""
    S -> NP VP
    NP -> DT N | DT N PP
    VP -> V NP | V NP PP
    PP -> P NP
    DT -> 'Oʻzbekiston' | 'poytaxti' | 'Toshkent' | 'shahridir'
    N -> 'Respublikasi' | 'Toshkent' | 'shahar'
    V -> 'boʻlishi'
    P -> 'of'
""")

# Parser yaratish
parser = ChartParser(grammar)

# Matnni tahlil qilish
sentence = 'Oʻzbekiston Respublikasi poytaxti Toshkent shahridir'.split()
print("Sintaktik tahlil daraxtlari:")
for tree in parser.parse(sentence):
    print(tree)
    tree.draw()  # Daraxtni grafik tarzda ko‘rsatadi

############################# Semantik Tahlil #################################

from gensim.models import Word2Vec
from gensim.models.word2vec import Word2Vec as W2V

# So‘zlar ro‘yxati
sentences = [["the", "capital", "of", "Uzbekistan", "is", "Tashkent"],
             ["Tashkent", "is", "the", "capital", "city", "of", "Uzbekistan"]]

# Word2Vec modelini yaratish va o‘rganish
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=0)

# Semantik o‘xshashlikni hisoblash
similarity = model.wv.similarity('capital', 'Tashkent')
print(f"'capital' va 'Tashkent' o‘xshashligi: {similarity}")

############################# NLP Asosiy Tahlil Misollari ############################

import spacy

# spaCy modelini yuklash
nlp = spacy.load("en_core_web_sm")

# Matnni tahlil qilish
doc = nlp("The capital of Uzbekistan is Tashkent.")

# Morfologik tahlil
print("Morfologik tahlil:")
for token in doc:
    print(f"{token.text}: {token.pos_}, {token.tag_}")

# Sintaktik tahlil
print("\nSintaktik tahlil:")
for token in doc:
    print(f"{token.text}: {token.dep_}, {token.head.text}")

# Semantik tahlil
print("\nSemantik o‘xshashlik:")
similarity = doc.vector
print(f"Matnning vektor ko‘rinishi: {similarity}")
