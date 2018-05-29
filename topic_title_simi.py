import os, sys, re
import numpy as np
import pandas as pd 
import nltk, random, spacy 
from spacy.en import English
from spacy.fr import French 
from scipy import spatial
import pickle
import gensim 
from gensim import corpora
from gensim.models import KeyedVectors
nltk.download('wordnet')
nltk.download('stopwords')
spacy.load('en')
spacy.load('fr')

en_stop = set(nltk.corpus.stopwords.words('english'))
fr_stop = set(nltk.corpus.stopwords.words('french'))
black_list = ['@card@']

lemma_text = sys.argv[1] # e.g. storyzy_yt_test2.corrected.tsv.text.lemma.txt
lang = sys.argv[2]  # 'en' or 'fr'
corrected_csv = sys.argv[3] # e.g. storyzy_yt_test2.corrected.tsv
output = sys.argv[4] # e.g. yt_test2_topic_title_simi.txt

def tokenize(text, lang):
	if lang == 'en':
		parser = English()
	elif lang == 'fr':
		parser = French()

	lda_tokens = []
	tokens = parser(text)
	# will split apostrophe 
	for token in tokens:
		if token.orth_.isspace():
			continue
		else:
			lda_tokens.append(token.lower_)
	return lda_tokens

def prepare_text_for_lda(text, lang):
	tokens = tokenize(text, lang)
	tokens = [token for token in tokens if len(token) > 2]
	if lang == 'en':
		tokens = [token for token in tokens if token not in en_stop]
	elif lang == 'fr':
		tokens = [token for token in tokens if token not in fr_stop]
	tokens = [token for token in tokens if token not in black_list]
	return tokens

def embed_model(lang):
	if lang == 'fr':
		model = KeyedVectors.load_word2vec_format('fr.vec', binary=False)
		return model 
	elif lang == 'en':
		model = KeyedVectors.load_word2vec_format('en.vec', binary=False)
		return model 

# better to run in jupyter notebook, to load only once the model and reuse it for each input file
if lang == 'en':
	model = embed_model('en')
elif lang == 'fr':
	model = embed_model('fr')

# --------------------------- average embedding of top five topic words for each text ------------------------
topic_embedding_average = []

with open(lemma_text) as f:
	next(f)
	for line in f:
		text_data = []
		line = line.strip('\n')
		tokens = prepare_text_for_lda(line, lang)
		text_data.append(tokens)

		dictionary = corpora.Dictionary(text_data)
		corpus = [dictionary.doc2bow(text) for text in text_data]
		if (len(dictionary)!=0) and if len(corpus) >=1 :
			pickle.dump(corpus, open('corpus.pkl', 'wb'))
			dictionary.save('dictionary.gensim')
			NUM_TOPICS = 5
			ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)
			ldamodel.save('model5.gensim')
			topics = ldamodel.print_topics(num_words=4)
		else:
			pass

		topics_dic = {}
		for topic in topics:
			proba = []
			word = []
			m = re.search(r'\(\d+\, (.*?) \+ (.*?) \+ (.*?) \+ (.*?)\'.*', str(topic))
			if m:
				t1 = m.group(1).replace('"','').lstrip('\'')
				t2 = m.group(2).replace('"','')
				t3 = m.group(3).replace('"','')
				t4 = m.group(4).replace('"','')

				proba1 = t1.split('*')[0]
				w1 = t1.split('*')[1]

				proba2 = t2.split('*')[0]
				w2 = t2.split('*')[1]

				proba3 = t3.split('*')[0]
				w3 = t3.split('*')[1]

				proba4 = t4.split('*')[0]
				w4 = t4.split('*')[1]

				proba.extend((proba1, proba2, proba3, proba4))
				word.extend((w1, w2, w3, w4)) 

				for (x, y) in zip(word, proba):
					if x not in topics_dic:
						topics_dic[x] = y 
					else:
						if y > topics_dic[x]:
							topics_dic[x] = y
			else:
				pass
		sorted_keys = sorted(topics_dic, key=topics_dic.get, reverse=True)
		result_topic = []
		for r in sorted_keys[0:5]:
			# topics_dic[r]: probability 
			result_topic.append(r)

		# dimension of current word embedding: english 300, french 100
		if lang == 'en':	
			tmp = np.zeros(300)  
		elif lang == 'fr':
			tmp = np.zeros(100)

		for word in result_topic:
			try:
				embedding = model.get_vector(word)  
				tmp = np.add(tmp, embedding)
			except:
				pass
		average = tmp/5 
		topic_embedding_average.append(average)

# ---------------------------- average embedding of each title -------------------------
title_embedding_average = []
data = pd.read_csv(corrected_csv, sep="\t")

if 'yt' in corrected_csv:
	titles = data["video-title"]
else:	
	titles = data["title"]

for title in titles:
	if lang == 'en':	
		tmp = np.zeros(300)  
	elif lang == 'fr':
		tmp = np.zeros(100)

	for word in str(title).split():
		try:
			embedding = model.get_vector(word)
			tmp = np.add(tmp, embedding)
		except:
			pass
	average = tmp/len(str(title).split())
	title_embedding_average.append(average)

# check if they have processed the same number of lines 
print(len(topic_embedding_average))
print(len(title_embedding_average))

# ---------------------------- write output -------------------------------------------
# when embedding est zero, its norm is also 0, distance = nan  
output = open(output, 'w')
 
for (topic, title) in zip(topic_embedding_average, title_embedding_average):
	cosine = 1 - spatial.distance.cosine(topic, title) 
	print(cosine, file=output)

