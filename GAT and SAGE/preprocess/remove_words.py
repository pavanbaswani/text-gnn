from nltk.corpus import stopwords
import nltk
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn

import sys
sys.path.append('../')
from utils.utils import clean_str, loadWord2Vec, clean_str_indic  


if len(sys.argv) != 2:
	sys.exit("Use: python remove_words.py <dataset>")

datasets = ['20ng', 'R8', 'R52', 'ohsumed', 'mr', 'hin', 'tel', 'bn', 'gu', 'kn', 'ml', 'ta','mar']
indic = ['hin', 'tel', 'bn', 'gu', 'kn', 'ml', 'mar', 'ta']

dataset = sys.argv[1]

if dataset not in datasets:
	sys.exit("wrong dataset name")

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

if dataset == 'hin':
    stop_words = ['तुम','मेरी','मुझे','क्योंकि','हम','प्रति','अबकी','आगे','माननीय','शहर','बताएं','कौनसी','क्लिक','किसकी','बड़े','मैं','and','रही','आज','लें','आपके','मिलकर','सब','मेरे','जी','श्री','वैसा','आपका','अंदर', 'अत', 'अपना', 'अपनी', 'अपने', 'अभी', 'आदि', 'आप', 'इत्यादि', 'इन', 'इनका', 'इन्हीं', 'इन्हें', 'इन्हों', 'इस', 'इसका', 'इसकी', 'इसके', 'इसमें', 'इसी', 'इसे', 'उन', 'उनका', 'उनकी', 'उनके', 'उनको', 'उन्हीं', 'उन्हें', 'उन्हों', 'उस', 'उसके', 'उसी', 'उसे', 'एक', 'एवं', 'एस', 'ऐसे', 'और', 'कई', 'कर','करता', 'करते', 'करना', 'करने', 'करें', 'कहते', 'कहा', 'का', 'काफ़ी', 'कि', 'कितना', 'किन्हें', 'किन्हों', 'किया', 'किर', 'किस', 'किसी', 'किसे', 'की', 'कुछ', 'कुल', 'के', 'को', 'कोई', 'कौन', 'कौनसा', 'गया', 'घर', 'जब', 'जहाँ', 'जा', 'जितना', 'जिन', 'जिन्हें', 'जिन्हों', 'जिस', 'जिसे', 'जीधर', 'जैसा', 'जैसे', 'जो', 'तक', 'तब', 'तरह', 'तिन', 'तिन्हें', 'तिन्हों', 'तिस', 'तिसे', 'तो', 'था', 'थी', 'थे', 'दबारा', 'दिया', 'दुसरा', 'दूसरे', 'दो', 'द्वारा', 'न', 'नहीं', 'ना', 'निहायत', 'नीचे', 'ने', 'पर', 'पर', 'पहले', 'पूरा', 'पे', 'फिर', 'बनी', 'बही', 'बहुत', 'बाद', 'बाला', 'बिलकुल', 'भी', 'भीतर', 'मगर', 'मानो', 'मे', 'में', 'यदि', 'यह', 'यहाँ', 'यही', 'या', 'यिह', 'ये', 'रखें', 'रहा', 'रहे', 'ऱ्वासा', 'लिए', 'लिये', 'लेकिन', 'व', 'वर्ग', 'वह', 'वह', 'वहाँ', 'वहीं', 'वाले', 'वुह', 'वे', 'वग़ैरह', 'संग', 'सकता', 'सकते', 'सबसे', 'सभी', 'साथ', 'साबुत', 'साभ', 'सारा', 'से', 'सो', 'ही', 'हुआ', 'हुई', 'हुए', 'है', 'हैं', 'हो', 'होता', 'होती', 'होते', 'होना', 'होने', 'अपनि', 'जेसे', 'होति', 'सभि', 'तिंहों', 'इंहों', 'दवारा', 'इसि', 'किंहें', 'थि', 'उंहों', 'ओर', 'जिंहें', 'वहिं', 'अभि', 'बनि', 'हि', 'उंहिं', 'उंहें', 'हें', 'वगेरह', 'एसे', 'रवासा', 'कोन', 'निचे', 'काफि', 'उसि', 'पुरा', 'भितर', 'हे', 'बहि', 'वहां', 'कोइ', 'यहां', 'जिंहों', 'तिंहें', 'किसि', 'कइ', 'यहि', 'इंहिं', 'जिधर', 'इंहें', 'अदि', 'इतयादि', 'हुइ', 'कोनसा', 'इसकि', 'दुसरे', 'जहां', 'अप', 'किंहों', 'उनकि', 'भि', 'वरग', 'हुअ', 'जेसा', 'नहिं']

# print(stopwords_hi)

# Read Word Vectors
# word_vector_file = 'data/glove.6B/glove.6B.200d.txt'
# vocab, embd, word_vector_map = loadWord2Vec(word_vector_file)
# word_embeddings_dim = len(embd[0])
# dataset = '20ng'

doc_content_list = []
#with open('data/wiki_long_abstracts_en_text.txt', 'r') as f:
with open('../data/corpus/' + dataset + '.txt', 'rb') as f:
    for line in f.readlines():
        if dataset not in indic:
            doc_content_list.append(line.strip().decode('latin1'))
        else:
            doc_content_list.append(line.strip().decode('utf8'))
            
word_freq = {}  # to remove rare words

print("Generating Word frequencies")
for doc_content in doc_content_list:
    if dataset not in indic:
        temp = clean_str(doc_content)
    else:
        temp = clean_str_indic(doc_content)
    words = temp.split()
    for word in words:
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1

print("Word Frequency Lenght: ", len(word_freq))

print("Cleaning the text and removing <5 frequency words from text")
min_len = 10000
aver_len = 0
max_len = 0 

clean_docs = []
for doc_content in doc_content_list:
    word_count = 0
    # doc_content = doc_content.split('\t',1)[1]
    if dataset not in indic:
        temp = clean_str(doc_content)
    else:
        temp = clean_str_indic(doc_content)
    words = temp.split()
    doc_words = []
    for word in words:
        # word not in stop_words and word_freq[word] >= 5               #word not in stop_words and  (remove stop words to check working)
        if dataset == 'R8':
            doc_words.append(word)
            word_count += 1
        elif word not in stop_words and word_freq[word] >= 5:                                             
            doc_words.append(word)
            word_count += 1
    
    aver_len = aver_len + word_count
    if word_count < min_len:
        min_len = word_count
    if word_count > max_len:
        max_len = word_count
        
    doc_str = ' '.join(doc_words).strip()
    #if doc_str == '':
        #doc_str = temp
    clean_docs.append(doc_str)

print("Total docs: ", len(clean_docs))
print(clean_docs[0])
clean_corpus_str = '\n'.join(clean_docs)


# #with open('../data/wiki_long_abstracts_en_text.clean.txt', 'w') as f:
with open('../data/corpus/' + dataset + '.clean.txt', 'w') as f:
    f.write(clean_corpus_str)

# # #dataset = '20ng'
# min_len = 10000
# aver_len = 0
# max_len = 0 

# # #with open('../data/wiki_long_abstracts_en_text.txt', 'r') as f:
# with open('../data/corpus/' + dataset + '.clean.txt', 'r') as f:
#     lines = f.readlines()
#     for line in lines:
#         line = line.strip()
#         temp = line.split()
#         aver_len = aver_len + len(temp)
#         if len(temp) < min_len:
#             min_len = len(temp)
#         if len(temp) > max_len:
#             max_len = len(temp)

aver_len = 1.0 * aver_len / len(clean_docs)
print('Min_len : ' + str(min_len))
print('Max_len : ' + str(max_len))
print('Average_len : ' + str(aver_len))