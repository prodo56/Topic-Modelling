import string
import gensim
from gensim import corpora
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os


datafolder = "./data/british-fiction-corpus/"
lemmatiser = WordNetLemmatizer()
punc = set(string.punctuation)

def readFiles(fi):
    try:
        with open(fi,"r") as f:
            return f.read()
    except Exception as e:
        print e.message
        raise


def cleanDocs(doc):
    stopword_removal = " ".join([word.lower() for word in doc.split() if word not in stopwords.words('english')])
    punctuation_removal = ''.join(char for char in stopword_removal if char not in punc)
    #print punctuation_removal
    normalise = " ".join([lemmatiser.lemmatize(word) for word in punctuation_removal.split()])
    return normalise


docs = []
for f in os.listdir(datafolder):
    try:
        docs.append(readFiles(datafolder+f))
    except Exception as e:
        print "error with reading file {} with error {}".format(f,e.message)

cleanedDocuments = [cleanDocs(doc).split() for doc in docs]
wordDict = corpora.Dictionary(cleanedDocuments)
docTermMatrix = [wordDict.doc2bow(doc) for doc in cleanedDocuments]
Lda = gensim.models.ldamodel.LdaModel
model = Lda(docTermMatrix,num_topics=3,id2word=wordDict,passes=100)
#print model.top_topics(doc_term_matrix,num_words=4)
print model.print_topics(num_topics=3,num_words=3)

