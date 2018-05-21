import re
import numpy as np
from sklearn import tree
from bs4 import BeautifulSoup
from pprint import pprint
from gensim import corpora, models, similarities
from sklearn.ensemble import RandomForestClassifier
from pprint import pprint
import jiebaa.jieba as jieba
import jieba.analyse
from gensim.models import word2vec
import logging
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.datasets import samples_generator
from sklearn.feature_selection import SelectKBest, f_regression,chi2
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.feature_extraction.text import CountVectorizer
print(__doc__)



'''
***对原始数据 进行分类到格相应的数组
'''
class  NLPCC_emotion_Detection:
    def __init__(self):
        self.content = []
        self.happiness = [ ]
        self.sadness = [ ]
        self.anger = [ ]
        self.fear = [ ]
        self.surprise = [ ]
        self.read_inf()


    def read_inf(self):
        '''得到信息 分入相应'''
        i = 1
        fpath = "train.txt"
        file = open(fpath,'r', encoding = 'utf-8')
        for line in file:
            conte = re.sub('<.*>|\t','',line.strip())
            if conte  != '':
                # print(conte)
                # print(i)
                if i == 1:
                    self.happiness.append(conte)
                elif i == 2:
                    self.sadness.append(conte)
                elif i == 3:
                    self.anger.append(conte)
                elif i == 4:
                    self.fear.append(conte)
                elif i == 5:
                    self.surprise.append(conte)
                else:
                    temp = ''
                    for code in conte:
                        if code != '':
                            pattern = re.compile('^[\u4e00-\u9fa5_a-zA-Z< >]+$')
                            item = re.findall(pattern,code )
                            if item != []:
                                temp += item[0].__str__()

                    # print(temp)
                    self.content.append(temp)
                    temp = ''
                    i = 0
                i += 1
            conte = ''
        data_processing(self,self.content)

'''
*** 对文本 进行分词 和加工需要的数据类型
'''
def data_processing(self,cont):
    texts_one = []
    stop_word = []
    texts = []

    stop_file = open('D:\Python_workspace\Emotion Detection\stop_words.txt', encoding = 'utf-8')
    for word in stop_file.readlines():
        stop_word.append(word[ :-1 ])

    for x in cont:
        temp = str(x).split()
        # print(str(x).split())
        texts_one = [ma_word for ma_word in temp if ma_word not in stop_word]
        texts.append(texts_one)
        print(texts)
    # pprint(texts)
    # LDA_LSI_hda_code(texts)
    # word2vec_NLPCC(texts)
    # selsectkbest_NLP(self,texts)

'''
***卡方统计 提取特征
'''


def selsectkbest_NLP(self,texts):
    cont = []
    string_s = ''
    for item in texts:
        for i in item:
            string_s += i
            string_s += ' '
        print(string_s)
        cont.append(string_s)
        string_s = ''
    print(cont)
    transfromer = TfidfTransformer()
    vectorizer = CountVectorizer()
    tfidf = transfromer.fit_transform(vectorizer.fit_transform(cont))
    print(tfidf.toarray())
    X = np.array(tfidf.toarray())
    # X = SelectKBest(f_regression,k = 3)
    y = self.happiness


    #pca  降维处理
    pca = PCA(n_components = 2, whiten = True, random_state = 0)
    X = pca.fit_transform(X)
    #卡方降维
    # anova = SelectKBest(chi2,k = 3)
    # fs = anova.fit_transform(X,y)
    print(type(X))
    print(type(y))
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    priors = np.array((1, 2, 4), dtype = float)
    priors /= priors.sum()
    # gnb = Pipeline([
    #     ('sc', StandardScaler()),
    #     ('poly', PolynomialFeatures(degree=1)),
    #     ('clf', GaussianNB())])
    #gnb = KNeighborsClassifier(n_neighbors=3).fit(x, y.ravel())
    # gnb = tree.DecisionTreeClassifier()
    # gnb = svm.SVC()

    gnb = RandomForestClassifier(n_estimators=200, criterion='entropy', max_depth=3)
    # anova = SelectKBest(f_regression,k = 3)
    # gnb =  make_pipeline(anova,gnb)
    # print(type(X_train))
    # print(type(y_train))

    gnb.fit(X_train, y_train)
    y_hat = gnb.predict(X_train)
    print ('训练集准确度: %.2f%%' % (100 * accuracy_score(y_train, y_hat)))
    y_test_hat = gnb.predict(X_test)
    print ('测试集准确度：%.2f%%' % (100 * accuracy_score(y_test, y_test_hat)))  # 画图


#初始化
def word2vec_NLPCC(sentences):

    logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',level=logging.INFO)

    model = word2vec.Word2Vec(sentences,size=200)#训练skip-gram模型，默认window=5
    print("输出模型",model)

    y2 = model.most_similar("同性恋", topn = 20)  # 20个最相关的
    print("与【happy】最相关的词有：\n")
    for word in y2:
        print(word[ 0 ], word[ 1 ])
    print("*********\n")
    y2 = model.most_similar("happy", topn = 20)  # 20个最相关的
    print("与kongju】最相关的词有：\n")
    for word in y2:
        print(word[ 0 ], word[ 1 ])
    print("*********\n")


'''
***利用gensim 特征 tfidf发  LDA LSI模型进行分类
'''

def LDA_LSI_hda_code(texts):
    dictionary = corpora.Dictionary(texts)
    print (dictionary)
    V = len(dictionary)
    corpus = [dictionary.doc2bow(text) for text in texts]
    corpus_tfidf = models.TfidfModel(corpus)[corpus]
    corpus_tfidf = corpus

    print ('TF-IDF:')
    for c in corpus_tfidf:
        print (c)

    print ('\nLSI Model:')
    lsi = models.LsiModel(corpus_tfidf, num_topics=20, id2word=dictionary)
    topic_result = [a for a in lsi[corpus_tfidf]]
    pprint(topic_result)
    print ('LSI Topics:')
    pprint(lsi.print_topics(num_topics=20, num_words=10))
    similarity = similarities.MatrixSimilarity(lsi[corpus_tfidf])   # similarities.Similarity()
    print ('Similarity:')
    pprint(list(similarity))

    print ('\nLDA Model:')
    num_topics = 2
    lda = models.LdaModel(corpus_tfidf, num_topics=num_topics, id2word=dictionary,
                          alpha='auto', eta='auto', minimum_probability=0.001, passes=10)
    doc_topic = [doc_t for doc_t in lda[corpus_tfidf]]
    print ('Document-Topic:\n')
    pprint(doc_topic)
    for doc_topic in lda.get_document_topics(corpus_tfidf):
        print (doc_topic)
    for topic_id in range(num_topics):
        print ('Topic', topic_id)
        # pprint(lda.get_topic_terms(topicid=topic_id))
        pprint(lda.show_topic(topic_id))
    similarity = similarities.MatrixSimilarity(lda[corpus_tfidf])
    print ('Similarity:')
    pprint(list(similarity))

    hda = models.HdpModel(corpus_tfidf, id2word=dictionary)
    topic_result = [a for a in hda[corpus_tfidf]]
    print ('\n\nUSE WITH CARE--\nHDA Model:')
    pprint(topic_result)
    print ('HDA Topics:')
    print (hda.print_topics(num_topics=20, num_words=10))



if __name__ == '__main__':
    carry_out = NLPCC_emotion_Detection()
    # carry_out.read_inf()