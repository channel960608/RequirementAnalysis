#!/usr/bin/python
import read_data_cchit
import read_data_smos
from nltk.text import TextCollection
import nltk
from gensim import corpora, models, similarities
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from nltk.tokenize import WordPunctTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def pretreatment(content, language):
        # 分词
        words = []
        words.extend(word_tokenize(content, language=language))
        # 小写处理
        words_lower=[i.lower() for i in words]
        # 去除标点以及停用词
        stopwords_ = stopwords.words(language)
        punctuations = ['*/', '/**', '"', ',', '.', ':', ';', '?', '(', ')', '[', ']', '!', '@', '#', '%', '$', '*', '...'] # 自定义英文表单符号列表
        words_clear=[]
        for i in words_lower:
                if i not in stopwords_: # 过滤停用词
                        if i not in punctuations: # 过滤标点符号
                                words_clear.append(i)
        # 词干化处理
        st = PorterStemmer()
        words_stem=[st.stem(word) for word in words_clear]
        
        return words_stem         

def calculateVector():
        source, target = read_data_cchit.load_data()
        for key in source.keys():
                source[key]['content'] = pretreatment(source[key]['content'], 'english')
        for key in target.keys():
                target[key]['content'] = pretreatment(target[key]['content'], 'english')
        
        art_source = []
        art_target = []
        for item in source.values():
                art_source.append(item)
        for item in target.values():
                art_target.append(item)
        len_source = len(art_source)

        documents = [item['content'] for item in list(source.values())+list(target.values())]

        texts = [document for document in documents]
        dictionary = corpora.Dictionary(texts)
        #由文档向量以及频率构成文档向量
        corpus = [dictionary.doc2bow(text) for text in texts]

        tfidf = models.TfidfModel(corpus)
        corpus_tfidf = tfidf[corpus]
        for index, doc in enumerate(corpus_tfidf):
                if index < len_source:
                        art_source[index]['vector'] = doc
                else:
                        art_target[index-len_source]['vector'] = doc

        return art_source, art_target

def calculateVector_SMOS():
        source, target = read_data_smos.load_data()
        for key in source.keys():
                source[key]['content'] = pretreatment(source[key]['content'], 'italian')
        for key in target.keys():
                target[key]['content'] = pretreatment(target[key]['content'], 'italian')
        
        art_source = []
        art_target = []
        for item in source.values():
                art_source.append(item)
        for item in target.values():
                art_target.append(item)
        len_source = len(art_source)

        documents = [item['content'] for item in list(source.values())+list(target.values())]

        texts = [document for document in documents]
        dictionary = corpora.Dictionary(texts)
        #由文档向量以及频率构成文档向量
        corpus = [dictionary.doc2bow(text) for text in texts]

        tfidf = models.TfidfModel(corpus)
        corpus_tfidf = tfidf[corpus]
        for index, doc in enumerate(corpus_tfidf):
                if index < len_source:
                        art_source[index]['vector'] = doc
                else:
                        art_target[index-len_source]['vector'] = doc

        return art_source, art_target

def similarity(vector_1, vector_2):
        # 计算两个向量的余弦相似度，关键是找到两个向量第几维上的值均不为0
        len_1 = len(vector_1)
        len_2 = len(vector_2)
        if (len_1 == 0) or (len_2 == 0):
                return 0
        index_1 = 0
        index_2 = 0
        m_sum = 0
        while (index_1<len_1) and (index_2<len_2):
                dif = vector_1[index_1][0] - vector_2[index_2][0]
                if dif == 0:
                        m_sum += vector_1[index_1][1] * vector_2[index_2][1]
                        index_1 = index_1 + 1
                        index_2 = index_2 + 1

                elif dif < 0:
                        # m_sum += vector_1[index_1][1]
                        index_1 = index_1 + 1
                else:
                        # m_sum += vector_2[index_2][1]
                        index_2 = index_2 + 1
        sum_1 = 0
        sum_2 = 0
        for item in vector_1:
                sum_1+=item[1]*item[1]
        for item in vector_2:
                sum_2+=item[1]*item[1]
        value = m_sum / math.sqrt(sum_1 * sum_2)
        return value


def evaluate(predict, answer):
        
        if len(predict) == 0:
                return 0, 0
        x = set(predict)
        y = set(answer)
        z = x & y
        A = len(z)
        R = A / len(answer)
        P = A / len(predict)
        # print("正确的链接的一共有:" + str(A))
        
        
        return R, P

# 计算给定的所有需求跟踪链接的文本相似度，以便更好得选择K值
def evaluate_labels(art_source, art_target, link):
        m_array = []
        for item in link:
                pre = item[0]
                aft = item[1]
                # type(item)
                source = getById(art_source, pre)
                target = getById(art_target, aft)
                sim = similarity(source['vector'], target['vector'])
                m_array.append((item, sim))
        m_array.sort(key=lambda item : -1 * item[1])
        sim_list = []
        for tup in m_array:
                print(tup)
                sim_list.append(tup[1])
        return sim_list

def getById(art_dic, id):
        for item in art_dic:
                if item['id'] == id:
                        return item
        return None

def draw(sim_list):
        sns.set_palette('deep', desat=.6)
        sns.set_context(rc={'figure.figsize': (8, 5) } )
        data =  sim_list
        plt.hist(data, bins=80, histtype="stepfilled", alpha=.8)

def run(dataset):
        # k = 0.5
        ans = []
        if dataset == 'cchit':
                art_source, art_target = calculateVector()
        elif dataset == 'smos':
                art_source, art_target = calculateVector_SMOS()
        else:
                print('dataset choose error')
                return -1
        
        
        for item_source in art_source:
                for item_target in art_target:
                        
                        similarityValue = similarity(item_source['vector'], item_target['vector'])
                        pre_id = item_source['id']
                        aft_id = item_target['id']
                        ans.append((pre_id, aft_id, similarityValue))
        if dataset == 'cchit':
                link = read_data_cchit.read_link()
        elif dataset == 'smos':
                link = read_data_smos.read_link()
        else:
                print('dataset choose error')
                return -1
        
        # 评估所有已标定链接的文本余弦相似度的分布密度，并通过图像呈现
        sim_list = evaluate_labels(art_source, art_target, link) 
        draw(sim_list)

        # 选择多个K值比较召回率与精确率
        points_R = []
        points_P = []
        R_x = []
        R_y = []
        P_x = []
        P_y = []
        for k in np.arange(0, 1, 0.01):
        # for k in [0.5]:
                choose = []
                for item in ans:
                        if item[2]>=k:
                                choose.append((item[0], item[1]))
                P, R = evaluate(choose, link)
                point_p = [k, P]
                point_r = [k, R]
                points_P.append(point_p)
                points_R.append(point_r)
                R_x.append(k)
                P_x.append(k)
                R_y.append(R)
                P_y.append(P)
                # print("做出的链接判断一共有" + str(len(choose)))
                # print("K值为：" + str(k))
                # print("召回率为：" + str(P))
                # print("精确率为：" + str(R))
                # print("-------------------------------") 
                print(str(k)+' '+str(P)+' '+str(R))

        plt.figure()
        # plt.plot(points_P)
        plt.plot(R_x, R_y, color='red', label='R')
        plt.plot(P_x, P_y, color='green', label='P')
        # plt.plot(P_y, R_y, color='black', label='P-R')
        
        # plt.xlim(0,1)
        plt.xlabel("k value")
        plt.ylabel("P and R")
        plt.title("Relationship between k and the value of R,P")
        
        
        plt.figure()
        plt.plot(P_y, R_y, color='black', label='P-R')
        plt.show()



def test():
        # vector_1 = [(0, 0.013964313710414039), (1, 0.0050365916055848566), (2, 0.5658574871396089), (3, 0.05919877923179541), (4, 0.23068753052765487), (5, 0.5104061135207874), (6, 0.18153858158049213), (7, 0.07518919045778549), (8, 0.5104061135207874), (9, 0.2513137101770754), (10, 0.004748305089177805), (11, 0.00664100935324287)]
        # vector_2 = [(0, 0.010564095578117955), (9, 0.38024096414020514), (10, 0.0035921241699640215), (11, 0.005023967450008889), (18, 0.09295495332396897),(19, 0.023824976421251205), (22, 0.017518375449431738), (23, 0.009654913365857413), (24, 0.06123500270346173), (25, 0.061376239852885324), (26, 0.06123500270346173), (40, 0.2602776054534852), (46, 0.14936960839139177), (214, 0.13733513631264738), (241, 0.08577378796808113), (254, 0.025992167759287914), (262, 0.20482360692243848), (330, 0.1727099368488064), (577, 0.31963748116069696), (627, 0.28872226789426964), (738, 0.3103082679804913), (750, 0.3441762664253164), (751, 0.38612559691123194), (752, 0.3103082679804913)]
        # vector_1_example = [(0, 1), (1, 1), (3, 1)]
        # vector_2_example = [(1, 1), (2, 1), (3, 1), (4, 1)]
        # value = similarity(vector_1_example, vector_2_example)
        # pre()

                
        print('pause')

if __name__ == "__main__":
    run('smos')
#     test()