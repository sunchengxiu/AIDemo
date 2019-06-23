
from sklearn.feature_extraction import DictVectorizer

from  sklearn.feature_extraction.text import  CountVectorizer

import jieba

from sklearn.preprocessing import  MinMaxScaler , StandardScaler , Imputer

from sklearn.feature_extraction.text import TfidfVectorizer

import  numpy as np


def stand():

    """
    标准化:目的是转化为均值为0，标准差为1

    :return:
    """

    std = StandardScaler()
    data = std.fit_transform([[2,-3,4] , [1,3,1] , [5,8,-1]])
    print(data)



def imputer():

    """
    缺失值处理:填补策略：mean(平局值),按照行或者列（axis,1 是 列， 0 是行）,按照平局值填补，所以，空缺位置为（1 + 9）/2 = 5
    :return:
    """

    im = Imputer(missing_values='NaN' , strategy='mean',axis=0)
    data = im.fit_transform([[1,2],[np.nan , 4],[9,8]])
    print(data)



def MinMax():
    # 归一化，所有特征同样重要,如果数据相差较大，那么算出来的距离也过大，影响较大，所以要归一化，使得某一个特征对最后结果不会造成过大的影响（10000-500)^2+(20-15)^2+(5-4)^2
    # 如果异常点较多，对最大值最小值影响大
    #  x' = (x - X_min) / (X_max - X_min)  x'' = x'*(mx-mi)*mi
    # 鲁棒性：适合数据量较小的场景，不稳定


    """
    标准化：x' = (x- mean(平局值)) / (标准差  )
    var : 方差 ： (x1 - mean)^2 + (x2 - mean)^2 + (x3 - mean)^2 + ... / n(样本个数) 标准差 = 根号方差
    :return:
    """

    mm = MinMaxScaler(feature_range=(5,6))
    data = mm.fit_transform([[80,5,30,50] , [60 , 90,5,2] , [80 , 50,30 , 20 ],[33,44,55,66]])
    print(data)

def countVec():
    c1, c2, c3 = jiebaCut()
    print(c1, c2, c3)

    dict = CountVectorizer();
    data = dict.fit_transform([c1,c2,c3])
    # 统计每一篇文章的字符，数组展示每一个字符出现的次数,单个字母不统计
    print(dict.get_feature_names());
    print(data.toarray())


def tfidf():
    c1, c2, c3 = jiebaCut()
    print(c1, c2, c3)

    tf = TfidfVectorizer();
    data = tf.fit_transform([c1, c2, c3])
    # 统计每一篇文章的字符，数组展示每一个字符出现的次数,单个字母不统计
    print(tf.get_feature_names());
    # 得出重要性
    print(data.toarray())

def jiebaCut():

    cut1 = jieba.cut("我是中国人，我喜欢写代码，我爱编程");
    cut2 = jieba.cut("我是男人，我喜欢女人")
    cut3 = jieba.cut("我是你爸爸，爸爸爱你")

    # 转化为列表
    list1 = list(cut1)
    list2 = list(cut2)
    list3 = list(cut3)

    c1 = " ".join(list1)
    c2 = " ".join(list2)
    c3 = " ".join(list3)

    return c1 , c2 , c3





def dictvec():

    """

    tf:Term frequency ，词的频率
    idf:inverse document frequency ， 逆文档频率
    重要性：ft*idf
    :return:
    """


    dict = DictVectorizer(sparse=False)
    # sparse 矩阵，节约内存，方便数据处理
    data = dict.fit_transform([{'city':'bj','t':100},{'city':'sh','t':80},{'city':'sj','t':60}])
    # 获取特征值
    # 字典数据抽取：把字典中的一些类别数据，转化为特征值
    print(dict.get_feature_names())
    print(data)
    return None;

if __name__ == "__main__":

    # dictvec()
    # countVec()
    # tfidf()
    # MinMax()
    # stand()
    imputer()