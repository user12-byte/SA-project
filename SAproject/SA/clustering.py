import pandas as pd
import jieba
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

# 加载数据
print("加载数据...")
df = pd.read_csv(r'D:\Desktop\datasets\weibo_senti_100k.csv')
df = df.drop(columns=['label'])
df = df.head(1000)  # 只取1000条，加快运行，去掉标签列： 因为无监督学习（聚类）不需要先验的类别信息

# 分词
print("分词中...")
def tokenize(text):
    return " ".join(jieba.cut(text))

df['cut_review'] = df['review'].astype(str).apply(tokenize)

# TF-IDF 特征提取
print("提取TF-IDF特征...")
vectorizer = TfidfVectorizer(max_features=300)
X = vectorizer.fit_transform(df['cut_review']).toarray()

# 快速多次初始化的KMeans，把文本分成3类，每一类里的句子内容相似。
# 用不同初始点运行20次，选择效果最好的那次
# 每次最多迭代300轮
n_clusters = 3
print("运行KMeans（多次初始化）...")
kmeans = KMeans(n_clusters=n_clusters, n_init=20, max_iter=300, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_
df['cluster'] = labels

# 降维可视化
print("降维可视化...")
X_embedded = TSNE(n_components=2, random_state=42).fit_transform(X)

plt.figure(figsize=(10, 6))
for i in range(n_clusters):
    plt.scatter(X_embedded[labels == i, 0], X_embedded[labels == i, 1], label=f"Cluster {i}")
plt.title("KMeans 快速聚类结果")
plt.legend()
plt.grid(True)
plt.show()

# 输出前几条结果，输出结果类似于“这句话系统觉得和哪一堆评论比较像”
print("\n前10条评论及其聚类编号：")
for i in range(10):
    print(f"[Cluster {df['cluster'][i]}] {df['review'][i]}")
