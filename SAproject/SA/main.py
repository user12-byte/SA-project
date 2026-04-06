#!/usr/bin/env python3
# sentiment_toolkit.py
# 综合情感分析、粗糙集关键词分析、文本聚类工具

import os
import sys
import random
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.feature_selection import chi2
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import KBinsDiscretizer
from roughset import RoughSet

try:
    from roughset import RoughSet
except ImportError:
    RoughSet = None  # 如果没有安装 roughset 库，后续功能会提示安装

# 简单的中文分词预处理函数（示例，可根据实际模型需要调整）
def preprocess_review(review):
    # 这里可以添加中文分词、去除标点符号、去停用词等处理
    # 为示例起见，我们直接返回原文
    return review

# 功能1：情感预测
def sentiment_prediction():
    print("\n=== 情感预测 ===")
    use_default = input("使用默认模型文件路径? (y/n): ").strip().lower()
    if use_default == 'y':
        model_path = "sentiment_model.pkl"
        vect_path = "vectorizer.pkl"
    else:
        model_path = input("请输入情感模型文件路径 (.pkl): ").strip()
        vect_path = input("请输入向量化器文件路径 (.pkl): ").strip()
    if not os.path.exists(model_path) or not os.path.exists(vect_path):
        print("模型文件或向量化器文件不存在，请检查路径。")
        return
    # 加载训练好的模型和向量化器:contentReference[oaicite:6]{index=6}
    model = joblib.load(model_path)
    vectorizer = joblib.load(vect_path)
    while True:
        review = input("请输入要分析的中文句子 (或输入 'quit' 返回菜单): ").strip()
        if review.lower() == 'quit':
            break
        # 文本预处理
        proc = preprocess_review(review)
        # 特征提取
        feat = vectorizer.transform([proc])
        # 情感预测
        try:
            pred = model.predict(feat)[0]
        except Exception as e:
            print("预测错误：", e)
            break
        # 简单格式化输出（根据模型类型，以下示例假定输出字符串）
        if isinstance(pred, str):
            print(f"预测标签: {pred}")
        else:
            # 若为数值标签，可自定义解释
            print(f"预测标签 (numeric): {pred}")

# 功能2：粗糙集关键词分析
def roughset_keyword_analysis():
    print("\n=== 粗糙集关键词分析 ===")
    if RoughSet is None:
        print("缺少 roughset 库，请先运行 `pip install roughset` 并重启程序。")
        return
    use_default = input("使用默认数据集路径? (y/n): ").strip().lower()
    if use_default == 'y':
        data_path = r"D:\Desktop\datasets\weibo_senti_100k.csv"
    else:
        data_path = input("请输入CSV数据集路径: ").strip()
    if not os.path.exists(data_path):
        print("数据集文件不存在，请检查路径。")
        return
    # 读取数据集，假设CSV文件包含列 'review' 和 'label'
    df = pd.read_csv(data_path)
    # 只取前几列适用，如果没有标签列名，可自行更换
    if 'review' not in df.columns or 'label' not in df.columns:
        print("数据集中未发现 'review' 或 'label' 列。")
        return
    # 随机采样部分数据以加速分析
    sample_df = df.sample(n=200, random_state=42) if len(df) > 200 else df.copy()
    reviews = sample_df['review'].astype(str).tolist()
    labels = sample_df['label'].astype(str).tolist()
    # 特征提取：使用 CountVectorizer 或 TfidfVectorizer
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(reviews)
    y = np.array([1 if lab == 'positive' or lab == '1' else 0 for lab in labels])
    # 计算卡方统计量选出前k个关键词:contentReference[oaicite:7]{index=7}
    chi2_stats, p_vals = chi2(X, y)
    feature_names = np.array(vectorizer.get_feature_names_out())
    # 取出分数前10的关键词
    top_k = 10
    top_idx = np.argsort(chi2_stats)[-top_k:]
    top_features = feature_names[top_idx]
    top_scores = chi2_stats[top_idx]
    # 绘制关键词重要性柱状图
    plt.figure(figsize=(8,4))
    plt.barh(range(top_k), top_scores, color='skyblue')
    plt.yticks(range(top_k), top_features)
    plt.xlabel("Chi-square 统计量")
    plt.title("关键词重要性")
    plt.tight_layout()
    plt.show()
    # 粗糙集分析：将关键词作为属性
    # 构造数据表: 属性列为关键词在文档中的 TF-IDF 值，决策列为标签




# 功能3：文本聚类分析
def review_clustering():
    print("\n=== 文本聚类分析 ===")
    use_default = input("使用默认数据集路径? (y/n): ").strip().lower()
    if use_default == 'y':
        data_path = r"D:\Desktop\datasets\weibo_senti_100k.csv"
    else:
        data_path = input("请输入CSV数据集路径: ").strip()
    if not os.path.exists(data_path):
        print("数据集文件不存在，请检查路径。")
        return
    df = pd.read_csv(data_path)
    if 'review' not in df.columns:
        print("数据集中未发现 'review' 列。")
        return
    sample_df = df.sample(n=200, random_state=42) if len(df) > 200 else df.copy()
    reviews = sample_df['review'].astype(str).tolist()
    # TF-IDF 向量化:contentReference[oaicite:9]{index=9}
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(reviews)
    # KMeans 聚类:contentReference[oaicite:10]{index=10}
    try:
        n_clusters = int(input("请输入聚类簇数 (默认3): ") or 3)
    except:
        n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    print("聚类完成，前10个文本及其簇标签：")
    for i, doc in enumerate(reviews[:10]):
        print(f"[簇{labels[i]}] {doc[:50]}...")
    # t-SNE 降维可视化:contentReference[oaicite:11]{index=11}
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X.toarray())
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='tab10', marker='o')
    plt.title("文本聚类 t-SNE 可视化")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.colorbar(scatter, label='Cluster')
    plt.show()

# 主菜单
def main():
    while True:
        print("\n===== 情感分析与文本工具 =====")
        print("1. 情感预测")
        print("2. 粗糙集关键词分析")
        print("3. 文本聚类分析")
        print("0. 退出")
        choice = input("请选择功能编号: ").strip()
        if choice == '1':
            sentiment_prediction()
        elif choice == '2':
            roughset_keyword_analysis()
        elif choice == '3':
            review_clustering()
        elif choice == '0':
            print("退出程序。")
            break
        else:
            print("无效选择，请重新输入。")

if __name__ == "__main__":
    main()
