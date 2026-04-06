import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import os

#分类器：
#TF-IDF： 关键词越“有代表性”，得分越高。
#朴素贝叶斯： 假设每个词对情感标签的贡献是独立的，计算每个标签的“概率”，选出最大值。
#使用 jieba 进行中文分词，通过 TfidfVectorizer 提取文本特征，然后使用 MultinomialNB 多项式朴素贝叶斯算法进行训练

def train_model(data_path, model_path='sentiment_model.pkl', vectorizer_path='vectorizer.pkl'):
    try:
        data = pd.read_csv(data_path, encoding='utf-8')
    except Exception as e:
        print(f"读取数据时出错: {e}")
        return

    if 'review' not in data.columns or 'label' not in data.columns:
        print("CSV 文件需要包含 'review' 和 'label' 两列。")
        return

    data['tokens'] = data['review'].apply(lambda x: " ".join(jieba.lcut(str(x))))

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data['tokens'])
    y = data['label']

    model = MultinomialNB()
    model.fit(X, y)

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"模型训练完成并保存为: {model_path}, {vectorizer_path}")

if __name__ == '__main__':
    data_path = input("请输入训练数据CSV路径（包含'review'和'label'列）: ")
    train_model(data_path.strip())