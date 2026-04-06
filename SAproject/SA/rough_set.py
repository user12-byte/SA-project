import pandas as pd
import jieba
import re
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

# 分词缓存字典
segmentation_cache = {}

def cached_jieba_lcut(text):
    """带缓存的分词函数"""
    if text in segmentation_cache:
        return segmentation_cache[text]
    
    result = jieba.lcut(text)
    segmentation_cache[text] = result
    return result

# 1. 加载数据（只取部分数据）
df = pd.read_csv(r"D:\Desktop\datasets\weibo_senti_100k.csv", encoding='utf-8')
df = df.sample(n=5000, random_state=42)

# 2. 清洗文本
def clean_text(text):
    text = re.sub(r"\[.*?\]", "", text)  # 去除 [表情]
    text = re.sub(r"[^\u4e00-\u9fa5]", "", text)  # 保留中文
    return text

df['clean'] = df['review'].apply(clean_text)

# 3. 加载停用词表
stopwords = set()
with open(r"D:\Desktop\datasets\stopwords_hit.txt", 'r', encoding='utf-8') as f:
    stopwords = set([line.strip() for line in f if line.strip()])

# 4. 分词 + 去停用词（使用缓存分词）
print("开始分词处理...")
df['tokens'] = df['clean'].apply(lambda x: [w for w in cached_jieba_lcut(x) if w and w not in stopwords])
print(f"分词完成，缓存了 {len(segmentation_cache)} 个不同文本的分词结果")

# 5. 构建词汇表（高频词）
all_words = Counter()
for tokens in df['tokens']:
    all_words.update(tokens)
vocab = [word for word, count in all_words.items() if count >= 5]

# 创建词汇索引字典，避免重复使用 list.index()
vocab_to_index = {word: idx for idx, word in enumerate(vocab)}

# 6. 构建二值词袋模型（优化版本）
N = len(df)
M = len(vocab)
X = np.zeros((N, M), dtype=int)

print("构建词袋模型...")
for i, tokens in enumerate(df['tokens']):
    unique_tokens = set(tokens)  # 去重，避免重复计算
    for w in unique_tokens:
        if w in vocab_to_index:
            X[i, vocab_to_index[w]] = 1

y = df['label'].values

# 7. 粗糙集正域与依赖度计算函数
def positive_region_size(X, y, attrs):
    n = X.shape[0]
    eq_classes = {}
    for i in range(n):
        key = tuple(X[i, attrs]) if attrs else tuple()
        eq_classes.setdefault(key, []).append(i)
    pos_count = 0
    for group in eq_classes.values():
        labels = [y[i] for i in group]
        if all(l == labels[0] for l in labels):
            pos_count += len(group)
    return pos_count

def dependency_degree(X, y, attrs):
    return positive_region_size(X, y, attrs) / X.shape[0]

# 8. 每个词单独计算依赖度
print("计算词汇依赖度...")
importances = []
for j, word in enumerate(vocab):
    gamma = dependency_degree(X, y, [j])
    importances.append((gamma, word))
importances.sort(reverse=True)
global_keywords = [word for gamma, word in importances[:10]]
print("全局高影响力关键词（前10）:", global_keywords)

# 9. 前向贪婪算法实现属性约简
print("执行属性约简...")
selected = []
current_dep = 0
attrs = list(range(len(vocab)))
while True:
    best_attr = None
    best_dep = current_dep
    for a in attrs:
        if a in selected:
            continue
        dep = dependency_degree(X, y, selected + [a])
        if dep > best_dep:
            best_dep = dep
            best_attr = a
    if best_attr is not None:
        selected.append(best_attr)
        current_dep = best_dep
        print(f"选择属性: {vocab[best_attr]}, 当前依赖度: {current_dep:.4f}")
    else:
        break

reduct_keywords = [vocab[a] for a in selected]
print("选出的属性约简关键词:", reduct_keywords)

# 10. 输入句子分析关键词（优化版本，使用缓存）
def extract_keywords_from_sentence(sentence, global_keywords):
    # 清洗文本
    cleaned_sentence = re.sub(r"\[.*?\]", "", sentence)
    cleaned_sentence = re.sub(r"[^\u4e00-\u9fa5]", "", cleaned_sentence)
    
    # 使用缓存分词
    tokens = [w for w in cached_jieba_lcut(cleaned_sentence) if w and w not in stopwords]
    
    # 使用集合交集操作，提高查找效率
    global_keywords_set = set(global_keywords)
    return [w for w in tokens if w in global_keywords_set]

# 示例句子，输入一句话，我们就可以从中提取出“对分类最有影响”的关键词
input_sentence = "电影好烂，太失望了"
result = extract_keywords_from_sentence(input_sentence, global_keywords)
print("该句中影响分类的关键词：", result)

# 11. 可视化依赖度前20词，可视化图展示哪些词在全局上最有影响力
top_n = 20
top_keywords = importances[:top_n]
words = [w for gamma, w in top_keywords]
scores = [gamma for gamma, w in top_keywords]

plt.figure(figsize=(10, 6))
plt.barh(words[::-1], scores[::-1], color='skyblue')
plt.xlabel("粗糙集依赖度 γ")
plt.title("关键词对情感分类的影响力（Top 20）")
plt.tight_layout()
plt.show()

# 输出优化统计信息
print(f"\n优化统计:")
print(f"总共缓存了 {len(segmentation_cache)} 个不同文本的分词结果")
print(f"词汇表大小: {len(vocab)}")
print(f"样本数量: {len(df)}")