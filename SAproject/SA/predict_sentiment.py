import jieba
import joblib
import os

#预测阶段，可以输入任意中文句子，程序将使用训练好的模型进行分析，并输出该句子是正向或负向的情感标签。

def predict_sentiment(model_path='sentiment_model.pkl', vectorizer_path='vectorizer.pkl'):
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        print("模型或向量化器文件不存在，请先训练模型。")
        return

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    print("输入中文句子进行情感预测（输入 'exit' 退出）：")
    while True:
        review = input("> ")
        if review.lower() == 'exit':
            break
        tokens = " ".join(jieba.lcut(review))
        X = vectorizer.transform([tokens])
        pred = model.predict(X)[0]
        print(f"预测情感标签: {pred}")

if __name__ == '__main__':
    predict_sentiment()
