import pandas as pd
import joblib
import os
import numpy as np
from sklearn.metrics import classification_report

def run_aligned_evaluation():
    # --- 1. 设置路径 ---
    # 获取脚本所在的目录
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 动态构建模型和数据的路径
    model_path = os.path.join(base_dir, "models", "emotion_classifier_pipe_lr.pkl")
    data_path = os.path.join(base_dir, "test_sent_emo.csv")

    # 检查文件是否存在
    if not os.path.exists(model_path):
        print(f"错误：未找到模型文件 {model_path}")
        return
    if not os.path.exists(data_path):
        print(f"错误：未找到测试集文件 {data_path}。请确保从 MELD 下载了 test_sent_emo.csv。")
        return

    # --- 2. 加载模型与测试集 ---
    print(f"正在从 {model_path} 加载模型...")
    pipe_lr = joblib.load(open(model_path, "rb"))

    print(f"正在从 {data_path} 加载数据...")
    test_df = pd.read_csv(data_path)
    
    # MELD 标准列名为 'Utterance' (文本) 和 'Emotion' (标签)
    X_test = test_df['Utterance'].astype(str)
    y_true = test_df['Emotion'].str.lower()

    # --- 3. 执行预测与标签映射 ---
    print("正在进行推理预测...")
    y_pred_raw = pipe_lr.predict(X_test)

    # 核心操作：将你的第8类 'shame' 映射到最接近的 'sadness'
    # MELD 仅含 7 类：neutral, surprise, fear, sadness, joy, disgust, anger
    y_pred_mapped = [label if label!= 'shame' else 'sadness' for label in y_pred_raw]

    # --- 4. 生成学术对标报告 ---
    meld_labels = ['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger']
    
    print("\n" + "="*45)
    print(" MELD 基准测试报告 (8类映射至7类后) ")
    print("="*45)
    
    # 使用映射后的结果计算 Weighted F1 (SCI 论文核心指标)
    report = classification_report(
        y_true, 
        y_pred_mapped, 
        labels=meld_labels, 
        digits=4
    )
    print(report)

if __name__ == "__main__":
    run_aligned_evaluation()