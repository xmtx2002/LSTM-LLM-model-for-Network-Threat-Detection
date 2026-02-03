import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ==========================================
# 1. 全局配置
# ==========================================
DATA_DIR = r"D:\LSTM\data"
IMG_SAVE_DIR = r"D:\LSTM\research_plots"
MODEL_SAVE_PATH = r"D:\LSTM\zeroday_lstm.h5"
SEQ_LENGTH = 10  # 时间步长
SEQ_OVERLAP = 5  # 滑动窗口重叠
MAX_SAMPLES_PER_CLASS = 50000  # 限制每类样本最大数量，防止内存爆炸

if not os.path.exists(IMG_SAVE_DIR):
    os.makedirs(IMG_SAVE_DIR)


# ==========================================
# 2. 数据加载与清洗
# ==========================================
def load_and_clean_data(data_dir):
    print(f">>> Scanning for CSV files in {data_dir}...")
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))

    if not csv_files:
        raise ValueError(f"No CSV files found in {data_dir}")

    df_list = []
    for file in csv_files:
        print(f"    Loading: {os.path.basename(file)}...")
        try:
            # CIC-IDS2017 数据量巨大，这里不限制读取行数，但稍后会采样
            temp_df = pd.read_csv(file, low_memory=False)

            # 清理列名（去除首尾空格，替换特殊字符）
            temp_df.columns = temp_df.columns.str.strip().str.replace(' ', '_')

            df_list.append(temp_df)
        except Exception as e:
            print(f"    Skipping {file}: {e}")

    if not df_list: raise ValueError("No valid data loaded.")

    print(">>> Concatenating DataFrames...")
    df = pd.concat(df_list, ignore_index=True)

    # 基础清洗
    print(">>> Cleaning Data (Inf/NaN)...")
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # 删除无关列
    cols_to_drop = ['Timestamp', 'Flow_ID', 'Src_IP', 'Dst_IP', 'Source_IP', 'Destination_IP']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

    print(f">>> Total Records Loaded: {len(df)}")
    print(">>> Available Raw Labels:", df['Label'].unique())
    return df


def create_zero_day_split(df):
    """
    构造 Zero-Day 场景：
    训练集 = Benign + DoS/DDoS (已知攻击)
    测试集 = Benign + Web Attacks (未知攻击)
    """
    print("\n>>> Constructing Zero-Day Scenario (CIC-IDS2017)...")

    # 1. 定义攻击类别关键词 (CIC-IDS2017 专用)
    # 训练集：流量型攻击
    train_keywords = ['DoS', 'DDoS', 'Heartbleed']
    # 测试集：Web 攻击 (通常包含 Web Attack, Sql, Xss)
    test_keywords = ['Web', 'Sql', 'XSS']

    # 2. 初始化类型标签
    df['Experiment_Type'] = 'Ignore'  # 默认为忽略（如 PortScan, Bot 等不参与此实验）

    # 标记 Benign
    df.loc[df['Label'].str.contains('BENIGN', case=False, na=False), 'Experiment_Type'] = 'Benign'

    # 标记 Train_Attack (DoS)
    for kw in train_keywords:
        df.loc[df['Label'].str.contains(kw, case=False, na=False), 'Experiment_Type'] = 'Train_Attack'

    # 标记 Test_Attack (Web)
    for kw in test_keywords:
        df.loc[df['Label'].str.contains(kw, case=False, na=False), 'Experiment_Type'] = 'Test_Attack'

    # 过滤掉不需要的数据 (如 Bot, PortScan)
    df = df[df['Experiment_Type'] != 'Ignore'].copy()

    print(">>> Selected Data Distribution:")
    print(df['Experiment_Type'].value_counts())

    # 3. 切分与平衡
    # 训练集：Benign + DoS
    df_train = df[df['Experiment_Type'].isin(['Benign', 'Train_Attack'])].copy()
    # 标签：Attack=1, Benign=0
    df_train['Label_Enc'] = df_train['Experiment_Type'].apply(lambda x: 1 if x == 'Train_Attack' else 0)

    # 测试集：Benign + Web
    df_test = df[df['Experiment_Type'].isin(['Benign', 'Test_Attack'])].copy()
    # 标签：Attack=1, Benign=0 (这里即便Web攻击我们没见过，但在测试集Ground Truth里它是1)
    df_test['Label_Enc'] = df_test['Experiment_Type'].apply(lambda x: 1 if x == 'Test_Attack' else 0)

    # 4. 下采样平衡数据 (防止 Benign 太多淹没攻击)
    def balance_data(temp_df):
        g = temp_df.groupby('Label_Enc')
        # 取每类最小数量，或者设置的上限
        target_size = min(g.size().min(), MAX_SAMPLES_PER_CLASS)
        return g.apply(lambda x: x.sample(n=target_size, random_state=42)).reset_index(drop=True)

    print("\n>>> Balancing Training Set...")
    df_train = balance_data(df_train)

    print(">>> Balancing Test Set...")
    df_test = balance_data(df_test)

    print(f"\n>>> Final Datasets:")
    print(f"    Train Set (DoS vs Benign): {len(df_train)} rows")
    print(f"    Test Set (Web vs Benign):  {len(df_test)} rows")

    return df_train, df_test


def generate_sequences(df, seq_len, overlap, scaler=None):
    # 提取特征列 (排除非数值列)
    feature_cols = [c for c in df.columns if c not in ['Label', 'Experiment_Type', 'Label_Enc']]

    # 确保所有列都是数值型
    X = df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values
    y = df['Label_Enc'].values

    # 归一化
    if scaler is None:
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)

    X_seq, y_seq = [], []
    for i in range(0, len(X) - seq_len + 1, overlap):
        X_seq.append(X[i: i + seq_len])
        # 标签取序列最后一个时间步的标签
        y_seq.append(y[i + seq_len - 1])

    return np.array(X_seq), np.array(y_seq), scaler


# ==========================================
# 3. 绘图与模型
# ==========================================
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pred Benign', 'Pred Attack'],
                yticklabels=['True Benign', 'True Attack'])
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(os.path.join(IMG_SAVE_DIR, f"{title.replace(' ', '_')}.png"))
    plt.close()


def save_history_plot(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Train Acc (DoS)', linestyle='--')
    plt.plot(history.history['val_accuracy'], label='Test Acc (Web/Zero-Day)', linewidth=2.5, color='red')
    plt.title('Zero-Day Detection Failure Proof')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(IMG_SAVE_DIR, "zeroday_training_curve.png"))
    plt.close()


# ==========================================
# 4. 主程序
# ==========================================
if __name__ == "__main__":
    try:
        # 1. 加载数据
        df = load_and_clean_data(DATA_DIR)

        # 2. 构造 Zero-Day 切分
        train_df, test_df = create_zero_day_split(df)

        if len(train_df) == 0 or len(test_df) == 0:
            raise ValueError("Data split failed. Check if CSVs contain 'DoS' and 'Web Attack' labels.")

        # 3. 生成序列
        print("\n>>> Generating sequences (this may take a moment)...")
        # 对训练集拟合 Scaler
        X_train, y_train, scaler = generate_sequences(train_df, SEQ_LENGTH, SEQ_OVERLAP, scaler=None)
        # 对测试集使用相同的 Scaler (模拟真实场景)
        X_test, y_test, _ = generate_sequences(test_df, SEQ_LENGTH, SEQ_OVERLAP, scaler=scaler)

        # *** 关键修复：强制指定 num_classes=2 ***
        # 这样即使测试集中某种类别样本很少，形状也能对齐
        y_train_enc = to_categorical(y_train, num_classes=2)
        y_test_enc = to_categorical(y_test, num_classes=2)

        print(f"Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")

        # 4. 构建模型
        print("\n>>> Building LSTM Model...")
        model = Sequential([
            Input(shape=(SEQ_LENGTH, X_train.shape[2])),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(2, activation='softmax')
        ])

        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        # 5. 训练
        print("\n>>> Starting Training (Training on DoS, Validating on Web Attack)...")
        history = model.fit(
            X_train, y_train_enc,
            epochs=20,  # 演示用 20 轮足够了
            batch_size=64,
            validation_data=(X_test, y_test_enc),
            verbose=1
        )

        # 6. 保存与评估
        model.save(MODEL_SAVE_PATH)
        save_history_plot(history)

        print("\n>>> Generating Final Report...")
        y_pred_prob = model.predict(X_test)
        y_pred = np.argmax(y_pred_prob, axis=1)

        print("\n=== Classification Report (Zero-Day Test Set) ===")
        # 注意：这里 label 1 代表 Web Attack
        print(classification_report(y_test, y_pred, target_names=['Benign', 'Web Attack']))

        plot_confusion_matrix(y_test, y_pred, "Zero-Day Confusion Matrix")
        print(f"\nResults saved to: {IMG_SAVE_DIR}")
        print("Experiment Complete. Check the plots!")

    except Exception as e:
        print(f"\n!!! An error occurred: {e}")
        import traceback

        traceback.print_exc()