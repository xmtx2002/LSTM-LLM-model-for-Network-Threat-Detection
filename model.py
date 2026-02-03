import pandas as pd
import numpy as np
import os
import glob
import requests
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import warnings
from tqdm import tqdm

# ===================== 0. 配置区域 =====================
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 路径配置
DATA_PATH = r"D:\LSTM\data"
OUTPUT_DIR = r"D:\LSTM\model_benchmark_v17_final"
LSTM_MODEL_PATH = r"D:\LSTM\zeroday_lstm.h5"
OLLAMA_API = "http://localhost:11434/api/generate"

# 待测试的 LLM 模型
LLM_CANDIDATES = [
    "llama3_LSTM:latest",
    "DeepSeek-R1_LSTM:latest",
    "gemma2_LSTM:latest",
    "qwen3_LSTM:latest"
]

# 样本设置：50攻击 + 50正常 = 100条
N_SAMPLES = 50

# 混合模型置信度触发区间 (0.3 ~ 0.7 之间交给 LLM)
CONF_LOW = 0.3
CONF_HIGH = 0.7

if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
# 设置绘图风格
plt.style.use('ggplot')

# ===================== 1. 核心工具函数 =====================

# --- 加载 LSTM 模型 ---
print(f"🔧 [Init] Loading LSTM Model...")
lstm_model = None
try:
    # 尝试标准加载
    lstm_model = tf.keras.models.load_model(LSTM_MODEL_PATH)
    print("✅ LSTM Model Loaded Successfully!")
except Exception:
    print(f"⚠️ Standard load failed, trying compile=False...")
    try:
        # 兼容性加载模式
        lstm_model = tf.keras.models.load_model(LSTM_MODEL_PATH, compile=False)
        print("✅ LSTM Model Loaded (No Compile Mode)!")
    except Exception as e:
        print(f"❌ LSTM Load FAILED. Predictions will be 0.")
        print(f"Error: {e}")

# --- 数据预处理 (归一化) ---
scaler = MinMaxScaler()
IS_SCALER_FITTED = False


def preprocess_for_lstm_optimized(df_features):
    global IS_SCALER_FITTED, scaler
    data = df_features.replace([np.inf, -np.inf], 0).fillna(0)

    # 模拟训练时的拟合
    if not IS_SCALER_FITTED:
        scaler.fit(data)
        IS_SCALER_FITTED = True

    scaled_data = scaler.transform(data)
    # Reshape to (Batch, TimeSteps, Features)
    return np.expand_dims(scaled_data, axis=1)


# --- LLM 提示词生成 ---
def get_smart_prompt(model_name, features):
    data_json = json.dumps(features, indent=2)

    # DeepSeek 专用提示词 (侧重推理)
    if "deepseek" in model_name.lower():
        return f"""
        [Role]: Cybersecurity Forensic Analyst.
        [Input Packet]: {data_json}
        [Task]: Determine if this is a 'Web Attack' or 'Benign'.
        [Logic]:
        - Check 'Avg Size'. Web attacks (SQLi/XSS) often have irregular lengths.
        - Check 'Duration'. Port scans are short; DoS is long.
        [Output]: If suspicious, output "ATTACK". If normal, output "BENIGN".
        """

    # Llama3/Qwen/Gemma 通用提示词
    return f"""
    Analyze the following network traffic features: {data_json}
    Classify as "ATTACK" or "BENIGN".
    Rules:
    1. Unexpected high ports + short duration -> Suspicious.
    2. Large total bytes but few packets -> Suspicious (Data Exfiltration).
    3. Standard web ports (80/443) -> Likely Benign unless pattern matches injection.

    Respond JSON: {{ "prediction": "ATTACK" or "BENIGN" }}
    """


# --- 查询 LLM ---
def query_llm(model_name, features):
    try:
        response = requests.post(OLLAMA_API, json={
            "model": model_name,
            "prompt": get_smart_prompt(model_name, features),
            "stream": False,
            "format": "json" if "llama" not in model_name else ""
        }, timeout=20)  # 20秒超时

        if response.status_code == 200:
            txt = response.json()['response'].upper()
            if "ATTACK" in txt: return "ATTACK"
            if "BENIGN" in txt: return "BENIGN"
    except:
        pass
    return "BENIGN"  # 兜底策略：报错当成正常


# --- 加载数据 ---
def load_and_prep_data():
    all_files = glob.glob(os.path.join(DATA_PATH, "*.csv"))
    attack_list, benign_list = [], []

    print("📂 [Init] Loading CSV Data...")
    for f in all_files:
        # 简单过滤文件名
        if "WebAttacks" not in f and "Thursday" not in f: continue
        try:
            df = pd.read_csv(f, encoding='cp1252')
            df.columns = [c.strip() for c in df.columns]

            # 标签识别
            mask_attack = df['Label'].astype(str).apply(lambda x: any(k in x for k in ['Web', 'Sql', 'XSS', 'Brute']))
            attacks = df[mask_attack].copy()
            benigns = df[df['Label'] == 'BENIGN'].copy()

            if not attacks.empty: attack_list.append(attacks)
            if not benigns.empty: benign_list.append(benigns)
        except:
            continue

    if not attack_list or not benign_list:
        print("❌ No valid data found!")
        return None, None, None

    # Isolation Forest 训练 (用前1000条正常数据)
    train_benign = pd.concat(benign_list).head(1000).select_dtypes(include=[np.number]).fillna(0)
    iso_forest = IsolationForest(n_estimators=1000, contamination=0.1, random_state=42)
    iso_forest.fit(train_benign)
    feature_cols = train_benign.columns.tolist()

    # 采样 (50 + 50)
    n_attack = min(len(pd.concat(attack_list)), N_SAMPLES)
    n_benign = min(len(pd.concat(benign_list)), N_SAMPLES)

    test_df = pd.concat([
        pd.concat(attack_list).sample(n=N_SAMPLES, replace=(n_attack < N_SAMPLES), random_state=42),
        pd.concat(benign_list).sample(n=N_SAMPLES, replace=(n_benign < N_SAMPLES), random_state=42)
    ]).sample(frac=1).reset_index(drop=True)

    print(f"📊 Dataset Ready: {len(test_df)} samples (Balanced)")
    return test_df, iso_forest, feature_cols


# ===================== 2. 绘图函数 =====================

def plot_single_model(model_name, metrics):
    """为每个模型画一张单独的图"""
    labels = list(metrics.keys())
    values = list(metrics.values())

    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, values, color=['#4c72b0', '#55a868', '#c44e52', '#8172b3'])

    plt.ylim(0, 1.15)
    plt.title(f"Performance: {model_name}\n(N={N_SAMPLES * 2})")
    plt.ylabel("Score")

    # 标数值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.2f}', ha='center', va='bottom', fontweight='bold')

    safe_name = model_name.split(":")[0]
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"1_Individual_{safe_name}.png"))
    plt.close()


def plot_summary(all_results):
    """汇总对比图"""
    df = pd.DataFrame(all_results).set_index("Model")

    # 使用 seaborn 调色板
    ax = df.plot(kind='bar', figsize=(14, 7), width=0.85, edgecolor='black', alpha=0.9)

    plt.title(f"Benchmark Summary: Hybrid LSTM-LLM vs Baseline (N={N_SAMPLES * 2})", fontsize=16)
    plt.ylabel("Score", fontsize=14)
    plt.ylim(0, 1.2)
    plt.xticks(rotation=0, fontsize=11)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4, fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # 在每个柱子上标数值
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3, fontsize=9, rotation=90)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "2_Summary_Comparison.png"), dpi=300)
    print(f"\n✅ Summary Plot Saved to: {os.path.join(OUTPUT_DIR, '2_Summary_Comparison.png')}")
    plt.close()


# ===================== 3. 主程序 =====================

def main():
    df, iso_model, feature_cols = load_and_prep_data()
    if df is None: return

    # --- 1. 基础模型预测 (LSTM + IsoForest) ---
    print("⚡ Calculating Base Metrics (LSTM & IsoForest)...")
    feat_data = df[feature_cols]

    # LSTM 预测 (带修复逻辑)
    if lstm_model:
        lstm_inputs = preprocess_for_lstm_optimized(feat_data)
        raw_preds = lstm_model.predict(lstm_inputs, verbose=0)

        # [核心修复] 判断输出维度
        if raw_preds.shape[-1] == 2:
            # 如果输出是 [Neg_Prob, Pos_Prob]，取第二列
            lstm_probs = raw_preds[:, 1]
        else:
            # 如果输出是 [Pos_Prob]，直接展平
            lstm_probs = raw_preds.flatten()

        print(f"   🔍 LSTM Output Processed. Shape: {lstm_probs.shape}")
    else:
        lstm_probs = np.zeros(len(df))  # 全0

    iso_preds = iso_model.predict(feat_data.fillna(0))

    # --- 2. 准备 LSTM 基准线数据 ---
    # 真实标签转换 (1=Attack, 0=Benign)
    y_true_global = [1 if any(k in str(label) for k in ['Web', 'Sql', 'XSS', 'Brute']) else 0 for label in df['Label']]
    # LSTM 预测转换
    lstm_preds_binary = [1 if p > 0.5 else 0 for p in lstm_probs]

    summary_metrics = []

    # 计算 LSTM 基准指标
    lstm_metrics = {
        "Accuracy": accuracy_score(y_true_global, lstm_preds_binary),
        "Recall": recall_score(y_true_global, lstm_preds_binary),
        "Precision": precision_score(y_true_global, lstm_preds_binary, zero_division=0),
        "F1": f1_score(y_true_global, lstm_preds_binary)
    }

    summary_entry = lstm_metrics.copy()
    summary_entry["Model"] = "LSTM (Baseline)"
    summary_metrics.append(summary_entry)

    # 画 LSTM 图
    plot_single_model("LSTM_Only", lstm_metrics)
    print(f"   📊 LSTM Baseline F1: {lstm_metrics['F1']:.3f}")

    # --- 3. 遍历 LLM 混合模型 ---
    print("\n🚀 Starting Hybrid LLM Testing...")

    # 预先计算触发条件，减少循环内的计算
    base_records = []
    for i in range(len(df)):
        p = lstm_probs[i]
        iso = iso_preds[i]
        # 触发逻辑: LSTM 不确定 (0.3~0.7) 或 LSTM 漏报但 IsoForest 报警
        trigger = (CONF_LOW < p < CONF_HIGH) or (p < CONF_LOW and iso == -1)
        base_records.append({"prob": p, "trigger": trigger, "row": df.iloc[i]})

    for model_name in LLM_CANDIDATES:
        safe_name = model_name.split(":")[0]
        print(f"👉 Testing: {safe_name}")

        y_pred_hybrid = []
        llm_call_count = 0

        # 进度条
        for rec in tqdm(base_records, leave=False, desc=safe_name):
            if rec['trigger']:
                llm_call_count += 1
                # 准备最关键的特征发给 LLM
                row = rec['row']
                feat = {
                    "Dst Port": int(row.get("Destination Port", 0)),
                    "Duration": float(row.get("Flow Duration", 0)),
                    "Avg Size": float(row.get("Total Length of Fwd Packets", 0)) / max(1, float(
                        row.get("Total Fwd Packets", 0)))
                }
                pred_label = query_llm(model_name, feat)
                y_pred_hybrid.append(1 if pred_label == "ATTACK" else 0)
            else:
                # 没触发 -> 听 LSTM 的
                y_pred_hybrid.append(1 if rec['prob'] > 0.5 else 0)

        # 计算指标
        curr_metrics = {
            "Accuracy": accuracy_score(y_true_global, y_pred_hybrid),
            "Recall": recall_score(y_true_global, y_pred_hybrid),
            "Precision": precision_score(y_true_global, y_pred_hybrid, zero_division=0),
            "F1": f1_score(y_true_global, y_pred_hybrid)
        }

        print(f"   📝 LLM Calls: {llm_call_count} | F1: {curr_metrics['F1']:.3f}")

        # 绘图 & 记录
        plot_single_model(safe_name, curr_metrics)

        summary_entry = curr_metrics.copy()
        summary_entry["Model"] = safe_name
        summary_metrics.append(summary_entry)

    # --- 4. 最终汇总 ---
    print("\n🎨 Generating Summary Chart...")
    plot_summary(summary_metrics)
    print("\n🎉 All Done! Results are in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()