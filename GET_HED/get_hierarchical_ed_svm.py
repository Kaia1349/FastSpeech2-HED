import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import glob
import numpy as np
import os
import pickle
from tqdm import tqdm

# ========== 配置 ==========
emos = ["Angry", "Happy", "Sad", "Surprise"]
emos.sort()
sil_phones = ["sil", "sp", "spn"]

feature_dir = "/scratch/s5858895/Summary-Hierarchical-ED/implementation/Dataset/ESD/features/"
scaler_path = "/scratch/s5858895/Summary-Hierarchical-ED/implementation/parameters/scaler_OpenSMILE.pkl"
hed_extractor_path = "/scratch/s5858895/Summary-Hierarchical-ED/implementation/parameters/linearsvm_OpenSMILE.pkl"
reset = False  # 是否强制重跑

# ========== 函数 ==========
def GetIntensity(scaler, models, feature):
    feature = scaler.transform(feature)
    length_array = []
    for emotion in emos:
        bool_list = np.isnan(feature.astype(float)).sum(axis=1).astype(bool)
        feature[bool_list] = 0
        array = models[emotion].decision_function(feature)
        array[bool_list] = np.nan
        length_array += [array]
    return np.array(length_array)

def get_words_indices(word_dir):
    words_indices = []
    for i, w in enumerate(word_dir):
        words_indices += [i] * len(word_dir[w])
    return np.array(words_indices)

def get_boollist_fastspeech2(words_dir):
    bl = [not (key in sil_phones) for key in [e for wd in words_dir.values() for e in wd]]
    bl1 = bl.copy()
    for idx in range(1, len(bl1)-1):
        if not bl1[idx] and bl1[idx-1]:
            bl1[idx] = True
    bl2 = bl[::-1].copy()
    for idx in range(1, len(bl2)-1):
        if not bl2[idx] and bl2[idx-1]:
            bl2[idx] = True
    return np.array(bl1) * np.array(bl2[::-1])

def GetMinMax_NoOutliers(outputs):
    q1, q3 = np.quantile(outputs, [0.25, 0.75])
    iqr = q3 - q1
    bool_list = (q1 - 1.5 * iqr <= outputs) * (q3 + 1.5 * iqr >= outputs)
    min_ = outputs[bool_list].min()
    max_ = outputs[bool_list].max()
    return min_, max_, bool_list

def normalize_svm(x, min_, max_):
    x[x>0] = x[x>0] / max_
    x[x<0] = -x[x<0] / min_
    return (x+1)/2

# ========== 加载模型 ==========
models = pickle.load(open(hed_extractor_path, 'rb'))
scaler = pickle.load(open(scaler_path, 'rb'))

# ========== Step 1: 生成 raw HED ==========
print("🔄 Step 1: Generating raw HED features...")

files = glob.glob(os.path.join(feature_dir, "opensmile", "**", "*.npy"), recursive=True)
files.sort()

nonexists = []
for path in tqdm(files):
    bn = os.path.basename(path)[:-4]
    savepath = os.path.join(feature_dir, "HED/raw", bn + ".npy")
    if not reset and os.path.exists(savepath):
        continue

    try:
        features = np.load(path, allow_pickle=True).item()
        words_dir = np.load(path.replace("opensmile", "words_phones_dir"), allow_pickle=True).item()
    except Exception:
        nonexists.append(bn)
        continue

    bl = get_boollist_fastspeech2(words_dir)
    words_indices = get_words_indices(words_dir)[bl]

    iw = GetIntensity(scaler, models, features["words"])
    ip = GetIntensity(scaler, models, features["phones"][:len(bl)])[:, bl]
    iu = GetIntensity(scaler, models, features["utterance"])

    iw = iw[:, words_indices]
    iu = np.repeat(iu, ip.shape[1], axis=1)

    full_feature = np.concatenate([ip, iw, iu], axis=0)

    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    np.save(savepath, full_feature)

print(f"❌ Missing or unreadable word_dir for: {nonexists}")

# ========== Step 2: 计算每个情绪的min/max ==========
print("\n🔎 Step 2: Computing min/max per emotion...")

training_files = glob.glob(os.path.join(feature_dir, "HED/raw", "**", "*.npy"), recursive=True)
training_files.sort()

arrays = []
for path in tqdm(training_files):
    feature = np.load(path)
    arrays.append(feature[8:12, 0])  # 取utterance-level emotion

arrays = np.array(arrays)

min_list = []
max_list = []
for e in range(len(emos)):
    bl = (1 - np.isnan(arrays).mean(axis=1).astype(bool)).astype(bool)
    min_, max_, _ = GetMinMax_NoOutliers(arrays[bl][:, e])
    min_list.append(min_)
    max_list.append(max_)
    print(f"Emotion: {emos[e]} | Min: {min_:.3f} | Max: {max_:.3f}")

# ========== Step 3: 归一化 HED特征 ==========
print("\n🎯 Step 3: Normalizing HED features...")

files = glob.glob(os.path.join(feature_dir, "HED/raw", "**", "*.npy"), recursive=True)
files.sort()

for path in tqdm(files):
    bn = os.path.basename(path)[:-4]
    savepath = os.path.join(feature_dir, "HED/normalized", bn + ".npy")
    if not reset and os.path.exists(savepath):
        continue

    try:
        a = np.load(path)
    except Exception:
        continue

    for s, segment in enumerate(["phones", "words"]):
        for e in range(len(emos)):
            b = normalize_svm(a[s*len(emos)+e], min_list[e], max_list[e])
            b = np.clip(b, 0, 1)
            b = pd.Series(b).interpolate(method="linear", limit_direction="both").values
            a[s*len(emos)+e] = b

    for e in range(len(emos)):
        iu = normalize_svm(a[8+e], min_list[e], max_list[e])
        a[8+e] = np.clip(iu, 0, 1)

    a[np.isnan(a)] = 0
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    np.save(savepath, a)

print("\n✅ HED特征提取与归一化完成！")