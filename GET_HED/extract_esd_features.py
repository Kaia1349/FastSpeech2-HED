import warnings
warnings.filterwarnings("ignore")

import os
import glob
import numpy as np
import librosa
import tgt
from tqdm import tqdm
import opensmile

# 设置采样率
sr = 22050

# 设置路径
wav_dir = "/scratch/s5858895/Summary-Hierarchical-ED/implementation/Dataset/Emotion Speech Dataset"
textgrid_dir = "/scratch/s5858895/FastSpeech2/preprocessed_data/ESD_HED/TextGrid"
feature_dir = "/scratch/s5858895/Summary-Hierarchical-ED/implementation/Dataset/ESD/features"
reset = False

# 初始化 opensmile
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
    sampling_rate=sr,
)

# 获取 TextGrid 的 word-phone 对应信息
def get_words_phones_dir(textgrid):
    tier_w = textgrid.get_tier_by_name("words")
    tier_p = textgrid.get_tier_by_name("phones")
    text_w = [[interval.end_time, interval.text] for interval in tier_w.intervals]
    text_p = [[interval.end_time, interval.text] for interval in tier_p.intervals]
    word_dir = {}
    idx = 0
    for i, (time, word) in enumerate(text_w):
        key = f"{i}-{word}"
        word_dir[key] = []
        while idx < len(text_p):
            time_p, phone = text_p[idx]
            word_dir[key].append(phone)
            idx += 1
            if time_p == time:
                break
    return word_dir

# 创建输出文件夹
os.makedirs(os.path.join(feature_dir, "opensmile"), exist_ok=True)
os.makedirs(os.path.join(feature_dir, "words_phones_dir"), exist_ok=True)

# 获取所有 wav 路径
wav_paths = sorted(glob.glob(os.path.join(wav_dir, "*", "*.wav")))
notexists = []

# 主处理流程
print("🔄 提取 utterance / word / phone 层级特征 ...")
print(f"🎧 采样率设为 {sr} Hz")

for wav_path in tqdm(wav_paths):
    filename = os.path.basename(wav_path).replace(".wav", "")
    speaker = os.path.basename(os.path.dirname(wav_path))
    tg_path = os.path.join(textgrid_dir, speaker, f"{filename}.TextGrid")
    feat_save_path = os.path.join(feature_dir, "opensmile", speaker, f"{filename}.npy")
    worddir_save_path = os.path.join(feature_dir, "words_phones_dir", speaker, f"{filename}.npy")

    if not reset and os.path.exists(feat_save_path) and os.path.exists(worddir_save_path):
        continue

    if not os.path.exists(tg_path):
        notexists.append(tg_path)
        continue

    os.makedirs(os.path.dirname(feat_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(worddir_save_path), exist_ok=True)

    textgrid = tgt.read_textgrid(tg_path)
    audio, sr_loaded = librosa.load(wav_path, sr=None)
    if sr_loaded != sr:
        audio = librosa.resample(audio, orig_sr=sr_loaded, target_sr=sr)

    tier_w = textgrid.get_tier_by_name("words")
    tier_p = textgrid.get_tier_by_name("phones")
    if not tier_w.intervals:
        continue

    alignments = {}

    # utterance-level
    utt_start = int(tier_w.intervals[0].start_time * sr)
    utt_end = int(tier_w.intervals[-1].end_time * sr)
    segment = audio[utt_start:utt_end]
    try:
        feat = np.array(smile.process_signal(segment, sr))
        alignments["utterance"] = feat
    except:
        notexists.append(filename)
        continue

    # word-level
    alignments["words"] = []
    for interval in tier_w.intervals:
        start = int(interval.start_time * sr)
        end = int(interval.end_time * sr)
        segment = audio[start:end]
        try:
            feat = np.array(smile.process_signal(segment, sr))[0]
            alignments["words"].append(feat)
        except:
            break  # 出错就跳过整个文件，保持 clean
    else:
        # phone-level
        alignments["phones"] = []
        for interval in tier_p.intervals:
            start = int(interval.start_time * sr)
            end = int(interval.end_time * sr)
            segment = audio[start:end]
            try:
                feat = np.array(smile.process_signal(segment, sr))[0]
                alignments["phones"].append(feat)
            except:
                break
        else:
            # 全部成功再保存
            np.save(feat_save_path, alignments)
            word_dir = get_words_phones_dir(textgrid)
            np.save(worddir_save_path, word_dir)

print("✅ 提取完成！")
if notexists:
    print("⚠️ 缺失或失败文件列表：")
    for name in notexists:
        print(" -", name)