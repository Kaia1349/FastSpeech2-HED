import re
import argparse
from string import punctuation

import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from g2p_en import G2p
from pypinyin import pinyin, Style

from utils.model import get_model, get_vocoder
from utils.tools import to_device, synth_samples
from dataset import TextDataset
from text import text_to_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


def preprocess_english(text, preprocess_config):
    text = text.rstrip(punctuation)
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)


def synthesize(model, step, configs, vocoder, batchs, control_values, hed_vector=None):
    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control = control_values

    for batch in batchs:
        batch = to_device(batch, device)
        with torch.no_grad():
            if hed_vector is not None:
                # 构造基础 HED 向量（12维，转为 [1, 12, 1]）
                hed_vector_tensor = torch.FloatTensor(hed_vector).unsqueeze(0).unsqueeze(2).to(device)

                # 第一次 forward（不加 hed）获取 mel 长度
                output_temp = model(
                    *(batch[2:]),
                    hed=None,
                    p_control=pitch_control,
                    e_control=energy_control,
                    d_control=duration_control
                )
                T_mel = output_temp[0].size(1)

                # 构造匹配长度的 HED
                hed = hed_vector_tensor.repeat(1, 1, T_mel)
            else:
                hed = None

            # 正式 forward，加入 HED
            if hed is not None:
                print("[DEBUG] hed repeat shape:", hed.shape)
            output = model(
                *(batch[2:]),
                hed=hed,
                p_control=pitch_control,
                e_control=energy_control,
                d_control=duration_control
            )
          

            synth_samples(
                batch,
                output,
                vocoder,
                model_config,
                preprocess_config,
                train_config["path"]["result_path"],
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, required=True)
    parser.add_argument("--mode", type=str, choices=["batch", "single"], required=True)
    parser.add_argument("--source", type=str, default=None)
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--speaker_id", type=int, default=0)
    parser.add_argument("-p", "--preprocess_config", type=str, required=True)
    parser.add_argument("-m", "--model_config", type=str, required=True)
    parser.add_argument("-t", "--train_config", type=str, required=True)
    parser.add_argument("--pitch_control", type=float, default=1.0)
    parser.add_argument("--energy_control", type=float, default=1.0)
    parser.add_argument("--duration_control", type=float, default=1.0)
    parser.add_argument("--hed_vector", type=float, nargs='+', help="12-dim emotion vector for HED injection")
    parser.add_argument("--baseline_ckpt", type=str, default=None)  # ✅ 添加这行
    args = parser.parse_args()

    assert (args.mode == "single") == (args.text is not None)
    preprocess_config = yaml.load(open(args.preprocess_config), Loader=yaml.FullLoader)
    model_config = yaml.load(open(args.model_config), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    model = get_model(args, configs, device, train=False)
    vocoder = get_vocoder(model_config, device)

    if args.mode == "single":
        ids = raw_texts = [args.text[:100]]
        speakers = np.array([args.speaker_id])
        if preprocess_config["preprocessing"]["text"]["language"] == "en":
            texts = np.array([preprocess_english(args.text, preprocess_config)])
        elif preprocess_config["preprocessing"]["text"]["language"] == "zh":
            texts = np.array([preprocess_mandarin(args.text, preprocess_config)])
        text_lens = np.array([len(texts[0])])
        batchs = [(ids, raw_texts, speakers, texts, text_lens, max(text_lens))]
    else:
        dataset = TextDataset(args.source, preprocess_config)
        batchs = DataLoader(dataset, batch_size=8, collate_fn=dataset.collate_fn)

    control_values = args.pitch_control, args.energy_control, args.duration_control
    synthesize(model, args.restore_step, configs, vocoder, batchs, control_values, args.hed_vector)