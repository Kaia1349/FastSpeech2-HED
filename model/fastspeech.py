import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import Encoder, Decoder, PostNet
from .modules import VarianceAdaptor
from utils.tools import get_mask_from_lengths


class FastSpeech2(nn.Module):
    """ FastSpeech2 with optional HED injection (delayed and scaled) """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2, self).__init__()
        self.model_config = model_config
        self.encoder = Encoder(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.postnet = PostNet()

        self.speaker_emb = None
        if model_config["multi_speaker"]:
            with open(
                os.path.join(
                    preprocess_config["path"]["preprocessed_path"], "speakers.json"
                ),
                "r",
            ) as f:
                n_speaker = len(json.load(f))
            self.speaker_emb = nn.Embedding(
                n_speaker,
                model_config["transformer"]["encoder_hidden"],
            )

        # ✅ HED projection module
        self.hed_proj = nn.Linear(12, model_config["transformer"]["encoder_hidden"])
        self.hed_act = nn.Tanh()
        self.hed_fuse = nn.Linear(
            model_config["transformer"]["encoder_hidden"] * 2,
            model_config["transformer"]["encoder_hidden"],
        )

        # ✅ initialize HED module weights
        nn.init.xavier_uniform_(self.hed_proj.weight)
        nn.init.xavier_uniform_(self.hed_fuse.weight)

    def forward(
        self,
        speakers,
        texts,
        src_lens,
        max_src_len,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        hed=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
        step=None  # ✅ new: current training step (int or None)
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )

        output = self.encoder(texts, src_masks)

        if self.speaker_emb is not None:
            output = output + self.speaker_emb(speakers).unsqueeze(1).expand(
                -1, max_src_len, -1
            )

        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
        )

        # ✅ HED fusion after variance adaptor (aligned to T_mel)
        if hed is not None:
            hed = hed.transpose(1, 2)  # [B, T, 12]
            hed_embed = self.hed_act(self.hed_proj(hed))  # [B, T, H]

            # ✅ Optional: scale HED based on step (warm-up)
            if step is not None and step < 36000:
                scale = min((step - 16000) / 16000, 1.0) if step >= 16000 else 0.0
                hed_embed = hed_embed * scale
        else:
            B = output.size(0)
            T = mel_lens.max().item() if mel_lens is not None else 100
            H = self.model_config["transformer"]["encoder_hidden"]
            hed_embed = torch.zeros(B, T, H).to(output.device)

        output = torch.cat([output, hed_embed], dim=-1)  # [B, T, 2H]
        output = self.hed_fuse(output)  # [B, T, H]

        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)
        postnet_output = self.postnet(output) + output

        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
        )
