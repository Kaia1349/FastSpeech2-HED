import argparse
import os
import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.model import get_model, get_vocoder, get_param_num
from utils.tools import to_device, log, synth_one_sample
from model import FastSpeech2Loss
from dataset import Dataset
from evaluate import evaluate
from utils.tools import plot_mel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args, configs):
    print("Prepare training ...")
    preprocess_config, model_config, train_config = configs

    # Get dataset
    dataset = Dataset("train.txt", preprocess_config, train_config, sort=True, drop_last=True)
    batch_size = train_config["optimizer"]["batch_size"]
    group_size = 4
    assert batch_size * group_size < len(dataset)

    loader = DataLoader(
        dataset,
        batch_size=batch_size * group_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        num_workers=4,
        pin_memory=True
    )

    # Prepare model
    model, optimizer = get_model(args, configs, device, train=True)
    model = nn.DataParallel(model)
    num_param = get_param_num(model)
    Loss = FastSpeech2Loss(preprocess_config, model_config).to(device)
    print("Number of FastSpeech2 Parameters:", num_param)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Init logger
    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)
    train_log_path = os.path.join(train_config["path"]["log_path"], "train")
    val_log_path = os.path.join(train_config["path"]["log_path"], "val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)
    train_logger = SummaryWriter(train_log_path)
    val_logger = SummaryWriter(val_log_path)

    # Training setup
    step = args.restore_step + 1
    epoch = 1
    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    total_step = train_config["step"]["total_step"]
    log_step = 50
    save_step = 1000
    synth_step = 1000
    val_step = 1000

    outer_bar = tqdm(total=total_step, desc="Training", position=0)
    outer_bar.n = args.restore_step
    outer_bar.update()

    while True:
        inner_bar = tqdm(total=len(loader), desc=f"Epoch {epoch}", position=1)
        for batch_group in loader:
            if batch_group is None:
                continue

            for batch in batch_group:
                if not isinstance(batch, (list, tuple)) or len(batch) < 13:
                    continue
                try:
                    batch = to_device(batch, device)

                    # ✅ 控制 HED 注入策略
                    if step < 16000:
                        hed_input = None
                    else:
                        hed_input = batch[12]

                    # ✅ Forward with step (用于 HED 缩放)
                    output = model(
                        speakers=batch[2],
                        texts=batch[3],
                        src_lens=batch[4],
                        max_src_len=batch[5],
                        mels=batch[6],
                        mel_lens=batch[7],
                        max_mel_len=batch[8],
                        p_targets=batch[9],
                        e_targets=batch[10],
                        d_targets=batch[11],
                        hed=hed_input,
                        step=step,  # ✅ 传 step 给模型
                    )

                    # Loss
                    losses = Loss(batch, output)
                    total_loss = losses[0] / grad_acc_step
                    total_loss.backward()

                    if step % grad_acc_step == 0:
                        nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)
                        optimizer.step_and_update_lr()
                        optimizer.zero_grad()

                    if step % log_step == 0:
                        loss_vals = [l.item() for l in losses]
                        msg1 = f"Step {step}/{total_step}, "
                        msg2 = (
                            "Total Loss: {:.4f}, Mel: {:.4f}, Postnet: {:.4f}, "
                            "Pitch: {:.4f}, Energy: {:.4f}, Duration: {:.4f}"
                        ).format(*loss_vals)
                        outer_bar.write(msg1 + msg2)
                        with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                            f.write(msg1 + msg2 + "\n")
                        log(train_logger, step, losses=loss_vals)

                    if step % synth_step == 0:
                        fig, wav_recon, wav_pred, tag = synth_one_sample(
                            batch, output, vocoder, model_config, preprocess_config
                        )
                        sr = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
                        log(train_logger, fig=fig, tag=f"Training/step_{step}_{tag}")
                        log(train_logger, audio=wav_recon, sampling_rate=sr, tag=f"Training/step_{step}_{tag}_reconstructed")
                        log(train_logger, audio=wav_pred, sampling_rate=sr, tag=f"Training/step_{step}_{tag}_synthesized")

                    if step % val_step == 0:
                        model.eval()
                        msg = evaluate(model, step, configs, val_logger, vocoder)
                        outer_bar.write(msg)
                        with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                            f.write(msg + "\n")
                        model.train()

                    if step % save_step == 0:
                        ckpt_path = os.path.join(train_config["path"]["ckpt_path"], f"{step}.pth.tar")
                        torch.save(
                            {
                                "model": model.module.state_dict(),
                                "optimizer": optimizer._optimizer.state_dict(),
                            },
                            ckpt_path,
                        )

                    if step == total_step:
                        print("✅ Training complete.")
                        return

                    step += 1
                    outer_bar.update(1)

                except Exception as e:
                    print(f"[ERROR] Exception during training loop: {e}")
                    continue
            inner_bar.update(1)
        epoch += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument("--baseline_ckpt", type=str, default=None)
    parser.add_argument("-p", "--preprocess_config", type=str, required=True)
    parser.add_argument("-m", "--model_config", type=str, required=True)
    parser.add_argument("-t", "--train_config", type=str, required=True)
    args = parser.parse_args()

    preprocess_config = yaml.load(open(args.preprocess_config, "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    main(args, configs)