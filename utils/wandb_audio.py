import wandb
import os


def log_audio(wav, tmp_path, wandb_result, delete_res=True):
    with open(tmp_path, "wb") as f:
        f.write(wav.data)
    wandb.log({wandb_result: wandb.Audio(tmp_path, sample_rate=22050)})
    if delete_res:
        os.remove(tmp_path)
