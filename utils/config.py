import torch
import dataclasses


@dataclasses.dataclass
class TaskConfig:
    work_dir: str = "/home/jupyter/work/resources/tts"  # because we work with ds
    work_dir_LJ: str = "/home/jupyter/mnt/s3/bucket-hse-rw/data/datasets"  # because we work with ds
    model_path: str = "/home/jupyter/work/resources/models"

    aligner: str = "fsa"

    one_batch: bool = False
    PAD_IDX: int = 0

    vocab_size: int = 10000
    mels: int = 80

    phonems: int = 51
    phonems_emb: int = 384

    encoder_N: int = 6
    encoder_attention_head: int = 2
    encoder_pos_emb: int = 384
    encoder_hidden_self_attention: int = 384
    encoder_attention_dropout: float = 0.1
    encoder_filter_size: int = 1536
    encoder_conv_kernel: int = 3

    length_regulator_conv_dim: int = 384
    length_regulator_filter_size: int = 384
    length_regulator_kernel: int = 3
    length_regulator_dropout: float = 0.1

    decoder_input_dim: int = 384
    decoder_N: int = 6
    decoder_attention_head: int = 2
    decoder_pos_emb: int = 384
    decoder_hidden_self_attention: int = 384
    decoder_filter_size: int = 1536
    decoder_conv_kernel: int = 3

    output_linear_in_dim: int = 384

    torch_seed: int = 42

    batch_size: int = 3

    batch_limit: int = -1

    learning_rate: float = 4e-4
    weight_decay: float = 1e-5

    num_epochs: int = 750
    device: torch.device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu')
    wandb_api: str = "99f2c4dae0db3099861ebd92a63e1194f42d16d9"
    wandb_project: str = "tts-final"
    log_audio: bool = True
    laep: int = 107
    laep_model: int = 1
    laep_val: int = 1
    log_final_audio: bool = True
    wandb: bool = True
