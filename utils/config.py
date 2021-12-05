import torch
import dataclasses


@dataclasses.dataclass
class TaskConfig:
    work_dir: str = "/home/jupyter/work/resources/tts"  # tts directory
    work_dir_LJ: str = "/home/jupyter/mnt/s3/bucket-hse-rw/data/datasets"  # directory with LJ dataset
    model_path: str = "/home/jupyter/work/resources/models"  # directory with model results (contain best model and last epoch model)

    aligner: str = "fsa"  # type of aligner fsa - pretrained Fast Speech Aligner; ga - Grapheme Aligner

    one_batch: bool = False  # if you want to train just on one batch

    # Config for Fast Speech model and its blocks

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

    torch_seed: int = 42  # set torch seed for reproduction purpose

    batch_size: int = 3  # set train batchsize, validation batchsize will be calculated as the number of lines in test_text.txt

    batch_limit: int = -1  # set number of batches for training (by setting it -1 all batches will be used)

    learning_rate: float = 4e-4  # max_lr for scheduler
    weight_decay: float = 1e-5  #weight_decay for optimizer

    num_epochs: int = 750  # number of epochs

    device: torch.device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu')  # set device
    wandb_project: str = "tts-final"  # set wandb project name
    log_audio: bool = True  # True if you want to log audio results while training
    laep: int = 107  # Data will be sent to wandb every laep iterations
    laep_model: int = 1  # Model will be saved every laep_model iterations (best model will be saved after every improvement)
    laep_val: int = 1  # Validation will be called every laep_val iterations during training
    wandb: bool = True  # Set False if you Don't want to send any data to wandb
