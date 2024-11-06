import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from models.translator import TranslatorModel
from dataset.MSLTDataset import MSLTDataset
from config import config

from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning import seed_everything
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm


import torch
torch.set_float32_matmul_precision('high')

def parse_arguments():
    parser = ArgumentParser()
    
    # General Training Arguments
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to load a checkpoint to resume training")
    parser.add_argument("--ckpt_dir", type=str, default="../checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--max_epochs", type=int, default=10, help="Maximum number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and validation")
    parser.add_argument("--gradient_clip_val", type=float, default=None, help="Value for gradient clipping")
    parser.add_argument("--accelerator", type=str, default="gpu", choices=["gpu", "cpu"], help="Accelerator to use")
    parser.add_argument("--log_type", type=str, default="wandb", choices=["wandb", "tensorboard"], help="Logger type")
    parser.add_argument("--learning_rate", type=float, default=config.model['adamw_config'].get('lr', 4e-4), help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=config.model['adamw_config'].get('weight_decay', 1e-4), help="Weight decay for optimizer")
    
    return parser.parse_args()


def setup_tokenizers():
    # Load tokenizers for source and target language based on config
    try:
        src_tokenizer = AutoTokenizer.from_pretrained(config.SRC_MODEL_NAME)
        tgt_tokenizer = AutoTokenizer.from_pretrained(config.TGT_MODEL_NAME)
    except Exception as e:
        print(f"Error loading tokenizers: {e}")
        raise
    return src_tokenizer, tgt_tokenizer


def setup_logger(log_type="wandb", project_name="Machine Translation", run_name="Transformer"):
    # Setup logging with options for Weights & Biases or TensorBoard
    if log_type == "wandb":
        logger = WandbLogger(project=project_name, name=run_name, version=0)
    elif log_type == "tensorboard":
        logger = TensorBoardLogger(save_dir="logs", name=run_name)
    else:
        raise ValueError(f"Unsupported logger type: {log_type}")
    return logger


def setup_data_module(src_tokenizer, tgt_tokenizer, max_length, batch_size):
    data_dir_en = "../data/MSLT_Corpus/Data/MSLT_Dev_EN"
    data_dir_zh = "../data/MSLT_Corpus/Data/MSLT_Dev_ZH"

    # Initialize MSLTDataset with tokenizers and configuration parameters
    dataset = MSLTDataset(
        data_dir_en=data_dir_en,
        data_dir_zh=data_dir_zh,
        tokenizer=src_tokenizer,
        max_length=max_length
    )
    # Return DataLoader with the dataset and batch_size
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)


def main():
    args = parse_arguments()
    
    # Set the random seed for reproducibility
    seed_everything(config.SEED)
    
    # Tokenizers for source and target languages
    src_tokenizer, tgt_tokenizer = setup_tokenizers()

    # Initialize model with parameters from config and optional overrides
    model = TranslatorModel()

    # Setup data module
    train_loader = setup_data_module(
        src_tokenizer, tgt_tokenizer, config.MAX_SEQ_LEN, args.batch_size
    )

    # Setup logger
    logger = setup_logger(args.log_type)

    # Configure checkpointing and learning rate monitoring
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.ckpt_dir, 
        monitor="val/acc", 
        mode="max", 
        save_top_k=1, 
        save_last=True,
        filename="best-checkpoint-{epoch:02d}-{val/acc:.2f}"
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Setup trainer
    trainer = Trainer(
        accelerator=args.accelerator,
        max_epochs=args.max_epochs,
        gradient_clip_val=args.gradient_clip_val,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        enable_checkpointing=True,
    )

    # Start training
    print("Starting training...")
    trainer.fit(model, train_loader, ckpt_path=args.ckpt_path)
    print("Training completed.")


if __name__ == "__main__":
    main()
