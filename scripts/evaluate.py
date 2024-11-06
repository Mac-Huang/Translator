import sys
import os
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from argparse import ArgumentParser

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from models.translator import TranslatorModel
from dataset.MSLTDataset import MSLTDataset
from config import config


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="../checkpoints/last.ckpt", help="Path to the model checkpoint for evaluation")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to run evaluation on")
    return parser.parse_args()


def load_data(tokenizer, batch_size):
    # Load training data as evaluation samples
    data_dir_en = "../data/MSLT_Corpus/Data/MSLT_Dev_EN"
    data_dir_zh = "../data/MSLT_Corpus/Data/MSLT_Dev_ZH"

    dataset = MSLTDataset(
        data_dir_en=data_dir_en,
        data_dir_zh=data_dir_zh,
        tokenizer=tokenizer,
        max_length=config.MAX_SEQ_LEN
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def evaluate(model, dataloader, device, num_samples=5):
    model.eval()
    model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            if idx >= num_samples:  # Limit to a few samples for quick testing
                break
            
            src_input_ids = batch["src_input_ids"].to(device)
            tgt_input_ids = batch["tgt_input_ids"].to(device)
            src_attention_mask = batch["src_attention_mask"].to(device)
            tgt_attention_mask = batch["tgt_attention_mask"].to(device)

            # Get model predictions
            outputs = model(src_input_ids, tgt_input_ids[:, :-1], src_attention_mask, tgt_attention_mask[:, :-1])

            # Calculate loss
            loss = loss_fn(outputs.view(-1, outputs.size(-1)), tgt_input_ids[:, 1:].reshape(-1))
            print(f"Sample {idx + 1} - Loss: {loss.item()}")

            # Generate predictions
            predictions = torch.argmax(outputs, dim=-1)

            # Decode and print the source, prediction, and ground truth
            src_text = model.src_tokenizer.decode(src_input_ids[0], skip_special_tokens=True)
            tgt_text = model.tgt_tokenizer.decode(tgt_input_ids[0], skip_special_tokens=True)
            pred_text = model.tgt_tokenizer.decode(predictions[0], skip_special_tokens=True)

            print(f"Source: {src_text}")
            print(f"Prediction: {pred_text}")
            print(f"Ground Truth: {tgt_text}")
            print("-" * 50)


def main():
    args = parse_arguments()

    # Load tokenizers and model
    src_tokenizer = AutoTokenizer.from_pretrained(config.SRC_MODEL_NAME)
    tgt_tokenizer = AutoTokenizer.from_pretrained(config.TGT_MODEL_NAME)
    
    # Load the model from checkpoint
    model = TranslatorModel.load_from_checkpoint(
        args.ckpt_path,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        **config.model
    )

    # Load a few training samples as evaluation samples
    dataloader = load_data(src_tokenizer, args.batch_size)

    # Run evaluation
    print("Evaluating on a few training samples...")
    evaluate(model, dataloader, args.device)


if __name__ == "__main__":
    main()
