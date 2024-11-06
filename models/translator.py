import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

import torch
import numpy as np
from config import config
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from models.transformer import Transformer
from transformers import AutoTokenizer
import torch.nn as nn
from scripts.utils.heap import PriorityQueue
from scripts.utils.text_processing import fix_text_content

class TranslatorModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # Initialize Transformer with configurations from config.py
        self.model = Transformer(
            src_vocab_size=config.VOCAB_SIZE,
            tgt_vocab_size=config.VOCAB_SIZE,
            N=config.NUM_LAYERS,
            d_model=config.D_MODEL,
            max_seq_len=config.MAX_SEQ_LEN,
            d_ff=config.D_MODEL * 4,
            head=config.NUM_HEADS,
            dropout=config.DROPOUT,
        )
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0) # ignore the [PAD]
        
        self.src_tokenizer = AutoTokenizer.from_pretrained(config.SRC_MODEL_NAME)
        self.tgt_tokenizer = AutoTokenizer.from_pretrained(config.TGT_MODEL_NAME)

    def forward(self, src_input_ids, tgt_input_ids, src_mask, tgt_mask):
        return self.model(src_input_ids, tgt_input_ids, src_mask, tgt_mask)

    def training_step(self, batch, batch_idx):
        # (batch_size, seq_len)
        src_input_ids, tgt_input_ids = batch["src_input_ids"], batch["tgt_input_ids"]
        src_attention_mask, tgt_attention_mask = batch["src_attention_mask"], batch["tgt_attention_mask"]
        
        # Create the subsequent mask for target input
        tgt_seq_len = tgt_input_ids.size(1) - 1
        tgt_subsequent_mask = self.subsequent_mask(tgt_seq_len).to(tgt_input_ids.device)
        
        # print("src_input_ids shape:", src_input_ids.shape)
        # print("tgt_input_ids shape:", tgt_input_ids.shape)
        # print("src_attention_mask shape:", src_attention_mask.shape)
        # print("tgt_attention_mask shape:", tgt_attention_mask.shape)
        # print("tgt_subsequent_mask shape:", tgt_subsequent_mask.shape)
        
        # logits: (batch_size, seq_length, vocab_size)
        tgt_attention_mask = tgt_attention_mask[:, None, :-1] & tgt_subsequent_mask # Combined with the subs msk
        logits = self(src_input_ids, tgt_input_ids[:, :-1], src_attention_mask, tgt_attention_mask)
        
        # Loss calculation
        loss = self.loss_fn(logits.view(-1, logits.size(-1)), tgt_input_ids[:, 1:].reshape(-1))
        
        if batch_idx % 200 == 0:
            self.log("train_loss", loss, on_epoch=True, on_step=True)
            
        return loss
    
    def subsequent_mask(self, size):
        attn_shape = (1, size, size)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).bool()
        return subsequent_mask
    
    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=config.LR,
            weight_decay=config.WEIGHT_DECAY,
        )
        scheduler = StepLR(optimizer, **config.model['scheduler_config'])
        return [optimizer], [scheduler]



    @torch.no_grad()
    def greedy_translate(self, text: str, direction: str = "[EN-TO-ZH]", max_translation_length: int = 100):
        self.eval()
        
        text = fix_text_content(direction + " " + text)

        src_token_ids, src_attention_mask = self.src_tokenizer(
            text, return_token_type_ids=False, return_tensors="pt"
        ).values()
        
        device = next(self.parameters()).device
        src_token_ids = src_token_ids.to(device)
        src_attention_mask = src_attention_mask.to(device)
        
        tgt_token_ids = torch.tensor([[self.tgt_tokenizer.cls_token_id]], device=device)
        tgt_attention_mask = torch.tensor([[1]], device=device)

        for _ in range(max_translation_length):
            logits = self(src_token_ids, tgt_token_ids, src_attention_mask, tgt_attention_mask)
            next_tgt_token_id = torch.argmax(logits[:, -1, :], keepdim=True, dim=-1)
            tgt_token_ids = torch.cat([tgt_token_ids, next_tgt_token_id], dim=-1)
            tgt_attention_mask = torch.cat(
                [tgt_attention_mask, torch.ones_like(next_tgt_token_id, dtype=torch.int64)], dim=-1
            )

            if next_tgt_token_id == self.tgt_tokenizer.sep_token_id:
                break

        return self.tgt_tokenizer.decode(tgt_token_ids[0], skip_special_tokens=True)

    @torch.no_grad()
    def beam_translate(self, text: str, direction: str = "[EN-TO-ZH]", max_translation_length: int = 50, beam_size: int = 3):
        self.eval()
        
        text = fix_text_content(direction + " " + text)

        src_token_ids, src_attention_mask = self.src_tokenizer(
            text, return_token_type_ids=False, return_tensors="pt"
        ).values()
        
        device = next(self.parameters()).device
        src_token_ids = src_token_ids.to(device)
        src_attention_mask = src_attention_mask.to(device)

        tgt_token_ids = torch.tensor([[self.tgt_tokenizer.cls_token_id]], device=device)
        tgt_attention_mask = torch.tensor([[1]], device=device)

        heap = PriorityQueue(key=lambda x: x[0], mode="min")
        heap.push((1.0, tgt_token_ids, tgt_attention_mask, False))

        ret = []
        for _ in range(max_translation_length):
            while len(heap) > beam_size:
                heap.pop()

            norm_prob = 0
            mem = []

            while not heap.empty():
                tgt_seq_prob, tgt_token_ids, tgt_attention_mask, completed = heap.pop()

                if completed:
                    ret.append(
                        (tgt_seq_prob.item(), self.tgt_tokenizer.decode(tgt_token_ids.squeeze_(), skip_special_tokens=True))
                    )

                    if len(ret) == beam_size:
                        return ret
                    continue

                norm_prob += tgt_seq_prob

                logits = self(src_token_ids, tgt_token_ids, src_attention_mask, tgt_attention_mask)

                # (vocab_size,)
                token_probs = torch.softmax(logits[:, -1, :], dim=-1).squeeze_()

                # P(Tn | T1:Tn-1)
                top_k_token_probs, top_k_token_ids = torch.topk(token_probs, beam_size, largest=True)

                for i in range(beam_size):
                    next_token_id = top_k_token_ids[i]
                    next_token_prob = top_k_token_probs[i]
                    completed = next_token_id == self.tgt_tokenizer.sep_token_id

                    mem.append(
                        (
                            tgt_seq_prob * next_token_prob,
                            torch.cat((tgt_token_ids, next_token_id.view(1, 1)), dim=-1),
                            torch.cat([tgt_attention_mask, torch.ones((1, 1), device=device)], dim=-1),
                            completed,
                        )
                    )

            for tgt_seq_prob, tgt_token_ids, tgt_attention_mask, completed in mem:
                tgt_seq_prob /= norm_prob  # normalize
                heap.push((tgt_seq_prob, tgt_token_ids, tgt_attention_mask, completed))

        while len(ret) < beam_size and not heap.empty():
            tgt_seq_prob, tgt_token_ids, tgt_attention_mask, completed = heap.pop()

            decoded_seq = self.tgt_tokenizer.decode(tgt_token_ids[0], skip_special_tokens=True)
            ret.append((tgt_seq_prob.item(), decoded_seq))

        return ret


# =================================================================================================================
def test_translator():
    # Create an instance of the TranslatorModel
    print("Starting test_translator...")
    model = TranslatorModel()
    
    # Create dummy input
    batch_size = 8
    src_seq_length = 64
    tgt_seq_length = 64
        
    # Create random input data
    src_input_ids = torch.randint(1, config.VOCAB_SIZE, (batch_size, src_seq_length))
    tgt_input_ids = torch.randint(1, config.VOCAB_SIZE, (batch_size, tgt_seq_length))
    src_mask = torch.ones((batch_size, src_seq_length), dtype=torch.bool)
    tgt_attention_mask = torch.ones((batch_size, tgt_seq_length), dtype=torch.bool)
    
    # Generate tgt_subsequent_mask
    tgt_seq_len = tgt_input_ids.size(1) - 1
    tgt_subsequent_mask = model.subsequent_mask(tgt_seq_len).to(tgt_input_ids.device)
    
    # Expand tgt_attention_mask and combine with tgt_subsequent_mask
    tgt_attention_mask = tgt_attention_mask[:, None, :-1] & tgt_subsequent_mask
    
    # Print shapes to confirm broadcasting works as expected
    print("tgt_attention_mask shape:", tgt_attention_mask.shape)
    print("Expected shape:", (batch_size, 1, tgt_seq_len, tgt_seq_len))
    
    # Forward pass to check for errors
    try:
        logits = model(src_input_ids, tgt_input_ids[:, :-1], src_mask, tgt_attention_mask)
        print("Logits shape:", logits.shape)
        assert logits.shape == (batch_size, tgt_seq_len, config.VOCAB_SIZE), "Unexpected logits shape"
        print("Forward pass successful, logits shape is as expected.")
    except Exception as e:
        print("Error during forward pass:", e)

    # Test loss calculation
    try:
        loss = model.loss_fn(logits.view(-1, logits.size(-1)), tgt_input_ids[:, 1:].reshape(-1))
        print("Loss value:", loss.item())
    except Exception as e:
        print("Error during loss computation:", e)



def __main__():
    test_translator()

if __name__ == "__main__":
    __main__()

