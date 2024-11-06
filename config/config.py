SEED = 188
D_MODEL = 256
MAX_SEQ_LEN = 64
VOCAB_SIZE = 105879
DROPOUT = 0.1
NUM_LAYERS = 6
NUM_HEADS = 8
LR = 4e-4
FINETUNE_LR = 1e-5
WEIGHT_DECAY = 1e-4
STEP_SIZE = 1
GAMMA = 0.2
SRC_MODEL_NAME = "../bert-base/bert-base-multilingual-cased"
TGT_MODEL_NAME = "../bert-base/bert-base-multilingual-cased"

model = dict(
    d_model=D_MODEL,
    encoder_config=dict(
        pos_encoding_config=dict(
            max_seq_len=MAX_SEQ_LEN,
            dropout=DROPOUT,
        ),
        num_layers=NUM_LAYERS,
        layer_config=dict(
            attention_config=dict(
                d_model=D_MODEL,
                num_heads=NUM_HEADS,
                dropout=DROPOUT,
            ),
            feed_forward_config=dict(
                d_model=D_MODEL,
                d_feed_forward=D_MODEL * 4,
                dropout=DROPOUT,
            ),
        ),
    ),
    decoder_config=dict(
        pos_encoding_config=dict(
            max_seq_len=MAX_SEQ_LEN,
            dropout=DROPOUT,
        ),
        num_layers=NUM_LAYERS,
        layer_config=dict(
            attention_config=dict(
                d_model=D_MODEL,
                num_heads=NUM_HEADS,
                dropout=DROPOUT,
            ),
            feed_forward_config=dict(
                d_model=D_MODEL,
                d_feed_forward=D_MODEL * 4,
                dropout=DROPOUT,
            ),
        ),
    ),
    adamw_config=dict(
        lr=LR,
        finetune_lr=FINETUNE_LR,
        weight_decay=WEIGHT_DECAY,
    ),
    scheduler_config=dict(
        step_size=STEP_SIZE,
        gamma=GAMMA,
    ),
)
