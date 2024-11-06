# Translation Project - Transformer-based English-Chinese Translator

This project is a Transformer-based English-Chinese translation model, designed to improve translation performance through custom configurations and efficient training strategies. The project leverages PyTorch, PyTorch Lightning, and Weights & Biases for streamlined training, evaluation, and logging.

## Project Structure

The main project files and directories are organized as follows:

```
.
├── bert-base                      # Pretrained BERT models (multilingual and uncased)
├── checkpoints                    # Directory to store model checkpoints
├── config                         # Configuration files for model parameters
├── data                           # Data directory containing the MSLT Corpus (EN-ZH dataset)
│   ├── MSLT_Corpus
│   │   ├── Data
│   │   │   ├── MSLT_Dev_EN
│   │   │   ├── MSLT_Dev_ZH
│   │   │   ├── MSLT_Test_EN
│   │   │   └── MSLT_Test_ZH
│   └── Paper                      # Additional resources and documentation for the MSLT Corpus
├── logs                           # Training and evaluation logs, including Weights & Biases tracking
├── models                         # Model architecture files
├── notebooks                      # Jupyter Notebooks for experiments and visualizations
├── scripts                        # Training, evaluation, and utility scripts
│   ├── dataset
│   │   └── MSLTDataset.py         # Custom dataset for MSLT Corpus
│   ├── utils                      # Utility scripts for data processing and priority queue
├── stopwords                      # Directory for language-specific stopwords
└── README.md                      # Project documentation
```

## Getting Started

### Prerequisites

- Python 3.8+
- [PyTorch](https://pytorch.org/) and [PyTorch Lightning](https://www.pytorchlightning.ai/)
- [Transformers](https://huggingface.co/transformers/) library from Hugging Face
- [Weights & Biases](https://wandb.ai/) for experiment tracking

### Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/translation-project.git
   cd translation-project
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up Weights & Biases (optional):

   ```bash
   wandb login
   ```

### Data Preparation

The project uses the MSLT (Microsoft Speech Language Translation) Corpus, with directories for both English and Chinese datasets (`data/MSLT_Corpus`). Organize your data files within `data/MSLT_Corpus/Data/MSLT_Dev_EN` and `data/MSLT_Corpus/Data/MSLT_Dev_ZH` for training and evaluation.

### Model Configuration

The model configurations are defined in `config/config.py`. Key settings include:
- **Model Dimensions**: `D_MODEL`, `NUM_LAYERS`, and `NUM_HEADS`
- **Training Hyperparameters**: `LR` (learning rate), `WEIGHT_DECAY`, and `DROPOUT`
- **Tokenizer Paths**: `SRC_MODEL_NAME` and `TGT_MODEL_NAME`

These configurations can be customized as needed to improve performance.

## Usage

### Training

To train the model from scratch, use the `train.py` script:

```bash
python scripts/train.py --max_epochs 10 --batch_size 16 --log_type "wandb"
```

Additional options for `train.py`:
- `--ckpt_path`: Load a checkpoint to resume training
- `--accelerator`: Use `"gpu"` or `"cpu"`
- `--learning_rate`: Override the default learning rate

### Evaluation

The `evaluate.py` script can be used to evaluate the model on the test set:

```bash
python scripts/evaluate.py --ckpt_path checkpoints/best-checkpoint.ckpt --batch_size 8
```

The evaluation script provides metrics such as translation loss and displays sample predictions.

### Translation Inference

Use the `greedy_translate` or `beam_translate` methods in `TranslatorModel` for generating translations. Here’s an example of using greedy translation:

```python
from models.translator import TranslatorModel

model = TranslatorModel.load_from_checkpoint("checkpoints/best-checkpoint.ckpt")
model.eval()
text = "Hello, how are you?"
translation = model.greedy_translate(text, direction="[EN-TO-ZH]")
print(f"Translation: {translation}")
```

## Logging and Monitoring

- **Weights & Biases**: For experiment tracking, the `wandb` directory stores logs and model metadata.
- **TensorBoard**: Optionally, TensorBoard logging can be enabled through `train.py` to visualize training metrics and model performance.

## Notebooks

The `notebooks` directory contains Jupyter notebooks for visualizing and experimenting with different model architectures, hyperparameters, and evaluation metrics.

## Key Considerations and Optimizations

1. **Masking Strategy**: Ensure correct implementation of source and target masks to optimize attention layers in the Transformer.
2. **Learning Rate Scheduling**: Adjust `LR` and scheduler settings to avoid overfitting.
3. **Regularization**: Consider adding label smoothing or scheduled sampling to improve generalization.
4. **Evaluation Metrics**: Track BLEU scores and other translation metrics to evaluate translation quality effectively.

## Future Work

- **Data Augmentation**: Experiment with additional preprocessing steps and augmentations.
- **Model Pruning and Quantization**: Optimize for deployment on resource-constrained devices.
- **Advanced Translation Techniques**: Implement more sophisticated techniques like Transformer-XL or MT-DNN for performance improvements.

## Contributing

Contributions are welcome! Please open an issue to discuss your ideas or create a pull request with proposed changes.

## License

This project is licensed under the MIT License. See `LICENSE` for details.

## Acknowledgments

Special thanks to the developers of PyTorch, PyTorch Lightning, and Hugging Face Transformers for their open-source tools and resources.