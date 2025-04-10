# Gemma Hindi Fine-tuning

This repository contains a script for fine-tuning the [Google Gemma 3-1B IT](https://huggingface.co/google/gemma-3-1b-it) causal language model on a bilingual English-Hindi dataset. The fine-tuning leverages [LoRA](https://arxiv.org/abs/2106.09685) for efficient training and uses Hugging Face's `transformers`, `datasets`, and `peft` libraries.

## Overview

- **Dataset:** Uses the [cfilt/iitb-english-hindi](https://huggingface.co/datasets/cfilt/iitb-english-hindi) dataset (train split).
- **Model:** Fine-tuning is performed on the [google/gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it) causal language model.
- **Training:** Employs a LoRA configuration to adapt the model parameters efficiently. Checkpointing is integrated into the training loop to allow for resume training.

## Features

- **Tokenization:** Custom tokenization that concatenates English and Hindi translations.
- **Efficient Fine-tuning:** Integration of LoRA for reducing memory usage and training time.
- **Checkpointing:** Automatic checkpoint detection and resume training functionality.
- **Custom Optimizer Option:** Uses `paged_adamw_8bit` (if supported) for training in 8-bit precision mode.

## Requirements

Ensure you have the following installed:
- Python 3.8+
- [PyTorch](https://pytorch.org/) (with CUDA support if available)
- [Transformers](https://huggingface.co/docs/transformers/)
- [Datasets](https://huggingface.co/docs/datasets/)
- [peft](https://github.com/huggingface/peft)
- [huggingface_hub](https://huggingface.co/docs/huggingface_hub) (for authentication)
- [huggingface_token](Usage in this script for injecting your token)
- Optionally, [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) if planning to use 8-bit optimizers

You can install these dependencies via pip. For example:

```bash
pip install torch transformers datasets peft huggingface_hub huggingface_token
```

*Note:* If using the `paged_adamw_8bit` optimizer, install `bitsandbytes` according to the official instructions.

## Getting Started

1. **Authentication:**  
   The script logs into the Hugging Face Hub with a token from `huggingface_token`. Make sure you have your token set up in the corresponding module or update the login segment appropriately.

2. **Dataset Preparation:**  
   The script downloads and prepares the CFILT English-Hindi dataset. It tokenizes the data by concatenating the English and Hindi translations, setting up the tokenized inputs and labels for training.

3. **Model Loading and Fine-tuning:**  
   - Loads the pre-trained Gemma model.
   - Applies a LoRA configuration to perform efficient fine-tuning on specific target modules (`q_proj` and `v_proj`).
   - Sets training parameters (e.g., batch sizes, number of epochs, logging steps, gradient accumulation).

4. **Checkpointing and Resuming Training:**  
   The training script automatically searches for the latest checkpoint in the output directory and resumes training if available.

5. **Saving the Model:**  
   Once training completes, the fine-tuned model is saved in the specified output directory.

## How to Run

Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/yourusername/gemma-hindi-finetuning.git
cd gemma-hindi-finetuning
```

Run the training script:

```bash
python finetune.py
```

*Replace `finetune.py` with the name of your Python file containing the provided code.*

## Script Breakdown

- **Hugging Face Authentication:**
  ```python
  from huggingface_hub import login
  from huggingface_token import token
  login(token=token)
  ```
- **Dataset and Tokenization:**
  The script loads the CFILT English-Hindi dataset, tokenizes the concatenated translations, and assigns the tokenized `input_ids` as labels.
  
- **Model and LoRA Setup:**
  The model is loaded with the Gemma configuration and then wrapped with a LoRA module for fine-tuning on the causal language modeling task.

- **Training and Checkpointing:**
  Uses Hugging Face's `Trainer` API with custom training arguments, logging, and evaluation strategies. The script supports resuming from the latest checkpoint if available.
  
- **Model Saving:**
  The final fine-tuned model is saved to disk for subsequent evaluation or inference.

## Contributing

Feel free to fork this repository and submit pull requests for improvements, bug fixes, or additional features. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Enjoy fine-tuning and experimenting with language models!
