# FINAL_OPTIMIZED0.py
import os
import random
import torch 
import numpy as np, json
from torch.utils.data import Dataset
from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments, EarlyStoppingCallback
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors, trainers as tok_trainers

# MODEL CONFIGURATION
OUTPUT_DIR = "FINAL_OPTIMIZED0_checkpoints"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_CHILD = os.path.join(BASE_DIR, "data", "babylm_child_corpus.txt")
CORPUS_ADULT = os.path.join(BASE_DIR, "data", "bookcorpus.txt")

VOCAB_SIZE = 16000
WINDOW_SIZE = 256
EMBD_DIM = 768 # Dimension of the token embeddings
TRANSF_BLOCKS = 12
ATT_HEAD = 12 # Number of self-attention heads per transformer layer
CHILD_EPOCHS = 6
ADULT_EPOCHS = 7
CHILD_LR = 5e-4
ADULT_LR = 1e-4
BATCH_SIZE = 2
GRAD_ACCUM = 16  # Effective batch size = 2 * 16 = 32
DROPOUT = 0.1
WARMUP_STEPS = 500 # Number of steps during which the lr increases from 0
SEED = 0

CHILD_TOKENS = 10000000
ADULT_TOKENS = 30000000
VAL_HOLDOUT = 0.05 # 5% of data is hold out for validation

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# TOKENIZER
def train_tokenizer():
    # Read the first 5 million tokens of both corpora to use for the tokenizer (to not have adult language overrule)
    with open(CORPUS_CHILD, "r", encoding="utf-8") as file:
        child_text = file.read(5000000)
    with open(CORPUS_ADULT, "r", encoding="utf-8") as file:
        adult_text = file.read(5000000)
    combined = child_text + " " + adult_text # Combine them so that the tokenizer is trained on both corpora

    # Set up the Byte-Pair Encoding tokenizer
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
    trainer = tok_trainers.BpeTrainer(vocab_size=VOCAB_SIZE,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tokenizer.train_from_iterator([combined], trainer=trainer)
    tokenizer.save(os.path.join(OUTPUT_DIR, "tokenizer.json"))
    return tokenizer

# PREPARING DATA
# To seperate the dataset into blocks and prepare it for use
class BlockDataset(Dataset):
    def __init__(self, token_ids, block_size=WINDOW_SIZE, shuffle=True):
        num_blocks = len(token_ids) // block_size
        self.blocks = [token_ids[(i * block_size) : ((i + 1) * block_size)] for i in range(num_blocks)] # Creating 256 token chuncks of data
        if shuffle:
            random.shuffle(self.blocks)

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, index):
        chunk = self.blocks[index]
        # The model gets "input_ids" as input and has to predict "labels"
        return {"input_ids": torch.tensor(chunk, dtype=torch.long), "labels": torch.tensor(chunk, dtype=torch.long)} 

# To load the data, tokenize it and split into the training and validation sets
def load_split_data(filepath, tokenizer, max_tokens, val_holdout=VAL_HOLDOUT):
    token_ids = []
    with open(filepath, "r", encoding="utf-8") as file:
        while len(token_ids) < max_tokens:
            chunk = file.read(1000000)
            if not chunk:
                break
            token_ids.extend(tokenizer.encode(chunk).ids) # Encoding the subwords into numbers and adding them to the token_ids list

    token_ids = token_ids[:max_tokens]
    split_index = int(len(token_ids) * (1 - val_holdout)) # Splitting index for train and validation sets
    train_set = token_ids[:split_index]
    val_set = token_ids[split_index:]
    return train_set, val_set

# TRAINING
def train_stage(model, train_ids, val_ids, stage, lr, epochs, device):
    training_args = TrainingArguments(output_dir=os.path.join(OUTPUT_DIR, stage),
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=lr,
        warmup_steps=WARMUP_STEPS,
        weight_decay=0.01, # L2 regularization to prevent overfitting
        bf16=True,
        evaluation_strategy="epoch", # Validate the model after every epoch
        save_strategy="epoch", # Save model checkpoints after every epoch
        load_best_model_at_end=True, # Uses the best performing checkpoints as the final model
        metric_for_best_model="eval_loss", # The checkpoints with the lowest validation loss (perplexity) is the best
        logging_strategy="epoch",
        save_total_limit=2,
        seed=SEED)

    trainer = Trainer(model=model,
        args=training_args,
        train_dataset=BlockDataset(train_ids, shuffle=True),
        eval_dataset=BlockDataset(val_ids, shuffle=False),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]) # If two consecutive epochs get a higher loss, stop training
    trainer.train()

    # Saving the losses and validation scores for plotting later
    with open(os.path.join(OUTPUT_DIR, f"{stage}_metrics.json"), "w") as file:
        json.dump(trainer.state.log_history, file, indent=4)

    # Reloading and saving the best performing model
    best_model_path = os.path.join(OUTPUT_DIR, f"{stage}_best")
    trainer.save_model(best_model_path)
    return GPT2LMHeadModel.from_pretrained(best_model_path).to(device)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = train_tokenizer()

    config = GPT2Config(vocab_size=VOCAB_SIZE,
            n_positions=WINDOW_SIZE,
            n_ctx=WINDOW_SIZE,
            n_embd=EMBD_DIM,
            n_layer=TRANSF_BLOCKS,
            n_head=ATT_HEAD,
            bos_token_id=tokenizer.token_to_id("<s>"),
            eos_token_id=tokenizer.token_to_id("</s>"),
            pad_token_id=tokenizer.token_to_id("<pad>"),
            resid_pdrop=DROPOUT,
            embd_pdrop=DROPOUT,
            attn_pdrop=DROPOUT)

    model = GPT2LMHeadModel(config).to(device)

    child_train, child_val = load_split_data(CORPUS_CHILD, tokenizer, CHILD_TOKENS)
    adult_train, adult_val = load_split_data(CORPUS_ADULT, tokenizer, ADULT_TOKENS)

    model = train_stage(model, child_train, child_val, "stage_child", CHILD_LR, CHILD_EPOCHS, device)
    model = train_stage(model, adult_train, adult_val, "stage_adult", ADULT_LR, ADULT_EPOCHS, device)

if __name__ == "__main__":
    main()
