#!/usr/bin/bash -l
#SBATCH --partition teaching
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --output=out_translate_sk_en.out

# === Load modules and activate environment ===
module load gpu
module load mamba
source activate atmt

# === 1. Preprocess Slovak-English data ===
python preprocess.py \
  --source-lang sk \
  --target-lang en \
  --raw-data sk-en/data/raw \
  --dest-dir sk-en/data/prepared \
  --model-dir cz-en/tokenizers \
  --test-prefix train \
  --train-prefix train \
  --valid-prefix train \
  --src-model cz-en/tokenizers/cz-bpe-8000.model \
  --tgt-model cz-en/tokenizers/en-bpe-8000.model

# === 2. Translate using trained Czech-English model ===
python translate.py \
    --cuda \
    --input sk-en/data/raw/train.sk \
    --src-tokenizer cz-en/tokenizers/cz-bpe-8000.model \
    --tgt-tokenizer cz-en/tokenizers/en-bpe-8000.model \
    --checkpoint-path cz-en/checkpoints/checkpoint_best.pt \
    --output sk-en/output.txt \
    --max-len 300

# === 3. Evaluate BLEU score ===
python evaluate.py \
  --reference sk-en/data/raw/train.en \
  --hypothesis sk-en/output.txt > sk-en/bleu_score.txt
