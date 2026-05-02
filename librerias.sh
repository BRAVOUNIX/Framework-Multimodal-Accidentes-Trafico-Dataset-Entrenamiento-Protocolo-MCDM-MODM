#!/bin/bash

apt update
apt install -y  vim nvtop
pip install  GPUtil ultralytics seaborn tqdm timm scikit-learn transformers peft nltk rouge_score 
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
pip install protobuf --break-system-packages
pip install sentencepiece tiktoken protobuf --break-system-packages
pip install bert-score mauve-text --break-system-packages
pip install flash-attn --no-build-isolation --break-system-packages
pip install csvkit --break-system-packages
ulimit -n 131072
ulimit -a
