# Transformer NLP Explorer

An interactive educational application combining three Transformer practicals:
1. Transformer architecture inspection with BERT
2. Masked Language Modeling (MLM) with BERT
3. Causal Language Modeling (CLM) with GPT-2

## Features
- Inspect BERT tokens, token IDs, hidden-state dimensions and attention heatmaps.
- Predict masked words and display Top-K probabilities.
- Generate text with GPT-2 using greedy, temperature, Top-K and Top-P decoding.
- Compare model outputs through an interactive Streamlit dashboard.

## Stack
Python, PyTorch, Hugging Face Transformers, Streamlit, Pandas, Plotly.

## Run locally
Python 3.10 or 3.11 recommended.

```bash
git clone https://github.com/Treaxon/transformer-nlp-explorer.git
cd transformer-nlp-explorer
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The first run downloads pretrained models from Hugging Face and requires internet access.

## Models
- `bert-base-uncased`: encoder model for architecture inspection and masked-token prediction.
- `gpt2`: autoregressive model for next-token prediction and text generation.

## Experiment mapping
| Practical | Application module | Output |
|---|---|---|
| Understanding Transformer Architecture | Architecture Explorer | Tokens, IDs, hidden states, attention heatmap |
| Implementing Masked Language Modeling | Masked Word Predictor | Top-K predictions and probabilities |
| Implementing Causal Language Modeling | Causal Text Generator | Next-token probabilities and generated continuation |

## Notes
- This project uses pretrained models for inference; it does not fine-tune them.
- Change one generation parameter at a time when comparing decoding strategies.
- Record actual outputs from your own execution in the report.
- Attention patterns are useful for inspection but are not definitive explanations.
- Generated text may contain factual errors or repetition.

## License
MIT
