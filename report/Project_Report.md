# Project Report — Transformer NLP Explorer

## 1. Title Page
**Transformer NLP Explorer**  
An Interactive Platform for Transformer Architecture, Masked Language Modeling and Causal Language Modeling

Submitted by: ____________________  
Roll Number: ____________________  
Class / Division: ________________  
Department: ____________________  
Institute: ______________________  
Academic Year: 2026–2027  
Guide: _________________________

## 2. Index
1. Title Page
2. Index
3. Abstract
4. Introduction
5. Application Workflow
6. Results / Observation
7. Conclusion

## 3. Abstract
This project presents an interactive Streamlit application called Transformer NLP Explorer. It combines three practical aspects of Transformer-based language models. The first module uses pretrained BERT to inspect tokenization, token IDs, contextual hidden-state dimensions and self-attention weights. The second implements Masked Language Modeling (MLM), where BERT predicts a missing token using surrounding context and displays Top-K predictions with probabilities. The third uses GPT-2 for Causal Language Modeling (CLM), displaying next-token candidates and generating text with greedy decoding, temperature, Top-K and Top-P sampling. The application uses Python, PyTorch, Hugging Face Transformers, Streamlit and Plotly.

## 4. Introduction
Transformers use attention mechanisms to model relationships between tokens. Text is tokenized and converted into numerical IDs before being processed by a model. BERT is an encoder model that can use context from both directions and is suitable for masked-token prediction. GPT-2 is an autoregressive model that predicts the next token from preceding context.

### Objectives
- Inspect BERT tokens and token IDs.
- Examine hidden-state dimensions and attention weights.
- Predict masked words and rank candidates by probability.
- Generate text with GPT-2.
- Compare greedy, temperature, Top-K and Top-P decoding.

### Tools and models
- Python and PyTorch
- Hugging Face Transformers
- BERT: bert-base-uncased
- GPT-2: gpt2
- Streamlit, Pandas and Plotly

## 5. Application Workflow
**Overall:** Input text → Tokenizer → Pretrained Transformer → Model outputs → Interactive visualization.

### Module 1: Architecture Explorer
Input sentence → BERT tokenizer → token IDs → BERT encoder → hidden states and attention weights → heatmap.

### Module 2: Masked Word Predictor
Sentence with one [MASK] token → BERT → masked-position logits → softmax → Top-K candidate tokens.

### Module 3: Causal Text Generator
Prompt → GPT-2 tokenizer → causal Transformer → next-token logits → decoding strategy → append token → repeat.

Insert screenshots from your actual application run:
- Figure 1: Architecture Explorer token table and attention heatmap.
- Figure 2: Masked Word Predictor Top-K output.
- Figure 3: Causal Text Generator output and decoding settings.

## 6. Results / Observation
Record actual outputs from your execution. The following are expected trends, not measured results:
- BERT-base hidden states normally have a hidden dimension of 768.
- Different attention heads/layers can show different attention distributions.
- BERT returns ranked vocabulary predictions for the masked position.
- Greedy decoding selects the highest-scoring token at each step.
- Lower temperature tends to make sampling more conservative; higher temperature generally increases variation.
- A larger Top-K or Top-P threshold generally permits a broader candidate set.

### Observation table
| Test | Setting | Actual output / observation |
|---|---|---|
| Architecture | Sentence and selected attention head | [Fill after running] |
| MLM | One [MASK], Top-K = 5 | [Fill after running] |
| CLM | Greedy | [Fill after running] |
| CLM | Temperature = 0.7 | [Fill after running] |
| CLM | Temperature = 1.5 | [Fill after running] |
| CLM | Top-K = 10 and 50 | [Fill after running] |
| CLM | Top-P = 0.80 and 0.95 | [Fill after running] |

Evaluate generated outputs for coherence, relevance, repetition, diversity, length and topic drift. Add actual screenshots and observations.

## 7. Conclusion
Transformer NLP Explorer combines three complementary practical studies in one application. The architecture module demonstrates tokenization, contextual representations and attention. The MLM module demonstrates bidirectional masked-token prediction. The CLM module demonstrates left-to-right next-token prediction and autoregressive text generation. The dashboard makes model outputs and decoding behavior easier to inspect. Final experimental conclusions should be based on actual observations from running the application.

GitHub: https://github.com/Treaxon/transformer-nlp-explorer
