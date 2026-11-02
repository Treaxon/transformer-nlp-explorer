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

The application workflow diagram and the screenshots captured from the running application are included in the accompanying Word report. The screenshot sequence documents:
- Figure 1: Transformer NLP Explorer home page.
- Figures 2–8: BERT Architecture Explorer, including attention layer 7/head 9 and layer 9/head 5.
- Figures 9–12: BERT Masked Language Modeling with Top-K = 9 and Top-K = 4.
- Figures 13–16: GPT-2 baseline output and comparisons of maximum new tokens (79 vs 32).
- Figures 17–18: GPT-2 Top-K sampling comparison (Top-K = 75 vs 25).
- Figures 19–20: GPT-2 temperature sampling comparison (temperature = 1.50 vs 0.50).
- Figures 21–22: GPT-2 Top-P sampling comparison (Top-P = 0.90 vs 0.50).

The full screenshot set and workflow diagram are embedded in the Word report supplied with the project.

## 6. Results / Observation
### Captured application observations
- For the BERT input sentence “i am testing my capstone project on my laptop”, the displayed last hidden-state shape was (1, 12, 768): batch size 1, 12 tokens and hidden dimension 768.
- The attention heatmaps differed when the selection changed from layer 7/head 9 to layer 9/head 5.
- For the masked sentence “i am testing my capstone project on my [MASK].”, “computer” was the highest-ranked prediction at 12.90%. The next displayed candidates included “laptop” (6.05%) and “phone” (5.29%).
- Reducing MLM Top-K from 9 to 4 reduced the number of displayed candidates; “computer” remained the top prediction in the captured results.
- GPT-2 generation comparisons used the prompt “i am testing my capstone project on my”. The baseline maximum-new-token setting was 79 and was compared with 32.
- The Top-K sampling comparison used 75 versus 25; the temperature sampling comparison used 1.50 versus 0.50; and the Top-P sampling comparison used 0.90 versus 0.50.
- The captured GPT-2 outputs demonstrate that generated text can repeat phrases or drift from the prompt.

These are observations from the supplied screenshots, not a general benchmark of model quality.

### Observation table
| Test | Setting | Actual output / observation |
|---|---|---|
| Architecture | Input sentence; layer 7/head 9 and layer 9/head 5 | Hidden-state shape (1, 12, 768); attention heatmaps differed by selected layer/head |
| MLM | One [MASK], Top-K = 9 | “computer” ranked first at 12.90% |
| MLM | One [MASK], Top-K = 4 | Four candidates displayed; “computer” remained first at 12.90% |
| CLM | Maximum new tokens = 79 vs 32 | Compared the maximum continuation length |
| CLM | Top-K sampling: 75 vs 25 | Compared the size of the candidate pool |
| CLM | Temperature sampling: 1.50 vs 0.50 | Compared sampling diversity / probability sharpness |
| CLM | Top-P sampling: 0.90 vs 0.50 | Compared nucleus candidate-pool thresholds |

For GPT-2, the baseline screenshot showed “iph” as the highest next-token candidate (88.25%). This is a subword token, not necessarily a complete word. Compare generated outputs for coherence, relevance, repetition, diversity, length and topic drift.

## 7. Conclusion
Transformer NLP Explorer combines three complementary practical studies in one application. The architecture module demonstrates tokenization, contextual representations and attention. The MLM module demonstrates bidirectional masked-token prediction. The CLM module demonstrates left-to-right next-token prediction and autoregressive text generation. The dashboard makes model outputs and decoding behavior easier to inspect. Final experimental conclusions should be based on actual observations from running the application.

GitHub: https://github.com/Treaxon/transformer-nlp-explorer
