import streamlit as st
import torch
import pandas as pd
import plotly.express as px
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM, AutoModelForCausalLM

st.set_page_config(page_title="Transformer NLP Explorer", page_icon="🧠", layout="wide")
st.title("🧠 Transformer NLP Explorer")
st.caption("Explore Transformer architecture, masked language modeling and causal text generation.")

@st.cache_resource
def bert_arch():
    t = AutoTokenizer.from_pretrained("bert-base-uncased")
    m = AutoModel.from_pretrained("bert-base-uncased", output_attentions=True, output_hidden_states=True)
    m.eval()
    return t, m

@st.cache_resource
def bert_mlm():
    t = AutoTokenizer.from_pretrained("bert-base-uncased")
    m = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
    m.eval()
    return t, m

@st.cache_resource
def gpt():
    t = AutoTokenizer.from_pretrained("gpt2")
    m = AutoModelForCausalLM.from_pretrained("gpt2")
    t.pad_token = t.eos_token
    m.config.pad_token_id = t.eos_token_id
    m.eval()
    return t, m

def topk_table(logits, tok, k):
    p = torch.softmax(logits, dim=-1)
    vals, ids = torch.topk(p, k)
    return pd.DataFrame({
        "Token": [tok.decode([int(i)]).replace("\n", "\\n") for i in ids],
        "Probability": [float(v) for v in vals]
    })

a, b, c = st.tabs(["Architecture Explorer", "Masked Word Predictor", "Causal Text Generator"])

with a:
    st.header("1. Transformer Architecture Explorer")
    sentence = st.text_input("Enter a sentence", "The animal did not cross the road because it was tired.")
    layer = st.slider("Attention layer", 1, 12, 12) - 1
    head = st.slider("Attention head", 1, 12, 1) - 1
    if st.button("Analyze sentence"):
        try:
            tok, model = bert_arch()
            x = tok(sentence, return_tensors="pt", truncation=True, max_length=128)
            with torch.no_grad():
                out = model(**x)
            tokens = tok.convert_ids_to_tokens(x["input_ids"][0])
            st.subheader("Tokens and token IDs")
            st.dataframe(pd.DataFrame({"Position": range(len(tokens)), "Token": tokens,
                                       "Token ID": x["input_ids"][0].tolist()}), use_container_width=True)
            st.metric("Last hidden-state shape", str(tuple(out.last_hidden_state.shape)))
            st.caption("BERT-base hidden dimension is 768. Shape: batch × sequence length × hidden dimension.")
            att = out.attentions[layer][0, head].cpu().numpy()
            fig = px.imshow(att, x=tokens, y=tokens, color_continuous_scale="Blues",
                            title=f"Attention heatmap — Layer {layer+1}, Head {head+1}",
                            labels={"x":"Key token","y":"Query token","color":"Attention"})
            fig.update_layout(height=550, xaxis_tickangle=-35)
            st.plotly_chart(fig, use_container_width=True)
            st.info("Attention values show the distribution for the selected head. They should not be treated as complete explanations of model reasoning.")
        except Exception as e:
            st.error(f"Model error: {e}")

with b:
    st.header("2. Masked Language Modeling")
    text = st.text_input("Enter one [MASK] token", "The cat is sitting on the [MASK].")
    k = st.slider("Top-K predictions", 1, 10, 5, key="mlm_k")
    if st.button("Predict missing word"):
        if text.count("[MASK]") != 1:
            st.warning("Include exactly one [MASK] token.")
        else:
            try:
                tok, model = bert_mlm()
                x = tok(text, return_tensors="pt")
                pos = (x["input_ids"][0] == tok.mask_token_id).nonzero(as_tuple=True)[0]
                with torch.no_grad():
                    out = model(**x)
                df = topk_table(out.logits[0, int(pos[0])], tok, k)
                st.dataframe(df.style.format({"Probability":"{:.4%}"}), use_container_width=True)
                st.plotly_chart(px.bar(df, x="Token", y="Probability", title="Top-K masked token probabilities"), use_container_width=True)
                st.success(f"Highest-ranked token: {df.iloc[0]['Token']}")
            except Exception as e:
                st.error(f"Model error: {e}")

with c:
    st.header("3. Causal Language Modeling with GPT-2")
    prompt = st.text_area("Enter a prompt", "The future of artificial intelligence is")
    mode = st.selectbox("Decoding strategy", ["Greedy", "Temperature", "Top-K", "Top-P", "Combined"])
    col1, col2, col3 = st.columns(3)
    with col1:
        n = st.slider("Maximum new tokens", 10, 100, 40)
        temperature = st.slider("Temperature", 0.2, 2.0, 1.0, 0.1)
    with col2:
        top_k = st.slider("Top-K", 1, 100, 50)
        top_p = st.slider("Top-P", 0.5, 1.0, 0.9, 0.05)
    with col3:
        seed = st.number_input("Random seed", 0, 99999, 42)
    if st.button("Generate text"):
        if not prompt.strip():
            st.warning("Enter a prompt.")
        else:
            try:
                tok, model = gpt()
                x = tok(prompt, return_tensors="pt")
                torch.manual_seed(int(seed))
                params = {"max_new_tokens": n, "pad_token_id": tok.eos_token_id}
                if mode == "Greedy":
                    params["do_sample"] = False
                else:
                    params.update({"do_sample": True, "temperature": temperature})
                    if mode in ("Top-K", "Combined"):
                        params["top_k"] = top_k
                    if mode in ("Top-P", "Combined"):
                        params["top_p"] = top_p
                with torch.no_grad():
                    generated = model.generate(**x, **params)
                    logits = model(**x).logits[0, -1, :]
                st.subheader("Generated continuation")
                st.write(tok.decode(generated[0], skip_special_tokens=True))
                st.subheader("Top 5 next-token candidates")
                st.dataframe(topk_table(logits, tok, 5).style.format({"Probability":"{:.4%}"}),
                             use_container_width=True)
            except Exception as e:
                st.error(f"Model error: {e}")

st.divider()
st.caption("Educational demonstration using pretrained BERT and GPT-2 models. Initial model downloads require internet access.")
