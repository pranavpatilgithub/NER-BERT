import os
import json
import torch
import streamlit as st
from transformers import pipeline

# ─────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "NER — BERT vs DistilBERT",
    page_icon  = None,
    layout     = "wide"
)

# ─────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
BERT_PATH       = os.path.join(BASE_DIR, "models", "bert_model",       "best")
DISTILBERT_PATH = os.path.join(BASE_DIR, "models", "distilbert_model", "best")
RESULTS_PATH    = os.path.join(BASE_DIR, "models", "results", "phase1_results.json")

# ─────────────────────────────────────────────────────────
# ENTITY CONFIG (emoji removed)
# ─────────────────────────────────────────────────────────
ENTITY_CONFIG = {
    "PER" : {"label": "PERSON",       "color": "#D6EAF8", "border": "#2E86C1"},
    "ORG" : {"label": "ORGANIZATION", "color": "#D5F5E3", "border": "#1E8449"},
    "LOC" : {"label": "LOCATION",     "color": "#FEF9E7", "border": "#D4AC0D"},
    "MISC": {"label": "MISCELLANEOUS","color": "#FDEDEC", "border": "#C0392B"},
}

# ─────────────────────────────────────────────────────────
# LOAD RESULTS
# ─────────────────────────────────────────────────────────
def load_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return {
        "BERT": {"f1": 0, "precision": 0, "recall": 0, "train_time": 0, "latency_ms": 0, "parameters": "110M"},
        "DistilBERT": {"f1": 0, "precision": 0, "recall": 0, "train_time": 0, "latency_ms": 0, "parameters": "66M"}
    }

MODEL_RESULTS = load_results()

# ─────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────
@st.cache_resource
def load_bert():
    return pipeline('token-classification', model=BERT_PATH, aggregation_strategy='simple', device=-1)

@st.cache_resource
def load_distilbert():
    return pipeline('token-classification', model=DISTILBERT_PATH, aggregation_strategy='simple', device=-1)

# ─────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────
def predict(text, pipe):
    raw = pipe(text)
    entities = []

    for ent in raw:
        word = ent['word']

        # FIX: remove wordpiece artifacts
        word = word.replace("##", "")

        entity_type = ent['entity_group'].replace('B-','').replace('I-','')

        if entity_type in ENTITY_CONFIG:
            entities.append({
                'word': word,
                'type': entity_type,
                'score': ent['score'],
                'start': ent['start'],
                'end': ent['end']
            })

    return entities

# ─────────────────────────────────────────────────────────
# HIGHLIGHT RENDER
# ─────────────────────────────────────────────────────────
def render_highlighted(text, entities):
    if not entities:
        return f"<p style='font-size:16px;line-height:2'>{text}</p>"

    result = text
    offset = 0

    for ent in sorted(entities, key=lambda x: x['start']):
        cfg   = ENTITY_CONFIG.get(ent['type'], {})
        color = cfg.get('color')
        bd    = cfg.get('border')
        label = cfg.get('label')

        highlight = (
            f"<mark style='background:{color};border:1.5px solid {bd};"
            f"border-radius:4px;padding:2px 6px;margin:0 2px;font-weight:500'>"
            f"{ent['word']} "
            f"<sup style='font-size:10px;color:{bd};font-weight:bold'>{label}</sup>"
            f"</mark>"
        )

        s = ent['start'] + offset
        e = ent['end'] + offset
        result = result[:s] + highlight + result[e:]
        offset += len(highlight) - (ent['end'] - ent['start'])

    return f"<p style='font-size:16px;line-height:2.4'>{result}</p>"

# ─────────────────────────────────────────────────────────
# TABLE RENDER
# ─────────────────────────────────────────────────────────
def render_table(entities):
    if not entities:
        st.info("No named entities found.")
        return

    rows = ""
    for ent in entities:
        cfg   = ENTITY_CONFIG.get(ent['type'], {})
        label = cfg.get('label')
        color = cfg.get('color')
        bd    = cfg.get('border')
        score = f"{ent['score']:.1%}"

        rows += f"""
        <tr>
            <td style='padding:8px 14px;font-weight:500;font-size:15px'>
                {ent['word']}
            </td>
            <td style='padding:8px 14px'>
                <span style='background:{color};border:1px solid {bd};
                border-radius:4px;padding:3px 10px;
                font-size:13px;font-weight:600;color:{bd}'>
                    {label}
                </span>
            </td>
            <td style='padding:8px 14px;color:#555;font-size:14px'>
                {score}
            </td>
        </tr>
        """

    html = f"""
    <table style='width:100%;border-collapse:collapse'>
        <thead>
            <tr style='background:#F0F2F6'>
                <th style='padding:10px 14px;text-align:left'>Entity</th>
                <th style='padding:10px 14px;text-align:left'>Type</th>
                <th style='padding:10px 14px;text-align:left'>Confidence</th>
            </tr>
        </thead>
        <tbody>
            {rows}
        </tbody>
    </table>
    """

    st.markdown(html, unsafe_allow_html=True)

def render_entities_text(entities):
    if not entities:
        st.info("No named entities found.")
        return

    st.markdown("### Detected Entities")

    for i, ent in enumerate(entities, 1):
        cfg   = ENTITY_CONFIG.get(ent['type'], {})
        label = cfg.get('label')
        score = f"{ent['score']:.1%}"

        st.write(f"{i}. {ent['word']} → {label} ({score})")
# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
def main():

    # Header
    st.markdown("""
    <h1 style='text-align:center;color:#1F4E79;margin-bottom:0'>
        Named Entity Recognition
    </h1>
    <p style='text-align:center;color:#888;font-size:16px;margin-top:6px'>
        BERT-base-cased vs DistilBERT-base-cased | CoNLL 2003
    </p>
    <hr style='margin:16px 0 24px'>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("## Model Selection")

        selected = st.radio(
            "Choose model:",
            ["BERT-base-cased", "DistilBERT"]
        )

        st.markdown("---")
        st.markdown("## Model Metrics")

        key = "BERT" if selected == "BERT-base-cased" else "DistilBERT"
        metrics = MODEL_RESULTS.get(key, {})

        c1, c2 = st.columns(2)
        c1.metric("F1", f"{metrics.get('f1',0):.3f}")
        c2.metric("Precision", f"{metrics.get('precision',0):.3f}")
        c1.metric("Recall", f"{metrics.get('recall',0):.3f}")
        c2.metric("Params", metrics.get('parameters'))

        # st.metric("Train Time", f"{metrics.get('train_time',0)}s")
        st.metric("Latency", f"{metrics.get('latency_ms',0)}ms")

        st.markdown("---")
        st.markdown("## Entity Types")

        for _, cfg in ENTITY_CONFIG.items():
            st.markdown(
                f"<div style='margin-bottom:6px'>"
                f"<span style='background:{cfg['color']};border:1px solid {cfg['border']};"
                f"border-radius:4px;padding:3px 10px;font-size:13px;"
                f"font-weight:600;color:{cfg['border']}'>"
                f"{cfg['label']}</span></div>",
                unsafe_allow_html=True
            )

    # Input
    input_text = st.text_area(
        "Enter a sentence:",
        value="Elon Musk founded SpaceX in California in 2002.",
        height=90
    )

    analyze = st.button("Analyze", type="primary")

    # Inference
    if analyze and input_text.strip():
        with st.spinner(f"Running {selected}..."):
            pipe = load_bert() if selected == "BERT-base-cased" else load_distilbert()
            entities = predict(input_text, pipe)

        st.markdown("---")
        st.markdown(f"### Results — {selected}")

        st.markdown("**Highlighted Text:**")
        st.markdown(
            f"<div style='background:black;border:1px solid #E0E0E0;"
            f"border-radius:8px;padding:16px 20px'>"
            f"{render_highlighted(input_text, entities)}</div>",
            unsafe_allow_html=True
        )

        left, right = st.columns([2,1])

        with left:
            st.markdown("**Detected Entities:**")
            render_entities_text(entities)

        with right:
            st.markdown("**Summary:**")
            if entities:
                counts = {}
                for e in entities:
                    counts[e['type']] = counts.get(e['type'], 0) + 1

                st.metric("Total", len(entities))
                for k, v in counts.items():
                    st.metric(k, v)
            else:
                st.metric("Total", 0)

    elif analyze:
        st.warning("Please enter a sentence.")

    # Footer
    st.markdown("---")
    st.markdown("""
    <p style='text-align:center;color:#BBB;font-size:13px'>
        MTech NLP Mini Project | CoNLL 2003 Benchmark
    </p>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()