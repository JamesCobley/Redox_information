import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import openai
import tempfile
import os
import subprocess

# --- Configuration ---
openai.api_key = st.secrets["OPENAI_API_KEY"]
MODEL = "gpt-4-0613"

# --- Helper: Call GPT to adjust code ---
def modify_code_with_gpt(source_code: str, user_request: str) -> str:
    prompt = (
        "You are a Python expert. The following code:\n"
        f"```python\n{source_code}\n```\n"
        f"User requests: {user_request}\n"
        "Modify the code accordingly and return the full updated script without explanations."
    )
    resp = openai.ChatCompletion.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that provides Python code updates."},
            {"role": "user",   "content": prompt},
        ],
        temperature=0
    )
    return resp.choices[0].message.content

# --- App UI ---
st.title("Interactive Information-Theoretic Analysis App")
st.markdown("Upload your dataset, choose a metric, and optionally edit the code via GPT.")

# 1) Upload data
uploaded_file = st.file_uploader("Upload CSV or XLSX data", type=["csv", "xlsx"])
if not uploaded_file:
    st.info("Please upload a data file to continue.")
    st.stop()

try:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    st.success("Data loaded successfully!")
    if st.checkbox("Show raw data"):
        st.dataframe(df)
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

# 2) Choose script
st.sidebar.header("Analysis Options")
script_map = {
    "Shannon Entropy":       "scripts/Oxi_Shannon.py",
    "KL Divergence":         "scripts/Oxi_KL.py",
    "Mutual Information":    "scripts/Oxi_MI.py",
    "Per-site MI":           "scripts/Oxi_MI_per_site.py",
    "Fisher Information":    "scripts/Oxi_FIM.py",
    "Fisher–Rao Distance":   "scripts/Oxi_Fisher_Rao_d.py",
}
choice = st.sidebar.selectbox("Select metric to run", list(script_map.keys()))

# 3) Load source
with open(script_map[choice], "r") as f:
    source_code = f.read()
st.sidebar.write(f"Loaded: `{script_map[choice]}`")

# 4) Optional GPT code edit
user_req = st.sidebar.text_input("Edit script? Describe changes here:")
if user_req:
    st.info("Modifying code via GPT…")
    try:
        source_code = modify_code_with_gpt(source_code, user_req)
        st.code(source_code, language="python")
    except Exception as e:
        st.error(f"GPT code modification failed: {e}")

# 5) Run
if st.button("Run Analysis"):
    tmp_py = tempfile.NamedTemporaryFile(delete=False, suffix=".py")
    tmp_py.write(source_code.encode())
    tmp_py.close()

    try:
        # ensure the uploaded dataframe is saved for the script to read
        df.to_csv("input_data.csv", index=False)
        # run the chosen script, passing the data filename if needed:
        result = subprocess.run(
            ["python", tmp_py.name, "input_data.csv"],
            capture_output=True, text=True, check=True
        )
        st.text("Script Output:")
        st.text(result.stdout)

        # look for known output images:
        for fname in [
            "shannon_entropy_violin.png",
            "per_peptide_kl_viridis.png",
            "brain_entropy.png",
            "kl_hist_all.png",
            "brain_entropy.png",
            "brain_entropy.png",
            "brain_entropy.png",
            "per_site_mi.png",
            "fisher_information_surface.png",
            "fim_heatmap.png",
        ]:
            if os.path.exists(fname):
                st.image(fname, use_column_width=True)
    except subprocess.CalledProcessError as e:
        st.error(f"Error running script:\n{e.stderr}")
    finally:
        os.unlink(tmp_py.name)
