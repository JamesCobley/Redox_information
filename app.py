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
    resp = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that provides Python code updates."},
            {"role": "user",   "content": prompt},
        ],
        temperature=0
    )
    return resp.choices[0].message["content"]

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
    "Shannon Entropy":       "Oxi_Shannon.py",
    "KL Divergence":         "Oxi_KL.py",
    "Mutual Information":    "Oxi_MI.py",
    "Per-site MI":           "Oxi_MI_per_site.py",
    "Fisher Information":    "Oxi_FIM.py",
    "Fisher–Rao Distance":   "Oxi_Fisher_Rao_d.py",
}
choice = st.sidebar.selectbox("Select metric to run", list(script_map.keys()))

# 3) Load source
source_path = script_map[choice]
st.sidebar.write(f"Loaded: `{source_path}`")
try:
    with open(source_path, "r") as f:
        source_code = f.read()
except FileNotFoundError:
    st.error(f"Script not found at `{source_path}`")
    st.stop()

# 4) Optional GPT code edit
user_req = st.sidebar.text_input("Edit script? Describe changes here:")
if user_req:
    st.info("Modifying code via GPT…")
    try:
        source_code = modify_code_with_gpt(source_code, user_req)
        st.code(source_code, language="python")
    except Exception as e:
        st.error(f"GPT code modification failed: {e}")

# 5) Run Analysis
if st.button("Run Analysis"):
    # Save user-uploaded data for the script
    df.to_csv("input_data.csv", index=False)

    # Write the (possibly edited) script to temp file
    tmp_py = tempfile.NamedTemporaryFile(delete=False, suffix=".py")
    tmp_py.write(source_code.encode())
    tmp_py.close()

    try:
        # Execute the script, passing the data path
        result = subprocess.run(
            ["python", tmp_py.name, "input_data.csv"],
            capture_output=True, text=True, check=True
        )
        st.subheader("Script Output")
        st.text(result.stdout)

        # Display any known output images
        for fname in [
            "shannon_entropy_violin.png",
            "per_peptide_kl_viridis.png",
            "brain_entropy.png",
            "kl_hist_all.png",
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

