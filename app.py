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
openai.api_key = st.secrets["OPENAI_API_KEY"]  # Store your key in Streamlit secrets
MODEL = "gpt-4-0613"

# --- Helper: Call GPT to adjust code ---
def modify_code_with_gpt(source_code: str, user_request: str) -> str:
    prompt = f"You are a Python expert. The following code:\n```python\n{source_code}\n```\nUser requests: {user_request}\nModify the code accordingly and return the full updated script without explanations."
    resp = openai.ChatCompletion.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that provides Python code updates."},
            {"role": "user", "content": prompt},
        ],
        functions=[],
        temperature=0
    )
    return resp.choices[0].message.content

# --- App UI ---
st.title("Interactive Information-Theoretic Analysis App")
st.markdown(
    "Upload your dataset, choose metrics, and optionally edit the analysis code via GPT." 
)

# File upload
uploaded_file = st.file_uploader("Upload CSV or Excel data", type=["csv", "xlsx"])
if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.success("Data loaded successfully!")
        if st.checkbox("Show raw data"):
            st.dataframe(df)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

    # Select analysis script
    st.sidebar.header("Analysis Options")
    default_script = st.sidebar.selectbox(
        "Choose a built-in script", ["Shannon Entropy Violin", "Custom Metric"]
    )
    if default_script == "Shannon Entropy Violin":
        with open("scripts/shannon_entropy.py") as f:
            source_code = f.read()
    else:
        source_code = st.sidebar.text_area("Paste your custom script", height=200)

    # GPT editing
    user_req = st.sidebar.text_input("Edit script? Describe changes here:")
    if user_req:
        st.info("Modifying code via GPT...")
        try:
            updated_code = modify_code_with_gpt(source_code, user_req)
            st.code(updated_code, language='python')
            source_code = updated_code
        except Exception as e:
            st.error(f"GPT code modification failed: {e}")

    # Execute script
    if st.button("Run Analysis"):
        # Save code to temp file
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".py")
        tmp.write(source_code.encode('utf-8'))
        tmp.close()
        try:
            # Run script
            result = subprocess.run(
                ["python", tmp.name], capture_output=True, text=True, check=True
            )
            st.text("Script Output:")
            st.text(result.stdout)
            # Display generated plot if exists
            img_path = "shannon_entropy_violin.png"
            if os.path.exists(img_path):
                st.image(img_path, use_column_width=True)
        except subprocess.CalledProcessError as e:
            st.error(f"Error running script: {e.stderr}")
        finally:
            os.unlink(tmp.name)
