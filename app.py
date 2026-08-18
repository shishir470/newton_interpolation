"""
Streamlit web app for Newton's Forward & Backward Difference Interpolation.

Run locally:
    pip install streamlit
    streamlit run app.py

Deploy for free (a real public URL) at https://share.streamlit.io :
    1. Push this file + newton_interpolation.py to a GitHub repo.
    2. Sign in to share.streamlit.io with GitHub.
    3. Click "New app", pick the repo, set main file to app.py, click Deploy.
"""

import streamlit as st
import pandas as pd

from newton_interpolation import (
    newton_forward, newton_backward, choose_method, check_equally_spaced
)

st.set_page_config(page_title="Newton's Difference Interpolation", page_icon="📈", layout="centered")

st.title("📈 Newton's Difference Interpolation")
st.caption("Forward & backward finite-difference interpolation for equally spaced data.")

with st.form("inputs"):
    col1, col2 = st.columns(2)
    with col1:
        x_raw = st.text_input("x values (comma separated)", "1, 2, 3, 4, 5")
    with col2:
        y_raw = st.text_input("y values (comma separated)", "1, 8, 27, 64, 125")

    x_target = st.number_input("Value of x to interpolate", value=1.5, format="%.6f")
    method = st.radio("Method", ["Auto", "Forward", "Backward"], horizontal=True)
    submitted = st.form_submit_button("Compute")

if submitted:
    try:
        x = [float(v) for v in x_raw.replace(",", " ").split()]
        y = [float(v) for v in y_raw.replace(",", " ").split()]
        if len(x) != len(y):
            st.error(f"Mismatched lengths: {len(x)} x-values vs {len(y)} y-values.")
            st.stop()

        h = check_equally_spaced(x)

        chosen = method.lower()
        if chosen == "auto":
            chosen = choose_method(x, x_target)
            st.info(f"Auto-selected **{chosen}** difference method (step h = {h:g}).")

        if chosen == "forward":
            result, terms, p, table = newton_forward(x, y, x_target)
        else:
            result, terms, p, table = newton_backward(x, y, x_target)

        # Build a display DataFrame for the difference table
        n = len(table)
        cols = ["x", "y"] + [f"Δ^{k}y" for k in range(1, n)]
        rows = []
        for i in range(len(x)):
            row = [x[i]]
            for k in range(n):
                row.append(table[k][i] if i < len(table[k]) else None)
            rows.append(row)
        df = pd.DataFrame(rows, columns=cols)

        st.subheader("Finite difference table")
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.subheader("Computation")
        st.latex(f"p = {p:.6f}")
        running = 0.0
        term_lines = []
        for k, t in enumerate(terms):
            running += t
            term_lines.append(f"term {k} = {t:.6f}  →  running total = {running:.6f}")
        st.code("\n".join(term_lines))

        st.subheader("Result")
        st.metric(label=f"Interpolated y at x = {x_target}", value=f"{result:.6f}")

    except ValueError as e:
        st.error(str(e))
