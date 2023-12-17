import streamlit as st

from llm import parse_invoice_or_po, pdf2img

st.title("Didero: Coding Challenge")

uploaded_file = st.file_uploader("Upload an invoice", type="pdf")

if uploaded_file is not None:
    pdf_contents = uploaded_file.read()
    pages = pdf2img(pdf_contents)
    parsed_data = parse_invoice_or_po(pages)
    st.header("Parse Results:")
    st.json(parsed_data)
