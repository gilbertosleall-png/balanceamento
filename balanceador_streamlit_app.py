import io
import pandas as pd
import streamlit as st
import plotly.express as px
from balance_engine import balance_line, generate_gantt, export_excel_report, export_pdf_report

st.set_page_config(page_title="Balanceador de Operadores", layout="wide")

st.title("Balanceador de Operadores — Streamlit App")

uploaded = st.file_uploader("Upload Excel/CSV", type=["xlsx", "xls", "csv"]) 

if uploaded is None:
    st.info("Faça upload de um arquivo ou baixe o modelo de exemplo.")
    st.stop()

try:
    if uploaded.type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/vnd.ms-excel"]:
        df = pd.read_excel(uploaded)
    else:
        df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Erro ao ler o arquivo: {e}")
    st.stop()

num_ops = st.number_input("Número de operadores", min_value=1, max_value=100, value=3)

result = balance_line(df, num_ops)

if result is None:
    st.error("Não foi possível balancear a linha com os parâmetros fornecidos.")
    st.stop()

assign, kpis, gantt_df = result

st.subheader("Atribuição de tarefas")
st.dataframe(assign)

st.subheader("KPIs")
st.json(kpis)

st.subheader("Gráfico de Gantt")
fig = generate_gantt(gantt_df)
st.plotly_chart(fig, use_container_width=True)

# Exportações
excel_bytes = export_excel_report(assign, gantt_df)
st.download_button("Exportar Excel", data=excel_bytes, file_name="balanceamento.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

pdf_bytes = export_pdf_report(assign)
st.download_button("Exportar PDF", data=pdf_bytes, file_name="balanceamento.pdf", mime="application/pdf")
