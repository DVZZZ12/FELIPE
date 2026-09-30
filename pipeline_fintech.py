"""
Pipeline de dados - Fintech
Limpeza, transformação, agregação, detecção de outliers (Z-Score) e visualização.
Entrada : transacoes.csv (gerado por gerar_dados.py)
Saídas  : anomalias_zscore.csv, pivot_risco.csv, grafico_transacoes_diarias.png
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option("display.width", 140)
pd.set_option("display.float_format", "{:,.2f}".format)

df = pd.read_csv(
    "transacoes.csv",
    encoding="latin1",
    usecols=["id_cliente", "data_transacao", "valor", "origem", "estado_cliente"],
    dtype={
        "id_cliente": "category",
        "origem": "category",
        "estado_cliente": "category",
        "valor": "float64",
    },
)
print(f"[1] Linhas carregadas: {len(df):,} | memória: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
print(f"    NaN em 'valor' antes da imputação: {df['valor'].isna().sum()}")

mediana_por_estado = df.groupby("estado_cliente", observed=True)["valor"].transform("median")
df["valor"] = df["valor"].fillna(mediana_por_estado)
print(f"    NaN em 'valor' depois da imputação: {df['valor'].isna().sum()}")

df["plataforma"] = "Mobile"

df["data_transacao"] = (
    pd.to_datetime(df["data_transacao"])
    .astype("datetime64[ns]")
    .dt.tz_localize("America/Sao_Paulo")
)

dias_pt = {0: "Segunda", 1: "Terça", 2: "Quarta", 3: "Quinta", 4: "Sexta", 5: "Sábado", 6: "Domingo"}
df["dia_semana_num"] = df["data_transacao"].dt.dayofweek
df["dia_semana"] = df["dia_semana_num"].map(dias_pt)
df["mes"] = df["data_transacao"].dt.month

n_antes = len(df)
df = df.drop_duplicates(keep="first").reset_index(drop=True)
print(f"[2] Duplicatas removidas: {n_antes - len(df)} | linhas restantes: {len(df):,}")
print(f"    dtype data_transacao: {df['data_transacao'].dtype}")

mask = (
    (df["mes"] == 9)
    & ((df["estado_cliente"] == "SP") | (df["estado_cliente"] == "RJ"))
    & (df["valor"] > 5000)
)
df_filtrado = df.loc[mask]
print(f"[3] Transações de setembro em SP/RJ com valor > R$ 5.000: {len(df_filtrado)}")
print(df_filtrado.head().to_string(index=False))

risco_dict = {
    "C100": "Baixo",
    "C101": "Alto",
    "C102": "Médio",
    "C103": "Baixo",
    "C104": "Alto",
}

df["nivel_risco"] = df["id_cliente"].astype(object).map(risco_dict)

pivot = pd.pivot_table(
    df,
    index="mes",
    columns="nivel_risco",
    values="valor",
    aggfunc="sum",
    margins=True,
    margins_name="Total",
    observed=True,
)
print("\n[4] Soma de valor por mês x nível de risco (R$):")
print(pivot.to_string())

def zscore_por_grupo(dados: pd.DataFrame, coluna: str, grupo: str) -> pd.Series:
    """Z = (x - mu) / sigma calculado dentro de cada grupo, sem laços.

    Usa desvio padrão populacional (ddof=0), coerente com a fórmula do enunciado.
    Grupos com sigma = 0 resultam em NaN (não podem ser outliers).
    """
    g = dados.groupby(grupo, observed=True)[coluna]
    mu = g.transform("mean")
    sigma = g.transform("std", ddof=0).replace(0, np.nan)
    return (dados[coluna] - mu) / sigma


df["z_score"] = zscore_por_grupo(df, "valor", "estado_cliente")
df_anomalias = df.loc[df["z_score"] > 2.5].copy()

print(f"\n[5] Potenciais anomalias (Z > 2.5): {len(df_anomalias)} de {len(df)} "
      f"({len(df_anomalias) / len(df):.1%})")
print(df_anomalias.groupby("estado_cliente", observed=True)["valor"]
      .agg(qtd="count", valor_total="sum").to_string())

diario = df.set_index("data_transacao")["valor"].resample("D").sum()
media_movel = diario.rolling(7).mean()

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(diario.index, diario.values, color="#9ecae1", linewidth=1.6,
        marker="o", markersize=3, label="Total diário")
ax.plot(media_movel.index, media_movel.values, color="#08519c", linewidth=2.6,
        label="Média móvel 7 dias")

ax.set_ylim(bottom=0)
ax.set_title("Valor total de transações por dia e média móvel de 7 dias", fontsize=14, pad=12)
ax.set_xlabel("Data")
ax.set_ylabel("Valor total (R$)")
ax.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda v, _: f"R$ {v:,.0f}".replace(",", "."))
)
ax.grid(alpha=0.3)
ax.legend(loc="lower right")
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig("grafico_transacoes_diarias.png", dpi=150)

df_anomalias.to_csv("anomalias_zscore.csv", index=False, encoding="utf-8")
pivot.to_csv("pivot_risco.csv", encoding="utf-8")
print("\nArquivos salvos: anomalias_zscore.csv, pivot_risco.csv, grafico_transacoes_diarias.png")

plt.show()
