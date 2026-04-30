import requests
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# =========================
# 1. SELIC (AGREGADA MENSAL)
# =========================
def get_selic():
    url = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.1178/dados"

    headers = {
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0"
    }

    params = {
        "formato": "json",
        "dataInicial": "01/01/2023"
    }

    r = requests.get(url, headers=headers, params=params)

    if r.status_code != 200:
        print("Erro SELIC:", r.text)
        return pd.DataFrame()

    data = r.json()

    df = pd.DataFrame(data)

    df["valor"] = df["valor"].astype(float)
    df.rename(columns={"valor": "selic"}, inplace=True)

    # 🔥 converter para data
    df["data"] = pd.to_datetime(df["data"], dayfirst=True)

    # 🔥 AGRUPAR POR MÊS
    df["data"] = df["data"].dt.to_period("M")
    df = df.groupby("data").mean().reset_index()

    # formato YYYYMM
    df["data"] = df["data"].astype(str).str.replace("-", "")

    return df.tail(24)


# =========================
# 2. BOLSA FAMÍLIA CAMPINAS
# =========================
def get_bolsa():
    url = "https://api.portaldatransparencia.gov.br/api-de-dados/bolsa-familia-por-municipio"

    headers = {
        "chave-api-dados": "SUA_CHAVE_AQUI"
    }

    meses = pd.date_range(end=pd.Timestamp.today(), periods=24, freq='ME')
    meses = [d.strftime("%Y%m") for d in meses]

    dados = []

    for i, mes in enumerate(meses):
        print(f"🔄 Buscando mês {mes} ({i+1}/24)")

        try:
            params = {
                "codigoIbge": "3509502",
                "mesAno": mes
            }

            r = requests.get(
                url,
                headers=headers,
                params=params,
                timeout=5  # 🔥 ESSENCIAL
            )

            if r.status_code != 200:
                print(f"⚠️ erro {r.status_code}")
                valor = np.nan
            else:
                data = r.json()

                if isinstance(data, list) and len(data) > 0:
                    valor = float(data[0].get("valor", np.nan))
                else:
                    valor = np.nan

        except Exception as e:
            print(f"❌ erro no mês {mes}: {e}")
            valor = np.nan

        dados.append({"data": mes, "bolsa": valor})

    return pd.DataFrame(dados)

# =========================
# 3. CARREGAR E ALINHAR
# =========================
df_selic = get_selic()
df_bolsa = get_bolsa()

print("\nSELIC:")
print(df_selic.head())

print("\nBOLSA:")
print(df_bolsa.head())

# merge correto
df = df_selic.merge(df_bolsa, on="data")

# remover dados faltantes
df = df.dropna(subset=["selic", "bolsa"])

print("\nDATASET FINAL:")
print(df.head())
print("Shape:", df.shape)


# =========================
# 4. CRIAR LAG (CORRELAÇÃO REAL)
# =========================
df["selic_lag1"] = df["selic"].shift(1)
df["bolsa_lag1"] = df["bolsa"].shift(1)

df = df.dropna()


# =========================
# 5. CORRELAÇÃO
# =========================
print("\n📊 CORRELAÇÃO REAL:")
print(df[["selic", "bolsa"]].corr())


# =========================
# 6. MODELO
# =========================
X = df[["selic_lag1", "bolsa_lag1"]]
y = df["bolsa"]

model = LinearRegression()
model.fit(X, y)


# =========================
# 7. PREVISÃO
# =========================
ultimo = df.iloc[-1]

entrada = pd.DataFrame([{
    "selic_lag1": ultimo["selic"],
    "bolsa_lag1": ultimo["bolsa"]
}])

previsao = model.predict(entrada)[0]

print(f"\n📈 Previsão próximo mês: R$ {previsao:.2f}")


# =========================
# 8. GRÁFICO
# =========================
plt.figure()

plt.plot(df["bolsa"].values, label="Bolsa Família (Campinas)")
plt.plot(df["selic"].values, label="SELIC")

plt.legend()
plt.title("SELIC x Bolsa Família - Campinas")

plt.show()