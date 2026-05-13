# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  IndPredict AI — Predição de Indicadores Econômicos                    ║
# ║  Rodar: streamlit run indicadores_predict.py                           ║
# ║  Instalar: pip install streamlit pandas numpy scikit-learn plotly      ║
# ║            python-bcb statsmodels                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

# ══════════════════════════════════════════════════════════════════════════════
#  BLOCO 1 — CARREGAMENTO DE DADOS (BCB + fallback simulado)
# ══════════════════════════════════════════════════════════════════════════════

# Códigos SGS do Banco Central
SERIES_BCB = {
    "IPCA":  433,   # % mensal
    "IGPM":  189,   # % mensal
    "INPC":  188,   # % mensal
    "SELIC": 432,   # % ao mês (over)
    "INCC":  192,   # Índice Nacional da Construção Civil
    "CDI":   4391,  # % ao mês
}

DESCRICOES = {
    "IPCA":  "Índice de Preços ao Consumidor Amplo",
    "IGPM":  "Índice Geral de Preços do Mercado",
    "INPC":  "Índice Nacional de Preços ao Consumidor",
    "SELIC": "Taxa SELIC (% a.m.)",
    "INCC":  "Índice Nac. da Construção Civil",
    "CDI":   "Certificado de Depósito Interbancário",
    # Regionais IBGE
    "IPCA-Campinas": "IPCA — Região de Campinas (Interior SP)",
    "IPCA-SP":       "IPCA — Município de São Paulo",
    "IPCA-Interior": "IPCA — Interior do Estado de SP",
}

UNIDADES = {
    "IPCA": "% a.m.", "IGPM": "% a.m.", "INPC": "% a.m.",
    "SELIC": "% a.m.", "INCC": "% a.m.", "CDI": "% a.m.",
    "IPCA-Campinas": "% a.m.", "IPCA-SP": "% a.m.", "IPCA-Interior": "% a.m.",
}

# Códigos de localidade IBGE para IPCA regional (tabela 7060)
# Fonte: IBGE SIDRA — Pesquisa Nacional por Amostra de Domicílios
IBGE_LOCALIDADES = {
    # Código N7 = Região Metropolitana / aglomeração
    "IPCA-Campinas": {"nivel": "N7", "cod": "3509502", "nome": "Campinas"},
    "IPCA-SP":       {"nivel": "N6", "cod": "3550308", "nome": "São Paulo"},
    "IPCA-Interior": {"nivel": "N7", "cod": "3509502", "nome": "Interior SP"},
}


@st.cache_data(ttl=3600, show_spinner=False)
def carregar_bcb(indicador: str, meses: int) -> pd.DataFrame | None:
    """Tenta carregar dado real do Banco Central via API REST."""
    cod  = SERIES_BCB[indicador]
    fim  = datetime.today()
    ini  = fim - pd.DateOffset(months=meses + 3)
    url  = (
        f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{cod}/dados"
        f"?formato=json"
        f"&dataInicial={ini.strftime('%d/%m/%Y')}"
        f"&dataFinal={fim.strftime('%d/%m/%Y')}"
    )
    try:
        r = requests.get(url, timeout=12)
        if r.status_code != 200:
            return None
        df = pd.DataFrame(r.json())
        df["data"]  = pd.to_datetime(df["data"], format="%d/%m/%Y", errors="coerce")
        df["data"]  = df["data"].dt.to_period("M").dt.to_timestamp()
        df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
        df = df[["data", "valor"]].dropna().sort_values("data").reset_index(drop=True)
        return df
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def carregar_ibge_regional(indicador: str, meses: int) -> pd.DataFrame | None:
    """
    Busca IPCA regional via IBGE SIDRA (tabela 7060 — IPCA por município).
    Campinas está na Região Metropolitana de Campinas (código 3509502).
    Variável 63 = variação mensal (%).
    """
    if indicador not in IBGE_LOCALIDADES:
        return None

    loc  = IBGE_LOCALIDADES[indicador]
    fim  = datetime.today()
    ini  = fim - pd.DateOffset(months=meses + 3)

    # Gera lista de períodos YYYYMM
    periodos = pd.date_range(ini, fim, freq="MS")
    periodo_str = "|".join(p.strftime("%Y%m") for p in periodos)

    # SIDRA API v3: tabela 7060, variável 63 (var. mensal), classificação N6 ou N7
    nivel = loc["nivel"]
    cod   = loc["cod"]
    url = (
        f"https://apisidra.ibge.gov.br/values/t/7060"
        f"/n6/{cod}"           # nível município
        f"/v/63"               # variação mensal
        f"/p/{periodo_str}"
        f"?formato=us&decimais=4"
    )

    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            # Tenta com nível N7 (região metropolitana)
            url2 = (
                f"https://apisidra.ibge.gov.br/values/t/7060"
                f"/n7/{cod}/v/63/p/{periodo_str}?formato=us&decimais=4"
            )
            r = requests.get(url2, timeout=15)
            if r.status_code != 200:
                return None

        dados = r.json()
        # Primeira linha é o cabeçalho
        if len(dados) < 2:
            return None

        registros = []
        for row in dados[1:]:   # pula header
            try:
                periodo = str(row.get("D3C") or row.get("D2C") or "")  # período YYYYMM
                valor   = row.get("V", "")
                if periodo and valor not in ("", "-", "..."):
                    dt = pd.Timestamp(f"{periodo[:4]}-{periodo[4:6]}-01")
                    registros.append({"data": dt, "valor": float(valor)})
            except Exception:
                continue

        if len(registros) < 6:
            return None

        df = pd.DataFrame(registros).sort_values("data").reset_index(drop=True)
        return df

    except Exception:
        return None


def simular_serie(indicador: str, meses: int, seed: int = 0) -> pd.DataFrame:
    """Gera série histórica sintética coerente para uso offline."""
    rng   = np.random.default_rng(seed)
    fim   = datetime.today().replace(day=1)
    datas = pd.date_range(end=fim, periods=meses, freq="MS")
    n     = len(datas)

    params = {
        "IPCA":          (0.55, 0.20, 0.12),
        "IGPM":          (0.60, 0.35, 0.20),
        "INPC":          (0.52, 0.18, 0.10),
        "SELIC":         (0.88, 0.05, 0.02),
        "INCC":          (0.65, 0.25, 0.15),
        "CDI":           (0.87, 0.05, 0.02),
        # Regionais: Campinas tende a ter IPCA ligeiramente abaixo da média nacional
        "IPCA-Campinas": (0.51, 0.18, 0.11),
        "IPCA-SP":       (0.56, 0.21, 0.12),
        "IPCA-Interior": (0.50, 0.17, 0.10),
    }
    media, amp, ruido = params.get(indicador, (0.5, 0.2, 0.1))
    tendencia = np.linspace(-amp * 0.3, amp * 0.3, n)
    sazon     = amp * np.sin(np.linspace(0, 4 * np.pi, n))
    valores   = media + tendencia + sazon + rng.normal(0, ruido, n)
    valores   = np.clip(valores, -1.5, 3.5)

    datas = pd.DatetimeIndex([pd.Timestamp(d) for d in datas])
    return pd.DataFrame({"data": datas, "valor": np.round(valores, 4)})


def get_serie(indicador: str, meses: int) -> tuple[pd.DataFrame, bool]:
    """
    Retorna (DataFrame, is_real).
    - Indicadores nacionais: tenta BCB → fallback simulado
    - Indicadores regionais IBGE: tenta SIDRA → fallback simulado
    """
    # Regionais IBGE
    if indicador in IBGE_LOCALIDADES:
        df = carregar_ibge_regional(indicador, meses)
        if df is not None and len(df) >= 6:
            df = df.tail(meses).reset_index(drop=True)
            return df, True
        # Fallback simulado com seed baseado no nome
        seed = sum(ord(c) for c in indicador)
        df = simular_serie(indicador, meses, seed=seed)
        return df, False

    # Nacionais BCB
    df = carregar_bcb(indicador, meses)
    if df is not None and len(df) >= 6:
        df = df.tail(meses).reset_index(drop=True)
        return df, True
    df = simular_serie(indicador, meses, seed=SERIES_BCB[indicador])
    return df, False


# ══════════════════════════════════════════════════════════════════════════════
#  BLOCO 2 — MODELO DE PREDIÇÃO
# ══════════════════════════════════════════════════════════════════════════════

ALGOS = {
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.06,
        subsample=0.8, min_samples_leaf=2, random_state=42),
    "Random Forest": RandomForestRegressor(
        n_estimators=300, max_depth=5, min_samples_leaf=2, random_state=42),
    "Ridge": Ridge(alpha=5.0),
}


def build_features(serie: pd.Series) -> pd.DataFrame:
    """Cria features de lags, médias móveis e sazonalidade a partir da série."""
    df = pd.DataFrame({"y": serie.values})
    df["mes"]       = range(len(df))                        # tendência
    df["mes_sin"]   = np.sin(2 * np.pi * np.arange(len(df)) / 12)
    df["mes_cos"]   = np.cos(2 * np.pi * np.arange(len(df)) / 12)
    for lag in [1, 2, 3, 6, 12]:
        df[f"lag_{lag}"] = df["y"].shift(lag)
    for win in [3, 6, 12]:
        df[f"ma_{win}"]  = df["y"].rolling(win, min_periods=1).mean()
        df[f"std_{win}"] = df["y"].rolling(win, min_periods=1).std().fillna(0)
    df["diff_1"]  = df["y"].diff(1).fillna(0)
    df["diff_12"] = df["y"].diff(12).fillna(0)
    return df


def prever(df_hist: pd.DataFrame, meses_pred: int, algo: str) -> dict:
    """
    Treina o modelo nos dados históricos e projeta `meses_pred` meses à frente.
    Retorna dict com histórico, previsão, IC e métricas.
    """
    serie = df_hist["valor"].values
    feat  = build_features(pd.Series(serie))
    fcols = [c for c in feat.columns if c != "y"]

    # Treino
    feat_clean = feat.ffill().bfill().fillna(0)
    X = feat_clean[fcols].values
    y = feat_clean["y"].values

    pipe = Pipeline([("sc", StandardScaler()), ("m", ALGOS[algo])])
    pipe.fit(X, y)

    # Métricas in-sample (últimos 20%)
    split = max(1, int(len(y) * 0.8))
    yhat_val = pipe.predict(X[split:])
    ytrue_val = y[split:]
    metricas = {
        "r2":   max(0.0, r2_score(ytrue_val, yhat_val)) if len(ytrue_val) > 1 else 1.0,
        "mae":  mean_absolute_error(ytrue_val, yhat_val) if len(ytrue_val) > 0 else 0.0,
        "mape": mean_absolute_percentage_error(ytrue_val, yhat_val) * 100 if len(ytrue_val) > 0 else 0.0,
    }

    # Previsão recursiva: usa predição anterior como lag
    extensao  = list(serie)
    preds     = []
    n_hist    = len(serie)

    for step in range(meses_pred):
        s_ext  = pd.Series(extensao)
        f_ext  = build_features(s_ext)
        f_ext  = f_ext.ffill().bfill().fillna(0)
        X_next = f_ext[fcols].iloc[[-1]].values
        pred   = float(pipe.predict(X_next)[0])
        preds.append(pred)
        extensao.append(pred)

    preds = np.array(preds)

    # Intervalo de confiança via perturbação nos resíduos
    residuos = ytrue_val - yhat_val if len(ytrue_val) > 1 else np.array([0.0])
    std_res  = np.std(residuos) if len(residuos) > 1 else abs(preds.mean()) * 0.05
    horizonte = np.arange(1, meses_pred + 1)
    margem    = 1.645 * std_res * np.sqrt(horizonte)   # IC 90%

    # Datas futuras — garante que ultima_data é sempre um Timestamp
    ultima_data  = pd.Timestamp(df_hist["data"].iloc[-1])
    datas_futuro = pd.date_range(
        ultima_data + pd.DateOffset(months=1), periods=meses_pred, freq="MS"
    )

    return {
        "historico": df_hist,
        "datas_fut": datas_futuro,
        "preds":     preds,
        "ci_lower":  preds - margem,
        "ci_upper":  preds + margem,
        "metricas":  metricas,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  BLOCO 3 — ANÁLISE COM CLAUDE API
# ══════════════════════════════════════════════════════════════════════════════

def analise_ia(indicadores_sel, resultados, meses_pred):
    """Chama Claude API para análise qualitativa das predições."""
    linhas = []
    for ind in indicadores_sel:
        r    = resultados[ind]
        hist = r["historico"]["valor"]
        pred = r["preds"]
        linhas.append(
            f"  • {ind} ({DESCRICOES[ind]}): "
            f"último valor histórico = {hist.iloc[-1]:.4f}% | "
            f"média prevista = {pred.mean():.4f}% | "
            f"projeção final ({meses_pred}m) = {pred[-1]:.4f}% | "
            f"tendência = {'↑ alta' if pred[-1] > hist.iloc[-1] else '↓ queda'} | "
            f"R² = {r['metricas']['r2']:.3f}"
        )

    prompt = f"""Você é um economista sênior especializado no mercado financeiro brasileiro.

Analise as predições abaixo geradas por um modelo de machine learning para os próximos {meses_pred} meses,
e forneça uma análise econômica concisa, técnica e orientada a decisões:

PREDIÇÕES DOS INDICADORES:
{chr(10).join(linhas)}

Forneça exatamente nessa estrutura:

**1. Cenário Macro**
(2-3 frases sobre o contexto geral das projeções)

**2. Destaques por Indicador**
(Para cada indicador: o que a tendência indica economicamente, em 1-2 frases)

**3. Correlações Relevantes**
(Como os indicadores se relacionam entre si nas projeções)

**4. Implicações Práticas**
(Para investidor, para contratos indexados, para decisões financeiras)

**5. Riscos e Incertezas**
(2-3 fatores que podem desviar as projeções)

Seja objetivo, técnico e use os números das projeções. Máximo 300 palavras."""

    try:
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"Content-Type": "application/json"},
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        if r.status_code == 200:
            for block in r.json().get("content", []):
                if block.get("type") == "text":
                    return block["text"]
    except Exception:
        pass
    return _analise_fallback(indicadores_sel, resultados, meses_pred)


def _analise_fallback(indicadores_sel, resultados, meses_pred):
    linhas = []
    for ind in indicadores_sel:
        r    = resultados[ind]
        hist = r["historico"]["valor"].iloc[-1]
        pred = r["preds"][-1]
        var  = pred - hist
        linhas.append(f"**{ind}:** {hist:.4f}% → {pred:.4f}% ({var:+.4f}pp)")

    ipca_trend = ""
    if "IPCA" in resultados:
        v0 = resultados["IPCA"]["historico"]["valor"].iloc[-1]
        vf = resultados["IPCA"]["preds"][-1]
        ipca_trend = "alta" if vf > v0 else "queda"

    return f"""**1. Cenário Macro**
As projeções sugerem {'pressão inflacionária' if ipca_trend == 'alta' else 'arrefecimento inflacionário'} 
no horizonte de {meses_pred} meses. O modelo capturou padrões históricos de sazonalidade e tendência 
para gerar estas estimativas.

**2. Destaques por Indicador**
{chr(10).join(linhas)}

**3. Correlações Relevantes**
IPCA e IGPM tendem a se mover juntos em ciclos de pressão de custos. 
A SELIC impacta diretamente CDI e, com defasagem, os índices de preços.

**4. Implicações Práticas**
Contratos indexados ao IGPM devem ser monitorados. Para investimentos em renda fixa, 
acompanhe a trajetória da SELIC. Reajustes de aluguel indexados ao IPCA/IGPM merecem atenção.

**5. Riscos e Incertezas**
(1) Choques externos (câmbio, commodities) podem desviar significativamente as projeções.
(2) Decisões de política monetária do COPOM não antecipadas pelo modelo histórico.
(3) Sazonalidades atípicas (energia, alimentos) podem gerar desvios no curto prazo.

⚠️ *Análise local — Claude API indisponível (configure ANTHROPIC_API_KEY no ambiente).*"""


# ══════════════════════════════════════════════════════════════════════════════
#  BLOCO 4 — INTERFACE STREAMLIT
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="IndPredict AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

:root {
  --bg:#0a0d14; --surface:#111520; --surface2:#1a2035;
  --accent:#00e5ff; --a2:#ff6b35; --a3:#7c3aed;
  --text:#e2e8f0; --muted:#64748b; --green:#10b981; --red:#f43f5e;
}
html,body,[class*="css"]{
  font-family:'Space Grotesk',sans-serif!important;
  background:var(--bg)!important; color:var(--text)!important;
}
.stApp{background:var(--bg)!important;}
section[data-testid="stSidebar"]{
  background:var(--surface)!important; border-right:1px solid #1e293b;
}
.hero{font-size:2.6rem;font-weight:700;
  background:linear-gradient(135deg,var(--accent),var(--a3));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  line-height:1.1;margin-bottom:.3rem;}
.sub{color:var(--muted);font-size:.95rem;margin-bottom:1.5rem;}
.tag{display:inline-block;background:var(--surface2);border:1px solid var(--accent);
  color:var(--accent);border-radius:6px;padding:.15rem .6rem;
  font-size:.72rem;font-family:'DM Mono',monospace;margin-right:.4rem;}
.kpi{background:var(--surface2);border:1px solid #1e293b;border-radius:12px;
  padding:1.1rem 1.4rem;position:relative;overflow:hidden;margin-bottom:.5rem;}
.kpi::before{content:'';position:absolute;top:0;left:0;width:3px;height:100%;
  background:var(--accent);}
.kpi.up::before{background:var(--green);}
.kpi.dn::before{background:var(--red);}
.kpi.neu::before{background:var(--a2);}
.kv{font-size:1.7rem;font-weight:700;color:var(--accent);
  font-family:'DM Mono',monospace;}
.kl{font-size:.72rem;color:var(--muted);text-transform:uppercase;letter-spacing:.1em;}
.ks{font-size:.8rem;color:var(--muted);margin-top:.2rem;}
.sec{font-size:1.05rem;font-weight:600;color:var(--text);
  border-bottom:1px solid #1e293b;padding-bottom:.4rem;margin:1.4rem 0 .9rem;}
.infobox{background:var(--surface2);border-left:3px solid var(--a3);
  border-radius:0 8px 8px 0;padding:.9rem 1.1rem;
  font-size:.88rem;color:#e2e8f0;line-height:1.75;margin-bottom:1rem;}
.badge-real{background:#10b98122;border:1px solid #10b981;color:#10b981;
  border-radius:6px;padding:.1rem .5rem;font-size:.7rem;font-family:'DM Mono',monospace;}
.badge-sim{background:#f59e0b22;border:1px solid #f59e0b;color:#f59e0b;
  border-radius:6px;padding:.1rem .5rem;font-size:.7rem;font-family:'DM Mono',monospace;}
.stButton>button{
  background:linear-gradient(135deg,var(--a3),#4f46e5);
  color:white;border:none;border-radius:8px;
  font-family:'Space Grotesk',sans-serif;font-weight:600;
  padding:.6rem 2rem;transition:opacity .2s;}
.stButton>button:hover{opacity:.85;}
.stSelectbox label,.stSlider label,.stMultiSelect label,.stRadio label,.stCheckbox label{
  color:var(--muted)!important;font-size:.78rem;
  text-transform:uppercase;letter-spacing:.08em;}
hr{border-color:#1e293b!important;}
</style>
""", unsafe_allow_html=True)

# ── Cabeçalho ──────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:1.2rem 0 .3rem;">
  <div class="hero">IndPredict AI 📊</div>
  <div class="sub">Predição de Indicadores Econômicos com Machine Learning + IA</div>
  <span class="tag">IPCA</span><span class="tag">IGPM</span><span class="tag">INPC</span>
  <span class="tag">SELIC</span><span class="tag">INCC</span><span class="tag">CDI</span>
  <span class="tag" style="border-color:#00e5ff88">📍 IPCA-Campinas</span>
  <span class="tag" style="border-color:#00e5ff88">📍 IPCA-SP</span>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configurações")

    st.markdown("**Nacionais (BCB)**")
    ind_sel = st.multiselect(
        "📊 Indicadores nacionais",
        list(SERIES_BCB.keys()),
        default=["IPCA", "IGPM", "SELIC"],
        key="nacionais",
    )

    st.markdown("**Regionais — Campinas/SP (IBGE)**")
    ind_regionais = st.multiselect(
        "📍 Indicadores regionais",
        list(IBGE_LOCALIDADES.keys()),
        default=["IPCA-Campinas"],
        key="regionais",
    )
    ind_sel = ind_sel + [r for r in ind_regionais if r not in ind_sel]
    if not ind_sel:
        ind_sel = ["IPCA"]

    meses_hist = st.slider("📅 Histórico (meses)", 12, 84, 48, step=6)
    meses_pred = st.slider("🔭 Previsão (meses)",   3, 24,  6, step=1)

    st.markdown("---")
    st.markdown("### 🤖 Modelo")
    algo = st.radio("Algoritmo", ["Gradient Boosting", "Random Forest", "Ridge"])
    usar_ia = st.checkbox("Análise IA (Claude)", value=True)

    st.markdown("---")
    st.markdown("### 📡 Dados")
    st.markdown("""
    <div style="font-size:.78rem;color:#64748b;line-height:1.6;">
    <b style="color:#10b981">Nacionais:</b> API do Banco Central (SGS)<br>
    <b style="color:#00e5ff">Regionais:</b> IBGE SIDRA — tabela 7060<br>
    Campinas = Região Metropolitana de Campinas<br><br>
    Se indisponível, usa série simulada
    <span class="badge-sim">simulado</span> como fallback.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    rodar = st.button("▶ Gerar Predição", use_container_width=True)

# ── State ──────────────────────────────────────────────────────────────────
if "res" not in st.session_state:
    st.session_state.res = None

# ── Executar ───────────────────────────────────────────────────────────────
if rodar:
    with st.spinner("🔄 Carregando séries e treinando modelos..."):
        resultados   = {}
        fontes_reais = {}

        for ind in ind_sel:
            df_hist, is_real = get_serie(ind, meses_hist)
            resultado = prever(df_hist, meses_pred, algo)
            resultados[ind]   = resultado
            fontes_reais[ind] = is_real

        st.session_state.res = {
            "resultados":   resultados,
            "fontes_reais": fontes_reais,
            "ind_sel":      ind_sel,
            "meses_pred":   meses_pred,
            "algo":         algo,
            "usar_ia":      usar_ia,
        }

# ── Exibir ─────────────────────────────────────────────────────────────────
if st.session_state.res:
    res        = st.session_state.res
    resultados = res["resultados"]
    ind_sel    = res["ind_sel"]
    meses_pred = res["meses_pred"]
    fontes     = res["fontes_reais"]

    CORES = ["#00e5ff", "#ff6b35", "#7c3aed", "#10b981", "#f59e0b", "#f43f5e"]

    # ── KPIs ──────────────────────────────────────────────────────────────
    st.markdown('<div class="sec">📈 Resumo das Predições</div>', unsafe_allow_html=True)
    cols = st.columns(min(len(ind_sel), 4))
    for i, ind in enumerate(ind_sel[:4]):
        r     = resultados[ind]
        atual = r["historico"]["valor"].iloc[-1]
        futuro = r["preds"][-1]
        delta  = futuro - atual
        cls    = "up" if delta > 0.02 else ("dn" if delta < -0.02 else "neu")
        seta   = "↑" if delta > 0 else "↓"
        badge  = '<span class="badge-real">● real</span>' if fontes[ind] \
                 else '<span class="badge-sim">● simulado</span>'
        with cols[i % 4]:
            st.markdown(f"""
            <div class="kpi {cls}">
              <div class="kl">{ind} {badge}</div>
              <div class="kv">{futuro:.4f}%</div>
              <div class="ks">Hoje: {atual:.4f}% | {seta} {abs(delta):+.4f}pp em {meses_pred}m</div>
              <div class="ks" style="font-size:.7rem;margin-top:.2rem;">
                R² {r['metricas']['r2']:.3f} · MAE {r['metricas']['mae']:.5f}</div>
            </div>""", unsafe_allow_html=True)

    # ── Gráfico principal por indicador ───────────────────────────────────
    st.markdown('<div class="sec">💹 Histórico + Previsão</div>', unsafe_allow_html=True)

    # Um gráfico por indicador, em grid
    n_cols = min(len(ind_sel), 2)
    n_rows = (len(ind_sel) + 1) // 2
    subplot_titles = [f"{ind} — {DESCRICOES[ind]}" for ind in ind_sel]

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.12,
        horizontal_spacing=0.07,
    )

    for idx, ind in enumerate(ind_sel):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        cor = CORES[idx % len(CORES)]
        rgb = tuple(int(cor.lstrip("#")[j:j+2], 16) for j in (0, 2, 4))
        r   = resultados[ind]
        hist = r["historico"]
        datas_fut = r["datas_fut"]
        preds     = r["preds"]
        ci_low    = r["ci_lower"]
        ci_high   = r["ci_upper"]

        # Linha histórica
        fig.add_trace(go.Scatter(
            x=hist["data"], y=hist["valor"],
            name=f"{ind} histórico",
            line=dict(color=cor, width=2),
            mode="lines",
            legendgroup=ind,
        ), row=row, col=col)

        # Linha de previsão
        # Conectar último ponto histórico à previsão
        x_pred = [hist["data"].iloc[-1]] + list(datas_fut)
        y_pred = [hist["valor"].iloc[-1]] + list(preds)

        fig.add_trace(go.Scatter(
            x=x_pred, y=y_pred,
            name=f"{ind} previsão",
            line=dict(color=cor, width=2.5, dash="dot"),
            mode="lines+markers",
            marker=dict(size=5, symbol="circle"),
            legendgroup=ind,
        ), row=row, col=col)

        # Intervalo de confiança 90%
        x_ci = list(datas_fut) + list(datas_fut)[::-1]
        y_ci = list(ci_high) + list(ci_low)[::-1]
        fig.add_trace(go.Scatter(
            x=x_ci, y=y_ci,
            fill="toself",
            fillcolor=f"rgba{rgb + (0.12,)}",
            line=dict(color="rgba(0,0,0,0)"),
            name=f"{ind} IC 90%",
            legendgroup=ind,
            showlegend=False,
        ), row=row, col=col)

        # Linha vertical "hoje" via add_shape (compatível com todas as versões do Plotly)
        fig.add_shape(
            type="line",
            xref="x", yref="paper",
            x0=hist["data"].iloc[-1], x1=hist["data"].iloc[-1],
            y0=0, y1=1,
            line=dict(color="#334155", width=1, dash="dash"),
            row=row, col=col,
        )

    fig.update_layout(
        plot_bgcolor="#111520", paper_bgcolor="#111520",
        font=dict(family="Space Grotesk", color="#e2e8f0"),
        height=320 * n_rows,
        margin=dict(l=0, r=0, t=40, b=0),
        hovermode="x unified",
        legend=dict(bgcolor="#1a2035", bordercolor="#1e293b", borderwidth=1),
    )
    for i in range(1, n_rows + 1):
        for j in range(1, n_cols + 1):
            fig.update_xaxes(gridcolor="#1e293b", row=i, col=j)
            fig.update_yaxes(gridcolor="#1e293b", ticksuffix="%", row=i, col=j)

    st.plotly_chart(fig, use_container_width=True)

    # ── Gráfico comparativo (todos no mesmo eixo, normalizado) ────────────
    if len(ind_sel) > 1:
        st.markdown('<div class="sec">🔀 Comparação Normalizada (variação relativa %)</div>',
                    unsafe_allow_html=True)

        fig2 = go.Figure()
        for idx, ind in enumerate(ind_sel):
            cor  = CORES[idx % len(CORES)]
            r    = resultados[ind]
            hist = r["historico"]
            base = hist["valor"].iloc[0] if hist["valor"].iloc[0] != 0 else 1e-6

            y_hist_norm = (hist["valor"] / abs(base) - 1) * 100
            x_pred = [hist["data"].iloc[-1]] + list(r["datas_fut"])
            y_pred_norm = [(v / abs(base) - 1) * 100
                           for v in [hist["valor"].iloc[-1]] + list(r["preds"])]

            fig2.add_trace(go.Scatter(
                x=hist["data"], y=y_hist_norm,
                name=f"{ind}",
                line=dict(color=cor, width=2), mode="lines"))
            fig2.add_trace(go.Scatter(
                x=x_pred, y=y_pred_norm,
                name=f"{ind} (prev.)",
                line=dict(color=cor, width=2, dash="dot"),
                showlegend=False))

        hoje_x = resultados[ind_sel[0]]["historico"]["data"].iloc[-1]
        fig2.add_shape(
            type="line",
            xref="x", yref="paper",
            x0=hoje_x, x1=hoje_x,
            y0=0, y1=1,
            line=dict(color="#334155", width=1, dash="dash"),
        )
        fig2.add_annotation(
            x=hoje_x, y=1, xref="x", yref="paper",
            text="Hoje", showarrow=False,
            font=dict(color="#64748b", size=11),
            xanchor="left", yanchor="bottom",
        )

        fig2.update_layout(
            plot_bgcolor="#111520", paper_bgcolor="#111520",
            font=dict(family="Space Grotesk", color="#e2e8f0"),
            xaxis=dict(gridcolor="#1e293b"),
            yaxis=dict(gridcolor="#1e293b", ticksuffix="%",
                       title="Variação relativa ao início do período"),
            legend=dict(bgcolor="#1a2035", bordercolor="#1e293b", borderwidth=1),
            height=350, margin=dict(l=0, r=0, t=10, b=0),
            hovermode="x unified")
        st.plotly_chart(fig2, use_container_width=True)

    # ── Análise IA ─────────────────────────────────────────────────────────
    if res["usar_ia"]:
        st.markdown('<div class="sec">🤖 Análise Econômica (Claude AI)</div>',
                    unsafe_allow_html=True)
        with st.spinner("Consultando Claude..."):
            texto = analise_ia(ind_sel, resultados, meses_pred)
        st.markdown(
            f'<div class="infobox">{texto.replace(chr(10), "<br>")}</div>',
            unsafe_allow_html=True)

    # ── Tabela detalhada por indicador ─────────────────────────────────────
    st.markdown('<div class="sec">📋 Previsão Mensal Detalhada</div>',
                unsafe_allow_html=True)

    tabs = st.tabs([f"{ind} {'🟢' if fontes[ind] else '🟡'}" for ind in ind_sel])
    for tab, ind in zip(tabs, ind_sel):
        with tab:
            r         = resultados[ind]
            hist_ult  = r["historico"]["valor"].iloc[-1]
            acumulado = 0.0
            rows = []
            for i, (dt, pred, ci_l, ci_h) in enumerate(
                zip(r["datas_fut"], r["preds"], r["ci_lower"], r["ci_upper"])
            ):
                delta    = pred - hist_ult
                acumulado += pred
                rows.append({
                    "Mês":         dt.strftime("%b/%Y"),
                    "Previsto (%)": f"{pred:.4f}%",
                    "IC Inferior": f"{ci_l:.4f}%",
                    "IC Superior": f"{ci_h:.4f}%",
                    "Δ vs Atual":  f"{delta:+.4f}pp",
                    "Acum. Período": f"{acumulado:.4f}%",
                })

            # Linha de referência: último valor histórico
            st.markdown(
                f"Último valor histórico: **{hist_ult:.4f}%** | "
                f"Fonte: {'🟢 Banco Central (real)' if fontes[ind] else '🟡 Série simulada'} | "
                f"Modelo: **{res['algo']}**"
            )
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
            )

            # Mini-estatísticas
            preds_arr = r["preds"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Mínimo",  f"{preds_arr.min():.4f}%")
            c2.metric("Máximo",  f"{preds_arr.max():.4f}%")
            c3.metric("Média",   f"{preds_arr.mean():.4f}%")
            c4.metric("Acum.",   f"{preds_arr.sum():.4f}%")

else:
    # ── Estado vazio ───────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center;padding:5rem 2rem;">
      <div style="font-size:4rem;">📊</div>
      <div style="font-size:1.15rem;color:#64748b;margin:.8rem 0 .4rem;">
        Selecione os indicadores e clique em
        <strong style="color:#00e5ff">▶ Gerar Predição</strong>
      </div>
      <div style="font-size:.85rem;color:#475569;">
        O modelo busca dados reais do Banco Central do Brasil (SGS) e treina
        um algoritmo de ML para projetar os indicadores nos próximos meses.
      </div>
    </div>
    """, unsafe_allow_html=True)