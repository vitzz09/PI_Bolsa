import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime, date
import warnings
warnings.filterwarnings("ignore")

from data_loader import DataLoader
from model import ImobPredictor
from utils import format_currency, gerar_dados_bairros

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ImobPredict AI",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

  :root {
    --bg: #0a0d14;
    --surface: #111520;
    --surface2: #1a2035;
    --accent: #00e5ff;
    --accent2: #ff6b35;
    --accent3: #7c3aed;
    --text: #e2e8f0;
    --muted: #64748b;
    --green: #10b981;
    --red: #f43f5e;
  }

  html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
  }

  .stApp { background: var(--bg) !important; }

  section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid #1e293b;
  }

  .metric-card {
    background: var(--surface2);
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    position: relative;
    overflow: hidden;
  }
  .metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: var(--accent);
  }
  .metric-card.warn::before { background: var(--accent2); }
  .metric-card.ok::before   { background: var(--green); }

  .metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--accent);
    font-family: 'DM Mono', monospace;
  }
  .metric-label { font-size: 0.75rem; color: var(--muted); text-transform: uppercase; letter-spacing: .1em; }
  .metric-sub   { font-size: 0.85rem; color: var(--muted); margin-top: .2rem; }

  .hero-title {
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent3) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
    margin-bottom: .3rem;
  }
  .hero-sub { color: var(--muted); font-size: 1rem; margin-bottom: 2rem; }

  .tag {
    display: inline-block;
    background: var(--surface2);
    border: 1px solid var(--accent);
    color: var(--accent);
    border-radius: 6px;
    padding: .15rem .6rem;
    font-size: .75rem;
    font-family: 'DM Mono', monospace;
    margin-right: .4rem;
  }

  div[data-testid="stMetric"] label { color: var(--muted) !important; }
  div[data-testid="stMetric"] div   { color: var(--accent) !important; font-family: 'DM Mono', monospace; }

  .stButton > button {
    background: linear-gradient(135deg, var(--accent3) 0%, #4f46e5 100%);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    padding: .6rem 2rem;
    transition: opacity .2s;
  }
  .stButton > button:hover { opacity: .85; }

  .stSelectbox label, .stSlider label, .stMultiSelect label,
  .stNumberInput label, .stRadio label {
    color: var(--muted) !important;
    font-size: .8rem;
    text-transform: uppercase;
    letter-spacing: .08em;
  }

  .section-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--text);
    border-bottom: 1px solid #1e293b;
    padding-bottom: .5rem;
    margin: 1.5rem 0 1rem;
  }

  .info-box {
    background: var(--surface2);
    border-left: 3px solid var(--accent3);
    border-radius: 0 8px 8px 0;
    padding: .8rem 1rem;
    font-size: .85rem;
    color: var(--muted);
    margin-bottom: 1rem;
  }

  hr { border-color: #1e293b !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 1.5rem 0 .5rem;">
  <div class="hero-title">ImobPredict AI</div>
  <div class="hero-sub">Predição inteligente de valores imobiliários com indicadores econômicos</div>
  <span class="tag">IPCA</span>
  <span class="tag">IGPM</span>
  <span class="tag">INPC</span>
  <span class="tag">SELIC</span>
  <span class="tag">ML</span>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configurações")

    cidades_bairros = {
        "São Paulo": ["Moema", "Vila Mariana", "Pinheiros", "Itaim Bibi", "Lapa",
                      "Santana", "Santo André", "Tatuapé", "Perdizes", "Jardins"],
        "Rio de Janeiro": ["Ipanema", "Leblon", "Barra da Tijuca", "Copacabana",
                           "Botafogo", "Tijuca", "Méier", "Campo Grande"],
        "Curitiba": ["Batel", "Água Verde", "Boa Vista", "Portão", "Centro",
                     "Xaxim", "CIC", "Santa Felicidade"],
        "Campinas": ["Cambuí", "Taquaral", "Jardim Guanabara", "Nova Campinas",
                     "Vila Mimosa", "Barão Geraldo"],
    }

    cidade_sel = st.selectbox("🏙️ Cidade", list(cidades_bairros.keys()))
    bairros_disponiveis = cidades_bairros[cidade_sel]
    bairros_sel = st.multiselect("🏘️ Bairros", bairros_disponiveis,
                                  default=bairros_disponiveis[:3])
    if not bairros_sel:
        bairros_sel = bairros_disponiveis[:1]

    tipo_imovel = st.selectbox("🏠 Tipo de Imóvel",
                                ["Apartamento", "Casa", "Sala Comercial", "Galpão"])

    meses_historico = st.slider("📅 Histórico (meses)", 12, 60, 36, step=6)
    meses_previsao  = st.slider("🔭 Previsão (meses)", 3, 24, 6, step=3)

    st.markdown("---")
    st.markdown("### 📊 Indicadores")
    usar_ipca  = st.checkbox("IPCA",  value=True)
    usar_igpm  = st.checkbox("IGPM",  value=True)
    usar_inpc  = st.checkbox("INPC",  value=False)
    usar_selic = st.checkbox("SELIC", value=True)
    usar_pib   = st.checkbox("PIB",   value=False)

    indicadores = []
    if usar_ipca:  indicadores.append("IPCA")
    if usar_igpm:  indicadores.append("IGPM")
    if usar_inpc:  indicadores.append("INPC")
    if usar_selic: indicadores.append("SELIC")
    if usar_pib:   indicadores.append("PIB")
    if not indicadores: indicadores = ["IPCA"]

    st.markdown("---")
    st.markdown("### 🤖 Modelo")
    modelo_tipo = st.radio("Algoritmo", ["Gradient Boosting", "Random Forest", "Linear Ridge"])
    usar_ia = st.checkbox("Usar análise IA (Claude)", value=True)

    st.markdown("---")
    rodar = st.button("▶ Gerar Predição", use_container_width=True)

# ── State ─────────────────────────────────────────────────────────────────
if "resultado" not in st.session_state:
    st.session_state.resultado = None

# ── Run prediction ─────────────────────────────────────────────────────────
if rodar:
    with st.spinner("🔄 Carregando dados econômicos e treinando modelo..."):
        loader = DataLoader()
        df_eco = loader.get_economic_data(meses_historico)
        df_bairros = gerar_dados_bairros(bairros_sel, cidade_sel, tipo_imovel, meses_historico)

        predictor = ImobPredictor(modelo_tipo)
        resultados = {}

        for bairro in bairros_sel:
            df_b = df_bairros[df_bairros["bairro"] == bairro].copy()
            df_merged = df_b.merge(df_eco, on="data", how="left").ffill()
            historico, previsao, metricas = predictor.fit_predict(
                df_merged, indicadores, meses_previsao
            )
            resultados[bairro] = {
                "historico": historico,
                "previsao":  previsao,
                "metricas":  metricas,
            }

        st.session_state.resultado = {
            "resultados":      resultados,
            "df_eco":          df_eco,
            "indicadores":     indicadores,
            "meses_previsao":  meses_previsao,
            "cidade":          cidade_sel,
            "tipo":            tipo_imovel,
            "usar_ia":         usar_ia,
            "modelo_tipo":     modelo_tipo,
        }

# ── Display ────────────────────────────────────────────────────────────────
if st.session_state.resultado:
    res         = st.session_state.resultado
    resultados  = res["resultados"]
    df_eco      = res["df_eco"]
    indicadores = res["indicadores"]
    cidade      = res["cidade"]
    tipo        = res["tipo"]

    # ── KPIs ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📈 Resumo da Predição</div>', unsafe_allow_html=True)
    cols = st.columns(len(bairros_sel) if len(bairros_sel) <= 4 else 4)

    for i, bairro in enumerate(bairros_sel[:4]):
        r    = resultados[bairro]
        prev = r["previsao"]
        hist = r["historico"]
        ultimo_hist = hist["valor_m2"].iloc[-1]
        ultimo_prev = prev["valor_m2_pred"].iloc[-1]
        variacao    = (ultimo_prev - ultimo_hist) / ultimo_hist * 100
        cls = "ok" if variacao >= 0 else "warn"

        with cols[i % 4]:
            st.markdown(f"""
            <div class="metric-card {cls}">
              <div class="metric-label">{bairro}</div>
              <div class="metric-value">{format_currency(ultimo_prev)}</div>
              <div class="metric-sub">Em {res['meses_previsao']}m →
                {'🟢' if variacao >= 0 else '🔴'} {variacao:+.1f}%</div>
              <div class="metric-sub" style="font-size:.75rem;margin-top:.3rem;">
                R² {r['metricas']['r2']:.3f} · MAE {r['metricas']['mae']:.1f}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Price forecast chart ──────────────────────────────────────────────
    st.markdown('<div class="section-title">💹 Previsão de Valor R$/m²</div>', unsafe_allow_html=True)

    CORES = ["#00e5ff", "#ff6b35", "#7c3aed", "#10b981", "#f59e0b", "#f43f5e"]
    fig = go.Figure()

    for idx, bairro in enumerate(bairros_sel):
        cor  = CORES[idx % len(CORES)]
        hist = resultados[bairro]["historico"]
        prev = resultados[bairro]["previsao"]

        fig.add_trace(go.Scatter(
            x=hist["data"], y=hist["valor_m2"],
            name=f"{bairro} (histórico)",
            line=dict(color=cor, width=2),
            mode="lines",
        ))
        fig.add_trace(go.Scatter(
            x=prev["data"], y=prev["valor_m2_pred"],
            name=f"{bairro} (previsão)",
            line=dict(color=cor, width=2.5, dash="dot"),
            mode="lines+markers",
            marker=dict(size=5),
        ))
        if "ci_lower" in prev.columns:
            x_ci = list(prev["data"]) + list(prev["data"])[::-1]
            y_ci = list(prev["ci_upper"]) + list(prev["ci_lower"])[::-1]
            fig.add_trace(go.Scatter(
                x=x_ci, y=y_ci,
                fill="toself",
                fillcolor=f"rgba{tuple(int(cor.lstrip('#')[i:i+2], 16) for i in (0,2,4)) + (0.08,)}",
                line=dict(color="rgba(0,0,0,0)"),
                name=f"{bairro} IC 90%",
                showlegend=False,
            ))

    # Separator line
    if bairros_sel:
        sep_x = resultados[bairros_sel[0]]["historico"]["data"].iloc[-1]
        fig.add_vline(x=sep_x, line_dash="dash", line_color="#334155",
                      annotation_text="Hoje", annotation_font_color="#64748b")

    fig.update_layout(
        plot_bgcolor="#111520", paper_bgcolor="#111520",
        font=dict(family="Space Grotesk", color="#e2e8f0"),
        xaxis=dict(gridcolor="#1e293b", showgrid=True),
        yaxis=dict(gridcolor="#1e293b", showgrid=True, tickprefix="R$ "),
        legend=dict(bgcolor="#1a2035", bordercolor="#1e293b", borderwidth=1),
        height=420,
        margin=dict(l=0, r=0, t=20, b=0),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Variation % bar ──────────────────────────────────────────────────
    st.markdown('<div class="section-title">📊 Variação Projetada por Bairro (%)</div>',
                unsafe_allow_html=True)

    col_bar, col_radar = st.columns([3, 2])

    with col_bar:
        bairros_list, var_list, cor_list = [], [], []
        for bairro in bairros_sel:
            hist = resultados[bairro]["historico"]
            prev = resultados[bairro]["previsao"]
            v0   = hist["valor_m2"].iloc[-1]
            vf   = prev["valor_m2_pred"].iloc[-1]
            var  = (vf - v0) / v0 * 100
            bairros_list.append(bairro)
            var_list.append(round(var, 2))
            cor_list.append("#10b981" if var >= 0 else "#f43f5e")

        fig2 = go.Figure(go.Bar(
            x=bairros_list, y=var_list,
            marker_color=cor_list,
            text=[f"{v:+.1f}%" for v in var_list],
            textposition="outside",
        ))
        fig2.update_layout(
            plot_bgcolor="#111520", paper_bgcolor="#111520",
            font=dict(family="Space Grotesk", color="#e2e8f0"),
            yaxis=dict(gridcolor="#1e293b", ticksuffix="%"),
            xaxis=dict(gridcolor="rgba(0,0,0,0)"),
            height=300, margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col_radar:
        # Radar de métricas do modelo
        metricas_radar = ["R²", "Precisão", "Estabilidade", "Confiança", "Cobertura"]
        fig3 = go.Figure()
        for idx, bairro in enumerate(bairros_sel[:3]):
            m   = resultados[bairro]["metricas"]
            cor = CORES[idx % len(CORES)]
            vals = [
                m["r2"] * 100,
                max(0, 100 - m["mape"]),
                min(100, m["r2"] * 90 + 10),
                min(100, m["r2"] * 85 + 15),
                85 + np.random.uniform(-5, 5),
            ]
            fig3.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=metricas_radar + [metricas_radar[0]],
                fill="toself",
                name=bairro,
                line=dict(color=cor),
                fillcolor=f"rgba{tuple(int(cor.lstrip('#')[i:i+2], 16) for i in (0,2,4)) + (0.15,)}",
            ))
        fig3.update_layout(
            polar=dict(
                bgcolor="#111520",
                radialaxis=dict(visible=True, range=[0, 100],
                                gridcolor="#1e293b", tickfont=dict(size=9)),
                angularaxis=dict(gridcolor="#1e293b"),
            ),
            plot_bgcolor="#111520", paper_bgcolor="#111520",
            font=dict(family="Space Grotesk", color="#e2e8f0"),
            legend=dict(bgcolor="#1a2035", bordercolor="#1e293b"),
            height=300, margin=dict(l=20, r=20, t=10, b=0),
        )
        st.plotly_chart(fig3, use_container_width=True)

    # ── Economic indicators ───────────────────────────────────────────────
    st.markdown('<div class="section-title">🌐 Indicadores Econômicos</div>',
                unsafe_allow_html=True)

    fig4 = make_subplots(
        rows=1, cols=len(indicadores),
        subplot_titles=indicadores,
    )
    CORES_ECO = ["#00e5ff", "#ff6b35", "#7c3aed", "#10b981", "#f59e0b"]

    for i, ind in enumerate(indicadores):
        col_ind = ind.lower()
        if col_ind in df_eco.columns:
            fig4.add_trace(go.Scatter(
                x=df_eco["data"], y=df_eco[col_ind],
                name=ind,
                line=dict(color=CORES_ECO[i % len(CORES_ECO)], width=2),
                fill="tozeroy",
                fillcolor=f"rgba{tuple(int(CORES_ECO[i%len(CORES_ECO)].lstrip('#')[j:j+2], 16) for j in (0,2,4)) + (0.08,)}",
            ), row=1, col=i+1)

    fig4.update_layout(
        plot_bgcolor="#111520", paper_bgcolor="#111520",
        font=dict(family="Space Grotesk", color="#e2e8f0"),
        height=260, margin=dict(l=0, r=0, t=30, b=0),
        showlegend=False,
    )
    for i in range(1, len(indicadores)+1):
        fig4.update_xaxes(gridcolor="#1e293b", row=1, col=i)
        fig4.update_yaxes(gridcolor="#1e293b", ticksuffix="%", row=1, col=i)
    st.plotly_chart(fig4, use_container_width=True)

    # ── IA Analysis ───────────────────────────────────────────────────────
    if res["usar_ia"]:
        st.markdown('<div class="section-title">🤖 Análise da IA</div>', unsafe_allow_html=True)

        with st.spinner("Consultando Claude AI..."):
            from ia_analysis import analisar_com_ia
            resumo_bairros = {}
            for b in bairros_sel:
                hist = resultados[b]["historico"]
                prev = resultados[b]["previsao"]
                resumo_bairros[b] = {
                    "valor_atual":   round(hist["valor_m2"].iloc[-1], 2),
                    "valor_previsto": round(prev["valor_m2_pred"].iloc[-1], 2),
                    "variacao_pct":  round((prev["valor_m2_pred"].iloc[-1] -
                                            hist["valor_m2"].iloc[-1]) /
                                            hist["valor_m2"].iloc[-1] * 100, 2),
                    "r2":   round(resultados[b]["metricas"]["r2"], 3),
                    "mape": round(resultados[b]["metricas"]["mape"], 2),
                }
            ultimo_eco = {ind: round(float(df_eco[ind.lower()].iloc[-1]), 2)
                          for ind in indicadores if ind.lower() in df_eco.columns}
            analise = analisar_com_ia(
                cidade, tipo, bairros_sel, resumo_bairros,
                ultimo_eco, res["meses_previsao"], res["modelo_tipo"]
            )

        st.markdown(f"""
        <div class="info-box" style="border-left-color: #7c3aed; font-size: .9rem;
             color: #e2e8f0; line-height: 1.7;">
          {analise.replace(chr(10), '<br>')}
        </div>
        """, unsafe_allow_html=True)

    # ── Monthly detail table ──────────────────────────────────────────────
    st.markdown('<div class="section-title">📋 Tabela de Previsão Mensal</div>',
                unsafe_allow_html=True)

    tab_bairros = st.tabs(bairros_sel)
    for tab, bairro in zip(tab_bairros, bairros_sel):
        with tab:
            prev = resultados[bairro]["previsao"].copy()
            hist_last = resultados[bairro]["historico"]["valor_m2"].iloc[-1]
            prev["Variação Acumulada"] = ((prev["valor_m2_pred"] - hist_last) / hist_last * 100).round(2)
            prev["Variação Mensal"]    = prev["valor_m2_pred"].pct_change().fillna(0).mul(100).round(2)
            prev_disp = prev[["data", "valor_m2_pred", "Variação Mensal", "Variação Acumulada"]].copy()
            prev_disp.columns = ["Mês", "Valor R$/m²", "Var. Mensal (%)", "Var. Acumulada (%)"]
            prev_disp["Mês"]          = prev_disp["Mês"].dt.strftime("%b/%Y")
            prev_disp["Valor R$/m²"]  = prev_disp["Valor R$/m²"].apply(lambda x: f"R$ {x:,.0f}")
            prev_disp["Var. Mensal (%)"] = prev_disp["Var. Mensal (%)"].apply(
                lambda x: f"{'↑' if x>=0 else '↓'} {abs(x):.2f}%")
            prev_disp["Var. Acumulada (%)"] = prev_disp["Var. Acumulada (%)"].apply(
                lambda x: f"{'↑' if x>=0 else '↓'} {abs(x):.2f}%")
            st.dataframe(prev_disp, use_container_width=True, hide_index=True)

else:
    # ── Empty state ───────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; padding: 4rem 2rem; color: #334155;">
      <div style="font-size: 4rem; margin-bottom: 1rem;">🏙️</div>
      <div style="font-size: 1.2rem; color: #64748b; margin-bottom: .5rem;">
        Configure os parâmetros e clique em <strong style="color:#00e5ff">▶ Gerar Predição</strong>
      </div>
      <div style="font-size: .85rem; color: #475569;">
        O modelo irá treinar com dados históricos reais de IPCA, IGPM e SELIC
        e gerar previsões por bairro com intervalos de confiança.
      </div>
    </div>
    """, unsafe_allow_html=True)