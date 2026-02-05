import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Valuation Master Pro", layout="wide", page_icon="💎")

# --- CSS PERSONNALISÉ ---
st.markdown("""
<style>
    .stApp { background-color: #0E1117; }
    .price-card-up {
        background: linear-gradient(135deg, rgba(0, 200, 83, 0.2), rgba(0, 200, 83, 0.05));
        border: 1px solid #00C853;
        border-radius: 15px;
        padding: 15px;
        text-align: right;
        box-shadow: 0 0 15px rgba(0, 200, 83, 0.2);
    }
    .price-card-down {
        background: linear-gradient(135deg, rgba(255, 23, 68, 0.2), rgba(255, 23, 68, 0.05));
        border: 1px solid #FF1744;
        border-radius: 15px;
        padding: 15px;
        text-align: right;
        box-shadow: 0 0 15px rgba(255, 23, 68, 0.2);
    }
    .price-big { font-size: 32px; font-weight: 800; color: white; margin: 0; line-height: 1.2; }
    .price-var { font-size: 18px; font-weight: 600; margin: 0; }
    .text-green { color: #00E676; }
    .text-red { color: #FF5252; }
    .metric-card {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #333;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 10px;
    }
    h1 { margin-top: 0px; padding-top: 0px; font-size: 2.5rem; }
    .score-box {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 20px;
        background-color: #1E1E1E;
        border: 1px solid #333;
    }
    .score-val { font-size: 3.5rem; font-weight: 900; margin: 0; line-height: 1; }
    .stTabs [data-baseweb="tab-list"] { gap: 20px; }
    .stTabs [data-baseweb="tab"] { height: 50px; background-color: transparent; border-radius: 4px; color: #aaa; font-weight: 600; }
    .stTabs [aria-selected="true"] { background-color: #262730; color: white; border-bottom: 2px solid #00C9FF; }
</style>
""", unsafe_allow_html=True)

# --- TA SÉLECTION DE NEWS MANUELLES ---
MY_CURATED_NEWS = {
    "NVDA": [
        {"title": "Nvidia dévoile sa nouvelle puce Blackwell", "source": "Site Perso / Note", "link": "https://www.nvidia.com/fr-fr/"},
        {"title": "Analyse : Pourquoi l'IA n'est qu'au début", "source": "Mon Blog", "link": "https://google.com"}
    ],
    "CSU": [ 
        {"title": "La stratégie d'acquisition de Constellation expliquée", "source": "Dossier Spécial", "link": "#"},
    ],
    "AMZN": [
        {"title": "Amazon AWS continue de dominer le cloud", "source": "TechCrunch", "link": "#"},
    ],
    "NBIS": [
        {"title": "Dernières avancées sur NBIS", "source": "Presse Spécialisée", "link": "#"},
    ],
    "FICO": [
        {"title": "FICO Score : Un monopole inébranlable ?", "source": "Analyse Moat", "link": "#"},
    ],
    "NOW": [ 
        {"title": "ServiceNow intègre l'IA générative", "source": "Bloomberg", "link": "#"},
    ],
    "NVO": [
        {"title": "Novo Nordisk et le succès du Wegovy", "source": "Santé Mag", "link": "#"},
    ]
}

# --- GESTION DE L'ÉTAT ---
if 'selected_metrics' not in st.session_state:
    st.session_state['selected_metrics'] = []

# --- FONCTION DE CHARGEMENT ROBUSTE (CACHE) ---
# C'est cette partie qui corrige ton problème de chargement
@st.cache_data(ttl=3600, show_spinner=False)
def get_stock_data(symbol):
    """Récupère toutes les données d'un coup et les met en cache pour 1h"""
    try:
        ticker = yf.Ticker(symbol)
        
        # On force le téléchargement ici pour attraper les erreurs
        info = ticker.info
        history = ticker.history(period="10y")
        
        # On récupère les états financiers sous forme de DataFrame (plus fiable pour le cache)
        financials = ticker.financials
        balance_sheet = ticker.balance_sheet
        cashflow = ticker.cashflow
        news = ticker.news
        
        # On essaie de récupérer le FCF ou l'Operating Cash Flow
        fcf_val = 0
        if cashflow is not None and not cashflow.empty:
            try:
                # Tentative 1 : Free Cash Flow explicite
                if 'Free Cash Flow' in cashflow.index:
                    fcf_val = cashflow.loc['Free Cash Flow'].iloc[0]
                # Tentative 2 : Operating CF + Capex
                elif 'Operating Cash Flow' in cashflow.index:
                    ocf = cashflow.loc['Operating Cash Flow'].iloc[0]
                    # Chercher CAPEX
                    capex = 0
                    for cbox in ['Capital Expenditure', 'Capital Expenditures', 'Payments For Property, Plant, And Equipment']:
                        if cbox in cashflow.index:
                            capex = cashflow.loc[cbox].iloc[0]
                            break
                    fcf_val = ocf + capex # Capex est souvent négatif
            except:
                fcf_val = 0

        return {
            "info": info,
            "history": history,
            "financials": financials,
            "balance_sheet": balance_sheet,
            "cashflow": cashflow,
            "news": news,
            "fcf": fcf_val
        }
    except Exception as e:
        return None

# --- ALGORITHME DE SCORING ---
def analyze_quality(info):
    score = 0
    details = []
    
    # 1. VALORISATION
    pe = info.get('trailingPE')
    fwd_pe = info.get('forwardPE')
    used_pe = fwd_pe if fwd_pe else (pe if pe else 0)
    
    if used_pe > 0:
        if used_pe < 25: score += 6; details.append("✅ Valo : Action bon marché (PE < 25)")
        elif used_pe < 45: score += 4; details.append("✅ Valo : Prix raisonnable pour croissance (PE < 45)")
        elif used_pe < 60: score += 2; details.append("⚠️ Valo : Prix élevé mais acceptable si hyper-croissance")
        else: details.append("❌ Valo : Action très chère (PE > 60)")
    else: details.append("⚠️ Valo : Pas de PE (Bénéfices négatifs ?)")

    # 2. RENTABILITÉ
    roe = info.get('returnOnEquity', 0)
    margins = info.get('profitMargins', 0)
    
    if roe > 0.20: score += 4; details.append(f"✅ Rentabilité : ROE excellent ({roe:.1%})")
    elif roe > 0.12: score += 2; details.append(f"✅ Rentabilité : ROE correct ({roe:.1%})")
    else: details.append(f"❌ Rentabilité : ROE faible ({roe:.1%})")
    
    if margins > 0.15: score += 4; details.append(f"✅ Marges : Très rentables ({margins:.1%})")
    elif margins > 0.08: score += 2; details.append(f"✅ Marges : Correctes ({margins:.1%})")
    else: details.append("❌ Marges : Faibles")

    # 3. SANTÉ & CROISSANCE
    debt_equity = info.get('debtToEquity', 0)
    rev_growth = info.get('revenueGrowth', 0)

    if debt_equity < 150: score += 3; details.append("✅ Bilan : Dette maîtrisée")
    else: details.append("⚠️ Bilan : Endettement à surveiller")
    
    if rev_growth > 0.10: score += 3; details.append(f"✅ Croissance : Dynamique ({rev_growth:.1%})")
    elif rev_growth > 0: score += 1; details.append("⚠️ Croissance : Molle")
    else: details.append("❌ Croissance : Chiffre d'affaires en baisse")

    return score, details

# --- FONCTIONS UTILITAIRES ---
def get_currency_rate(base_currency, target_currency="USD"):
    if base_currency == target_currency: return 1.0
    try:
        pair = f"{base_currency}{target_currency}=X"
        # On ne cache pas ça car c'est rapide
        hist = yf.Ticker(pair).history(period="1d")
        return hist['Close'].iloc[-1] if not hist.empty else 1.0
    except: return 1.0

def format_usd(value):
    if value is None or pd.isna(value): return "N/A"
    if abs(value) >= 1e9: return f"$ {value / 1e9:.2f} B"
    elif abs(value) >= 1e6: return f"$ {value / 1e6:.2f} M"
    return f"$ {value:.2f}"

def format_percent(val):
    if val is None or pd.isna(val): return "N/A"
    if abs(val) < 0.01 and abs(val) > 0:
        return f"{val*100:.4f}%"
    return f"{val*100:.2f}%"

# --- SIDEBAR ---
with st.sidebar:
    st.header("🔍 Recherche")
    ticker_input = st.text_input("Symbole (ex: LVMH.PA, NVDA)", value="NVDA").upper()
    st.markdown("---")
    st.info("💡 Ajoutez `.PA` pour la France.")
    if st.button("🔄 Rafraîchir les données"):
        st.cache_data.clear() # Bouton pour vider le cache si besoin

# --- MAIN ---
if ticker_input:
    
    with st.spinner(f'Chargement des données pour {ticker_input}...'):
        data_bundle = get_stock_data(ticker_input)

    if data_bundle is None or not data_bundle['info'] or len(data_bundle['info']) < 2:
        st.error(f"❌ Oups ! Impossible de trouver les infos pour '{ticker_input}'.")
        st.warning("👉 Vérifiez le symbole ou réessayez dans quelques instants.")
        st.stop()
    
    # On "déballe" les données
    info = data_bundle['info']
    hist = data_bundle['history']
    news_list = data_bundle['news']
    
    currency_base = info.get('currency', 'USD')
    exchange_rate = get_currency_rate(currency_base, "USD")
    
    # --- EN-TÊTE ---
    col_infos, col_price = st.columns([0.75, 0.25])
    
    with col_infos:
        st.title(info.get('longName', ticker_input))
        st.markdown(f"**Secteur:** {info.get('sector', 'N/A')} &nbsp; | &nbsp; **Devise d'origine:** {currency_base} (Converti en USD)")

    with col_price:
        curr_price_orig = info.get('currentPrice', 0)
        curr_price_usd = curr_price_orig * exchange_rate
        prev_close = info.get('previousClose', curr_price_orig)
        
        if prev_close: change_pct = ((curr_price_orig - prev_close) / prev_close) * 100
        else: change_pct = 0
            
        if change_pct >= 0: css_class = "price-card-up"; color_class = "text-green"; sign = "+"
        else: css_class = "price-card-down"; color_class = "text-red"; sign = ""
        
        st.markdown(f"""
        <div class="{css_class}">
            <p class="price-big">${curr_price_usd:.2f}</p>
            <p class="price-var {color_class}">{sign}{change_pct:.2f}% (1J)</p>
        </div>
        """, unsafe_allow_html=True)

    # --- ONGLETS ---
    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Vue d'ensemble", "📈 Graphique", "🔮 Simulateur", "🏢 Fondamentaux", "⚔️ Comparaison"])

    # --- TAB 1 : VUE D'ENSEMBLE ---
    with tab1:
        col_desc, col_score = st.columns([2, 1])
        
        with col_desc:
            st.subheader("📝 À propos")
            summary = info.get('longBusinessSummary', "Description non disponible.")
            with st.expander("📖 Lire la description complète de l'entreprise", expanded=False):
                st.write(summary)
            
            st.markdown("---")
            
            div_rate = info.get('dividendRate', 0)
            curr_price = info.get('currentPrice', 1)
            
            if div_rate and curr_price:
                real_div_yield = div_rate / curr_price
            else:
                real_div_yield = info.get('dividendYield', 0)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Market Cap", format_usd(info.get('marketCap', 0) * exchange_rate))
            c2.metric("P/E (Est.)", f"{info.get('forwardPE', info.get('trailingPE', 0)):.1f}")
            c3.metric("Beta", f"{info.get('beta', 0):.2f}")
            c4.metric("Dividende %", format_percent(real_div_yield))

        with col_score:
            score, reasons = analyze_quality(info)
            color_score = "#00E676" if score >= 15 else ("#FFAB00" if score >= 10 else "#FF5252")
            
            st.markdown(f"""
            <div class="score-box">
                <h4 style="margin:0; color:#aaa;">Score IA</h4>
                <p class="score-val" style="color:{color_score};">{score}/20</p>
                <p style="margin-top:5px; font-size:0.9em; color:{color_score};">
                    { "EXCELLENT" if score >= 15 else ("CORRECT" if score >= 10 else "RISQUÉ") }
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("**Détails de la notation :**")
            for r in reasons:
                st.caption(r)

        st.markdown("---")

        # === SECTION ACTUALITÉS & ANALYSES ===
        st.subheader("📰 Actualités & Analyses")

        clean_ticker = ticker_input.split('.')[0] 
        
        if ticker_input in MY_CURATED_NEWS or clean_ticker in MY_CURATED_NEWS:
            news_to_show = MY_CURATED_NEWS.get(ticker_input, MY_CURATED_NEWS.get(clean_ticker))
            st.markdown(f"#### ⭐ Sélection de l'expert pour {ticker_input}")
            for item in news_to_show:
                st.markdown(f"""
                <div style="background-color: #1a2e1a; padding: 15px; border-radius: 10px; border-left: 4px solid #00E676; margin-bottom: 10px;">
                    <div style="font-size: 0.85em; color: #00E676; margin-bottom: 5px;">
                        <strong>📌 SÉLECTION MANUELLE • {item['source']}</strong>
                    </div>
                    <a href="{item['link']}" target="_blank" style="text-decoration: none; color: white; font-size: 1.1em; font-weight: 600;">
                        {item['title']}
                    </a>
                </div>
                """, unsafe_allow_html=True)
            
        with st.expander("🌍 Voir les actualités automatiques du web", expanded=True):
            if news_list:
                for item in news_list[:3]:
                    title = item.get('title', 'Sans titre')
                    publisher = item.get('publisher', 'Inconnu')
                    link = item.get('link', '#')
                    st.markdown(f"""
                    <div style="background-color: #1E1E1E; padding: 10px; border-radius: 8px; margin-bottom: 8px; border: 1px solid #333;">
                        <small style="color: #888;">{publisher}</small><br>
                        <a href="{link}" target="_blank" style="text-decoration: none; color: #ddd; font-weight: 500;">{title}</a>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Pas de news automatiques récentes.")

    # --- TAB 2 : GRAPHIQUE ---
    with tab2:
        if not hist.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hist.index, y=hist['Close'], mode='lines', name='Prix', 
                line=dict(color='#2962FF', width=2), fill='tozeroy', fillcolor='rgba(41, 98, 255, 0.1)'
            ))
            fig.update_xaxes(rangeslider_visible=False, gridcolor='#333')
            fig.update_yaxes(gridcolor='#333')
            fig.update_layout(template="plotly_dark", margin=dict(l=0, r=0, t=30, b=0), height=500, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        else:
            st.warning("Pas de données graphiques disponibles.")

    # --- TAB 3 : SIMULATEUR DCF ---
    with tab3:
        st.markdown("### 🧮 Modèle DCF Simplifié")
        
        # On utilise le FCF calculé dans la fonction de chargement
        auto_fcf = (data_bundle['fcf'] * exchange_rate) / 1e6
        shares_out = info.get('sharesOutstanding', 1) / 1e9

        col_sim_input, col_sim_viz = st.columns([1, 2])
        with col_sim_input:
            st.markdown('<div class="metric-card">#### 🛠️ Hypothèses', unsafe_allow_html=True)
            in_fcf = st.number_input("FCF départ (M $)", value=float(auto_fcf), step=100.0)
            in_growth = st.slider("Croissance (%)", 0, 40, 10)
            in_shares_chg = st.slider("Var. Actions (%)", -5, 5, -1)
            in_pe = st.number_input("Multiple sortie P/FCF", value=20.0)
            in_discount = st.number_input("Taux d'actualisation (%)", value=10.0)
            st.markdown('</div>', unsafe_allow_html=True)

        years_proj = 5
        years_list = list(range(datetime.now().year, datetime.now().year + years_proj + 1))
        sim_prices, curr_fcf_sim, curr_shares = [curr_price_usd], in_fcf, shares_out
        for i in range(1, years_proj + 1):
            curr_fcf_sim *= (1 + in_growth/100)
            curr_shares *= (1 + in_shares_chg/100)
            # Protection division par zero
            if curr_shares <= 0: curr_shares = 0.001
            sim_prices.append((curr_fcf_sim / (curr_shares * 1000)) * in_pe)
        
        price_target = sim_prices[-1]
        fair_value = price_target / ((1 + in_discount/100) ** years_proj)
        if curr_price_usd > 0:
            upside = ((fair_value - curr_price_usd) / curr_price_usd) * 100
        else: upside = 0

        with col_sim_viz:
            fig_proj = go.Figure()
            fig_proj.add_trace(go.Scatter(x=years_list, y=sim_prices, mode='lines+markers', name='Projection', line=dict(color='#00E676', width=3, dash='dash')))
            fig_proj.add_trace(go.Scatter(x=[years_list[0]], y=[curr_price_usd], mode='markers', name='Actuel', marker=dict(color='white', size=12)))
            fig_proj.update_layout(title="Trajectoire estimée", template="plotly_dark", height=400, paper_bgcolor='rgba(0,0,0,0)', yaxis_title="Prix ($)")
            st.plotly_chart(fig_proj, use_container_width=True, config={'displayModeBar': False})
            r1, r2, r3 = st.columns(3)
            r1.metric("Cible (5 ans)", f"${price_target:.2f}")
            r2.metric("Juste Valeur", f"${fair_value:.2f}", f"{upside:+.1f}%")
            r3.metric("CAGR", f"{((price_target/max(0.1, curr_price_usd))**(1/years_proj)-1)*100:.1f}%")

    # --- TAB 4 : FONDAMENTAUX ---
    with tab4:
        st.subheader("📊 Analyse Fondamentale Historique")
        st.caption("Sélectionnez les métriques pour les comparer.")
        
        # Récupération depuis le bundle
        df_inc = data_bundle['financials'].T
        df_bs = data_bundle['balance_sheet'].T
        df_cf = data_bundle['cashflow'].T
        
        if not df_inc.empty:
            # Fusion des données pour le graph
            try:
                df_fund = df_inc.join(df_bs, lsuffix='_inc', rsuffix='_bs', how='outer')
                df_fund = df_fund.join(df_cf, lsuffix='', rsuffix='_cf', how='outer')
                df_fund = df_fund.sort_index(ascending=True)
                df_fund.index = pd.to_datetime(df_fund.index).year
                plot_data = pd.DataFrame(index=df_fund.index)

                if 'Total Revenue' in df_fund: plot_data['Chiffre d\'Affaires ($B)'] = (df_fund['Total Revenue'] * exchange_rate) / 1e9
                if 'Net Income' in df_fund: plot_data['Résultat Net ($B)'] = (df_fund['Net Income'] * exchange_rate) / 1e9
                if 'Total Assets' in df_fund: plot_data['Total Actifs ($B)'] = (df_fund['Total Assets'] * exchange_rate) / 1e9
                if 'Operating Cash Flow' in df_fund: plot_data['Operating Cash Flow ($B)'] = (df_fund['Operating Cash Flow'] * exchange_rate) / 1e9
                
                # Calcul marges
                if 'Gross Profit' in df_fund and 'Total Revenue' in df_fund: 
                     plot_data['Marge Brute (%)'] = (df_fund['Gross Profit'] / df_fund['Total Revenue'])
                
                plot_data = plot_data.fillna(0)
                
                # Affichage interactif
                metrics_list = plot_data.columns.tolist()
                cols = st.columns(3)
                for i, metric in enumerate(metrics_list):
                    is_checked = metric in st.session_state['selected_metrics']
                    if cols[i % 3].checkbox(metric, key=f"chk_{metric}", value=is_checked):
                        if metric not in st.session_state['selected_metrics']: st.session_state['selected_metrics'].append(metric)
                    else:
                        if metric in st.session_state['selected_metrics']: st.session_state['selected_metrics'].remove(metric)

                selected = st.session_state['selected_metrics']
                if selected:
                    df_melted = plot_data[selected].reset_index().melt(id_vars='index', var_name='Métrique', value_name='Valeur')
                    df_melted.rename(columns={'index': 'Année'}, inplace=True)
                    fig_comp = px.line(df_melted, x='Année', y='Valeur', color='Métrique', markers=True, title="Historique", color_discrete_sequence=px.colors.qualitative.Bold)
                    fig_comp.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400)
                    st.plotly_chart(fig_comp, use_container_width=True)
            except Exception as e:
                st.error(f"Erreur de traitement des données financières : {e}")
        else:
            st.info("Données financières détaillées non disponibles.")

    # --- TAB 5 : COMPARATEUR ---
    with tab5:
        st.subheader("⚖️ Comparaison sectorielle")
        comp_input = st.text_input("Concurrents (ex: MSFT, GOOGL)", "MSFT, GOOGL, AMZN").upper()
        if comp_input and st.button("Lancer la comparaison"):
            tickers_list = [t.strip() for t in comp_input.split(',')]
            if ticker_input not in tickers_list: tickers_list.insert(0, ticker_input)
            comp_data = []
            
            progress_bar = st.progress(0)
            for idx, t in enumerate(tickers_list):
                try:
                    # On utilise pas le cache ici pour l'instant pour ne pas alourdir, ou on pourrait
                    s_comp = yf.Ticker(t)
                    i_comp = s_comp.info # Risque de lenteur ici, mais c'est à la demande
                    p_usd = i_comp.get('currentPrice', 0) * get_currency_rate(i_comp.get('currency', 'USD'), "USD")
                    comp_data.append({
                        "Ticker": t, "Prix ($)": p_usd, "P/E": i_comp.get('trailingPE', 0),
                        "P/E Fwd": i_comp.get('forwardPE', 0), "Marge Nette": format_percent(i_comp.get('profitMargins', 0)),
                    })
                except: pass
                progress_bar.progress((idx + 1) / len(tickers_list))
            
            if comp_data:
                df_comp = pd.DataFrame(comp_data)
                st.dataframe(df_comp.set_index("Ticker"), use_container_width=True)
                fig_comp = px.bar(df_comp, x='Ticker', y='P/E Fwd', title="Comparaison P/E Forward", color='Ticker')
                fig_comp.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_comp, use_container_width=True)