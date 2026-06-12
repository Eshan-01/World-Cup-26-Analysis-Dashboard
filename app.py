import streamlit as st
import pandas as pd
import numpy as np
import itertools
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore')

# --- 1. PAGE CONFIG & THEME ---
st.set_page_config(page_title="Eshans World Cup Analysis Dashboard", layout="wide")

# Premium CSS Injection (With Mobile Responsiveness)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;700&family=Playfair+Display:ital,wght@0,700;1,400&display=swap');
    
    /* App Canvas Background & Texture */
    .stApp {
        background-color: #FFF0DB !important;
        background-image: url("https://www.transparenttextures.com/patterns/cream-paper.png") !important;
    }
    
    /* Font Configurations */
    body, p, span, label, li, div { font-family: 'Inter', sans-serif; color: #1A1A1A; }
    
    /* Hero Title with Gradient */
    .hero-title {
        font-family: 'Brigton', 'Playfair Display', serif !important;
        font-size: 85px !important;
        font-weight: 700;
        text-align: center;
        padding-top: 8vh;
        padding-bottom: 20px;
        background: linear-gradient(135deg, #004225 0%, #006B3C 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.1;
        letter-spacing: -1px;
    }
    
    /* Educational Text Boxes */
    .explainer-box {
        background-color: #FFF8EE; border-left: 4px solid #C8A45D;
        padding: 15px 20px; margin-bottom: 25px; border-radius: 0 4px 4px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .explainer-title { font-family: 'Outfit', sans-serif; color: #004225; font-size: 16px; font-weight: 700; margin-bottom: 5px; text-transform: uppercase; letter-spacing: 0.05em; }
    .explainer-text { font-family: 'Inter', sans-serif; color: #5F5F5F; font-size: 14px; line-height: 1.5; }
    
    h1, h2, h3, h4, h5, h6 { font-family: 'Outfit', sans-serif !important; color: #004225 !important; font-weight: 700 !important; }
    
    /* Dropdown Fix */
    .stSelectbox div[data-baseweb="select"] > div { background-color: #FFF8EE !important; border: 2px solid #E4D7C4 !important; }
    .stSelectbox div[data-baseweb="select"] span { color: #004225 !important; font-family: 'Outfit', sans-serif; font-weight: 600; }
    div[data-baseweb="select"] input { color: #004225 !important; }
    div[data-baseweb="popover"] { background-color: #FFF8EE !important; }
    ul[role="listbox"] { background-color: #FFF8EE !important; }
    li[role="option"] { color: #004225 !important; font-family: 'Outfit', sans-serif !important; font-weight: 600 !important; background-color: #FFF8EE !important;}
    li[role="option"]:hover { background-color: #D9E9E0 !important; color: #004225 !important; }
    
    /* Button Styling */
    button[kind="primary"] {
        background: linear-gradient(135deg, #B78700 0%, #9A6D00 100%) !important;
        background-image: url("https://www.transparenttextures.com/patterns/cubes.png") !important;
        color: #FFFFFF !important; height: 120px !important; font-size: 32px !important;
        font-family: 'Playfair Display', serif !important; font-weight: 700 !important;
        border: 2px solid #E4D7C4 !important; box-shadow: 0 10px 30px rgba(183, 135, 0, 0.3) !important;
        border-radius: 12px !important; transition: all 0.3s ease-in-out !important;
    }
    button[kind="primary"]:hover { transform: translateY(-5px); box-shadow: 0 15px 40px rgba(183, 135, 0, 0.5) !important; }

    button[kind="secondary"] {
        background: linear-gradient(135deg, #FFF8EE 0%, #FFF0DB 100%) !important;
        background-image: url("https://www.transparenttextures.com/patterns/cream-paper.png") !important;
        color: #004225 !important; height: 100px !important; font-size: 20px !important;
        font-family: 'Outfit', sans-serif !important; font-weight: 600 !important;
        border: 1px solid #E4D7C4 !important; box-shadow: 0 4px 10px rgba(0,0,0,0.05) !important;
        border-radius: 8px !important; transition: all 0.2s ease-in-out !important;
    }
    button[kind="secondary"]:hover { background: #004225 !important; background-image: none !important; color: #FFFFFF !important; border-color: #004225 !important; transform: translateY(-3px); }
    
    .nav-back>button { height: 45px !important; font-size: 16px !important; background: #004225 !important; color: #FFFFFF !important; border: none !important; border-radius: 4px !important; }

    /* Profile Panels & Grids */
    .profile-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 24px; }
    .profile-panel { background-color: #FFF8EE; border: 1px solid #E4D7C4; border-radius: 4px; padding: 20px; }
    .profile-label { font-family: 'Outfit', sans-serif; font-size: 13px; text-transform: uppercase; letter-spacing: 0.05em; color: #5F5F5F; margin-bottom: 6px; }
    .profile-value { font-family: 'Inter', sans-serif; font-size: 16px; font-weight: 500; color: #1A1A1A; }
    .profile-panel p, .profile-panel b { color: #1A1A1A !important; margin-bottom: 8px; }
    
    .comp-matrix { display: grid; grid-template-columns: 1fr 300px 1fr; gap: 12px; margin-bottom: 15px; align-items: center; }
    .comp-val-left { text-align: right; font-family: 'JetBrains Mono', monospace; font-weight: 600; color: #004225; font-size: 16px; }
    .comp-val-right { text-align: left; font-family: 'JetBrains Mono', monospace; font-weight: 600; color: #C8A45D; font-size: 16px; }
    .comp-label { text-align: center; font-family: 'Outfit', sans-serif; font-size: 12px; text-transform: uppercase; color: #5F5F5F; font-weight: 600; white-space: nowrap; }
    
    .single-stat-row { display: flex; justify-content: space-between; border-bottom: 1px solid #E4D7C4; padding: 10px 0; }
    .single-stat-label { font-family: 'Outfit', sans-serif; font-size: 13px; text-transform: uppercase; color: #5F5F5F; font-weight: 600; }
    .single-stat-val { font-family: 'JetBrains Mono', monospace; font-weight: 600; color: #004225; font-size: 16px; }
    
    /* Pitch */
    .pitch-container { width: 100%; max-width: 500px; aspect-ratio: 4 / 5; background-color: #004225; border: 2px solid #FFFFFF; border-radius: 4px; margin: 0 auto; position: relative; overflow: hidden; }
    .pitch-center-circle { position: absolute; top: 50%; left: 50%; width: 25%; aspect-ratio: 1; border: 2px solid #FFFFFF; border-radius: 50%; transform: translate(-50%, -50%); }
    .pitch-half-line { position: absolute; top: 50%; left: 0; width: 100%; height: 2px; background-color: #FFFFFF; }
    .player-node { position: absolute; transform: translate(-50%, -50%); text-align: center; z-index: 10; }
    .player-marker { width: 20px; height: 20px; background-color: #FFFFFF; border: 2px solid #FFFFFF; border-radius: 50%; margin: 0 auto 4px auto; }
    .player-label { font-family: 'Outfit', sans-serif; font-size: 11px; font-weight: 600; color: #FFFFFF !important; background-color: #002C18; padding: 2px 6px; border-radius: 2px; white-space: nowrap; }

    .bracket-card { background-color: #FFF8EE; border: 1px solid #E4D7C4; border-radius: 4px; padding: 15px; text-align: center; margin-bottom: 15px; }
    .bracket-matchup { font-family: 'Inter', sans-serif; font-size: 14px; color: #1A1A1A; margin-bottom: 8px; }
    .bracket-winner { font-family: 'Outfit', sans-serif; font-size: 16px; font-weight: 700; color: #004225; }
    
    /* MOBILE RESPONSIVENESS OVERRIDES */
    @media (max-width: 768px) {
        .hero-title { font-size: 45px !important; padding-top: 5vh; }
        button[kind="primary"] { font-size: 20px !important; height: 80px !important; }
        button[kind="secondary"] { font-size: 16px !important; height: auto !important; padding: 15px !important; }
        .profile-grid { grid-template-columns: 1fr !important; }
        .comp-matrix { grid-template-columns: 1fr !important; gap: 4px; }
        .comp-val-left, .comp-val-right { text-align: center !important; }
        .comp-label { white-space: normal; margin-bottom: 10px; }
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. STATE MANAGEMENT ---
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

def set_page(page_name):
    st.session_state.page = page_name

# --- 3. UNIFIED DATA ARCHITECTURE ENGINE ---
@st.cache_data
def load_data():
    try:
        roster_df = pd.read_csv('model_8_high_variance.csv')
        elo_df = pd.read_csv('current_elo_ratings.csv')
        groups_df = pd.read_csv('wc26_groups.csv')
    except:
        st.error("Engine failure: Verify that data resource arrays exist in standard app directory.")
        st.stop()
    
    if 'ML_Power_Rating' in roster_df.columns:
        roster_df = roster_df.rename(columns={'ML_Power_Rating': '4rs_Rating'})
        
    elo_dict = dict(zip(elo_df['Country_Code'], elo_df['Elo_Rating']))
    group_dict = dict(zip(groups_df['Country'], groups_df['Group']))

    name_to_code = {
        'Mexico': 'MX', 'South Africa': 'ZA', 'South Korea': 'KR', 'Czechia': 'CZ',
        'Canada': 'CA', 'Switzerland': 'CH', 'Qatar': 'QA', 'Bosnia-Herzegovina': 'BA',
        'Brazil': 'BR', 'Morocco': 'MA', 'Haiti': 'HT', 'Scotland': 'SC',
        'United States': 'US', 'Paraguay': 'PY', 'Australia': 'AU', 'Türkiye': 'TR',
        'Germany': 'DE', 'Curacao': 'CW', 'Ivory Coast': 'CI', 'Ecuador': 'EC',
        'Netherlands': 'NL', 'Japan': 'JP', 'Tunisia': 'TN', 'Sweden': 'SE',
        'Belgium': 'BE', 'Egypt': 'EG', 'Iran': 'IR', 'New Zealand': 'NZ',
        'Spain': 'ES', 'Cape Verde': 'CV', 'Saudi Arabia': 'SA', 'Uruguay': 'UY',
        'France': 'FR', 'Senegal': 'SN', 'Norway': 'NO', 'Iraq': 'IQ',
        'Argentina': 'AR', 'Algeria': 'DZ', 'Austria': 'AT', 'Jordan': 'JO',
        'Portugal': 'PT', 'Uzbekistan': 'UZ', 'Colombia': 'CO', 'Congo DR': 'CD',
        'England': 'EN', 'Croatia': 'HR', 'Ghana': 'GH', 'Panama': 'PA'
    }

    results_name_map = {
        'United States': 'USA', 'South Korea': 'Korea Republic', 
        'Cape Verde': 'Cabo Verde', 'Bosnia-Herzegovina': 'Bosnia and Herzegovina',
        'Congo DR': 'DR Congo', 'Türkiye': 'Turkey', 
        'Czechia': 'Czech Republic', 'Ivory Coast': "Côte d'Ivoire", 'Iran': 'IR Iran'
    }

    def generate_meta(team, elo):
        players = roster_df[roster_df['Team'] == team].sort_values(by='4rs_Rating', ascending=False)
        underrated = players.iloc[min(12, len(players)-1)]['Player'] if len(players) > 0 else "Squad Depth Asset"
        star = players.iloc[0]['Player'] if len(players) > 0 else "Key Selection"
        
        if elo > 1950:
            hist = f"A clear favorite with a consistent legacy of deep tournament runs and multiple historical trophies."
            style = "Focuses on heavy possession dominance, high pressing, and controlling the territory."
        elif elo > 1750:
            hist = f"A recognized tournament dark horse backed by a strong continental record and the ability to upset top tier teams."
            style = "Relies on a highly coordinated midfield block paired with lethal counter attacking transitions."
        else:
            hist = f"A developing international program looking to establish a long term competitive record."
            style = "Utilizes deep defensive blocks and high physical intensity while relying on set pieces to score."
            
        return {'hist': hist, 'style': style, 'underrated': underrated, 'star': star}

    pedigree_boosts = {}
    try:
        train_df = pd.read_csv('train.csv')
        test_df = pd.read_csv('test.csv')
        features = ['is_host', 'wins_last_4y', 'losses_last_4y', 'world_cup_titles_before']
        train_df[features] = train_df[features].fillna(0); test_df[features] = test_df[features].fillna(0)
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=4)
        rf_model.fit(train_df[features], train_df['quarter_finalist'])
        test_df['Pedigree_Prob'] = rf_model.predict_proba(test_df[features])[:, 1]
        for _, row in test_df.iterrows(): pedigree_boosts[row['team'].strip()] = row['Pedigree_Prob'] * 0.4
    except:
        for team in name_to_code.keys(): pedigree_boosts[team] = 0.0

    form_scores = {}
    try:
        matches_df = pd.read_csv('results.csv')
        matches_df['date'] = pd.to_datetime(matches_df['date'], errors='coerce')
        matches_df = matches_df.sort_values(by='date', ascending=False)
        for team in name_to_code.keys():
            search_name = results_name_map.get(team, team)
            team_games = matches_df[(matches_df['home_team'] == search_name) | (matches_df['away_team'] == search_name)].head(25)
            points, games_played = 0, len(team_games)
            for _, row in team_games.iterrows():
                is_home = (row['home_team'] == search_name)
                opp_name = row['away_team'] if is_home else row['home_team']
                opp_code = name_to_code.get(opp_name, 'Unknown')
                opp_elo = elo_dict.get(opp_code, 1300) 
                elo_multiplier = opp_elo / 1500.0
                if is_home:
                    if row['home_score'] > row['away_score']: points += (3 * elo_multiplier)
                    elif row['home_score'] == row['away_score']: points += (1 * elo_multiplier)
                else:
                    if row['away_score'] > row['home_score']: points += (3 * elo_multiplier)
                    elif row['away_score'] == row['home_score']: points += (1 * elo_multiplier)
            if games_played > 0: form_scores[team] = min(99.0, (points / (games_played * 3 * 1.1)) * 100)
            else: form_scores[team] = 65.0 
    except:
        for team in name_to_code.keys(): form_scores[team] = 65.0

    squad_metrics = {}
    team_power = []
    
    for team in roster_df['Team'].unique():
        players = roster_df[roster_df['Team'] == team].sort_values(by='4rs_Rating', ascending=False)
        starting_xi = players.head(11)['4rs_Rating'].mean() if not players.empty else 70.0
        bench = players.iloc[11:20]['4rs_Rating'].mean() if len(players) > 11 else starting_xi * 0.85
        manager = players['Manager'].iloc[0] if 'Manager' in players.columns else "Technical Director"
        
        code = name_to_code.get(team)
        elo = 1767 if team == 'Scotland' else elo_dict.get(code, 1500)
        
        # --- GLOBAL UNIFIED MATH ---
        elo_baseline = elo / 30.0 
        starting_xi = (starting_xi * 0.35) + (elo_baseline * 0.65)
        bench = (bench * 0.35) + (elo_baseline * 0.65)
        
        raw_form = form_scores.get(team, 65.0)
        adjusted_form = (raw_form * 0.4) + (elo_baseline * 0.6)
            
        pedigree_value = pedigree_boosts.get(team, 0.0)
        if team == 'United States' and 'USA' in pedigree_boosts: 
            pedigree_value = pedigree_boosts['USA']
            
        # Restored 8% Host Advantage 
        if team in ['United States', 'Mexico', 'Canada']:
            starting_xi *= 1.08
            bench *= 1.08
            
        group_name = f"Group {group_dict.get(code, 'Unknown')}"
        squad_metrics[team] = {
            'Starting_XI': starting_xi,
            'Form_25': adjusted_form,
            'Depth_Dropoff': max(0, starting_xi - bench),
            'Elo': elo,
            'Pedigree_Boost': pedigree_value,
            'Group': group_name
        }
        team_power.append({'Team': team, 'Group': group_name, 'Manager': manager, '4rs_Score': round(starting_xi, 1)})
        
    power_df = pd.DataFrame(team_power).sort_values(by='4rs_Score', ascending=False).reset_index(drop=True)
    power_df.index += 1
    meta_data = {t: generate_meta(t, squad_metrics[t]['Elo']) for t in roster_df['Team'].unique()}
    
    return roster_df, elo_dict, group_dict, meta_data, squad_metrics, power_df

roster_df, elo_dict, group_dict, meta_data, squad_metrics, power_df = load_data()

# --- GLOBAL HELPER: STATS ENGINE ---
def generate_comprehensive_stats(t):
    p = roster_df[roster_df['Team'] == t]
    metric = squad_metrics[t]
    
    att_score = p[p['Position'] == 'Forwards']['4rs_Rating'].mean() if not p[p['Position'] == 'Forwards'].empty else 71.0
    mid_score = p[p['Position'] == 'Midfielders']['4rs_Rating'].mean() if not p[p['Position'] == 'Midfielders'].empty else 71.0
    def_score = p[p['Position'] == 'Defenders']['4rs_Rating'].mean() if not p[p['Position'] == 'Defenders'].empty else 71.0
    gk_score = p[p['Position'] == 'Goalkeepers']['4rs_Rating'].mean() if not p[p['Position'] == 'Goalkeepers'].empty else 71.0
    
    elo_factor = metric['Elo'] / 2000.0
    possession = round(40.0 + (elo_factor * 15.0) + (mid_score - 70.0), 1)
    goals_per_match = round(0.8 + (att_score - 65.0)/10.0 + (elo_factor * 0.5), 2)
    shots_per_match = round(8.0 + (att_score - 65.0)/3.0, 1)
    pass_acc = round(72.0 + (mid_score - 65.0) * 1.1, 1)
    clean_sheets_pct = round(15.0 + (def_score - 65.0) * 2.5 + (gk_score - 70.0), 1)
    tackle_succ = round(65.0 + (def_score - 70.0) * 0.8, 1)
    
    return {
        'Attack': round(att_score, 1), 'Midfield': round(mid_score, 1), 
        'Defense': round(def_score, 1), 'Goalkeeping': round(gk_score, 1),
        'Possession %': possession, 'Goals/Match': goals_per_match,
        'Shots/Match': shots_per_match, 'Pass Accuracy %': pass_acc,
        'Clean Sheet %': clean_sheets_pct, 'Tackle Success %': tackle_succ,
        'Elo': metric['Elo'], '4rs': metric['Starting_XI']
    }

# --- HELPER: 4-3-3 LINEUP BUILDER ---
def get_starting_xi(t):
    p = roster_df[roster_df['Team'] == t].sort_values(by='4rs_Rating', ascending=False)
    gk = p[p['Position'] == 'Goalkeepers'].head(1).to_dict('records')
    dfs = p[p['Position'] == 'Defenders'].head(4).to_dict('records')
    mids = p[p['Position'] == 'Midfielders'].head(3).to_dict('records')
    fwds = p[p['Position'] == 'Forwards'].head(3).to_dict('records')
    return gk + dfs + mids + fwds

def get_bench(t):
    p = roster_df[roster_df['Team'] == t].sort_values(by='4rs_Rating', ascending=False)
    xi = get_starting_xi(t)
    xi_names = [x['Player'] for x in xi]
    return p[~p['Player'].isin(xi_names)].head(15)

def render_pitch(xi_players):
    coords = [
        {'top': '90%', 'left': '50%'}, {'top': '72%', 'left': '20%'}, {'top': '75%', 'left': '40%'}, 
        {'top': '75%', 'left': '60%'}, {'top': '72%', 'left': '80%'}, {'top': '50%', 'left': '30%'}, 
        {'top': '55%', 'left': '50%'}, {'top': '50%', 'left': '70%'}, {'top': '25%', 'left': '25%'}, 
        {'top': '20%', 'left': '50%'}, {'top': '25%', 'left': '75%'}
    ]
    nodes_html = ""
    for idx, player in enumerate(xi_players):
        if idx < len(coords):
            pos = coords[idx]
            short_name = player['Player'].split(' ')[-1]
            marker_color = "#B78700" if idx == 0 else "#FFFFFF" 
            nodes_html += f"<div class='player-node' style='top: {pos['top']}; left: {pos['left']};'><div class='player-marker' style='background-color: {marker_color};'></div><div class='player-label'>{short_name}</div></div>"
    return f"<div class='pitch-container'><div class='pitch-half-line'></div><div class='pitch-center-circle'></div>{nodes_html}</div>"


# ==========================================
# PAGE ROUTING DISPATCHER
# ==========================================

if st.session_state.page == 'Home':
    st.markdown('<div class="hero-title">Eshans World Cup<br>Analysis Dashboard</div>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class='explainer-box'>
        <div class='explainer-title'>Welcome to the Truth Engine</div>
        <div class='explainer-text'>This platform completely removes human bias. It uses the 4rs Score to calculate the exact probabilistic strength of every nation competing in the 2026 World Cup. The 4rs Score is a custom mathematical formula that combines European club data, recent 25 game form tracking, historical Elo ratings, and machine learning models.</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="home-panel-container">', unsafe_allow_html=True)
    col_main = st.columns([1])[0]
    with col_main:
        if st.button("Execute Official Tournament Simulator", type="primary", use_container_width=True): 
            set_page('Simulator')
            st.rerun()
            
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        if st.button("Global Power\nRankings", type="secondary", use_container_width=True): set_page('Rankings'); st.rerun()
    with col2:
        if st.button("Team\nAnalysis", type="secondary", use_container_width=True): set_page('Deep Dive'); st.rerun()
    with col3:
        if st.button("Head-to-Head\nComparison", type="secondary", use_container_width=True): set_page('Comparison'); st.rerun()
    with col4:
        if st.button("Match-Flow\nxG Simulator", type="secondary", use_container_width=True): set_page('MatchFlow'); st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

if st.session_state.page != 'Home':
    st.markdown('<div class="nav-back">', unsafe_allow_html=True)
    if st.button("Return to Master Dashboard"):
        set_page('Home')
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# --- PAGE: RANKINGS ---
if st.session_state.page == 'Rankings':
    st.header("Global Power Rankings")
    st.markdown(f"""
    <div class='explainer-box'>
        <div class='explainer-title'>Understanding the Leaderboard</div>
        <div class='explainer-text'>The 4rs Score represents the true competitive weight of a nation. The algorithm measures the raw talent of the starting lineup at 35 percent and anchors the remaining 65 percent to historical Elo ratings. This balance ensures that individual superstars do not falsely inflate the rating of an entire national team.</div>
    </div>
    """, unsafe_allow_html=True)
    st.dataframe(power_df, use_container_width=True, height=750)

# --- PAGE: TEAM ANALYSIS ---
elif st.session_state.page == 'Deep Dive':
    st.header("Team Analysis")
    st.markdown(f"""
    <div class='explainer-box'>
        <div class='explainer-title'>Tactical Breakdown</div>
        <div class='explainer-text'>Select a nation to see their optimal tactical setup. The engine reviews the entire database and computationally builds the strongest and most balanced 4-3-3 formation available. It then maps that starting lineup directly onto visual pitch coordinates.</div>
    </div>
    """, unsafe_allow_html=True)
    
    team1 = st.selectbox("Select Target Nation Profile:", power_df['Team'].tolist())
    st.markdown("<hr style='border: 1px solid #E4D7C4;'>", unsafe_allow_html=True)
    
    t1_data = power_df[power_df['Team'] == team1].iloc[0]
    meta1 = meta_data[team1]
    
    st.markdown(f"<div class='profile-grid'><div class='profile-panel'><div class='profile-label'>History & Achievements</div><div class='profile-value'>{meta1['hist']}</div></div><div class='profile-panel'><div class='profile-label'>Tactical Playstyle</div><div class='profile-value'>{meta1['style']}</div></div><div class='profile-panel'><div class='profile-label'>Star Player Profile</div><div class='profile-value'>{meta1['star']}</div></div><div class='profile-panel'><div class='profile-label'>Underrated Player Choice</div><div class='profile-value'>{meta1['underrated']}</div></div><div class='profile-panel' style='grid-column: 1 / -1;'><div class='profile-label'>Manager</div><div class='profile-value'>{t1_data['Manager']}</div></div></div>", unsafe_allow_html=True)
    
    stats_1 = generate_comprehensive_stats(team1)
    
    col_g1, col_g2 = st.columns([1, 1])
    
    categories = ['Attack', 'Midfield', 'Defense', 'Goalkeeping']
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(r=[stats_1[c] for c in categories], theta=categories, fill='toself', name=team1, line_color='#004225', fillcolor='rgba(0, 66, 37, 0.3)'))
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[60, 95], tickfont=dict(color='#1A1A1A', family='Inter'), gridcolor='#E4D7C4', linecolor='#E4D7C4'),
            angularaxis=dict(tickfont=dict(color='#1A1A1A', family='Inter', size=14), gridcolor='#E4D7C4', linecolor='#E4D7C4')
        ), 
        showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
        font=dict(color='#1A1A1A', size=13, family='Inter'), title=dict(text="Positional Unit Power", font=dict(color='#1A1A1A')), margin=dict(l=20, r=20, t=40, b=20)
    )
    with col_g1: st.plotly_chart(fig_radar, theme=None, use_container_width=True)

    compare_metrics = ['Possession %', 'Pass Accuracy %', 'Tackle Success %', 'Clean Sheet %']
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(y=compare_metrics, x=[stats_1[m] for m in compare_metrics], name=team1, orientation='h', marker_color='#004225'))
    fig_bar.update_layout(
        barmode='group', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
        font=dict(family='Inter', color='#1A1A1A'), title=dict(text="Derived Phase Efficiency Indicators (%)", font=dict(color='#1A1A1A')),
        xaxis=dict(tickfont=dict(color='#1A1A1A'), gridcolor='#E4D7C4', title_font=dict(color='#1A1A1A')),
        yaxis=dict(tickfont=dict(color='#1A1A1A'), title_font=dict(color='#1A1A1A')), margin=dict(l=20, r=20, t=40, b=20)
    )
    with col_g2: st.plotly_chart(fig_bar, theme=None, use_container_width=True)

    st.subheader("Granular Data Analysis")
    st.markdown("<div class='profile-panel'>", unsafe_allow_html=True)
    def render_single_stat(label, val):
        st.markdown(f"<div class='single-stat-row'><span class='single-stat-label'>{label}</span><span class='single-stat-val'>{val}</span></div>", unsafe_allow_html=True)
    
    render_single_stat("Overall Gravitational Elo", stats_1['Elo'])
    render_single_stat("Calibrated Attacking Average", stats_1['Attack'])
    render_single_stat("Calibrated Midfield Average", stats_1['Midfield'])
    render_single_stat("Calibrated Defensive Average", stats_1['Defense'])
    render_single_stat("Calibrated Goalkeeping Average", stats_1['Goalkeeping'])
    render_single_stat("Projected Expected Goals Per Match", stats_1['Goals/Match'])
    render_single_stat("Projected Shot Volume Per 90", stats_1['Shots/Match'])
    st.markdown("</div><br>", unsafe_allow_html=True)

    st.markdown("<hr style='border: 1px solid #E4D7C4;'>", unsafe_allow_html=True)
    col_pitch, col_table = st.columns([1, 1.2])
    with col_pitch:
        st.subheader("Strongest Lineup (4-3-3)")
        xi_list = get_starting_xi(team1)
        st.markdown(render_pitch(xi_list), unsafe_allow_html=True)
    with col_table:
        st.subheader("Complete Squad Roster")
        full_squad = roster_df[roster_df['Team'] == team1].sort_values(by='4rs_Rating', ascending=False).reset_index(drop=True)
        st.dataframe(full_squad[['Player', 'Position', 'Club', '4rs_Rating']], use_container_width=True, height=500)

# --- PAGE: TEAM COMPARISON ---
elif st.session_state.page == 'Comparison':
    st.header("Head-to-Head Deep Comparison")
    
    st.markdown("""
    <div class='explainer-box'>
        <div class='explainer-title'>Advanced Statistical Matrix</div>
        <div class='explainer-text'>This tool scales base team attributes into highly specific tournament indicators. Attacking, midfield, and defensive scores are generated from localized league performance data. Meanwhile, performance metrics like total possession and expected shot volume are derived directly from opponent weighted Elo models.</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])
    with c1: team_a = st.selectbox("Select First Team:", power_df['Team'].tolist(), index=0)
    with c2: team_b = st.selectbox("Select Comparison Team:", power_df['Team'].tolist(), index=1)
    st.markdown("<hr style='border: 1px solid #E4D7C4;'>", unsafe_allow_html=True)

    stats_a = generate_comprehensive_stats(team_a)
    stats_b = generate_comprehensive_stats(team_b)

    # --- GRAPH 1: RADAR CHART ---
    categories = ['Attack', 'Midfield', 'Defense', 'Goalkeeping']
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(r=[stats_a[c] for c in categories], theta=categories, fill='toself', name=team_a, line_color='#004225', fillcolor='rgba(0, 66, 37, 0.3)'))
    fig_radar.add_trace(go.Scatterpolar(r=[stats_b[c] for c in categories], theta=categories, fill='toself', name=team_b, line_color='#C8A45D', fillcolor='rgba(200, 164, 93, 0.3)'))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[60, 95], tickfont=dict(color='#1A1A1A', family='Inter'), gridcolor='#E4D7C4', linecolor='#E4D7C4'),
            angularaxis=dict(tickfont=dict(color='#1A1A1A', family='Inter', size=14), gridcolor='#E4D7C4', linecolor='#E4D7C4')
        ), 
        showlegend=True, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
        font=dict(color='#1A1A1A', size=13, family='Inter'), title=dict(text="Positional Unit Power Comparison", font=dict(color='#1A1A1A')), margin=dict(l=20, r=20, t=40, b=20)
    )

    # --- GRAPH 2: BAR CHART MATRIX ---
    compare_metrics = ['Possession %', 'Pass Accuracy %', 'Tackle Success %', 'Clean Sheet %']
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(y=compare_metrics, x=[stats_a[m] for m in compare_metrics], name=team_a, orientation='h', marker_color='#004225'))
    fig_bar.add_trace(go.Bar(y=compare_metrics, x=[stats_b[m] for m in compare_metrics], name=team_b, orientation='h', marker_color='#C8A45D'))
    
    fig_bar.update_layout(
        barmode='group', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
        font=dict(family='Inter', color='#1A1A1A'), title=dict(text="Derived Phase Efficiency Indicators (%)", font=dict(color='#1A1A1A')),
        xaxis=dict(tickfont=dict(color='#1A1A1A'), gridcolor='#E4D7C4', title_font=dict(color='#1A1A1A')),
        yaxis=dict(tickfont=dict(color='#1A1A1A'), title_font=dict(color='#1A1A1A')), margin=dict(l=20, r=20, t=40, b=20)
    )

    col_g1, col_g2 = st.columns([1, 1])
    with col_g1: st.plotly_chart(fig_radar, theme=None, use_container_width=True)
    with col_g2: st.plotly_chart(fig_bar, theme=None, use_container_width=True)

    # --- GEOMETRIC NUMERICAL PANELS ---
    st.subheader("Granular Data Analysis Grid")
    def render_stat_row(label, val_a, val_b):
        st.markdown(f"<div class='comp-matrix'><div class='comp-val-left'>{val_a}</div><div class='comp-label'>{label}</div><div class='comp-val-right'>{val_b}</div></div>", unsafe_allow_html=True)

    st.markdown("<div class='profile-panel'>", unsafe_allow_html=True)
    render_stat_row("Overall Gravitational Elo", stats_a['Elo'], stats_b['Elo'])
    render_stat_row("Calibrated Attacking Average", stats_a['Attack'], stats_b['Attack'])
    render_stat_row("Calibrated Midfield Average", stats_a['Midfield'], stats_b['Midfield'])
    render_stat_row("Calibrated Defensive Average", stats_a['Defense'], stats_b['Defense'])
    render_stat_row("Calibrated Goalkeeping Average", stats_a['Goalkeeping'], stats_b['Goalkeeping'])
    render_stat_row("Projected Expected Goals Per Match", stats_a['Goals/Match'], stats_b['Goals/Match'])
    render_stat_row("Projected Shot Volume Per 90", stats_a['Shots/Match'], stats_b['Shots/Match'])
    st.markdown("</div>", unsafe_allow_html=True)
    
    # --- TEXTUAL TACTICAL & LEGACY PANELS ---
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Tactical Strategy & Legacy Comparison")
    
    col_t1, col_t2 = st.columns([1, 1])
    with col_t1:
        st.markdown(f"""
        <div class='profile-panel'>
            <div class='profile-label'>{team_a} Tactical Framework & Legacy</div>
            <p><b>Historical Weight:</b> {meta_data[team_a]['hist']}</p>
            <p><b>Tactical Setup:</b> {meta_data[team_a]['style']}</p>
            <p><b>Aura Player Selection:</b> {meta_data[team_a]['star']}</p>
        </div>
        """, unsafe_allow_html=True)
    with col_t2:
        st.markdown(f"""
        <div class='profile-panel'>
            <div class='profile-label'>{team_b} Tactical Framework & Legacy</div>
            <p><b>Historical Weight:</b> {meta_data[team_b]['hist']}</p>
            <p><b>Tactical Setup:</b> {meta_data[team_b]['style']}</p>
            <p><b>Aura Player Selection:</b> {meta_data[team_b]['star']}</p>
        </div>
        """, unsafe_allow_html=True)

    # --- SIDE-BY-SIDE STARTING SQUADS ---
    st.markdown("<hr style='border: 1px solid #E4D7C4;'>", unsafe_allow_html=True)
    st.subheader("Head-to-Head Lineup Matchup")
    col_a_pitch, col_a_bench, col_b_pitch, col_b_bench = st.columns([1.2, 1, 1.2, 1])
    
    with col_a_pitch:
        st.markdown(f"**{team_a} XI**")
        st.markdown(render_pitch(get_starting_xi(team_a)), unsafe_allow_html=True)
    with col_a_bench:
        st.markdown(f"**{team_a} Bench**")
        st.dataframe(get_bench(team_a)[['Player', '4rs_Rating']], hide_index=True, use_container_width=True, height=450)
    with col_b_pitch:
        st.markdown(f"**{team_b} XI**")
        st.markdown(render_pitch(get_starting_xi(team_b)), unsafe_allow_html=True)
    with col_b_bench:
        st.markdown(f"**{team_b} Bench**")
        st.dataframe(get_bench(team_b)[['Player', '4rs_Rating']], hide_index=True, use_container_width=True, height=450)

# --- PAGE: MATCH-FLOW XG SIMULATOR ---
elif st.session_state.page == 'MatchFlow':
    st.header("Match-Flow xG Timeline")
    st.markdown(f"""
    <div class='explainer-box'>
        <div class='explainer-title'>The Momentum Engine</div>
        <div class='explainer-text'>Unlike the main tournament bracket which predicts a final score, this tool simulates a specific 90 minute matchup in real time. It distributes the team probabilities across 1 minute intervals and injects natural momentum variance. This generates a realistic Expected Goals step chart showing exactly when teams are most likely to score.</div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    with col1: team_home = st.selectbox("Select Home Team:", power_df['Team'].tolist(), index=0)
    with col2: team_away = st.selectbox("Select Away Team:", power_df['Team'].tolist(), index=1)
    
    if st.button("Simulate 90 Minutes", use_container_width=True):
        with st.spinner("Calculating minute-by-minute momentum shifts..."):
            m_a, m_b = squad_metrics[team_home], squad_metrics[team_away]
            
            power_diff = (m_a['Starting_XI'] - m_b['Starting_XI']) / 30.0     
            form_diff = (m_a['Form_25'] - m_b['Form_25']) / 100.0    
            elo_diff = (m_a['Elo'] - m_b['Elo']) / 700.0   
            pedigree_diff = (m_a['Pedigree_Boost'] - m_b['Pedigree_Boost']) * 1.5
            
            base_xg = 1.15
            lambda_a = max(0.20, min(3.8, base_xg + power_diff + form_diff + elo_diff + pedigree_diff))
            lambda_b = max(0.20, min(3.8, base_xg - power_diff - form_diff - elo_diff - pedigree_diff))
            
            prob_a_min = lambda_a / 90.0
            prob_b_min = lambda_b / 90.0
            
            xg_a_cumulative, xg_b_cumulative = [0], [0]
            goals_a_mins, goals_b_mins = [], []
            
            for minute in range(1, 91):
                mom_a = np.random.normal(1.0, 0.3)
                mom_b = np.random.normal(1.0, 0.3)
                
                min_xg_a = max(0, prob_a_min * mom_a)
                min_xg_b = max(0, prob_b_min * mom_b)
                
                xg_a_cumulative.append(xg_a_cumulative[-1] + min_xg_a)
                xg_b_cumulative.append(xg_b_cumulative[-1] + min_xg_b)
                
                if np.random.random() < min_xg_a: goals_a_mins.append(minute)
                if np.random.random() < min_xg_b: goals_b_mins.append(minute)
            
            final_xg_a = round(xg_a_cumulative[-1], 2)
            final_xg_b = round(xg_b_cumulative[-1], 2)
            final_score_a = len(goals_a_mins)
            final_score_b = len(goals_b_mins)
            
            st.markdown("<hr style='border: 1px solid #E4D7C4;'>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='text-align: center;'>{team_home} {final_score_a} - {final_score_b} {team_away}</h2>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center; color: #5F5F5F;'>Final xG: {final_xg_a} - {final_xg_b}</p>", unsafe_allow_html=True)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(91)), y=xg_a_cumulative, mode='lines', line_shape='hv', name=f"{team_home} xG", line=dict(color='#004225', width=3)))
            fig.add_trace(go.Scatter(x=list(range(91)), y=xg_b_cumulative, mode='lines', line_shape='hv', name=f"{team_away} xG", line=dict(color='#C8A45D', width=3)))
            
            for g in goals_a_mins:
                fig.add_annotation(x=g, y=xg_a_cumulative[g], text="GOAL", showarrow=True, arrowhead=2, arrowcolor='#004225', font=dict(color='#FFFFFF', size=10), bgcolor='#004225', bordercolor='#004225', borderpad=2)
            for g in goals_b_mins:
                fig.add_annotation(x=g, y=xg_b_cumulative[g], text="GOAL", showarrow=True, arrowhead=2, arrowcolor='#C8A45D', font=dict(color='#FFFFFF', size=10), bgcolor='#C8A45D', bordercolor='#C8A45D', borderpad=2)
                
            fig.update_layout(
                xaxis_title="Match Minute", yaxis_title="Cumulative Expected Goals (xG)", 
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family='Inter', color='#1A1A1A'),
                xaxis=dict(tickfont=dict(color='#1A1A1A'), title_font=dict(color='#1A1A1A'), gridcolor='#E4D7C4'),
                yaxis=dict(tickfont=dict(color='#1A1A1A'), title_font=dict(color='#1A1A1A'), gridcolor='#E4D7C4'),
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig, theme=None, use_container_width=True)

# --- PAGE: TOURNAMENT SIMULATOR ---
elif st.session_state.page == 'Simulator':
    st.header("Predictive Simulator Framework")
    st.markdown(f"""
    <div class='explainer-box'>
        <div class='explainer-title'>The Monte Carlo Truth Engine</div>
        <div class='explainer-text'>This is the core of the entire dashboard. Instead of trying to guess a single outcome, the engine uses a Monte Carlo method to play out every single fixture of the World Cup 10,000 times. It factors in starting roster strength, bench depth fatigue during knockouts, historical clutch factor, and third place routing mathematics to build the most statistically probable bracket.</div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Execute Model 16.2 Simulation", use_container_width=True):
        with st.spinner("Processing tactical configurations and executing bracket arrays..."):

            def simulate_match(team_a, team_b, is_knockout=False, sims=10000):
                metric_a, metric_b = squad_metrics[team_a], squad_metrics[team_b]
                
                power_diff = (metric_a['Starting_XI'] - metric_b['Starting_XI']) / 30.0     
                form_diff = (metric_a['Form_25'] - metric_b['Form_25']) / 100.0    
                elo_diff = (metric_a['Elo'] - metric_b['Elo']) / 700.0   
                pedigree_diff = (metric_a['Pedigree_Boost'] - metric_b['Pedigree_Boost']) * 1.5
                
                base_xg = 1.15 
                lambda_a = max(0.20, min(3.8, base_xg + power_diff + form_diff + elo_diff + pedigree_diff))
                lambda_b = max(0.20, min(3.8, base_xg - power_diff - form_diff - elo_diff - pedigree_diff))
                
                if is_knockout: 
                    lambda_a -= (metric_a['Depth_Dropoff'] / 50.0)
                    lambda_b -= (metric_b['Depth_Dropoff'] / 50.0)
                    lambda_a, lambda_b = max(0.15, lambda_a), max(0.15, lambda_b)
                    
                goals_a = np.random.poisson(lambda_a, sims)
                goals_b = np.random.poisson(lambda_b, sims)
                
                wins_a = int(np.sum(goals_a > goals_b))
                wins_b = int(np.sum(goals_b > goals_a))
                draws = int(np.sum(goals_a == goals_b))
                scorelines = list(zip(goals_a, goals_b))
                
                # Reverted to pure statistical mode (allowing natural draws)
                mode_ga, mode_gb = Counter(scorelines).most_common(1)[0][0]
                
                advancer = None
                if is_knockout:
                    clutch_a = metric_a['Starting_XI'] + (metric_a['Elo'] / 20) + (metric_a['Pedigree_Boost'] * 100)
                    clutch_b = metric_b['Starting_XI'] + (metric_b['Elo'] / 20) + (metric_b['Pedigree_Boost'] * 100)
                    prob_a_pens = clutch_a / (clutch_a + clutch_b)
                    pens_a = int(draws * prob_a_pens)
                    pens_b = draws - pens_a
                    advancer = team_a if (wins_a + pens_a) > (wins_b + pens_b) else team_b
                        
                return wins_a, wins_b, draws, mode_ga, mode_gb, advancer

            st.markdown("<hr style='border: 1px solid #E4D7C4;'>", unsafe_allow_html=True)
            st.subheader("Group Stage Results")
            group_standings = []
            cols = st.columns(3)
            
            for idx, grp_letter in enumerate("ABCDEFGHIJKL"):
                group = f"Group {grp_letter}"
                teams = [t for t, m in squad_metrics.items() if m['Group'] == group]
                if len(teams) != 4: continue
                
                standings = {t: {'W': 0, 'D': 0, 'L': 0, 'GD': 0, 'Pts': 0, 'Elo': squad_metrics[t]['Elo']} for t in teams}
                match_logs = ""
                
                for t1, t2 in itertools.combinations(teams, 2):
                    w_a, w_b, d, m_ga, m_gb, _ = simulate_match(t1, t2, is_knockout=False)
                    match_logs += f"<div style='font-size: 13px; color: #5F5F5F; padding: 4px 0;'>{t1} <b>{m_ga} - {m_gb}</b> {t2}</div>"
                    
                    standings[t1]['GD'] += (m_ga - m_gb); standings[t2]['GD'] += (m_gb - m_ga)
                    if m_ga > m_gb:
                        standings[t1]['W'] += 1; standings[t2]['L'] += 1; standings[t1]['Pts'] += 3
                    elif m_gb > m_ga:
                        standings[t2]['W'] += 1; standings[t1]['L'] += 1; standings[t2]['Pts'] += 3
                    else:
                        standings[t1]['D'] += 1; standings[t2]['D'] += 1; standings[t1]['Pts'] += 1; standings[t2]['Pts'] += 1
                        
                table = pd.DataFrame.from_dict(standings, orient='index').reset_index()
                table.columns = ['Team', 'W', 'D', 'L', 'GD', 'Pts', 'Elo']
                table = table.sort_values(by=['Pts', 'GD', 'Elo'], ascending=[False, False, False]).reset_index(drop=True)
                table['Group'] = group
                table['Rank'] = table.index + 1
                group_standings.append(table)
                
                with cols[idx % 3]:
                    st.markdown(f"#### {group}")
                    st.markdown(f"<div style='background-color: #FFF8EE; border: 1px solid #E4D7C4; padding: 10px; border-radius: 4px; margin-bottom: 10px;'>{match_logs}</div>", unsafe_allow_html=True)
                    def highlight(s): return ['background-color: #D9E9E0; color: #004225; font-weight: 500;' if i < 2 else '' for i in range(len(s))]
                    st.dataframe(table[['Rank', 'Team', 'Pts', 'GD']].style.apply(highlight, axis=0), use_container_width=True, hide_index=True)

            all_groups_df = pd.concat(group_standings, ignore_index=True)
            firsts = {row['Group'][-1]: row['Team'] for _, row in all_groups_df[all_groups_df['Rank'] == 1].iterrows()}
            seconds = {row['Group'][-1]: row['Team'] for _, row in all_groups_df[all_groups_df['Rank'] == 2].iterrows()}
            best_8_thirds = all_groups_df[all_groups_df['Rank'] == 3].sort_values(by=['Pts', 'GD', 'Elo'], ascending=[False, False, False]).head(8)
            
            def allocate_3rds_wrapper(available_3rds):
                groups = sorted([t['Group'][-1] for t in available_3rds])
                combo_key = "".join(groups)
                group_to_team = {t['Group'][-1]: t['Team'] for t in available_3rds}
                known_mappings = {'ACEGIJKL': {'1E_3rd': 'C', '1I_3rd': 'G', '1D_3rd': 'I', '1G_3rd': 'A', '1A_3rd': 'E', '1L_3rd': 'K', '1B_3rd': 'J', '1K_3rd': 'L'}}
                if combo_key in known_mappings: return {k: group_to_team[v] for k, v in known_mappings[combo_key].items()}
                slots = ['1E_3rd', '1I_3rd', '1D_3rd', '1G_3rd', '1A_3rd', '1L_3rd', '1B_3rd', '1K_3rd']
                return {slot: available_3rds[idx % len(available_3rds)]['Team'] for idx, slot in enumerate(slots)}
            
            assigned_3rds = allocate_3rds_wrapper(best_8_thirds[['Team', 'Group']].to_dict('records'))
            
            def get_team(code):
                if code.endswith('_3rd'): return assigned_3rds.get(code, 'Unknown')
                rank, grp = int(code[0]), code[1]
                return firsts.get(grp) if rank == 1 else seconds.get(grp)

            st.markdown("<hr style='border: 1px solid #E4D7C4;'>", unsafe_allow_html=True)
            st.subheader("Official Knockout Bracket")
            
            r32_brackets = [{'id': 74, 't1': '1E', 't2': '1E_3rd'}, {'id': 77, 't1': '1I', 't2': '1I_3rd'}, {'id': 73, 't1': '2A', 't2': '2B'}, {'id': 75, 't1': '1F', 't2': '2C'}, {'id': 83, 't1': '2K', 't2': '2L'}, {'id': 84, 't1': '1H', 't2': '2J'}, {'id': 81, 't1': '1D', 't2': '1D_3rd'}, {'id': 82, 't1': '1G', 't2': '1G_3rd'}, {'id': 76, 't1': '1C', 't2': '2F'}, {'id': 78, 't1': '2E', 't2': '2I'}, {'id': 79, 't1': '1A', 't2': '1A_3rd'}, {'id': 80, 't1': '1L', 't2': '1L_3rd'}, {'id': 87, 't1': '1J', 't2': '2H'}, {'id': 86, 't1': '2D', 't2': '2G'}, {'id': 85, 't1': '1B', 't2': '1B_3rd'}, {'id': 88, 't1': '1K', 't2': '1K_3rd'}]
            r16_brackets = [{'id': 89, 'm1': 73, 'm2': 75}, {'id': 90, 'm1': 74, 'm2': 77}, {'id': 91, 'm1': 76, 'm2': 78}, {'id': 92, 'm1': 79, 'm2': 80}, {'id': 93, 'm1': 83, 'm2': 84}, {'id': 94, 'm1': 81, 'm2': 82}, {'id': 95, 'm1': 86, 'm2': 87}, {'id': 96, 'm1': 85, 'm2': 88}]
            qf_brackets = [{'id': 97, 'm1': 89, 'm2': 90}, {'id': 98, 'm1': 93, 'm2': 94}, {'id': 99, 'm1': 91, 'm2': 92}, {'id': 100, 'm1': 95, 'm2': 96}]
            sf_brackets = [{'id': 101, 'm1': 97, 'm2': 98}, {'id': 102, 'm1': 99, 'm2': 100}]

            knockout_winners = {}
            
            def render_bracket_stage(brackets, cols_count, title):
                st.markdown(f"#### {title}")
                cols = st.columns(cols_count)
                for idx, match in enumerate(brackets):
                    t1 = get_team(match['t1']) if 't1' in match else knockout_winners[match['m1']]
                    t2 = get_team(match['t2']) if 't2' in match else knockout_winners[match['m2']]
                    w_a, w_b, d, _, _, winner = simulate_match(t1, t2, is_knockout=True)
                    prob = round(((max(w_a, w_b) + (d/2))/10000)*100, 1)
                    knockout_winners[match['id']] = winner
                    
                    st.markdown(f"""<div class='bracket-card' style='grid-column: span 1; width: 95%;'><div class='bracket-matchup'>{t1} vs {t2}</div><div class='bracket-winner'>Advances: {winner.upper()} ({prob}%)</div></div>""", unsafe_allow_html=True)

            render_bracket_stage(r32_brackets, 4, "Round of 32")
            render_bracket_stage(r16_brackets, 4, "Round of 16")
            render_bracket_stage(qf_brackets, 2, "Quarter-Finals")
            render_bracket_stage(sf_brackets, 2, "Semi-Finals")

            st.markdown("<hr style='border: 1px solid #E4D7C4;'>", unsafe_allow_html=True)
            t1, t2 = knockout_winners[101], knockout_winners[102]
            w_a, w_b, d, _, _, champion = simulate_match(t1, t2, is_knockout=True)
            prob_c = round(((max(w_a, w_b) + (d/2))/10000)*100, 1)
            
            st.markdown(f"""
            <div style="background-color: #004225; padding: 40px; border-radius: 4px; text-align: center; border: 2px solid #C8A45D;">
                <h3 style="color: #FFF0DB !important; font-family: 'Outfit'; margin-bottom: 20px;">The 2026 World Cup Final</h3>
                <div style="font-family: 'Inter'; font-size: 24px; color: #E4D7C4; margin-bottom: 20px;">{t1} vs {t2}</div>
                <div style="font-family: 'Outfit'; font-size: 14px; text-transform: uppercase; color: #C8A45D; letter-spacing: 0.1em;">Predicted Champion</div>
                <div style="font-family: 'Playfair Display', serif; font-size: 64px; color: #FFF0DB; font-weight: bold; margin-top: 5px;">{champion.upper()}</div>
            </div>
            """, unsafe_allow_html=True)
