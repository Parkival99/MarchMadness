import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64

st.set_page_config(page_title="March Madness Model", layout="wide")

st.markdown("""
<style>
.block-container {
    padding-top: 3rem;
}
</style>
""", unsafe_allow_html=True)

header_col1, header_col2 = st.columns([4,1])

with header_col1:
    st.title("March Madness Team Strength Model")

with header_col2:
    st.image("March_Madness_logo.svg.png", width=170)

# ----------------------------
# Load data
# ----------------------------

df = pd.read_parquet("model_dataset2.parquet")
tour = pd.read_csv("march_model26.csv")
mascot = pd.read_csv("march_model26mascot.csv")

df["Team"] = df["Team"].str.replace("\xa0","").str.strip()
tour["Team"] = tour["Team"].str.replace("\xa0","").str.strip()

df = df.merge(tour[["Team","Seed","Region","Conf"]], on="Team")

mascot["Team"] = mascot["Team"].str.replace("\xa0","").str.strip()
df = df.merge(mascot, on="Team", how="left")

# ----------------------------
# Numeric cleanup
# ----------------------------

numeric_cols = [
"AdjOE_x","AdjDE_x","3P%","2P%","FT%","AdjTempo", "PowerRating"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ----------------------------
# Net efficiency
# ----------------------------

df["NetEff"] = df["AdjOE_x"] - df["AdjDE_x"]

# ----------------------------
# Tempo balance
# ----------------------------

tempo_optimal = 67.5
df["TempoBalance"] = -(df["AdjTempo"] - tempo_optimal)**2

# ----------------------------
# Conference strength
# ----------------------------

conf_strength = {
"SEC":19.56,"B10":18.27,"B12":18.21,"ACC":15.17,"BE":13.54,
"MWC":9.31,"A10":4.71,"WCC":4.03,"MVC":3.32,"Amer":1.39,
"BW":-0.67,"WAC":-0.80,"CUSA":-1.93,"Ivy":-2.11,"BSky":-2.30,
"CAA":-2.55,"MAC":-3.02,"Slnd":-4.14,"Horz":-4.68,"SB":-5.87,
"Sum":-7.10,"BSth":-7.36,"SC":-8.25,"ASun":-8.73,"MAAC":-9.86,
"OVC":-10.99,"PL":-11.46,"SWAC":-14.08,"NEC":-14.59,
"AE":-16.56,"MEAC":-20.96
}

df["ConfStrength"] = df["Conf"].map(conf_strength).fillna(0)

# ----------------------------
# Standardize features
# ----------------------------

features = [
"AdjOE_x","AdjDE_x","NetEff","3P%","2P%","FT%",
"TempoBalance","ConfStrength", "PowerRating"
]

for col in features:
    df[col+"_z"] = (df[col] - df[col].mean()) / df[col].std()

# ----------------------------
# Sidebar sliders
# ----------------------------

st.sidebar.header("Model Weights")

w_net = st.sidebar.slider("Net Efficiency",0,100,50)
w_adjo = st.sidebar.slider("Offensive Efficiency",0,100,0)
w_adjd = st.sidebar.slider("Defensive Efficiency",0,100,0)
w_3p = st.sidebar.slider("3PT Shooting",0,100,15)
w_2p = st.sidebar.slider("2PT Shooting",0,100,15)
w_ft = st.sidebar.slider("Free Throw %",0,100,5)
w_tempo = st.sidebar.slider("Balanced Tempo",0,100,5)
w_conf = st.sidebar.slider("Conference Strength",0,100,10)
w_mascot = st.sidebar.slider("Mascot Power",0,100,0)

weights = [w_net,w_adjo,w_adjd,w_3p,w_2p,w_ft,w_tempo,w_conf,w_mascot]
total = sum(weights)

if total == 0:
    total = 1

w_net/=total
w_adjo/=total
w_adjd/=total
w_3p/=total
w_2p/=total
w_ft/=total
w_tempo/=total
w_conf/=total
w_mascot /= total

# ----------------------------
# Score calculation
# ----------------------------

df["Score"] = (
w_net * df["NetEff_z"]
+ w_adjo * df["AdjOE_x_z"]
- w_adjd * df["AdjDE_x_z"]
+ w_3p * df["3P%_z"]
+ w_2p * df["2P%_z"]
+ w_ft * df["FT%_z"]
+ w_tempo * df["TempoBalance_z"]
+ w_conf * df["ConfStrength_z"]
+ w_mascot * df["PowerRating_z"]
)

df["Score"] = df["Score"] * 25 + 50

df = df.sort_values("Score",ascending=False).reset_index(drop=True)
df["Model Rank"] = df.index + 1

# ----------------------------
# Helper functions
# ----------------------------

def win_prob(a,b):
    s1 = df[df.Team==a]["Score"].values[0]
    s2 = df[df.Team==b]["Score"].values[0]
    return 1/(1+np.exp(-(s1-s2)/8))

def seed_slot(region_df, seed):
    teams = region_df[region_df.Seed==seed]

    if len(teams) == 1:
        return teams.iloc[0]["Team"], teams.iloc[0]["Score"]

    # play-in case
    names = " / ".join(teams["Team"])
    score = teams["Score"].mean()

    return names, score

def slot_win_prob(scoreA, scoreB):
    return 1/(1+np.exp(-(scoreA-scoreB)/8))

def simulate_game(teamA, teamB):

    s1 = df[df.Team==teamA]["Score"].values[0]
    s2 = df[df.Team==teamB]["Score"].values[0]

    p = 1/(1+np.exp(-(s1-s2)/8))

    return teamA if np.random.random() < p else teamB

# ----------------------------
# Tabs
# ----------------------------

tab1, tab2, tab3 = st.tabs(["Model Rankings","Bracket Projection", "Upset Finder & Simulator"])

# ====================================================
# TAB 1
# ====================================================

with tab1:

    st.subheader("Model Rankings")

    st.dataframe(
        df[["Model Rank","Team","Seed","Region","Score"]],
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Matchup Predictor")

    team_list = df["Team"].sort_values()

    team1 = st.selectbox("Team 1",team_list)
    team2 = st.selectbox("Team 2",team_list,index=1)

    if team1 and team2:

        p = win_prob(team1,team2)

        col1,col2,col3 = st.columns(3)

        col1.metric(team1,f"{p*100:.1f}%")
        col3.metric(team2,f"{(1-p)*100:.1f}%")

    # ------------------------
    # Trapezoid
    # ------------------------

    st.subheader("Trapezoid of Excellence")
    
    fig, ax = plt.subplots(figsize=(9,6))
    
    ax.scatter(df["AdjTempo"], df["NetEff"], alpha=0.7)
    
    ax.set_xlabel("Adjusted Tempo")
    ax.set_ylabel("Net Efficiency")
    
    ax.set_xlim(61.5,77.5)
    
    top_eff = df["NetEff"].max()+2
    ax.set_ylim(5,top_eff)
    
    bottom_y = 26
    top_y = top_eff-2
    
    bottom_left = 66.5
    bottom_right = 71
    top_left = 64
    top_right = 73.5
    
    trap_x = [top_left, top_right, bottom_right, bottom_left, top_left]
    trap_y = [top_y, top_y, bottom_y, bottom_y, top_y]
    
    ax.plot(trap_x, trap_y)
    
    # ---------------------------------
    # Identify teams inside trapezoid
    # ---------------------------------
    
    def in_trapezoid(row):
    
        x = row["AdjTempo"]
        y = row["NetEff"]
    
        if y < bottom_y or y > top_y:
            return False
    
        # left boundary slopes inward
        left_bound = bottom_left - (bottom_left - top_left) * ((y-bottom_y)/(top_y-bottom_y))
    
        # right boundary slopes outward
        right_bound = bottom_right + (top_right-bottom_right) * ((y-bottom_y)/(top_y-bottom_y))
    
        return left_bound <= x <= right_bound
    
    
    trap_teams = df[df.apply(in_trapezoid, axis=1)]

    # ---------------------------------
    # Add team labels
    # ---------------------------------
    
    for _, row in trap_teams.iterrows():
        ax.text(
            row["AdjTempo"],
            row["NetEff"],
            row["Team"],
            fontsize=8
        )
    
    st.pyplot(fig)

# ====================================================
# TAB 2
# ====================================================

with tab2:

    st.subheader("Projected Bracket")

    regions = ["East","South","West","Midwest"]

    matchups = [
    (1,16),(8,9),(5,12),(4,13),
    (6,11),(3,14),(7,10),(2,15)
    ]
    
    def get_score(team):
        # Handle play-in teams like "Boise State/Colorado"
        if "/" in team:
    
            t1, t2 = team.split("/")
    
            s1 = df[df.Team==t1]["Score"]
            s2 = df[df.Team==t2]["Score"]
    
            s1 = s1.values[0] if len(s1) > 0 else 0
            s2 = s2.values[0] if len(s2) > 0 else 0
    
            return max(s1, s2)
    
        else:
    
            s = df[df.Team==team]["Score"]
    
            if len(s) == 0:
                return 0
    
            return s.values[0]
    
    def slot_win_prob(s1,s2):
        return 1/(1+np.exp(-(s1-s2)/8))

    def seed_slot(region_df, seed):

        teams = region_df[region_df.Seed==seed]

        if len(teams) == 1:

            team = teams.iloc[0]["Team"]
            score = get_score(team)

            return team, score

        elif len(teams) == 2:

            t1 = teams.iloc[0]["Team"]
            t2 = teams.iloc[1]["Team"]

            team_string = f"{t1}/{t2}"

            score = max(get_score(t1), get_score(t2))

            return team_string, score

        else:
            return "Unknown", 0

    # ----------------------------
    # Round of 64
    # ----------------------------

    round32 = []

    for region in regions:

        st.write(f"### {region} Region")

        r = df[df.Region==region]

        winners = []

        for s1,s2 in matchups:

            t1,score1 = seed_slot(r,s1)
            t2,score2 = seed_slot(r,s2)

            p = slot_win_prob(score1,score2)

            winner = t1 if p > .5 else t2

            st.write(f"{t1} vs {t2} → **{winner}**")

            winners.append(winner)

        round32.append(winners)

    # ----------------------------
    # Round of 32
    # ----------------------------

    st.subheader("Round of 32")

    sweet16 = []

    for region_winners in round32:

        winners = []

        for i in range(0,8,2):

            t1 = region_winners[i]
            t2 = region_winners[i+1]

            s1 = get_score(t1)
            s2 = get_score(t2)

            p = slot_win_prob(s1,s2)

            winner = t1 if p>.5 else t2

            st.write(f"{t1} vs {t2} → **{winner}**")

            winners.append(winner)

        sweet16.append(winners)

    # ----------------------------
    # Sweet 16
    # ----------------------------

    st.subheader("Sweet 16")

    elite8 = []

    for region in sweet16:

        winners = []

        for i in range(0,4,2):

            t1 = region[i]
            t2 = region[i+1]

            s1 = get_score(t1)
            s2 = get_score(t2)

            p = slot_win_prob(s1,s2)

            winner = t1 if p>.5 else t2

            st.write(f"{t1} vs {t2} → **{winner}**")

            winners.append(winner)

        elite8.append(winners)

    # ----------------------------
    # Elite 8
    # ----------------------------

    st.subheader("Elite 8")

    final4 = []

    for region in elite8:

        t1 = region[0]
        t2 = region[1]

        s1 = get_score(t1)
        s2 = get_score(t2)

        p = slot_win_prob(s1,s2)

        winner = t1 if p>.5 else t2

        st.write(f"{t1} vs {t2} → **{winner}**")

        final4.append(winner)

    # ----------------------------
    # Final Four
    # ----------------------------

    st.subheader("Final Four")

    championship = []

    for i in range(0,4,2):

        t1 = final4[i]
        t2 = final4[i+1]

        s1 = get_score(t1)
        s2 = get_score(t2)

        p = slot_win_prob(s1,s2)

        winner = t1 if p>.5 else t2

        st.write(f"{t1} vs {t2} → **{winner}**")

        championship.append(winner)

    # ----------------------------
    # Championship
    # ----------------------------

    st.subheader("National Championship")

    t1 = championship[0]
    t2 = championship[1]

    s1 = get_score(t1)
    s2 = get_score(t2)

    p = slot_win_prob(s1,s2)

    champ = t1 if p>.5 else t2

    st.write(f"{t1} vs {t2} → 🏆 **{champ}**")

with tab3:
    # ------------------------
    # Upset detection
    # ------------------------

    st.subheader("Most Likely Upsets")

    upsets = []

    for region in regions:

        r = df[df.Region==region]

        for s1,s2 in matchups:

            fav, fav_score = seed_slot(r,s1)
            dog, dog_score = seed_slot(r,s2)

            p = slot_win_prob(dog_score,fav_score)

            if p > 0.35:
                upsets.append((dog,fav,p))

    upsets = sorted(upsets,key=lambda x:x[2],reverse=True)

    for u in upsets[:10]:
        st.write(f"{u[0]} over {u[1]} ({u[2]*100:.1f}%)")

    st.subheader("Tournament Simulator")

    if st.button("Simulate Tournament"):
    
        champions = []
    
        regions = ["East","South","West","Midwest"]
    
        matchups = [(1,16),(8,9),(5,12),(4,13),(6,11),(3,14),(7,10),(2,15)]
    
        final_four = []
    
        for region in regions:
    
            r = df[df.Region==region]
    
            winners = []
    
            for s1,s2 in matchups:
    
                teams1 = r[r.Seed==s1]["Team"].tolist()
                teams2 = r[r.Seed==s2]["Team"].tolist()
    
                t1 = np.random.choice(teams1)
                t2 = np.random.choice(teams2)
    
                winners.append(simulate_game(t1,t2))
    
            # Sweet 16
            r2 = []
            for i in range(0,len(winners),2):
                r2.append(simulate_game(winners[i], winners[i+1]))
    
            # Elite 8
            r3 = []
            for i in range(0,len(r2),2):
                r3.append(simulate_game(r2[i], r2[i+1]))
    
            final_four.append(simulate_game(r3[0], r3[1]))
    
        # National Semifinals
        champ_game = [
            simulate_game(final_four[0], final_four[1]),
            simulate_game(final_four[2], final_four[3])
        ]
    
        champion = simulate_game(champ_game[0], champ_game[1])
    
        st.write("### Final Four")
        for t in final_four:
            st.write(t)
    
        st.write("### Championship")
        st.write(f"{champ_game[0]} vs {champ_game[1]}")
    
        st.write("## 🏆 Champion")
        st.write(champion)
