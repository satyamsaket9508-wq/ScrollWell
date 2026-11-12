"""Scrollwell: a fitness-style wellbeing coach. Predicts a student's mental-health score from daily habits."""
import json
import time
from datetime import date, datetime, timedelta
from datetime import time as dtime
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

HERE = Path(__file__).parent
MODEL_PATH, STORE = HERE / "Mental_Health_Model.pkl", HERE / "scrollwell_data.json"
BG, CARD, LINE, TXT, MUTED = "#0B0F14", "#151B23", "#232C38", "#F2F5F8", "#8A97A8"
VOLT, CYAN, AMBER, RED, ORANGE = "#C8FF3D", "#2DD4E6", "#FFB020", "#FF4D4D", "#FF6A3D"

_cfg = HERE / ".streamlit" / "config.toml"
if not _cfg.exists():  # dark theme (applies after one restart)
    _cfg.parent.mkdir(exist_ok=True)
    _cfg.write_text('[theme]\nbase = "dark"\nprimaryColor = "#C8FF3D"\nbackgroundColor = "#0B0F14"\n'
                    'secondaryBackgroundColor = "#151B23"\ntextColor = "#F2F5F8"\n')

PLATFORMS = ["Instagram", "Facebook", "YouTube", "Twitter", "Snapchat", "LinkedIn", "WhatsApp", "LINE",
             "VKontakte", "KakaoTalk", "WeChat"]
COUNTRIES = ["India", "USA", "Canada", "Australia", "UK", "Germany", "Mexico", "France", "Denmark", "Other"]
STRESS = ["Low", "Medium", "High", "Very High"]
FEATURES = ["Study_Hours", "Age", "Avg_Daily_Usage_Hours", "Daily_Unlocks", "Physical_Activity_Hours",
            "Sleep_Hours_Per_Night", "Stress_Level", "Gender", "Academic_Level", "Most_Used_Platform",
            "Purpose_Of_Use", "grouped_countries"]

st.set_page_config(page_title="Scrollwell", page_icon="💪", layout="wide", initial_sidebar_state="expanded")
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700;800&family=DM+Sans:wght@400;500;700&display=swap');
html, body, [class*="css"], .stApp { font-family: 'DM Sans', sans-serif; color: __TXT__; }
.stApp { background: radial-gradient(800px 500px at 92% -8%, rgba(200,255,61,.11), transparent 60%),
  radial-gradient(700px 500px at -6% 0%, rgba(45,212,230,.10), transparent 55%), __BG__; }
.stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp label, .stApp li, [data-testid="stMarkdownContainer"] { color: __TXT__; }
h1, h2, h3 { font-family: 'Barlow Condensed', sans-serif !important; text-transform: uppercase; letter-spacing: .03em; }
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.block-container { padding-top: 1.4rem; max-width: 1220px; }
[data-testid="stSidebar"] { background: #0F141B; border-right: 1px solid __LINE__; }
.hero { position: relative; overflow: hidden; border-radius: 28px; padding: 2.2rem 2.6rem; margin-bottom: 1.4rem;
  background: linear-gradient(120deg, #10161E 0%, #1B2A12 100%); border: 1px solid __LINE__; }
.hero::after { content: ""; position: absolute; right: -70px; top: -90px; width: 340px; height: 340px; border-radius: 50%;
  background: radial-gradient(circle, rgba(200,255,61,.45), transparent 65%); }
.hero > * { position: relative; z-index: 1; }
.stApp .hero h1 { font-size: clamp(2.2rem, 5vw, 3.8rem); font-weight: 800; line-height: 1; margin: 0 0 .5rem; }
.stApp .hero h1 em { color: __VOLT__; font-style: normal; }
.stApp .hero p { color: __MUTED__; max-width: 58ch; margin: 0; }
.chips { display: flex; gap: .55rem; flex-wrap: wrap; margin-top: 1.2rem; }
.chip { background: rgba(255,255,255,.07); border: 1px solid rgba(255,255,255,.14); padding: .38rem .9rem; border-radius: 99px;
  font-size: .86rem; font-weight: 700; }
.chip.v { background: __VOLT__; color: #0B0F14; border-color: __VOLT__; }
div[data-baseweb="tab-list"] { gap: .5rem; border-bottom: 0; }
div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] { display: none; }
button[data-baseweb="tab"] { background: __CARD__; border: 1px solid __LINE__; border-radius: 99px; padding: .5rem 1.3rem; height: auto; }
button[data-baseweb="tab"] p { font-weight: 700; }
.stApp button[aria-selected="true"] { background: __VOLT__; border-color: __VOLT__; }
.stApp button[aria-selected="true"] p { color: #0B0F14; }
div[data-testid="stVerticalBlockBorderWrapper"], .panel { background: __CARD__; border: 1px solid __LINE__; border-radius: 22px; }
div[data-testid="stVerticalBlockBorderWrapper"] { padding: .8rem 1rem; }
.panel { padding: 1.3rem 1.4rem; margin-bottom: 1rem; }
.panel h3 { margin: 0 0 .6rem; }
.panel.gauge { text-align: center; border-top: 5px solid var(--c); box-shadow: 0 30px 70px -40px var(--c); }
.gauge .arc { animation: grow 1s ease-out; }
@keyframes grow { from { stroke-dasharray: 0 283; } }
.gauge .score { font-family: 'Barlow Condensed'; font-size: 5rem; font-weight: 800; line-height: 1; margin-top: -3.6rem; }
.gauge .score small { font-size: 1.3rem; color: __MUTED__; font-weight: 600; }
.gauge .band { font-family: 'Barlow Condensed'; font-weight: 800; font-size: 1.5rem; text-transform: uppercase; letter-spacing: .05em; }
.gauge .sub, .note { color: __MUTED__ !important; font-size: .88rem; }
.rings { display: grid; grid-template-columns: repeat(4, 1fr); gap: .6rem; text-align: center; }
.ring { width: 78px; height: 78px; margin: 0 auto .4rem; border-radius: 50%; display: grid; place-items: center; position: relative;
  background: conic-gradient(var(--c) calc(var(--p) * 1%), #232C38 0); }
.ring::before { content: ""; position: absolute; inset: 8px; background: __CARD__; border-radius: 50%; }
.ring span { position: relative; font-weight: 800; font-size: .95rem; }
.rw b { display: block; font-size: .9rem; } .rw small { color: __MUTED__; font-size: .75rem; }
.lever { padding: .7rem 0; border-bottom: 1px solid __LINE__; } .lever:last-of-type { border-bottom: 0; }
.lt { display: flex; justify-content: space-between; align-items: center; }
.lt b { color: __VOLT__; font-family: 'Barlow Condensed'; font-size: 1.25rem; }
.bar { height: 6px; background: #232C38; border-radius: 9px; margin-top: .5rem; }
.bar i { display: block; height: 100%; border-radius: 9px; background: linear-gradient(90deg, __CYAN__, __VOLT__); }
.best { background: __VOLT__; color: #0B0F14; font-size: .68rem; font-weight: 800; padding: .1rem .5rem; border-radius: 99px; margin-left: .5rem; }
.tip { display: flex; gap: .8rem; padding: .6rem 0; } .tip .ic { font-size: 1.5rem; } .tip b { display: block; }
.badges { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: .8rem; }
.badge { background: __CARD__; border: 1px solid __LINE__; border-radius: 18px; padding: 1rem; text-align: center; }
.badge.on { border-color: __VOLT__; box-shadow: 0 0 24px -10px __VOLT__; } .badge.off { opacity: .35; filter: grayscale(1); }
.badge .e { font-size: 2rem; } .badge b { display: block; font-size: .85rem; margin-top: .3rem; }
div[data-testid="stMetric"] { background: __CARD__; border: 1px solid __LINE__; border-radius: 18px; padding: 1rem 1.2rem; }
div[data-testid="stMetric"] * { color: __TXT__; }
div[role="slider"] { background: __VOLT__ !important; } div[data-testid="stSliderThumbValue"] { color: __VOLT__ !important; }
.stButton > button, .stDownloadButton > button { background: #1B232D; color: __TXT__; border: 1px solid __LINE__; border-radius: 12px; font-weight: 700; }
.stButton > button:hover, .stDownloadButton > button:hover { border-color: __VOLT__; color: __VOLT__; }
button[data-testid="stBaseButton-primary"] { background: __VOLT__; color: #0B0F14; border: 0; }
button[data-testid="stBaseButton-primary"]:hover { color: #0B0F14; filter: brightness(1.08); }
[data-testid="stPopover"] button { border-radius: 50%; width: 34px; height: 34px; min-height: 0; padding: 0; color: __VOLT__; border: 1px solid __VOLT__; font-weight: 800; }
.hm { display: grid; grid-template-columns: 2.2fr repeat(7, 1fr); gap: 6px; align-items: center; font-size: .8rem; }
.hm .c { height: 22px; border-radius: 6px; background: #232C38; } .hm .c.on { background: __VOLT__; } .hm small { color: __MUTED__; text-align: center; }
.cal { display: grid; grid-template-columns: repeat(14, 1fr); gap: 5px; }
.cal span { aspect-ratio: 1; border-radius: 8px; background: #232C38; display: grid; place-items: center; font-size: .72rem; color: __MUTED__; }
.cal span.on { background: __VOLT__; color: #0B0F14; font-weight: 800; }
.breath { width: 150px; height: 150px; border-radius: 50%; margin: 1rem auto; display: grid; place-items: center; font-weight: 700;
  border: 2px solid __CYAN__; background: radial-gradient(circle, rgba(45,212,230,.45), rgba(45,212,230,.06)); animation: br 8s ease-in-out infinite; }
@keyframes br { 0%, 100% { transform: scale(.7); } 50% { transform: scale(1.15); } }
.clock { font-family: 'Barlow Condensed'; font-size: 4rem; font-weight: 800; color: __VOLT__; line-height: 1; }
@media (max-width: 760px) { .rings { grid-template-columns: repeat(2, 1fr); } }
</style>"""
for tok, val in dict(TXT=TXT, BG=BG, LINE=LINE, VOLT=VOLT, MUTED=MUTED, CARD=CARD, CYAN=CYAN).items():
    CSS = CSS.replace(f"__{tok}__", val)
st.markdown(CSS, unsafe_allow_html=True)


# ---------- model + storage ----------
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    return joblib.load(MODEL_PATH)  # saved with joblib


def predict(model, row):
    return float(model.predict(pd.DataFrame([row])[FEATURES])[0])


GKEYS = ("g_screen", "g_sleep", "g_move", "g_unlock")


def save_store():
    try:
        d = {k: st.session_state[k] for k in ("name", "history", "checks", "focus_total")}
        d["goals"] = {k: st.session_state[k] for k in GKEYS}
        try:
            d["focus_total"] = max(d["focus_total"], json.loads(STORE.read_text()).get("focus_total", 0))
        except (OSError, ValueError):
            pass
        STORE.write_text(json.dumps(d))
    except OSError:
        pass


if not MODEL_PATH.exists():
    st.error(f"Missing file: **{MODEL_PATH.name}**. Put it in the same folder as main.py.")
    st.stop()
model = load_model()

if "history" not in st.session_state:
    data = {"name": "", "history": [], "checks": {}, "focus_total": 0}
    try:
        data.update(json.loads(STORE.read_text()))
    except (OSError, ValueError):
        pass
    st.session_state.update({**{k: v for k, v in data.items() if k != "goals"}, **data.get("goals", {})})
S = st.session_state

DEFAULTS = dict(usage=5.0, sleep=6.5, unlocks=170, study=3.0, activity=1.5, stress="Medium")
PRESETS = {
    "🧘 Balanced": dict(usage=2.5, sleep=8.0, unlocks=100, study=4.0, activity=2.0, stress="Low"),
    "📱 Heavy scroller": dict(usage=8.0, sleep=5.5, unlocks=250, study=2.0, activity=0.3, stress="High"),
    "📚 Exam week": dict(usage=4.0, sleep=5.0, unlocks=150, study=8.0, activity=0.5, stress="Very High"),
}
for k, v in {**DEFAULTS, "g_screen": 3.0, "g_sleep": 8.0, "g_move": 2.0, "g_unlock": 100, "mood": "😐", "note": ""}.items():
    S.setdefault(k, v)

# ---------- pop-out pages (open in a new browser tab) ----------
def bump_focus(mins):
    try:
        d = json.loads(STORE.read_text())
    except (OSError, ValueError):
        d = {}
    d["focus_total"] = d.get("focus_total", 0) + mins
    try:
        STORE.write_text(json.dumps(d))
    except OSError:
        pass
    S.focus_total = d["focus_total"]


def chips_html(items):
    return '<div class="chips" style="justify-content:center">' + "".join(f'<span class="chip{" v" if v else ""}">{t}</span>' for t, v in items) + "</div>"


VIEW = st.query_params.get("view")
if VIEW:
    st.markdown("<style>[data-testid='stSidebar'], [data-testid='stSidebarCollapsedControl'] { display: none; }</style>"
                f'<a href="?" target="_self" style="color:{VOLT};font-weight:700;text-decoration:none">← Back to Scrollwell</a>', unsafe_allow_html=True)
    if VIEW == "focus":
        st.markdown("<h1 style='text-align:center'>Focus mode</h1>", unsafe_allow_html=True)
        S.setdefault("focus_end", 0.0); S.setdefault("focus_len", 25)
        st.select_slider("Session length (minutes)", [5, 10, 15, 25, 45, 60], key="focus_len")
        f1, f2 = st.columns(2)
        if f1.button("Start", type="primary", width="stretch", key="vstart"):
            S.focus_end = time.time() + S.focus_len * 60; st.rerun()
        if f2.button("Stop", width="stretch", key="vstop"):
            S.focus_end = 0.0; st.rerun()

        @st.fragment(run_every=1 if S.focus_end else None)
        def vclock():
            rem = S.focus_end - time.time()
            if S.focus_end and rem <= 0:
                bump_focus(S.focus_len); S.focus_end = 0.0; st.balloons(); st.rerun()
            m, sec = divmod(int(max(rem, 0)), 60) if S.focus_end else divmod(S.focus_len * 60, 60)
            st.markdown(f'<div class="panel" style="text-align:center;padding:3rem 1rem"><div class="clock" style="font-size:8rem">{m:02d}:{sec:02d}</div>'
                        f'<p class="note">{"Stay off your phone until the timer ends." if S.focus_end else "Press Start to begin."}</p></div>', unsafe_allow_html=True)

        vclock()
        st.caption(f"Total focused time: {S.focus_total} min")
    elif VIEW == "breathe":
        st.markdown("<h1 style='text-align:center'>Breathing space</h1>", unsafe_allow_html=True)
        P = {"Quick (3 s in · 3 s out)": 6, "Calm (4 s in · 4 s out)": 8, "Deep (6 s in · 6 s out)": 12}
        pat = st.selectbox("Pattern", list(P))
        st.markdown(f'<div style="text-align:center;padding:2rem 0"><div class="breath" style="width:280px;height:280px;font-size:1.6rem;animation-duration:{P[pat]}s">Breathe</div>'
                    '<p class="note">Inhale as the circle grows, exhale as it shrinks. Keep going for a few minutes.</p></div>', unsafe_allow_html=True)
    elif VIEW == "bedtime":
        st.markdown("<h1 style='text-align:center'>Bedtime calculator</h1>", unsafe_allow_html=True)
        mode = st.radio("I want to", ["Wake up at a set time", "Go to sleep right now"], horizontal=True)
        if mode.startswith("Wake"):
            wk_t = st.time_input("Wake-up time", dtime(7, 0)); b0 = datetime.combine(date.today(), wk_t)
            out = [(f"Sleep at {b0 - timedelta(minutes=90 * c + 15):%I:%M %p} · {c * 1.5:g} h", c == 5) for c in (6, 5, 4)]
        else:
            b0 = datetime.now()
            out = [(f"Wake at {b0 + timedelta(minutes=90 * c + 15):%I:%M %p} · {c * 1.5:g} h", c == 5) for c in (4, 5, 6)]
        st.markdown(chips_html(out) + '<p class="note" style="text-align:center;margin-top:.8rem">Includes about 15 minutes to fall asleep. '
                    '5 cycles (7.5 h) is a good target.</p>', unsafe_allow_html=True)
    elif VIEW == "report":
        st.markdown("<h1 style='text-align:center'>Progress report</h1>", unsafe_allow_html=True)
        H = S.history
        if not H:
            st.info("No check-ins yet. Check in on the Today tab and your report will appear here.")
        else:
            df_h = pd.DataFrame(H); df_h.insert(0, "#", range(1, len(df_h) + 1))
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Check-ins", len(H)); r2.metric("Average score", f"{df_h.score.mean():.1f}")
            r3.metric("Best score", f"{df_h.score.max():.1f}"); r4.metric("Latest", f"{H[-1]['score']:.1f}", f"{H[-1]['score'] - H[0]['score']:+.1f} since first")
            fig = px.line(df_h, x="#", y="score", markers=True, color_discrete_sequence=[VOLT], labels={"score": "Score"})
            fig.update_layout(height=340, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=TXT), margin=dict(l=0, r=0, t=10, b=0))
            fig.update_yaxes(range=[1, 10], gridcolor=LINE); fig.update_xaxes(dtick=1, gridcolor=LINE)
            st.plotly_chart(fig, width="stretch")
            st.dataframe(df_h.drop(columns=[c for c in ("note",) if c in df_h]), hide_index=True, width="stretch")
            st.download_button("⬇ Download (CSV)", df_h.to_csv(index=False), "scrollwell_report.csv", "text/csv")
            st.caption("Tip: press Ctrl + P to print or save this page as a PDF.")
    else:
        st.warning("Unknown page.")
    st.stop()

# ---------- sidebar: profile ----------
with st.sidebar:
    st.markdown("### 👤 Your profile")
    st.text_input("Your name", key="name", on_change=save_store, placeholder="e.g. Aryan")
    st.number_input("Age", 18, 24, 21, key="age")
    st.selectbox("Gender", ["Female", "Male"], key="gender")
    st.selectbox("Academic level", ["High School", "Undergraduate", "Graduate"], index=1, key="level")
    st.selectbox("Most-used platform", PLATFORMS, key="platform")
    st.selectbox("Main purpose", ["Entertainment", "Education", "Networking", "News"], key="purpose")
    st.selectbox("Country", COUNTRIES + ["Another country"], key="country")
    with st.expander("My daily goals"):
        st.number_input("Screen limit (h)", 1.0, 12.0, step=0.5, key="g_screen", on_change=save_store)
        st.number_input("Sleep goal (h)", 5.0, 10.0, step=0.5, key="g_sleep", on_change=save_store)
        st.number_input("Activity goal (h)", 0.5, 6.0, step=0.5, key="g_move", on_change=save_store)
        st.number_input("Unlock limit", 30, 300, step=10, key="g_unlock", on_change=save_store)
    st.caption("Your check-ins are stored locally in scrollwell_data.json on this computer.")

row = dict(Study_Hours=S.study, Age=S.age, Avg_Daily_Usage_Hours=S.usage, Daily_Unlocks=S.unlocks,
           Physical_Activity_Hours=S.activity, Sleep_Hours_Per_Night=S.sleep, Stress_Level=S.stress,
           Gender=S.gender, Academic_Level=S.level, Most_Used_Platform=S.platform, Purpose_Of_Use=S.purpose,
           grouped_countries=S.country if S.country in COUNTRIES else "other")
usage, sleep, unlocks, study, activity, stress = S.usage, S.sleep, S.unlocks, S.study, S.activity, S.stress
score = predict(model, row)


# ---------- info (i) helper ----------
TIPS = {
 "score": "Your predicted mental-health score from 1 to 10, calculated live by the trained model from the habits you set. Higher is better.",
 "rings": "Four progress rings showing how close you are to your daily goals. You can change the goals in the sidebar under **My daily goals**.",
 "best": "The model re-scores you with just one habit changed at a time. The habit at the top will lift your score the most.",
 "coach": "Short tips picked from your own numbers: screen time, sleep, unlocks, activity and stress.",
 "checkin": "Saves today's score, mood and an optional note. Check in daily to build your streak and earn XP and badges.",
 "share": "A ready-to-copy line with your score, rank and streak. Tap the copy icon on the box.",
 "quick": "One tap fills all the sliders with a typical day, so you can see how different lifestyles score. Reset goes back to the defaults.",
 "day": "Describe a typical day. The score, rings and plans update instantly as you move the sliders.",
 "drivers": "Green bars are habits that would raise your score if moved to a healthy level. Red bars would lower it.",
 "real": "Your daily numbers converted into real-life terms: full days per year on social media, weekly sleep debt, and yearly phone unlocks.",
 "curves": "Each curve shows your predicted score as only that habit changes. The green dot is where you are now.",
 "scenario": "Combine several changes at once, for example 1 hour less scrolling and 1 hour more sleep, and see the total effect.",
 "goal": "Pick a target score. The planner takes small steps, always choosing the habit change the model likes best, until it reaches the target.",
 "ramp": "The same plan spread over 7 days so the change is gradual and realistic.",
 "micro": "Five small daily habits. Your ticks are saved per day and shown in the weekly heatmap on the Progress tab.",
 "badges": "Achievements you unlock by building streaks, sleeping well, moving more and using your phone less. Greyed-out badges are still locked.",
 "trend": "Your score at each check-in, so you can see whether your habits are paying off.",
 "week": "A summary of your last 7 days compared with the 7 days before. Open the full report in a new tab to print or save it as a PDF.",
 "heat": "Which micro-habits you completed on each of the last 7 days. Green means done.",
 "cal": "Green squares are the days you checked in during the last 14 days.",
 "records": "Your personal bests across all check-ins.",
 "compare": "How your latest check-in compares with your very first one.",
 "insight": "Patterns found in your own check-ins. Needs at least 5 check-ins.",
 "mood": "Your mood at each check-in. Compare it with your score to see what lifts you.",
 "backup": "Download your data as a file and restore it later, for example on another computer.",
 "breath": "Box-style breathing to calm stress. Breathe in as the circle grows and out as it shrinks. Use the button to open it full-screen in a new tab.",
 "focus": "A phone-free focus timer. Finished sessions are added to your total focused time. Use the button to open a distraction-free focus page in a new tab.",
 "budget": "Enter the social media time you've used today to see how much of your daily limit is left.",
 "bed": "Sleep comes in 90-minute cycles. Pick a bedtime that lets you wake at the end of a cycle and feel fresher. Open it in a new tab for a bigger view.",
 "reset": "A quick 5-minute activity chosen for your current stress level.",
 "remind": "Download a daily calendar reminder for your check-in. Open the file to add it to Google, Apple or Outlook calendar.",
}
MOODS = ["😞", "😕", "😐", "🙂", "😄"]


def head(title, tip):
    c1, c2 = st.columns([14, 1], vertical_alignment="center")
    c1.markdown(f"<h3 style='margin:0'>{title}</h3>", unsafe_allow_html=True)
    with c2.popover("ⓘ"):
        st.markdown(tip)


def checkin(rec):
    S.history.append(rec)
    S.note = ""
    save_store()
    st.toast("Checked in! Keep the streak alive 🔥")


# ---------- helpers ----------
def band(s):
    if s >= 7.5: return "Thriving", VOLT, "Habits look healthy. Keep the balance."
    if s >= 6.0: return "Steady", CYAN, "Doing okay, with room to improve."
    if s >= 4.5: return "Strained", AMBER, "Some habits are weighing on wellbeing."
    return "At risk", RED, "Small changes can turn this around."


def rank(s):
    for lim, t in [(3.5, "Rookie"), (4.5, "Starter"), (6.0, "Rising"), (7.5, "Strong"), (8.5, "Champion")]:
        if s < lim: return t
    return "Legend"


def gauge(s):
    label, color, sub = band(s)
    frac = max(0.0, min(1.0, (s - 1) / 9)) * 282.7
    return f"""<div class="panel gauge" style="--c:{color}"><svg viewBox="0 0 220 125" width="100%" style="max-width:380px">
<path d="M20 110 A90 90 0 0 1 200 110" fill="none" stroke="#232C38" stroke-width="16" stroke-linecap="round"/>
<path class="arc" d="M20 110 A90 90 0 0 1 200 110" fill="none" stroke="{color}" stroke-width="16" stroke-linecap="round"
 stroke-dasharray="{frac:.1f} 283"/></svg><div class="score">{s:.1f}<small>/10</small></div>
<div class="band" style="color:{color}">{label}</div><div class="sub">{sub}</div></div>"""


def rings_html():
    items = [("Screen", min(100, S.g_screen / usage * 100), CYAN, f"{usage:g}h / {S.g_screen:g}h"),
             ("Sleep", min(100, sleep / S.g_sleep * 100), VOLT, f"{sleep:g}h / {S.g_sleep:g}h"),
             ("Move", min(100, activity / S.g_move * 100), ORANGE, f"{activity:g}h / {S.g_move:g}h"),
             ("Unlocks", min(100, S.g_unlock / max(unlocks, 1) * 100), AMBER, f"{unlocks} / {S.g_unlock}")]
    cells = "".join(f'<div class="rw"><div class="ring" style="--p:{p:.0f};--c:{c}"><span>{p:.0f}%</span></div><b>{n}</b><small>{t}</small></div>'
                    for n, p, c, t in items)
    return f'<div class="panel"><div class="rings">{cells}</div></div>'


def tips():
    t = []
    if usage > 5: t.append(("📵", "Set app timers", "Over 5 h/day. A 3 h limit on your top app is a strong first step."))
    if sleep < 7: t.append(("😴", "Protect your sleep", "Under 7 h/night. Charge your phone outside the bedroom and keep a fixed bedtime."))
    if unlocks > 200: t.append(("🔓", "Fewer pickups", "Turn off non-essential notifications to cut your unlocks."))
    if activity < 1: t.append(("🏃", "Move a little", "Even a 20-minute walk fills your activity ring."))
    if stress in ("High", "Very High"): t.append(("🧘", "Reset your stress", "Try 5 minutes of slow breathing, or talk to someone you trust."))
    return t[:3] or [("🌟", "Great balance", "Keep your routine steady and check in daily to hold your streak.")]


def streak():
    days, d, n = {h["date"] for h in S.history}, date.today(), 0
    if d.isoformat() not in days: d -= timedelta(days=1)
    while d.isoformat() in days:
        n, d = n + 1, d - timedelta(days=1)
    return n


# ---------- progress / gamification ----------
today = date.today().isoformat()
HABITS = ["Phone-free first 30 minutes of the day", "10-minute walk or stretch", "A phone-free meal",
          "Screens off 1 hour before bed", "In bed on time for 8 hours of sleep"]
n_chk, hist = len(S.history), S.history
ctx = dict(n=n_chk, streak=streak(), sleep=sleep, usage=usage, act=activity, score=score,
           gain=(hist[-1]["score"] - hist[0]["score"]) if hist else 0, done=sum(len(v) for v in S.checks.values()))
BADGES = [("🚀", "First check-in", ctx["n"] >= 1), ("🔥", "3-day streak", ctx["streak"] >= 3), ("🗓️", "7 check-ins", ctx["n"] >= 7),
          ("😴", "Sleep champion", sleep >= 8), ("📵", "Digital minimalist", usage <= 2.5), ("🏃", "Active mover", activity >= 2),
          ("🌟", "Thriving", score >= 7.5), ("📈", "Comeback (+1)", ctx["gain"] >= 1), ("✅", "Habit hero", ctx["done"] >= 5)]
unlocked = sum(b[2] for b in BADGES)
xp = n_chk * 10 + unlocked * 25 + ctx["done"] * 5
label, color, _ = band(score)
hr = datetime.now().hour
greet = "Good morning" if hr < 12 else "Good afternoon" if hr < 18 else "Good evening"
QUOTE = {"Thriving": "You're in the zone. Protect what's working.", "Steady": "Small habits, stacked daily, beat big resolutions.",
         "Strained": "One better night's sleep changes everything. Start there.", "At risk": "Be kind to yourself. Tiny steps count."}[label]

st.markdown(f"""<div class="hero"><h1>{greet}, <em>{S.name.strip() or "champ"}</em></h1><p>{QUOTE}</p>
<div class="chips"><span class="chip v">🏅 {rank(score)}</span><span class="chip">⚡ Level {xp // 100 + 1} · {xp} XP</span>
<span class="chip">🔥 {ctx["streak"]}-day streak</span><span class="chip">🎖️ {unlocked}/{len(BADGES)} badges</span></div></div>""",
            unsafe_allow_html=True)

tab_today, tab_sim, tab_goal, tab_well, tab_prog = st.tabs(["Today", "Simulator", "Goal & Plan", "Wellness", "Progress"])

# ---------- TODAY ----------
with tab_today:
    left, right = st.columns([6, 5], gap="large")
    with left.container(border=True):
        head("Quick start", TIPS["quick"])
        pc = st.columns(len(PRESETS) + 1)
        for c, name in zip(pc, PRESETS):
            c.button(name, on_click=S.update, args=(PRESETS[name],), key="p_" + name, width="stretch")
        pc[-1].button("↺ Reset", on_click=S.update, args=(DEFAULTS,), key="p_reset", width="stretch")
        head("Your day", TIPS["day"])
        a, b = st.columns(2)
        a.slider("Social media (hours/day)", 1.0, 10.0, step=0.1, key="usage")
        b.slider("Sleep (hours/night)", 3.5, 10.0, step=0.1, key="sleep")
        a.slider("Phone unlocks per day", 50, 300, step=5, key="unlocks")
        b.slider("Study (hours/day)", 0.0, 10.0, step=0.1, key="study")
        a.slider("Physical activity (hours/day)", 0.0, 6.0, step=0.1, key="activity")
        b.select_slider("Stress level", STRESS, key="stress")
    scen = {"Spend 1 hour less on social media": ("Avg_Daily_Usage_Hours", max(1.0, usage - 1), usage > 1.0),
            "Sleep 1 hour more": ("Sleep_Hours_Per_Night", min(10.0, sleep + 1), sleep < 10.0),
            "Unlock your phone 30 fewer times": ("Daily_Unlocks", max(50, unlocks - 30), unlocks > 50),
            "Add 30 minutes of activity": ("Physical_Activity_Hours", min(6.0, activity + 0.5), activity < 6.0)}
    gains = sorted(((predict(model, {**row, k: v}) - score, t) for t, (k, v, ok) in scen.items() if ok), reverse=True)
    with right:
        head("Wellbeing score", TIPS["score"])
        st.markdown(gauge(score), unsafe_allow_html=True)
        head("Daily rings", TIPS["rings"])
        st.markdown(rings_html(), unsafe_allow_html=True)
        head("What would help most", TIPS["best"])
        top = max(gains[0][0], 0.01)
        lev = "".join(f'<div class="lever"><div class="lt"><span>{t}{"<span class=best>Best move</span>" if n == 0 and g > 0.05 else ""}</span>'
                      f'<b>{g:+.2f}</b></div><div class="bar"><i style="width:{max(4, min(100, g / top * 100)):.0f}%"></i></div></div>'
                      for n, (g, t) in enumerate(gains))
        st.markdown(f'<div class="panel">{lev}<p class="note" style="margin-top:.6rem">'
                    f'Estimated score change if only that habit changed. Model estimates, not guarantees.</p></div>', unsafe_allow_html=True)
        head("Your coach says", TIPS["coach"])
        tp = "".join(f'<div class="tip"><span class="ic">{i}</span><div><b>{h}</b><span class="note">{d}</span></div></div>' for i, h, d in tips())
        st.markdown(f'<div class="panel">{tp}</div>', unsafe_allow_html=True)
        report = "\n".join(["SCROLLWELL REPORT", f"Date: {datetime.now():%Y-%m-%d %H:%M}", "",
                            f"Predicted mental-health score: {score:.1f} / 10 ({label})", "", "YOUR DAY",
                            f"- Social media: {usage} h/day", f"- Sleep: {sleep} h/night", f"- Phone unlocks: {unlocks}/day",
                            f"- Study: {study} h/day", f"- Physical activity: {activity} h/day", f"- Stress: {stress}", "",
                            "WHAT WOULD HELP MOST", *[f"- {t}: {g:+.2f}" for g, t in gains], "",
                            "Statistical estimate from survey data. Not medical advice."])
        head("Daily check-in", TIPS["checkin"])
        st.select_slider("How do you feel today?", MOODS, key="mood")
        st.text_input("One-line journal (optional)", key="note", placeholder="What went well today?")
        rec = {"date": today, "time": f"{datetime.now():%H:%M}", "score": round(score, 2), "band": label, "usage": usage, "sleep": sleep,
               "unlocks": unlocks, "activity": activity, "stress": stress, "mood": MOODS.index(S.mood) + 1, "note": S.note.strip()}
        b1, b2 = st.columns(2)
        b1.button("Check in today", type="primary", width="stretch", on_click=checkin, args=(rec,))
        b2.download_button("⬇ Report", report, "scrollwell_report.txt", width="stretch")
        head("Share your score", TIPS["share"])
        st.code(f"💪 My Scrollwell score: {score:.1f}/10 ({label}) · {rank(score)} · {ctx['streak']}-day streak", language=None)

# ---------- SIMULATOR ----------
with tab_sim:
    head("What drives your score", TIPS["drivers"])
    st.caption("Score change if each habit moved to a healthy level, with everything else as you set it.")
    REF = {"Avg_Daily_Usage_Hours": ("Social media → 2.5 h", 2.5), "Sleep_Hours_Per_Night": ("Sleep → 8 h", 8.0),
           "Daily_Unlocks": ("Unlocks → 100", 100), "Physical_Activity_Hours": ("Activity → 2 h", 2.0),
           "Study_Hours": ("Study → 4 h", 4.0), "Stress_Level": ("Stress → Low", "Low")}
    eff = pd.DataFrame([(n, predict(model, {**row, k: v}) - score) for k, (n, v) in REF.items()], columns=["Habit", "Change"]).sort_values("Change")
    eff["Effect"] = ["Helps" if c > 0 else "Hurts" for c in eff.Change]

    def style_fig(fig, h=300):
        fig.update_layout(height=h, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(family="DM Sans", color=TXT), showlegend=False)
        fig.update_xaxes(gridcolor=LINE, zeroline=False); fig.update_yaxes(gridcolor=LINE, zeroline=False)
        return fig

    fig = px.bar(eff, x="Change", y="Habit", orientation="h", color="Effect", color_discrete_map={"Helps": VOLT, "Hurts": RED}, labels={"Habit": ""})
    st.plotly_chart(style_fig(fig, 320), width="stretch")
    head("Your numbers, in real life", TIPS["real"])
    m1, m2, m3 = st.columns(3)
    m1.metric("Social media per year", f"{usage * 365 / 24:.0f} days", help="Full 24-hour days spent on social media in a year")
    m2.metric("Sleep debt per week", f"{max(0, (8 - sleep) * 7):.1f} h", help="Compared with 8 hours a night")
    m3.metric("Phone unlocks per year", f"{unlocks * 365:,}")
    head("Slide a habit, watch the score", TIPS["curves"])
    st.caption("The dot is where you are right now.")

    def sweep(key, xs, now, xlabel):
        ys = model.predict(pd.DataFrame([{**row, key: x} for x in xs])[FEATURES])
        fig = px.line(x=xs, y=ys, labels={"x": xlabel, "y": "Predicted score"}, color_discrete_sequence=[CYAN])
        fig.add_scatter(x=[now], y=[score], mode="markers", marker=dict(size=15, color=VOLT, line=dict(width=3, color=BG)))
        fig.update_yaxes(range=[1, 10])
        return style_fig(fig, 270)

    c1, c2 = st.columns(2, gap="large")
    c1.plotly_chart(sweep("Avg_Daily_Usage_Hours", [1 + i * 0.25 for i in range(37)], usage, "Social media (h/day)"), width="stretch")
    c2.plotly_chart(sweep("Sleep_Hours_Per_Night", [3.5 + i * 0.25 for i in range(27)], sleep, "Sleep (h/night)"), width="stretch")
    c3, c4 = st.columns(2, gap="large")
    c3.plotly_chart(sweep("Daily_Unlocks", list(range(50, 301, 10)), unlocks, "Phone unlocks per day"), width="stretch")
    c4.plotly_chart(sweep("Physical_Activity_Hours", [i * 0.25 for i in range(25)], activity, "Physical activity (h/day)"), width="stretch")

    head("Build your own scenario", TIPS["scenario"])
    q1, q2 = st.columns(2)
    du = q1.slider("Social media change (h)", -4.0, 4.0, 0.0, 0.5, key="du")
    dsl = q2.slider("Sleep change (h)", -3.0, 3.0, 0.0, 0.5, key="dsl")
    dk = q1.slider("Phone unlocks change", -100, 100, 0, 10, key="dk")
    da = q2.slider("Activity change (h)", -1.0, 3.0, 0.0, 0.5, key="da")
    sc2 = predict(model, {**row, "Avg_Daily_Usage_Hours": min(10, max(1, usage + du)), "Sleep_Hours_Per_Night": min(10, max(3.5, sleep + dsl)),
                          "Daily_Unlocks": min(300, max(50, unlocks + dk)), "Physical_Activity_Hours": min(6, max(0, activity + da))})
    r1, r2, r3 = st.columns(3)
    r1.metric("Today", f"{score:.1f}"); r2.metric("Scenario", f"{sc2:.1f}"); r3.metric("Change", f"{sc2 - score:+.2f}")

# ---------- GOAL & PLAN ----------
with tab_goal:
    gl, gr = st.columns([6, 5], gap="large")
    with gl:
        head("Plan your way to a target score", TIPS["goal"])
        S.setdefault("target", 8.0)
        target = st.slider("My target score", 1.0, 10.0, step=0.1, key="target")
        STEPS = [("Social media", "Avg_Daily_Usage_Hours", -0.5, 1.0, 10.0, "h/day"), ("Sleep", "Sleep_Hours_Per_Night", 0.5, 3.5, 10.0, "h/night"),
                 ("Phone unlocks", "Daily_Unlocks", -20, 50, 300, "/day"), ("Physical activity", "Physical_Activity_Hours", 0.25, 0.0, 6.0, "h/day")]
        cur, cur_score = dict(row), score
        for _ in range(40):
            if cur_score >= target: break
            cands = [(k, min(hi, max(lo, cur[k] + d))) for _, k, d, lo, hi, _u in STEPS]
            cands = [(k, v) for k, v in cands if v != cur[k]]
            if not cands: break
            preds = model.predict(pd.DataFrame([{**cur, k: v} for k, v in cands])[FEATURES])
            i = int(preds.argmax())
            if preds[i] <= cur_score + 1e-6: break
            cur[cands[i][0]], cur_score = cands[i][1], float(preds[i])
        g1, g2, g3 = st.columns(3)
        g1.metric("Now", f"{score:.1f}"); g2.metric("Projected", f"{cur_score:.1f}", f"{cur_score - score:+.1f}"); g3.metric("Target", f"{target:.1f}")
        st.progress(max(0.0, min(1.0, (cur_score - 1) / max(target - 1, 0.1))))
        moves = [(n, row[k], cur[k], u) for n, k, _d, _lo, _hi, u in STEPS if abs(cur[k] - row[k]) > 1e-9]
        if not moves:
            st.success("You're already at or above your target. 🎉") if score >= target else st.info("These four habits can't raise your score further.")
        else:
            body = "".join(f'<div class="lever"><div class="lt"><span>{n}</span><b>{a:g} → {b:g} {u}</b></div></div>' for n, a, b, u in moves)
            hd = "🎯 Plan reaches your target" if cur_score >= target else "⚠️ Best possible with these habits"
            st.markdown(f'<div class="panel"><h3>{hd}</h3>{body}</div>', unsafe_allow_html=True)
            head("Your 7-day ramp", TIPS["ramp"])
            ramp = [{**row, **{k: row[k] + (cur[k] - row[k]) * d / 7 for _, k, *_ in STEPS}} for d in range(1, 8)]
            rp = model.predict(pd.DataFrame(ramp)[FEATURES])
            st.dataframe(pd.DataFrame({"Day": [f"Day {d}" for d in range(1, 8)], "Social media (h)": [round(r["Avg_Daily_Usage_Hours"], 1) for r in ramp],
                                       "Sleep (h)": [round(r["Sleep_Hours_Per_Night"], 1) for r in ramp], "Unlocks": [round(r["Daily_Unlocks"]) for r in ramp],
                                       "Activity (h)": [round(r["Physical_Activity_Hours"], 2) for r in ramp], "Score": rp.round(2)}), hide_index=True, width="stretch")
            saved = (usage - cur["Avg_Daily_Usage_Hours"]) * 365 / 24
            if saved > 0.5:
                st.caption(f"⏱️ Following this plan gives you back about {saved:.0f} full days a year.")
    with gr:
        head("Today's micro-habits", TIPS["micro"])
        done_now = []
        for i, h in enumerate(HABITS):
            if st.checkbox(h, value=i in S.checks.get(today, []), key=f"chk{i}"):
                done_now.append(i)
        if done_now != S.checks.get(today, []):
            S.checks[today] = done_now; save_store()
        st.progress(len(done_now) / len(HABITS))
        st.caption(f"{len(done_now)} of {len(HABITS)} done today · {ctx['done']} completed overall")

# ---------- WELLNESS ----------
with tab_well:
    wl, wr = st.columns(2, gap="large")
    with wl:
        head("Guided breathing", TIPS["breath"])
        st.link_button("↗ Open full-screen in a new tab", "?view=breathe")
        st.markdown('<div class="panel" style="text-align:center"><div class="breath">Breathe</div><p class="note">Inhale as the circle grows (4 s), '
                    'exhale as it shrinks (4 s). Repeat for one minute.</p></div>', unsafe_allow_html=True)
        head("Focus timer", TIPS["focus"])
        st.link_button("↗ Open focus mode in a new tab", "?view=focus")
        S.setdefault("focus_end", 0.0); S.setdefault("focus_len", 25)
        st.select_slider("Session length (minutes)", [5, 10, 15, 25, 45, 60], key="focus_len")
        f1, f2 = st.columns(2)
        if f1.button("Start", width="stretch", key="fstart"):
            S.focus_end = time.time() + S.focus_len * 60; st.rerun()
        if f2.button("Stop", width="stretch", key="fstop"):
            S.focus_end = 0.0; st.rerun()

        @st.fragment(run_every=1 if S.focus_end else None)
        def focus_clock():
            rem = S.focus_end - time.time()
            if S.focus_end and rem <= 0:
                S.focus_total += S.focus_len; S.focus_end = 0.0; save_store(); st.balloons(); st.rerun()
            m, sec = divmod(int(max(rem, 0)), 60) if S.focus_end else divmod(S.focus_len * 60, 60)
            st.markdown(f'<div class="panel" style="text-align:center"><div class="clock">{m:02d}:{sec:02d}</div>'
                        f'<p class="note">{"Stay off your phone until the timer ends." if S.focus_end else "Press Start to begin."}</p></div>', unsafe_allow_html=True)

        focus_clock()
        st.caption(f"Total focused time: {S.focus_total} min")
        head("Screen-time budget", TIPS["budget"])
        used = st.number_input("Social media used so far today (h)", 0.0, 16.0, 0.0, 0.25, key="used")
        st.progress(min(1.0, used / S.g_screen))
        st.caption(f"{S.g_screen - used:.1f} h left of your {S.g_screen:g} h budget" if used <= S.g_screen else f"Over budget by {used - S.g_screen:.1f} h")
    with wr:
        head("Bedtime calculator", TIPS["bed"])
        st.link_button("↗ Open in a new tab", "?view=bedtime")
        wake = st.time_input("I need to wake up at", dtime(7, 0), key="wake")
        base = datetime.combine(date.today(), wake)
        chips = "".join(f'<span class="chip{" v" if c == 5 else ""}">{base - timedelta(minutes=90 * c + 15):%I:%M %p} · {c * 1.5:g} h</span>' for c in (6, 5, 4))
        st.markdown(f'<div class="chips" style="margin:0 0 .4rem">{chips}</div><p class="note">Includes about 15 minutes to fall asleep. 5 cycles (7.5 h) is a good target.</p>',
                    unsafe_allow_html=True)
        head("5-minute reset", TIPS["reset"])
        RESET = {"Low": "Take a short walk outside and leave your phone behind.", "Medium": "Stretch your neck, shoulders and back for 5 minutes.",
                 "High": "Do 5 minutes of slow breathing, then write down what's worrying you.",
                 "Very High": "Pause. Drink water, breathe slowly for 2 minutes, then message someone you trust."}
        st.markdown(f'<div class="panel"><b>For stress: {stress}</b><p class="note" style="margin:.3rem 0 0">{RESET[stress]}</p></div>', unsafe_allow_html=True)
        head("Daily reminder", TIPS["remind"])
        rt = st.time_input("Remind me at", dtime(20, 0), key="remind_t")
        ics = ("BEGIN:VCALENDAR\r\nVERSION:2.0\r\nPRODID:-//Scrollwell//EN\r\nBEGIN:VEVENT\r\nUID:scrollwell-daily@local\r\n"
               f"DTSTAMP:{datetime.now():%Y%m%dT%H%M%S}\r\nDTSTART:{date.today():%Y%m%d}T{rt:%H%M}00\r\nRRULE:FREQ=DAILY\r\n"
               "SUMMARY:Scrollwell daily check-in\r\nBEGIN:VALARM\r\nTRIGGER:PT0M\r\nACTION:DISPLAY\r\nDESCRIPTION:Time for your check-in\r\nEND:VALARM\r\n"
               "END:VEVENT\r\nEND:VCALENDAR\r\n")
        st.download_button("⬇ Add to calendar (.ics)", ics, "scrollwell_reminder.ics", "text/calendar", width="stretch")

# ---------- PROGRESS ----------
with tab_prog:
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Check-ins", n_chk); p2.metric("Current streak", f"{ctx['streak']} 🔥"); p3.metric("XP", xp); p4.metric("Badges", f"{unlocked}/{len(BADGES)}")
    head("Badges", TIPS["badges"])
    st.markdown('<div class="badges">' + "".join(f'<div class="badge {"on" if ok else "off"}"><div class="e">{e}</div><b>{n}</b></div>' for e, n, ok in BADGES) + "</div>",
                unsafe_allow_html=True)
    head("Your trend", TIPS["trend"])
    if not hist:
        st.info("No check-ins yet. Set your day on the first tab and press **Check in today** to start your streak.")
    else:
        h = pd.DataFrame(hist); h.insert(0, "#", range(1, len(h) + 1))
        fig = px.line(h, x="#", y="score", markers=True, color_discrete_sequence=[VOLT], labels={"score": "Score"})
        fig.update_yaxes(range=[1, 10]); fig.update_xaxes(dtick=1)
        st.plotly_chart(style_fig(fig, 300), width="stretch")
        st.dataframe(h, hide_index=True, width="stretch")
        d1, d2 = st.columns(2)
        d1.download_button("⬇ Download history (CSV)", h.to_csv(index=False), "scrollwell_history.csv", "text/csv", width="stretch")
        if d2.button("🗑 Clear history", width="stretch"):
            S.history = []; save_store(); st.rerun()
    t0, t1 = date.today() - timedelta(days=6), date.today() - timedelta(days=13)
    wk = [h for h in hist if h["date"] >= t0.isoformat()]
    pw = [h for h in hist if t1.isoformat() <= h["date"] < t0.isoformat()]
    avg = lambda L: sum(h["score"] for h in L) / len(L)
    head("This week", TIPS["week"])
    st.link_button("↗ Open full report in a new tab", "?view=report")
    w1, w2, w3, w4 = st.columns(4)
    w1.metric("Check-ins (7 days)", len(wk))
    w2.metric("Average score", f"{avg(wk):.1f}" if wk else "–", f"{avg(wk) - avg(pw):+.1f} vs last week" if wk and pw else None)
    w3.metric("Best score", f"{max(h['score'] for h in wk):.1f}" if wk else "–")
    w4.metric("Focused time", f"{S.focus_total} min")
    head("Habit heatmap", TIPS["heat"])
    days7 = [date.today() - timedelta(days=i) for i in range(6, -1, -1)]
    cells = "<span></span>" + "".join(f"<small>{d:%a}</small>" for d in days7)
    for i, ht in enumerate(HABITS):
        cells += f"<span>{ht[:26]}</span>" + "".join(f'<span class="c{" on" if i in S.checks.get(d.isoformat(), []) else ""}"></span>' for d in days7)
    st.markdown(f'<div class="panel"><div class="hm">{cells}</div></div>', unsafe_allow_html=True)
    head("Check-in calendar", TIPS["cal"])
    got = {h["date"] for h in hist}
    st.markdown('<div class="panel"><div class="cal">' + "".join(f'<span class="{"on" if (date.today() - timedelta(days=i)).isoformat() in got else ""}">'
                f'{(date.today() - timedelta(days=i)).day}</span>' for i in range(13, -1, -1)) + "</div></div>", unsafe_allow_html=True)
    if hist:
        head("Personal records", TIPS["records"])
        ds_ = sorted({h["date"] for h in hist}); best_run = run = 0; prev = None
        for d in ds_:
            dd = date.fromisoformat(d); run = run + 1 if prev and (dd - prev).days == 1 else 1; best_run = max(best_run, run); prev = dd
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Best score", f"{max(h['score'] for h in hist):.1f}"); k2.metric("Longest streak", f"{best_run} days")
        k3.metric("Lowest screen time", f"{min(h['usage'] for h in hist):g} h"); k4.metric("Most sleep", f"{max(h['sleep'] for h in hist):g} h")
        head("First vs latest", TIPS["compare"])
        a0, a1 = hist[0], hist[-1]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Score", f"{a1['score']:.1f}", f"{a1['score'] - a0['score']:+.1f}")
        c2.metric("Social media", f"{a1['usage']:g} h", f"{a1['usage'] - a0['usage']:+.1f}", delta_color="inverse")
        c3.metric("Sleep", f"{a1['sleep']:g} h", f"{a1['sleep'] - a0['sleep']:+.1f}")
        c4.metric("Unlocks", f"{a1['unlocks']:g}", f"{a1['unlocks'] - a0['unlocks']:+.0f}", delta_color="inverse")
        c5.metric("Activity", f"{a1['activity']:g} h", f"{a1['activity'] - a0['activity']:+.1f}")
        head("Personal insight", TIPS["insight"])
        if len(hist) >= 5:
            med = sorted(h["sleep"] for h in hist)[len(hist) // 2]
            hi = [h["score"] for h in hist if h["sleep"] >= med]; lo = [h["score"] for h in hist if h["sleep"] < med]
            st.info(f"On check-ins with {med:g}+ hours of sleep your score averaged {sum(hi) / len(hi):.1f}, versus {sum(lo) / len(lo):.1f} on shorter nights."
                    if lo else "Your sleep has been steady, so there's no sleep pattern to compare yet.")
        else:
            st.caption(f"Do {5 - len(hist)} more check-in(s) to unlock personal insights.")
        moods = [h for h in hist if "mood" in h]
        if moods:
            head("Mood and journal", TIPS["mood"])
            fig = px.line(pd.DataFrame({"Check-in": range(1, len(moods) + 1), "Mood (1-5)": [h["mood"] for h in moods]}), x="Check-in", y="Mood (1-5)",
                          markers=True, color_discrete_sequence=[CYAN])
            fig.update_yaxes(range=[0.5, 5.5]); fig.update_xaxes(dtick=1)
            st.plotly_chart(style_fig(fig, 240), width="stretch")
            for h in [m for m in reversed(moods) if m.get("note")][:5]:
                st.markdown(f"**{h['date']}** {MOODS[h['mood'] - 1]}  {h['note']}")
        wrep = "\n".join(["SCROLLWELL WEEKLY SUMMARY", f"Generated: {datetime.now():%Y-%m-%d}", f"Check-ins: {len(wk)}",
                          f"Average score: {avg(wk):.1f}" if wk else "No check-ins this week", "", *[f"{h['date']}  {h['score']:.1f}  ({h['band']})" for h in wk]])
        st.download_button("⬇ Weekly summary (.txt)", wrep, "scrollwell_week.txt")
    head("Backup and restore", TIPS["backup"])
    bk = json.dumps({"history": S.history, "checks": S.checks, "focus_total": S.focus_total})
    st.download_button("⬇ Download backup (.json)", bk, "scrollwell_backup.json", "application/json")
    up = st.file_uploader("Restore from a backup", type="json")
    if up is not None and st.button("Restore this backup"):
        try:
            d = json.loads(up.getvalue())
            S.history, S.checks, S.focus_total = d.get("history", []), d.get("checks", {}), d.get("focus_total", 0)
            save_store(); st.rerun()
        except ValueError:
            st.error("That file isn't a valid Scrollwell backup.")
    st.caption("Scrollwell is a statistical estimate and not medical advice. If you are struggling, please talk to a doctor, counsellor or someone you trust.")