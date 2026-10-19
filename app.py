import html
import os
import re
import urllib.parse

import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="ChurnIQ | Customer Churn Predictor", page_icon="📡",
                   layout="wide", initial_sidebar_state="collapsed")

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Telco-Customer-Churn.csv")

GREEN, RED = "#06C167", "#FF5A4D"

# ---------- background: faint telecom / customer icons tiled on black ----------
ICONS = {
    "signal": '<path d="M4 20v-4M10 20v-8M16 20V8M22 20V4"/>',
    "wifi": '<path d="M2 9a15 15 0 0 1 20 0M5 12.5a10 10 0 0 1 14 0M8.5 16a5 5 0 0 1 7 0"/><circle cx="12" cy="19.5" r="1"/>',
    "phone": '<rect x="7" y="2" width="10" height="20" rx="2"/><path d="M11 18h2"/>',
    "user": '<circle cx="12" cy="8" r="4"/><path d="M4 21c0-4.4 3.6-8 8-8s8 3.6 8 8"/>',
    "chat": '<path d="M4 5h16v11H9l-5 4z"/>',
    "card": '<rect x="2" y="5" width="20" height="14" rx="2"/><path d="M2 10h20"/>',
    "trend": '<path d="M3 17l6-6 4 4 8-8M15 7h6v6"/>',
    "shield": '<path d="M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z"/>',
}
PLACEMENT = [  # (icon, x, y, rotation, scale)
    ("signal", 20, 24, -10, 1.5), ("wifi", 150, 30, 8, 1.6), ("user", 270, 20, 0, 1.5),
    ("phone", 60, 130, 12, 1.5), ("chat", 190, 120, -8, 1.6), ("trend", 300, 140, 6, 1.5),
    ("card", 20, 250, 0, 1.5), ("shield", 150, 240, -6, 1.5), ("signal", 275, 255, 10, 1.4),
]


def background_css():
    parts = []
    for name, x, y, rot, s in PLACEMENT:
        parts.append(f'<g transform="translate({x} {y}) rotate({rot} 12 12) scale({s})">{ICONS[name]}</g>')
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="340" height="320" viewBox="0 0 340 320" '
           'fill="none" stroke="#ffffff" stroke-opacity="0.07" stroke-width="1.2" '
           'stroke-linecap="round" stroke-linejoin="round">' + "".join(parts) + "</svg>")
    return "url(\"data:image/svg+xml;utf8," + urllib.parse.quote(svg) + "\")"


st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp {{ font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif; }}
.stApp {{ background-color:#000; background-image:{background_css()}; background-size:340px 320px; }}
#MainMenu, footer {{ visibility:hidden; }}
header[data-testid="stHeader"] {{ background:transparent; }}
.block-container {{ padding-top:2rem; padding-bottom:3rem; max-width:1180px; }}

.brand {{ display:flex; align-items:center; gap:10px; margin-bottom:6px; }}
.brand .logo {{ width:34px; height:34px; border-radius:9px; background:#fff; color:#000; display:flex;
  align-items:center; justify-content:center; font-weight:800; font-size:18px; }}
.brand .name {{ font-weight:700; font-size:20px; letter-spacing:-0.3px; }}
.hero h1 {{ font-size:46px; line-height:1.05; font-weight:800; letter-spacing:-1.5px; margin:14px 0 8px; }}
.hero p {{ color:#A6A6A6; font-size:17px; margin:0 0 22px; max-width:640px; }}
.pills {{ display:flex; gap:8px; flex-wrap:wrap; margin-bottom:26px; }}
.pill {{ background:#141414; border:1px solid #262626; color:#d6d6d6; border-radius:999px; padding:6px 14px;
  font-size:13px; font-weight:500; }}

div[data-testid="stVerticalBlockBorderWrapper"] {{ background:rgba(16,16,16,.92); border:1px solid #262626 !important;
  border-radius:16px; backdrop-filter:blur(2px); }}
.card-title {{ font-size:13px; font-weight:700; letter-spacing:1.2px; text-transform:uppercase; color:#8c8c8c;
  margin:2px 0 6px; }}
label p, .stSlider label p {{ color:#cfcfcf !important; font-size:13.5px !important; font-weight:500 !important; }}
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div, .stNumberInput input {{
  background:#1c1c1c !important; border:1px solid #2e2e2e !important; border-radius:10px !important; }}
div[data-baseweb="select"] > div:hover {{ border-color:#5a5a5a !important; }}
div[data-testid="stSlider"] [role="slider"] {{ background:#fff !important; box-shadow:0 0 0 4px rgba(255,255,255,.15); }}

.st-key-result {{ position:sticky; top:1rem; }}
.res-label {{ color:#8c8c8c; font-size:13px; font-weight:700; letter-spacing:1.2px; text-transform:uppercase; }}
.res-big {{ font-size:76px; font-weight:800; letter-spacing:-3px; line-height:1; margin:6px 0 14px; }}
.bar {{ position:relative; height:10px; border-radius:999px; background:#262626; overflow:visible; margin:6px 0 4px; }}
.bar .fill {{ height:100%; border-radius:999px; }}
.bar .mark {{ position:absolute; top:-5px; width:2px; height:20px; background:#fff; }}
.bar-legend {{ display:flex; justify-content:space-between; color:#7a7a7a; font-size:11.5px; margin-bottom:14px; }}
.chip {{ display:inline-block; padding:7px 14px; border-radius:999px; font-weight:700; font-size:14px; margin-bottom:10px; }}
.advice {{ color:#bdbdbd; font-size:14px; line-height:1.5; margin-bottom:18px; }}
.sep {{ height:1px; background:#262626; margin:6px 0 14px; }}
.frow {{ display:flex; align-items:center; gap:10px; margin:9px 0; font-size:13px; }}
.frow .fl {{ width:56%; color:#d8d8d8; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.frow .ft {{ position:relative; flex:1; height:8px; background:#1d1d1d; border-radius:4px; }}
.frow .ft::after {{ content:""; position:absolute; left:50%; top:-3px; width:1px; height:14px; background:#444; }}
.frow .fb {{ position:absolute; top:0; height:8px; border-radius:4px; }}
.note {{ color:#6f6f6f; font-size:12px; margin-top:10px; line-height:1.45; }}
.foot {{ color:#5c5c5c; font-size:12.5px; text-align:center; margin-top:34px; }}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def train_model():
    """Same pipeline as the notebook: clean -> encode -> split -> scale -> balanced Logistic Regression."""
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df = df.drop("customerID", axis=1).drop_duplicates()
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    X_raw = df.drop("Churn", axis=1)
    y = df["Churn"]
    X = pd.get_dummies(X_raw, drop_first=True).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    return model, scaler, X_raw, list(X.columns)


def predict(model, scaler, X_raw, columns, customer: dict):
    row = pd.DataFrame([customer])
    combined = pd.concat([X_raw, row], ignore_index=True)  # encode together so dummy columns line up
    encoded = pd.get_dummies(combined, drop_first=True).astype(int)
    encoded = encoded.reindex(columns=columns, fill_value=0).iloc[[-1]]
    scaled = scaler.transform(encoded)
    prob = model.predict_proba(scaled)[0, 1]
    contrib = pd.Series(model.coef_[0] * scaled[0], index=columns)
    return prob, contrib, encoded.iloc[0]


def spaced(t: str) -> str:
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", t)


def pretty(name: str, value: int) -> str:
    """Readable label. A 0 on a yes/no style column means the customer does NOT have that option."""
    if "_" not in name:
        return {"tenure": "Tenure", "MonthlyCharges": "Monthly charges", "TotalCharges": "Total charges",
                "SeniorCitizen": "Senior citizen"}.get(name, spaced(name))
    a, b = name.split("_", 1)
    a = spaced(a)
    if b == "Yes":
        return f"{a}: {'Yes' if value else 'No'}"
    return f"{a}: {b}" if value else f"{a}: not {b}"


def factor_rows(contrib: pd.Series, encoded: pd.Series, n: int = 6) -> str:
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index)[:n]
    mx = max(top.abs().max(), 1e-9)
    rows = []
    for name, v in top.items():
        w = abs(v) / mx * 50  # % of track, from the centre line
        color = RED if v > 0 else GREEN
        left = 50 if v > 0 else 50 - w
        rows.append(f'<div class="frow"><div class="fl">{html.escape(pretty(name, int(encoded[name])))}</div>'
                    f'<div class="ft"><div class="fb" style="left:{left:.1f}%;width:{w:.1f}%;background:{color}"></div></div></div>')
    return "".join(rows)


model, scaler, X_raw, columns = train_model()

# ---------------- header ----------------
st.markdown("""
<div class="brand"><div class="logo">C</div><div class="name">ChurnIQ</div></div>
<div class="hero">
<h1>Know who's leaving<br>before they do.</h1>
<p>Enter a telecom customer's details and get an instant churn risk score, with the factors behind it.</p>
</div>
<div class="pills"><span class="pill">Logistic Regression</span><span class="pill">Churn recall 78%</span><span class="pill">7,000+ customers</span></div>
""", unsafe_allow_html=True)

left, right = st.columns([3, 2], gap="large")

with left:
    with st.container(border=True):
        st.markdown('<div class="card-title">Customer</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        gender = c1.selectbox("Gender", ["Female", "Male"])
        senior = c2.selectbox("Senior citizen", ["No", "Yes"])
        partner = c1.selectbox("Has partner", ["No", "Yes"])
        dependents = c2.selectbox("Has dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)

    with st.container(border=True):
        st.markdown('<div class="card-title">Services</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        phone = c1.selectbox("Phone service", ["Yes", "No"])
        multi = c2.selectbox("Multiple lines",
                             ["No", "Yes", "No phone service"] if phone == "Yes" else ["No phone service"])
        internet = c1.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
        addon = ["No internet service"] if internet == "No" else ["No", "Yes"]
        online_sec = c2.selectbox("Online security", addon)
        online_bkp = c1.selectbox("Online backup", addon)
        device_prot = c2.selectbox("Device protection", addon)
        tech_support = c1.selectbox("Tech support", addon)
        stream_tv = c2.selectbox("Streaming TV", addon)
        stream_mov = c1.selectbox("Streaming movies", addon)

    with st.container(border=True):
        st.markdown('<div class="card-title">Billing</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        contract = c1.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = c2.selectbox("Paperless billing", ["Yes", "No"])
        payment = c1.selectbox("Payment method", ["Electronic check", "Mailed check",
                                                  "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly = c2.number_input("Monthly charges", 18.0, 120.0, 70.0, 0.5)
        total = tenure * monthly

customer = {
    "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0, "Partner": partner, "Dependents": dependents,
    "tenure": tenure, "PhoneService": phone, "MultipleLines": multi, "InternetService": internet,
    "OnlineSecurity": online_sec, "OnlineBackup": online_bkp, "DeviceProtection": device_prot,
    "TechSupport": tech_support, "StreamingTV": stream_tv, "StreamingMovies": stream_mov, "Contract": contract,
    "PaperlessBilling": paperless, "PaymentMethod": payment, "MonthlyCharges": monthly, "TotalCharges": total,
}
prob, contrib, encoded = predict(model, scaler, X_raw, columns, customer)

with right:
    with st.container(border=True, key="result"):
        threshold = st.session_state.get("thr", 0.50)
        at_risk = prob >= threshold
        color = RED if at_risk else GREEN
        chip_bg = "rgba(255,90,77,.15)" if at_risk else "rgba(6,193,103,.15)"
        verdict = "At risk of churning" if at_risk else "Likely to stay"
        advice = ("Offer a retention deal, for example a move to an annual contract, or a proactive support call."
                  if at_risk else "No action needed. Keep the service quality steady and check in at renewal.")
        st.markdown(f"""
<div class="res-label">Churn probability</div>
<div class="res-big" style="color:{color}">{prob:.1%}</div>
<div class="bar"><div class="fill" style="width:{prob*100:.1f}%;background:{color}"></div><div class="mark" style="left:{threshold*100:.0f}%"></div></div>
<div class="bar-legend"><span>0%</span><span>threshold {threshold:.0%}</span><span>100%</span></div>
<span class="chip" style="color:{color};background:{chip_bg}">{verdict}</span>
<div class="advice">{advice}</div>
<div class="sep"></div>
<div class="res-label" style="margin-bottom:6px">What's driving this</div>
{factor_rows(contrib, encoded)}
<div class="note"><span style="color:{RED}">Red</span> pushes toward churn, <span style="color:{GREEN}">green</span> toward staying.
Total charges are estimated as tenure × monthly charges.</div>
""", unsafe_allow_html=True)
        st.slider("Alert threshold", 0.10, 0.90, 0.50, 0.05, key="thr",
                  help="Flag a customer as at risk above this probability. Lower = catch more churners, "
                       "but more false alarms.")

st.markdown('<div class="foot">Built with Python, scikit-learn and Streamlit · Telco Customer Churn dataset</div>',
            unsafe_allow_html=True)
