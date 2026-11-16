"""Warm, restrained styling for the Streamlit chat interface."""
CSS = """
<style>
:root { --forest:#435c46; --ink:#2e3b32; --clay:#98684c; --paper:#f7f4ee; --line:#dce0d5; }
.stApp { background:var(--paper); color:var(--ink); }
.block-container { max-width:1120px; padding:4.5rem 2rem 5rem; }
header[data-testid="stHeader"] { background:rgba(247,244,238,.92); }
[data-testid="stSidebar"] { background:#efede5; border-right:1px solid var(--line); }
[data-testid="stSidebar"] .block-container { padding-top:1.4rem; }
h1,h2,h3 { color:var(--ink); letter-spacing:-.035em; }
h1 { font-size:clamp(2rem,4.5vw,3.4rem) !important; font-weight:650 !important; }
h2 { font-size:1.3rem !important; }
h3 { font-size:1.05rem !important; }
p,li { line-height:1.65; }
.brand { display:flex; align-items:center; gap:12px; margin:0 0 28px; }
.brand-mark { width:42px; height:42px; border-radius:14px; display:grid; place-items:center;
 background:var(--forest); color:#fff; font-size:15px; font-weight:700; letter-spacing:1px; }
.brand-name { font-size:15px; font-weight:650; }
.brand-caption { color:#677164; font-size:12px; margin-top:1px; }
.eyebrow { color:#777d68; letter-spacing:.13em; font-size:.71rem; font-weight:700; margin:8px 0 5px; }
.welcome-copy { color:#526052; font-size:1.08rem; line-height:1.75; max-width:780px; margin:0 0 12px; }
.scope-note { color:#697362; font-size:.88rem; margin:0 0 24px; }
[class*="st-key-topic_card_"] { background:#fffdf9; border:1px solid var(--line) !important;
 border-radius:20px !important; padding:1.15rem !important; box-shadow:0 5px 18px rgba(50,62,43,.025);
 min-height:250px; transition:border-color .18s ease,box-shadow .18s ease; }
[class*="st-key-topic_card_"]:hover { border-color:#a8b49c !important; box-shadow:0 7px 22px rgba(50,62,43,.06); }
[class*="st-key-topic_card_"] h3 { margin:0 0 3px !important; }
[class*="st-key-topic_card_"] p { font-size:.92rem; color:#586552; }
.stButton>button { border:1px solid #cfd7c7; background:#f2f4ec; color:#344d37;
 border-radius:12px; padding:.65rem .85rem; font-weight:500; transition:all .15s; }
.stButton>button:hover { border-color:var(--forest); background:#e7ecdf; color:#263e2a; }
.stButton>button:focus-visible { outline:3px solid #9aaa86; outline-offset:3px; }
[class*="st-key-topic_card_"] .stButton>button { min-height:62px; text-align:left; justify-content:flex-start; }
.stButton>button[kind="primary"] { background:var(--forest); color:white; border-color:var(--forest); }
[data-testid="stChatMessage"] { border:1px solid #e1e4db; border-radius:20px; padding:1.3rem;
 background:#fffdf9; margin-bottom:1rem; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { background:#e9eddf; border-color:#d9dfce; }
[data-testid="stChatMessage"] p { color:var(--ink); }
[data-testid="stChatInput"] { border:1px solid #bac6b1; border-radius:22px; background:#fffdf9;
 box-shadow:0 7px 26px rgba(38,57,36,.06); }
[data-testid="stBottomBlockContainer"] { background:var(--paper); max-width:1060px; padding-top:.6rem; }
[data-testid="stExpander"] { border:1px solid var(--line); border-radius:15px; background:#fcfbf7; }
[data-testid="stExpander"] summary { color:#46553e; }
[data-testid="stAlert"] { border-radius:14px; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color:#5c6955; }
[data-testid="stAppDeployButton"] { display:none; }
@media(max-width:700px) {
 .block-container { padding:4.5rem 1rem 5rem; }
 [class*="st-key-topic_card_"] { min-height:0; }
 .brand { margin-bottom:20px; }
}
@media(prefers-reduced-motion:reduce) { * { transition:none !important; } }
</style>
"""
