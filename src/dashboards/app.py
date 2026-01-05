import streamlit as st

st.set_page_config(
    page_title="UX Démo — Système décisionnel",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===== Header produit (sobre & pro) =====
st.title("UX Démo — Système IA décisionnel (PME SaaS)")
st.caption(
    "Produit : décision assistée (explicable) • Pas de décision automatique • "
    "Navigation E1→E5→E4 • Réutilisable V1→V3"
)

st.divider()

# ===== Rappels garde-fous (toujours visibles, sans jargon) =====
with st.expander("Règles du système (garde-fous)", expanded=False):
    st.write(
        "- Le système **oriente l’attention** et **explique** (E5), mais **ne décide pas**.\n"
        "- Aucune action métier n’est déclenchée automatiquement.\n"
        "- E4 sert à **décider & tracer** (responsabilité humaine).\n"
        "- Les écrans sont conçus pour être compris en **≤ 30 secondes**."
    )

st.info(
    "Utilise la **sidebar** pour naviguer : "
    "E1 À surveiller → E2 Santé globale → E3 Clients à risque → "
    "E5 Comprendre la situation → E4 Décider & tracer."
)
