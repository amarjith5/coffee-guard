import dataset
import detection
import disease_tracking
import streamlit as st
from streamlit_extras.stylable_container import stylable_container
from streamlit_option_menu import option_menu

# Setting page layout
st.set_page_config(
    page_title="Coffee Guard",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Define color scheme - only light mode with orange accent
THEME_COLORS = {
    "primaryColor": "#FF7E00",  # Vibrant orange
    "backgroundColor": "white",
    "secondaryBackgroundColor": "#f8f9fa",
    "textColor": "#333333",
    "accentColor": "#FFA500",  # Secondary orange
    "logo": "🌿",
}

# Apply theme to Streamlit config
for key, value in THEME_COLORS.items():
    st.config.set_option(f"theme.{key}", value)

# Apply theme using custom CSS
st.markdown(
    f"""
    <style>
        :root {{
            --primary-color: {THEME_COLORS["primaryColor"]};
            --background-color: {THEME_COLORS["backgroundColor"]};
            --secondary-background-color: {THEME_COLORS["secondaryBackgroundColor"]};
            --text-color: {THEME_COLORS["textColor"]};
            --accent-color: {THEME_COLORS["accentColor"]};
        }}
        .stApp {{
            background-color: var(--background-color);
            color: var(--text-color);
            font-family: 'Inter', 'Segoe UI', sans-serif;
        }}
        .sidebar .sidebar-content {{
            background-color: var(--secondary-background-color);
        }}
        div[data-testid="stMarkdownContainer"] h1,
        div[data-testid="stMarkdownContainer"] h2,
        div[data-testid="stMarkdownContainer"] h3 {{
            color: var(--primary-color);
            font-weight: 600;
            letter-spacing: -0.02em;
        }}
        .stTabs [data-baseweb="tab-list"] {{
            gap: 2px;
            background-color: var(--secondary-background-color);
            border-radius: 12px;
            padding: 4px;
        }}
        .stTabs [data-baseweb="tab"] {{
            height: 40px;
            white-space: pre-wrap;
            border-radius: 10px;
            padding: 0 16px;
            color: var(--text-color);
            transition: all 0.2s ease;
        }}
        .stTabs [aria-selected="true"] {{
            background-color: var(--primary-color) !important;
            color: white !important;
            font-weight: 600;
            transform: translateY(-2px);
        }}
        div[data-testid="stExpander"] {{
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
            margin-bottom: 1.2rem;
            border: 1px solid rgba(0, 0, 0, 0.03);
        }}
        div[data-testid="stExpander"] details summary p {{
            font-weight: 600;
        }}
        div[data-testid="stVerticalBlock"] div[data-testid="stHorizontalBlock"] {{
            align-items: center;
            gap: 1rem;
        }}
        button[kind="primaryFormSubmit"] {{
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            font-weight: 600;
            letter-spacing: 0.01em;
            text-transform: uppercase;
            font-size: 0.85rem;
            transition: all 0.3s ease;
        }}
        button[kind="primaryFormSubmit"]:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(0,0,0,0.15);
        }}
        div[data-testid="stSidebarUserContent"] {{
            padding-top: 1.5rem;
        }}
        .stSelectbox [data-baseweb="select"] {{
            border-radius: 10px;
        }}
        .stSelectbox [data-baseweb="select"]:focus {{
            border-color: var(--primary-color);
            box-shadow: 0 0 0 2px var(--primary-color);
        }}
        .stSlider [data-baseweb="slider"] {{
            height: 6px;
        }}
        .stSlider [data-baseweb="thumb"] {{
            height: 20px;
            width: 20px;
            background-color: var(--primary-color);
        }}
    </style>
""",
    unsafe_allow_html=True,
)

# Modern header with glassmorphism effect
st.markdown(
    f"""
    <div style="
        background: linear-gradient(135deg, {THEME_COLORS["primaryColor"]}15, {THEME_COLORS["accentColor"]}25);
        border-radius: 16px;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid {THEME_COLORS["primaryColor"]}30;
        padding: 2rem 1rem;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.1);
    ">
        <div style="display: flex; justify-content: center; align-items: center; margin-bottom: 1rem;">
            <img src="{THEME_COLORS["logo"]}" alt="" style="max-height: 85px; margin-right: 15px;">
            <div style="
                font-size: 3rem; 
                font-weight: 800; 
                background: linear-gradient(to right, {THEME_COLORS["primaryColor"]}, {THEME_COLORS["accentColor"]});
                -webkit-background-clip: text;
                background-clip: text;
                color: transparent;
                letter-spacing: -0.03em;
            ">Coffee Guard</div>
        </div>
        <p style="
            font-size: 1.2rem; 
            margin: 0 auto; 
            max-width: 600px; 
            color: {THEME_COLORS["textColor"]};
            opacity: 0.9;
        ">
            Coffee Leaf Disease Detection System
        </p>
    </div>
""",
    unsafe_allow_html=True,
)

# Floating navigation with pill design
with stylable_container(
    key="menu_container",
    css_styles=f"""
        background: {THEME_COLORS["secondaryBackgroundColor"]}CC;
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        padding: 0.6rem;
        border-radius: 18px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -2px rgba(0, 0, 0, 0.02);
        border: 1px solid {THEME_COLORS["primaryColor"]}20;
        position: sticky;
        z-index: 100;
    """,
):
    tab = option_menu(
        None,
        ["Detector", "Disease Library", "Disease Map"],
        icons=["binoculars-fill", "journal-medical", "geo-alt-fill"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": THEME_COLORS["primaryColor"], "font-size": "18px"},
            "nav-link": {
                "font-size": "1rem",
                "text-align": "center",
                "padding": "12px 20px",
                "margin": "0px 5px",
                "border-radius": "12px",
                "--hover-color": f"{THEME_COLORS['primaryColor']}10",
                "font-family": "'Inter', 'sans-serif'",
                "color": THEME_COLORS["textColor"],
                "font-weight": "500",
                "letter-spacing": "0.01em",
                "transition": "all 0.3s ease",
            },
            "nav-link-selected": {
                "color": "white",
                "font-weight": "600",
                "background-image": f"linear-gradient(to right, {THEME_COLORS['primaryColor']}, {THEME_COLORS['accentColor']})",
                "box-shadow": f"0 4px 12px 0 {THEME_COLORS['primaryColor']}60",
                "transform": "translateY(-2px)",
            },
        },
    )

# Content rendering based on selected tab
if tab == "Detector":
    detection.main(THEME_COLORS)
elif tab == "Disease Library":
    dataset.main()
elif tab == "Disease Map":
    disease_tracking.main(THEME_COLORS)
