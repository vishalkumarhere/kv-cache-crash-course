import streamlit as st

def apply_custom_css():
    """Apply custom CSS for sleek, modern UI."""
    st.markdown("""
    <style>
        /* Main app background */
        .stApp {
            background: #ffffff;
            color: #333333;
        }
        
        /* Main content area */
        .main {
            background: #ffffff;
            padding: 2rem;
            margin: 1rem;
        }
        
        /* Header styling */
        .main-header {
            color: #333333;
            font-size: 3rem;
            font-weight: 800;
            text-align: center;
            margin-bottom: 1rem;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }
        
        .sub-header {
            color: #666666;
            text-align: center;
            font-size: 1.1rem;
            margin-bottom: 2rem;
            font-weight: 300;
            letter-spacing: 1px;
        }
        
        /* Metric cards */
        .stMetric {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid #e9ecef;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Buttons */
        .stButton > button {
            background: #007bff;
            color: white !important;
            border: none;
            padding: 0.75rem 2rem;
            font-weight: 600;
            border-radius: 6px;
            transition: all 0.3s ease;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            width: 100%;
        }
        
        .stButton > button:hover {
            background: #0056b3;
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }
        
        /* Sliders */
        .stSlider > div > div > div {
            background: #007bff;
        }
        
        .stSlider > div > div > div > div {
            background-color: white;
            border: 2px solid #007bff;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 0.5rem;
            gap: 0.5rem;
            border: 1px solid #e9ecef;
        }
        
        .stTabs [data-baseweb="tab"] {
            background: transparent;
            border-radius: 6px;
            color: #666666;
            font-weight: 500;
            padding: 0.5rem 1rem;
            transition: all 0.3s ease;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background: #e9ecef;
            color: #333333;
        }
        
        .stTabs [aria-selected="true"] {
            background: #007bff;
            color: white !important;
        }
        
        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: #f8f9fa;
            border-right: 1px solid #e9ecef;
        }
        
        section[data-testid="stSidebar"] .stMarkdown {
            color: #333333;
        }
        
        /* Expander */
        .streamlit-expanderHeader {
            background: #f8f9fa;
            border-radius: 6px;
            border: 1px solid #e9ecef;
            color: #333333;
        }
        
        .streamlit-expanderHeader:hover {
            background: #e9ecef;
        }
        
        /* Text input */
        .stTextInput > div > div > input {
            background: white;
            border: 1px solid #ced4da;
            border-radius: 6px;
            color: #333333;
            padding: 0.5rem;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: #007bff;
            box-shadow: 0 0 0 2px rgba(0, 123, 255, 0.25);
        }
        
        /* Text area */
        .stTextArea > div > div > textarea {
            background: white;
            border: 1px solid #ced4da;
            border-radius: 6px;
            color: #333333;
        }
        
        /* Select box */
        .stSelectbox > div > div {
            background: white;
            border: 1px solid #ced4da;
            border-radius: 6px;
        }
        
        /* Multiselect */
        .stMultiSelect > div {
            background: white;
            border: 1px solid #ced4da;
            border-radius: 6px;
        }
        
        /* Progress bar */
        .stProgress > div > div > div {
            background: #007bff;
        }
        
        /* Info/Warning/Error/Success boxes */
        .stAlert {
            background: white;
            border-radius: 6px;
            border: 1px solid #e9ecef;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Dataframe */
        .dataframe {
            background: white;
            border-radius: 6px;
            border: 1px solid #e9ecef;
        }
        
        /* Code blocks */
        code {
            background: #f8f9fa;
            padding: 0.2rem 0.4rem;
            border-radius: 4px;
            color: #e83e8c;
            border: 1px solid #e9ecef;
        }
        
        /* Animation for pulse effect */
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        
        .pulse-animation {
            animation: pulse 2s infinite;
        }
        
        /* Text effects */
        .gradient-text {
            color: #007bff;
            font-weight: bold;
        }
        
        /* Clean containers */
        .glass-container {
            background: white;
            border-radius: 8px;
            border: 1px solid #e9ecef;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 1.5rem;
            margin: 1rem 0;
        }
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: #f8f9fa;
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #ced4da;
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #adb5bd;
        }
        
        /* Hover effects for interactive elements */
        .hover-lift {
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        
        .hover-lift:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        
        /* Loading spinner custom color */
        .stSpinner > div {
            border-color: #007bff !important;
        }
    </style>
    """, unsafe_allow_html=True)

def show_header():
    """Display the main header."""
    st.markdown("""
    <h1 class="main-header">KV-Cache Playground</h1>
    <p class="sub-header">Interactive Visualization of Key-Value Caching in Transformers</p>
    """, unsafe_allow_html=True)
    
    # Add branding footer
    st.markdown("""
    <div style="position: fixed; bottom: 10px; right: 10px; background: white; padding: 8px 12px; 
                border-radius: 6px; border: 1px solid #e9ecef; box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
                font-size: 0.8rem; color: #666666; z-index: 1000;">
        Built by <a href="https://aianytime.net" target="_blank" style="color: #007bff; text-decoration: none;">AI Anytime</a> | 
        Contact: <a href="mailto:sonu@aianytime.net" style="color: #007bff; text-decoration: none;">sonu@aianytime.net</a>
    </div>
    """, unsafe_allow_html=True)

def create_metric_card(title, value, delta=None, delta_color="normal"):
    """Create a custom metric card."""
    delta_html = ""
    if delta is not None:
        color = "#28a745" if delta_color == "normal" else "#dc3545"
        delta_html = f'<div style="color: {color}; font-size: 0.9rem;">Δ {delta}</div>'
    
    return f"""
    <div class="glass-container hover-lift">
        <div style="font-size: 0.9rem; color: #666666; margin-bottom: 0.5rem;">{title}</div>
        <div style="font-size: 1.8rem; font-weight: bold; color: #333333;">{value}</div>
        {delta_html}
    </div>
    """

def create_info_box(title, content, color="#007bff"):
    """Create an information box with custom styling."""
    return f"""
    <div class="glass-container" style="border-left: 4px solid {color};">
        <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
            <span style="font-weight: bold; font-size: 1.1rem; color: #333333;">{title}</span>
        </div>
        <div style="color: #666666; line-height: 1.6;">
            {content}
        </div>
    </div>
    """

def create_comparison_card(title1, value1, title2, value2, speedup=None):
    """Create a comparison card showing two metrics side by side."""
    speedup_html = ""
    if speedup:
        speedup_html = f"""
        <div style="text-align: center; margin-top: 1rem;">
            <span class="gradient-text" style="font-size: 1.2rem;">
                {speedup}x Speedup
            </span>
        </div>
        """
    
    return f"""
    <div class="glass-container">
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
            <div style="text-align: center;">
                <div style="color: #dc3545; font-size: 0.9rem; margin-bottom: 0.5rem;">{title1}</div>
                <div style="font-size: 1.5rem; font-weight: bold; color: #333333;">{value1}</div>
            </div>
            <div style="text-align: center;">
                <div style="color: #28a745; font-size: 0.9rem; margin-bottom: 0.5rem;">{title2}</div>
                <div style="font-size: 1.5rem; font-weight: bold; color: #333333;">{value2}</div>
            </div>
        </div>
        {speedup_html}
    </div>
    """

def show_loading_animation(text="Loading..."):
    """Show a custom loading animation."""
    return st.markdown(f"""
    <div style="text-align: center; padding: 2rem;">
        <div style="color: #666666; margin-top: 1rem;">{text}</div>
    </div>
    """, unsafe_allow_html=True)
