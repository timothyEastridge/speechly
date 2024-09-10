import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Set page config
st.set_page_config(page_title="Health Dashboard", layout="wide")

# Custom CSS to style the app
st.markdown("""
<style>
    .stApp {
        max-width: 400px;
        margin: 0 auto;
        font-family: 'Arial', sans-serif;
    }
    .main {
        padding: 2rem;
        background-color: #f0f2f6;
        border-radius: 20px;
    }
    .header {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .date {
        color: #888;
        font-size: 14px;
        margin-bottom: 20px;
    }
    .calendar {
        display: flex;
        justify-content: space-between;
        margin-bottom: 20px;
    }
    .day {
        text-align: center;
        font-size: 12px;
    }
    .current-day {
        background-color: #000;
        color: #fff;
        border-radius: 50%;
        width: 24px;
        height: 24px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
    }
    .mic-button {
        background-color: #4CAF50;
        color: white;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-container {
        display: flex;
        justify-content: space-between;
        margin-bottom: 20px;
    }
    .metric {
        text-align: center;
        flex: 1;
        padding: 10px;
        border-radius: 10px;
        margin: 5px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .metric-title {
        font-size: 12px;
        color: #888;
    }
    .metric-value {
        font-size: 18px;
        font-weight: bold;
    }
    .stat-box {
        background-color: white;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .stat-title {
        font-size: 12px;
        color: #888;
    }
    .stat-value {
        font-size: 18px;
        font-weight: bold;
    }
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: white;
        padding: 10px;
        text-align: center;
        box-shadow: 0 -2px 5px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Main content
st.markdown('<div class="main">', unsafe_allow_html=True)

# Header
st.markdown('<div class="header">Hello, Tim</div>', unsafe_allow_html=True)

# Date
today = datetime.now()
st.markdown(f'<div class="date">{today.strftime("%B %d, %Y")}</div>', unsafe_allow_html=True)

# Calendar week view
week_days = [(today + timedelta(days=i)).strftime('%a') for i in range(7)]
week_dates = [(today + timedelta(days=i)).day for i in range(7)]
st.markdown('<div class="calendar">', unsafe_allow_html=True)
for i, (day, date) in enumerate(zip(week_days, week_dates)):
    day_class = "current-day" if i == 0 else ""
    st.markdown(f"""
    <div class="day">
        <div>{day}</div>
        <div class="{day_class}">{date}</div>
    </div>
    """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Microphone button
st.markdown('<div class="mic-button">🎤</div>', unsafe_allow_html=True)

# Metrics
st.markdown('<div class="metric-container">', unsafe_allow_html=True)
metrics = [
    {"name": "Meals", "value": 45, "color": "#4CAF50"},
    {"name": "Activity", "value": 75, "color": "#FFA500"},
    {"name": "Mood", "value": 100, "color": "#2196F3"}
]
for metric in metrics:
    st.markdown(f"""
    <div class="metric" style="background-color: {metric['color']}22;">
        <div class="metric-title">{metric['name']}</div>
        <div class="metric-value">{metric['value']}%</div>
    </div>
    """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Health stats
col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    <div class="stat-box">
        <div class="stat-title">Protein Intake</div>
        <div class="stat-value">82g</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="stat-box">
        <div class="stat-title">Daily Calories</div>
        <div class="stat-value">859 KCal</div>
    </div>
    """, unsafe_allow_html=True)

# Steps chart
steps_data = {
    'Day': ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'],
    'Steps': [2000, 5000, 7000, 8000, 6000, 7500, 9000]
}
df = pd.DataFrame(steps_data)

fig = go.Figure()
fig.add_trace(go.Bar(
    x=df['Day'],
    y=df['Steps'],
    marker_color='#4CAF50'
))
fig.update_layout(
    title='Average Steps',
    yaxis_title='Steps',
    plot_bgcolor='white',
    showlegend=False,
    height=300,
    margin=dict(l=0, r=0, t=30, b=0)
)
st.plotly_chart(fig, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# Bottom navigation
st.markdown("""
<div class="footer">
    🏠 📷 📊
</div>
""", unsafe_allow_html=True)
