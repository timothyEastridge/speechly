import os
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import pyaudio
import wave
import openai
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError
import plotly.graph_objects as go
import numpy as np
from tempfile import NamedTemporaryFile
import keyring
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

# Initialize OpenAI API key
os.environ['OPENAI_API_KEY'] = keyring.get_password('eastridge', 'openai')
openai.api_key = os.getenv('OPENAI_API_KEY')

# Setup audio recording parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 8

# Custom CSS for app styling
st.markdown("""
<style>
    .reportview-container .main .block-container {
        max-width: 1200px;
        padding: 2rem;
        border: 2px solid #333333;
        border-radius: 10px;
        margin: 2rem auto;
    }
    .dataframe {
        width: 100% !important;
    }
    .dataframe td:nth-child(2) {
        min-width: 300px !important;
    }
    .stButton > button {
        width: 200px;
        height: 200px;  /* Increased height to make it more square-like */
        background-color: #FFCCCB;  /* Light red background */
        color: black;  /* Changed text color to black for better contrast */
        font-size: 18px;
        display: block;
        margin: 0 auto;
        border-radius: 50%;  /* Makes the button circular */
    }
</style>
""", unsafe_allow_html=True)

# Initialize the GPT model for calorie estimation and meal type classification
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def create_chat_llm():
    return ChatOpenAI(temperature=0.0, model="gpt-4o")

chat_llm = create_chat_llm()

# Define the prompt template for calorie estimation and meal type classification
calories_prompt_template = """
Analyze the following meal description and provide two pieces of information:
1. Estimate the number of calories for this meal. Reply with NUMBER only in 1-3 tokens. If unsure, reply with "0".
2. Classify the meal type (Breakfast, Lunch, Dinner, or Snack).

Reply with ONLY these two pieces of information, separated by a comma. For example: "500, Lunch" or "2500, Snack".
If unsure about the calories, use your best estimate.

Meal description: {meal_description}
"""

calories_prompt = PromptTemplate(template=calories_prompt_template, input_variables=["meal_description"])
calories_chain = LLMChain(llm=chat_llm, prompt=calories_prompt)

# Function to generate calorie estimate and meal type using GPT
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def estimate_calories_and_type(transcription_text):
    data = {"meal_description": transcription_text}
    try:
        response_dict = calories_chain.invoke(data)
        response_text = response_dict.get('text', '').strip()
        calories, meal_type = response_text.split(',')
        return calories.strip(), meal_type.strip()
    except Exception:
        return '0', 'Unknown'

# Audio recording function with temporary file path
def record_audio():
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = []

    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    audio.terminate()

    timestamp = datetime.now()
    temp_file = NamedTemporaryFile(delete=False, suffix=".wav")
    filepath = temp_file.name

    with wave.open(filepath, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    return filepath, timestamp

def transcribe_audio(filepath):
    try:
        with open(filepath, 'rb') as audio_file:
            transcript = openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
            return transcript.text
    except Exception as e:
        st.error(f"Transcription failed: {str(e)}")
        return None

# Email sending function
def send_email(to_address, subject, body):
    try:
        from_address = os.getenv('EMAIL_ADDRESS')
        password = os.getenv('EMAIL_PASSWORD')

        message = MIMEMultipart()
        message['From'] = from_address
        message['To'] = to_address
        message['Subject'] = subject

        message.attach(MIMEText(body, 'plain'))

        with smtplib.SMTP('smtp.gmail.com', 587) as session:
            session.starttls()
            session.login(from_address, password)
            text = message.as_string()
            session.sendmail(from_address, to_address, text)

        st.success("Email sent successfully!")
    except Exception as e:
        st.error(f"Error sending email: {str(e)}")

# Function to load mock data for the last 3 days
def load_mock_data():
    end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=3)
    
    mock_data = []
    meal_types = ['Breakfast', 'Lunch', 'Dinner', 'Snack']
    
    current_date = start_date
    while current_date < end_date:
        for meal in meal_types:
            timestamp = current_date + timedelta(hours=np.random.randint(6, 22))
            calories = np.random.randint(200, 800)
            mock_data.append({
                "Timestamp": timestamp,
                "Transcription": f"{meal.lower()}",
                "Calories": calories,
                "MealType": meal
            })
        current_date += timedelta(days=1)
    
    today = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)
    mock_data.append({
        "Timestamp": today,
        "Transcription": "Bowl of Cheerios and a yogurt",
        "Calories": 250,
        "MealType": "Breakfast"
    })
    
    return pd.DataFrame(mock_data)

# Function to create stacked bar chart
def create_stacked_bar_chart(df):
    df_copy = df.copy()
    
    df_copy['Date'] = df_copy['Timestamp'].dt.date
    df_copy['MealType'] = pd.Categorical(df_copy['MealType'], categories=['Breakfast', 'Lunch', 'Dinner', 'Snack'], ordered=True)
    daily_data = df_copy.groupby(['Date', 'MealType'])['Calories'].sum().unstack(fill_value=0)

    daily_totals = daily_data.sum(axis=1)

    fig = go.Figure()

    colors = {'Breakfast': '#FFA07A', 'Lunch': '#98FB98', 'Dinner': '#87CEFA', 'Snack': '#DDA0DD'}

    for meal_type in daily_data.columns:
        fig.add_trace(go.Bar(
            x=daily_data.index,
            y=daily_data[meal_type],
            name=meal_type,
            marker_color=colors[meal_type]
        ))

    for date, total in daily_totals.items():
        fig.add_annotation(
            x=date,
            y=total,
            text=f"{total:.0f}",
            showarrow=False,
            yshift=10,
            font=dict(size=12, color="black")
        )

    fig.update_layout(
        title='Daily Calorie Intake by Meal Type',
        barmode='stack',
        xaxis_title='Date',
        yaxis_title='Calories',
        legend_title='Meal Type',
        height=600
    )

    return fig

def main():
    if 'df' not in st.session_state:
        st.session_state.df = load_mock_data()
    
    today = datetime.now().date()
    if 'initial_data_cleaned' not in st.session_state:
        st.session_state.df = st.session_state.df[
            (st.session_state.df['Timestamp'].dt.date < today) | 
            ((st.session_state.df['Timestamp'].dt.date == today) & 
             (st.session_state.df['MealType'] == 'Breakfast') &
             (st.session_state.df['Transcription'] == "Bowl of Cheerios and a yogurt"))
        ]
        st.session_state.initial_data_cleaned = True

    # Center the "Record Audio" button
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("Record Audio", key="record_audio_btn"):
            with st.spinner("Recording audio..."):
                audio_file_path, timestamp = record_audio()
            
            if audio_file_path:
                with st.spinner("Transcribing audio..."):
                    transcription = transcribe_audio(audio_file_path)
                
                if transcription:
                    with st.spinner("Estimating calories and meal type..."):
                        calories, meal_type = estimate_calories_and_type(transcription)

                    new_entry = pd.DataFrame({
                        "Timestamp": [timestamp],
                        "Transcription": [transcription],
                        "Calories": [int(calories)],
                        "MealType": [meal_type]
                    })
                    st.session_state.df = pd.concat([st.session_state.df, new_entry], ignore_index=True)

                    st.success(f"Recorded: {transcription}")
                    st.info(f"Estimated Calories: {calories}, Meal Type: {meal_type}")
    
    # Create and display the stacked bar chart
    fig = create_stacked_bar_chart(st.session_state.df)
    st.plotly_chart(fig, use_container_width=True)

    # Display the dataframe
    sorted_df = st.session_state.df.sort_values(by='Timestamp', ascending=False)
    display_df = sorted_df[['Timestamp', 'Transcription', 'Calories', 'MealType']]
    display_df['Timestamp'] = display_df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

    st.dataframe(
        display_df.style.hide(axis="index"),
        width=1200,
        height=400
    )

    # Email Input Box at the Bottom
    email = st.text_input("Enter your email (optional):")
    if email:
        send_email(email, "Your Meal Data", "Thank you for using the app. Here is your meal data.")

if __name__ == "__main__":
    main()
