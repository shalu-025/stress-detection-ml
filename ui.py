"""
UI components for the stress detection system
"""

import streamlit as st
from typing import Dict, Any
import numpy as np

def create_sidebar(defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Create the sidebar for input collection"""
    st.sidebar.header("📊 Input Parameters")
    
    # Physiological inputs
    st.sidebar.subheader("💓 Physiological Data")
    
    heart_rate = st.sidebar.slider(
        "Heart Rate (bpm)",
        min_value=60,
        max_value=200,
        value=75,
        help="Current heart rate in beats per minute"
    )
    
    spo2 = st.sidebar.slider(
        "SpO2 (%)",
        min_value=85,
        max_value=100,
        value=98,
        help="Blood oxygen saturation level"
    )
    
    temperature = st.sidebar.slider(
        "Temperature (°C)",
        min_value=35.0,
        max_value=40.0,
        value=37.0,
        step=0.1,
        help="Body temperature"
    )
    
    age = st.sidebar.slider(
        "Age",
        min_value=18,
        max_value=100,
        value=30,
        help="Age in years"
    )
    
    gender = st.sidebar.selectbox(
        "Gender",
        options=["male", "female"],
        help="Biological gender"
    )
    
    # Derived parameters
    st.sidebar.subheader("📈 Derived Parameters")
    
    heart_rate_variability = st.sidebar.slider(
        "Heart Rate Variability",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="Heart rate variability score"
    )
    
    oxygen_saturation_risk = st.sidebar.slider(
        "Oxygen Saturation Risk",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.1,
        help="Risk level based on oxygen saturation"
    )
    
    temperature_deviation = st.sidebar.slider(
        "Temperature Deviation",
        min_value=-2.0,
        max_value=2.0,
        value=0.0,
        step=0.1,
        help="Deviation from normal temperature"
    )
    
    # Image and Audio inputs
    st.sidebar.subheader("📷 Image & Audio")
    
    # Placeholder for image upload
    uploaded_image = st.sidebar.file_uploader(
        "Upload Facial Image",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a facial image for stress analysis"
    )
    
    if uploaded_image is not None:
        st.sidebar.success("Image uploaded successfully!")
        # Store image in session state for processing
        st.session_state.captured_image = uploaded_image
    
    # Placeholder for audio upload
    uploaded_audio = st.sidebar.file_uploader(
        "Upload Audio Sample",
        type=['wav', 'mp3'],
        help="Upload an audio sample for voice stress analysis"
    )
    
    if uploaded_audio is not None:
        st.sidebar.success("Audio uploaded successfully!")
        # Store audio in session state for processing
        st.session_state.audio_data = uploaded_audio
    
    # Return collected inputs
    return {
        'heart_rate': heart_rate,
        'spo2': spo2,
        'temperature': temperature,
        'age': age,
        'gender': gender,
        'heart_rate_variability': heart_rate_variability,
        'oxygen_saturation_risk': oxygen_saturation_risk,
        'temperature_deviation': temperature_deviation
    }

def create_main_interface():
    """Create the main interface display"""
    st.header("📊 Stress Analysis Results")
    
    # Placeholder for results display
    if 'analysis_results' in st.session_state:
        results = st.session_state.analysis_results
        
        # Display stress level
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Stress Level",
                value=results.get('final_stress_level', 'Unknown'),
                delta=None
            )
        
        with col2:
            st.metric(
                label="Confidence",
                value=f"{results.get('confidence', 0.0):.2f}",
                delta=None
            )
        
        with col3:
            st.metric(
                label="Heart Rate",
                value=f"{results.get('physiological', {}).get('heart_rate', 0)} bpm",
                delta=None
            )
        
        # Display detailed results
        st.subheader("📋 Detailed Analysis")
        
        # Physiological metrics
        if 'physiological' in results:
            phys = results['physiological']
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Physiological Metrics:**")
                st.write(f"• Heart Rate: {phys.get('heart_rate', 0)} bpm")
                st.write(f"• SpO2: {phys.get('spo2', 0)}%")
                st.write(f"• Temperature: {phys.get('temperature', 0)}°C")
            
            with col2:
                st.write("**Additional Metrics:**")
                st.write(f"• Age: {phys.get('age', 0)} years")
                st.write(f"• HRV: {phys.get('heart_rate_variability', 0):.2f}")
                st.write(f"• O2 Risk: {phys.get('oxygen_saturation_risk', 0):.2f}")
        
        # Image and Audio scores
        if 'image_score' in results and results['image_score'] is not None:
            st.write(f"**Image Analysis Score:** {results['image_score']:.3f}")
        
        if 'audio_score' in results and results['audio_score'] is not None:
            st.write(f"**Audio Analysis Score:** {results['audio_score']:.3f}")
        
        # Display suggestions
        stress_level = results.get('final_stress_level', 'medium')
        display_suggestions(stress_level)
    
    else:
        st.info("Click 'Analyze Stress Level' to see results here.")

def display_suggestions(stress_level: str):
    """Display stress management suggestions"""
    st.subheader("💡 Recommendations")
    
    if stress_level == 'low':
        st.success("**Low Stress Level** - You're doing great!")
        st.write("""
        **Suggestions:**
        • Maintain your current healthy lifestyle
        • Continue regular exercise and good sleep habits
        • Practice mindfulness or meditation for maintenance
        • Stay hydrated and eat balanced meals
        """)
    
    elif stress_level == 'medium':
        st.warning("**Medium Stress Level** - Some attention needed")
        st.write("""
        **Suggestions:**
        • Take short breaks throughout the day
        • Practice deep breathing exercises
        • Consider light physical activity
        • Ensure adequate sleep (7-9 hours)
        • Limit caffeine and alcohol intake
        """)
    
    else:  # high stress
        st.error("**High Stress Level** - Immediate attention recommended")
        st.write("""
        **Suggestions:**
        • Take immediate breaks from stressful activities
        • Practice relaxation techniques (deep breathing, meditation)
        • Consider talking to a mental health professional
        • Prioritize sleep and rest
        • Engage in physical activity to release tension
        • Consider stress management techniques
        """)
    
    st.info("💡 **Note:** These are general suggestions. For personalized advice, consult with healthcare professionals.")

def display_stress_gauge(stress_score: float):
    """Display a visual stress gauge"""
    st.subheader("📊 Stress Level Gauge")
    
    # Create a progress bar
    progress = st.progress(0)
    
    # Update progress based on stress score
    if stress_score <= 0.3:
        progress.progress(stress_score)
        st.success("Low Stress")
    elif stress_score <= 0.6:
        progress.progress(stress_score)
        st.warning("Medium Stress")
    else:
        progress.progress(stress_score)
        st.error("High Stress")
    
    st.write(f"Stress Score: {stress_score:.3f}") 