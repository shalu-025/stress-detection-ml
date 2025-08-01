import streamlit as st
import os
import sys
from datetime import datetime
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add the current directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inputs import collect_inputs, load_defaults
from image_module import ImageStressDetector
from audio_module import AudioStressDetector
from model import MultimodalStressPredictor
from ui import create_sidebar, create_main_interface
from utils import setup_logging, log_session, export_history
from train import TrainingPipeline

# Page configuration
st.set_page_config(
    page_title="Stress Detection System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main application entry point"""
    
    # Initialize session state
    if 'session_history' not in st.session_state:
        st.session_state.session_history = []
    if 'current_session' not in st.session_state:
        st.session_state.current_session = {}
    
    # Setup logging
    logger = setup_logging()
    
    # Load default values
    defaults = load_defaults()
    
    # Initialize models
    try:
        image_detector = ImageStressDetector()
        audio_detector = AudioStressDetector()
        stress_predictor = MultimodalStressPredictor()
    except Exception as e:
        st.error(f"Error initializing models: {str(e)}")
        return
    
    # Main title
    st.title("🧠 Multimodal Stress Detection System")
    st.markdown("---")
    
    # Create sidebar for input collection
    user_inputs = create_sidebar(defaults)
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📊 Stress Analysis")
        
        # Collect all inputs
        if st.button("🔍 Analyze Stress Level", type="primary"):
            with st.spinner("Analyzing stress level..."):
                try:
                    # Collect physiological inputs
                    feature_vector = collect_inputs(user_inputs)
                    
                    # Initialize results
                    results = {
                        'timestamp': datetime.now(),
                        'physiological': feature_vector,
                        'image_score': None,
                        'audio_score': None,
                        'final_stress_level': None,
                        'confidence': None
                    }
                    
                    # Image analysis
                    if st.session_state.get('captured_image') is not None:
                        st.info("🔍 Analyzing facial expressions...")
                        image_score = image_detector.predict_stress(st.session_state.captured_image)
                        results['image_score'] = image_score
                        st.success(f"Image analysis complete: {image_score:.2f}")
                    
                    # Audio analysis
                    if st.session_state.get('audio_data') is not None:
                        st.info("🎤 Analyzing voice tone...")
                        audio_score = audio_detector.predict_stress(st.session_state.audio_data)
                        results['audio_score'] = audio_score
                        st.success(f"Audio analysis complete: {audio_score:.2f}")
                    
                    # Final prediction
                    final_stress, confidence = stress_predictor.predict(
                        physiological=feature_vector,
                        image_score=results['image_score'],
                        audio_score=results['audio_score']
                    )
                    
                    results['final_stress_level'] = final_stress
                    results['confidence'] = confidence
                    
                    # Store results
                    st.session_state.current_session = results
                    st.session_state.session_history.append(results)
                    
                    # Log session
                    log_session(logger, results)
                    
                    st.success("✅ Analysis complete!")
                    
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    logger.error(f"Analysis error: {str(e)}")
            
            # Display results
            if st.session_state.current_session:
                display_results(st.session_state.current_session)
    
    with col2:
        st.header("📈 Session History")
        display_history()
    
    # Bottom section for additional features
    st.markdown("---")
    col3, col4, col5 = st.columns(3)
    
    with col3:
        st.header("📊 Analytics")
        if st.session_state.session_history:
            display_analytics()
    
    with col4:
        st.header("💡 Suggestions")
        if st.session_state.current_session:
            display_suggestions(st.session_state.current_session['final_stress_level'])
    
    with col5:
        st.header("📤 Export")
        if st.session_state.session_history:
            # CSV Export
            try:
                csv_data, csv_filename = export_history(st.session_state.session_history, "csv")
                if csv_data and csv_filename:
                    st.download_button(
                        label="📊 Download CSV",
                        data=csv_data,
                        file_name=csv_filename,
                        mime="text/csv",
                        help="Download session history as CSV file"
                    )
                else:
                    st.error("Failed to generate CSV export")
            except Exception as e:
                st.error(f"Error generating CSV: {str(e)}")
            
            # PDF Export
            try:
                pdf_data, pdf_filename = export_history(st.session_state.session_history, "pdf")
                if pdf_data and pdf_filename:
                    st.download_button(
                        label="📄 Download PDF",
                        data=pdf_data,
                        file_name=pdf_filename,
                        mime="application/pdf",
                        help="Download session history as PDF report"
                    )
                else:
                    st.error("Failed to generate PDF export")
            except Exception as e:
                st.error(f"Error generating PDF: {str(e)}")
        else:
            st.info("No session history available for export")

def display_results(results):
    """Display analysis results"""
    st.subheader("📋 Analysis Results")
    
    # Stress level with color coding
    stress_level = results['final_stress_level']
    confidence = results['confidence']
    
    if stress_level == 'low':
        color = "green"
        emoji = "😌"
    elif stress_level == 'medium':
        color = "orange"
        emoji = "😐"
    else:
        color = "red"
        emoji = "😰"
    
    st.markdown(f"""
    <div style='text-align: center; padding: 20px; border: 2px solid {color}; border-radius: 10px;'>
        <h2>{emoji} Stress Level: {stress_level.upper()}</h2>
        <p>Confidence: {confidence:.1%}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Detailed scores
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Physiological", "Available", "✓")
    
    with col2:
        if results['image_score'] is not None:
            st.metric("Image Analysis", f"{results['image_score']:.2f}", "✓")
        else:
            st.metric("Image Analysis", "Not used", "✗")
    
    with col3:
        if results['audio_score'] is not None:
            st.metric("Audio Analysis", f"{results['audio_score']:.2f}", "✓")
        else:
            st.metric("Audio Analysis", "Not used", "✗")

def display_history():
    """Display session history"""
    if not st.session_state.session_history:
        st.info("No sessions recorded yet.")
        return
    
    # Create a simple history display
    for i, session in enumerate(st.session_state.session_history[-5:]):  # Show last 5
        timestamp = session['timestamp'].strftime("%H:%M")
        stress_level = session['final_stress_level']
        
        if stress_level == 'low':
            color = "🟢"
        elif stress_level == 'medium':
            color = "🟡"
        else:
            color = "🔴"
        
        st.write(f"{color} {timestamp} - {stress_level.upper()}")

def display_analytics():
    """Display analytics and charts"""
    if not st.session_state.session_history:
        return
    
    # Create stress level distribution
    stress_levels = [s['final_stress_level'] for s in st.session_state.session_history]
    stress_counts = pd.Series(stress_levels).value_counts()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    stress_counts.plot(kind='bar', ax=ax, color=['green', 'orange', 'red'])
    ax.set_title('Stress Level Distribution')
    ax.set_xlabel('Stress Level')
    ax.set_ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    st.pyplot(fig)

def display_suggestions(stress_level):
    """Display personalized suggestions"""
    suggestions = {
        'low': [
            "Great! Keep up the good work!",
            "Consider some light exercise",
            "Stay hydrated and well-rested"
        ],
        'medium': [
            "Take a few deep breaths",
            "Try a 5-minute walk",
            "Listen to calming music"
        ],
        'high': [
            "Practice deep breathing exercises",
            "Take a short break from work",
            "Consider talking to someone",
            "Try progressive muscle relaxation"
        ]
    }
    
    st.subheader("💡 Personalized Suggestions")
    for suggestion in suggestions.get(stress_level, []):
        st.write(f"• {suggestion}")

if __name__ == "__main__":
    main() 