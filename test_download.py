"""
Test script for download functionality
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from utils import export_history

# Mock session history for testing
mock_history = [
    {
        'timestamp': datetime.now(),
        'final_stress_level': 'low',
        'confidence': 0.85,
        'physiological': {
            'heart_rate': 75,
            'spo2': 98,
            'temperature': 37.0,
            'age': 30
        },
        'image_score': 0.2,
        'audio_score': 0.3
    },
    {
        'timestamp': datetime.now(),
        'final_stress_level': 'medium',
        'confidence': 0.72,
        'physiological': {
            'heart_rate': 85,
            'spo2': 96,
            'temperature': 37.2,
            'age': 30
        },
        'image_score': 0.5,
        'audio_score': 0.6
    }
]

def test_download_functionality():
    """Test the download functionality"""
    st.title("Download Functionality Test")
    
    st.write("Testing CSV and PDF download functionality...")
    
    # Test CSV export
    csv_data, csv_filename = export_history(mock_history, "csv")
    if csv_data and csv_filename:
        st.success("✅ CSV export successful")
        st.download_button(
            label="📊 Download Test CSV",
            data=csv_data,
            file_name=csv_filename,
            mime="text/csv"
        )
    else:
        st.error("❌ CSV export failed")
    
    # Test PDF export
    pdf_data, pdf_filename = export_history(mock_history, "pdf")
    if pdf_data and pdf_filename:
        st.success("✅ PDF export successful")
        st.download_button(
            label="📄 Download Test PDF",
            data=pdf_data,
            file_name=pdf_filename,
            mime="application/pdf"
        )
    else:
        st.error("❌ PDF export failed")
    
    # Display the mock data
    st.subheader("Mock Session History")
    df = pd.DataFrame(mock_history)
    st.dataframe(df)

if __name__ == "__main__":
    test_download_functionality() 