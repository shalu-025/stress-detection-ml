"""
Utility functions for stress detection system
"""

import logging
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
from fpdf import FPDF
import streamlit as st

def setup_logging(level: str = 'INFO') -> logging.Logger:
    """Setup logging configuration"""
    logger = logging.getLogger('stress_detection')
    logger.setLevel(getattr(logging, level.upper()))
    
    # Create console handler if it doesn't exist
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(console_handler)
    
    return logger

def log_session(logger: logging.Logger, session_data: Dict[str, Any]):
    """Log session data"""
    try:
        logger.info(f"Session completed - Stress Level: {session_data.get('final_stress_level', 'Unknown')}")
        logger.info(f"Confidence: {session_data.get('confidence', 0.0):.3f}")
        logger.info(f"Timestamp: {session_data.get('timestamp', datetime.now())}")
        
        # Log physiological data if available
        if 'physiological' in session_data:
            phys = session_data['physiological']
            logger.info(f"Physiological - HR: {phys.get('heart_rate', 0)} bpm, SpO2: {phys.get('spo2', 0)}%")
        
        # Log image and audio scores if available
        if session_data.get('image_score') is not None:
            logger.info(f"Image Score: {session_data['image_score']:.3f}")
        
        if session_data.get('audio_score') is not None:
            logger.info(f"Audio Score: {session_data['audio_score']:.3f}")
            
    except Exception as e:
        logger.error(f"Error logging session: {e}")

def export_history(session_history: List[Dict[str, Any]], format_type: str = "csv"):
    """Export session history to file and return data for download"""
    try:
        if not session_history:
            print("No session history to export")
            return None, None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format_type.lower() == "csv":
            # Convert to DataFrame and export as CSV
            df_data = []
            for session in session_history:
                row = {
                    'timestamp': session.get('timestamp', ''),
                    'stress_level': session.get('final_stress_level', ''),
                    'confidence': session.get('confidence', 0.0),
                    'heart_rate': session.get('physiological', {}).get('heart_rate', 0),
                    'spo2': session.get('physiological', {}).get('spo2', 0),
                    'temperature': session.get('physiological', {}).get('temperature', 0),
                    'age': session.get('physiological', {}).get('age', 0),
                    'image_score': session.get('image_score', 0),
                    'audio_score': session.get('audio_score', 0)
                }
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            filename = f"stress_history_{timestamp}.csv"
            csv_data = df.to_csv(index=False)
            return csv_data, filename
            
        elif format_type.lower() == "pdf":
            # Create PDF report
            pdf = FPDF()
            pdf.add_page()
            
            # Title
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(0, 10, 'Stress Detection Session History', ln=True, align='C')
            pdf.ln(10)
            
            # Summary
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, f'Total Sessions: {len(session_history)}', ln=True)
            
            # Stress level distribution
            stress_counts = {'low': 0, 'medium': 0, 'high': 0}
            for session in session_history:
                level = session.get('final_stress_level', 'medium')
                if level in stress_counts:
                    stress_counts[level] += 1
            
            pdf.ln(5)
            pdf.cell(0, 10, 'Stress Level Distribution:', ln=True)
            for level, count in stress_counts.items():
                pdf.cell(0, 10, f'{level.capitalize()}: {count}', ln=True)
            
            # Recent sessions
            pdf.ln(10)
            pdf.cell(0, 10, 'Recent Sessions:', ln=True)
            pdf.set_font('Arial', '', 10)
            
            for i, session in enumerate(session_history[-5:]):  # Last 5 sessions
                timestamp = session.get('timestamp', '')
                stress_level = session.get('final_stress_level', 'Unknown')
                confidence = session.get('confidence', 0.0)
                
                pdf.cell(0, 8, f'{i+1}. {timestamp} - {stress_level} (Confidence: {confidence:.2f})', ln=True)
            
            filename = f"stress_report_{timestamp}.pdf"
            try:
                pdf_data = pdf.output(dest='S').encode('latin-1')
                return pdf_data, filename
            except Exception as e:
                print(f"Error encoding PDF: {e}")
                # Fallback: try UTF-8 encoding
                try:
                    pdf_data = pdf.output(dest='S').encode('utf-8')
                    return pdf_data, filename
                except Exception as e2:
                    print(f"Error with UTF-8 encoding: {e2}")
                    return None, None
            
        else:
            print(f"Unsupported export format: {format_type}")
            return None, None
            
    except Exception as e:
        print(f"Error exporting history: {e}")
        return None, None

def generate_session_summary(sessions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate summary statistics from session data"""
    if not sessions:
        return {}
    
    summary = {}
    
    # Count stress levels
    stress_counts = {'low': 0, 'medium': 0, 'high': 0}
    confidences = []
    
    for session in sessions:
        stress_level = session.get('final_stress_level', 'medium')
        confidence = session.get('confidence', 0.5)
        
        if stress_level in stress_counts:
            stress_counts[stress_level] += 1
        confidences.append(confidence)
    
    # Calculate statistics
    summary['total_sessions'] = len(sessions)
    summary['stress_distribution'] = stress_counts
    summary['avg_confidence'] = sum(confidences) / len(confidences) if confidences else 0.0
    summary['max_confidence'] = max(confidences) if confidences else 0.0
    summary['min_confidence'] = min(confidences) if confidences else 0.0
    
    # Most common stress level
    if stress_counts:
        most_common = max(stress_counts, key=stress_counts.get)
        summary['most_common_stress_level'] = most_common
    
    # Physiological averages if available
    physiological_data = []
    for session in sessions:
        if 'physiological' in session:
            physiological_data.append(session['physiological'])
    
    if physiological_data:
        avg_hr = sum(d.get('heart_rate', 0) for d in physiological_data) / len(physiological_data)
        avg_spo2 = sum(d.get('spo2', 0) for d in physiological_data) / len(physiological_data)
        avg_temp = sum(d.get('temperature', 0) for d in physiological_data) / len(physiological_data)
        
        summary['avg_heart_rate'] = avg_hr
        summary['avg_spo2'] = avg_spo2
        summary['avg_temperature'] = avg_temp
    
    return summary

def validate_input_data(data: Dict[str, Any]) -> bool:
    """Validate input data for required fields and ranges"""
    required_fields = ['heart_rate', 'spo2', 'age']
    
    # Check for required fields
    for field in required_fields:
        if field not in data:
            return False
    
    # Validate ranges
    heart_rate = data.get('heart_rate')
    spo2 = data.get('spo2')
    age = data.get('age')
    
    # Heart rate validation (60-200 bpm)
    if not isinstance(heart_rate, (int, float)) or heart_rate < 60 or heart_rate > 200:
        return False
    
    # SpO2 validation (0-100%)
    if not isinstance(spo2, (int, float)) or spo2 < 0 or spo2 > 100:
        return False
    
    # Age validation (0-120 years)
    if not isinstance(age, (int, float)) or age < 0 or age > 120:
        return False
    
    # Optional temperature validation if present
    if 'temperature' in data:
        temp = data['temperature']
        if not isinstance(temp, (int, float)) or temp < 30 or temp > 45:
            return False
    
    return True

def save_session_data(session_data: Dict[str, Any], filename: Optional[str] = None) -> str:
    """Save session data to JSON file"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"session_{timestamp}.json"
    
    try:
        with open(filename, 'w') as f:
            json.dump(session_data, f, indent=2, default=str)
        return filename
    except Exception as e:
        print(f"Error saving session data: {e}")
        return ""

def load_session_data(filename: str) -> Dict[str, Any]:
    """Load session data from JSON file"""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading session data: {e}")
        return {} 