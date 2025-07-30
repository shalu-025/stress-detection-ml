import streamlit as st
import joblib
from app.suggestions import get_suggestion
import os
# --- Simple Login Setup ---
USERS = {
    "shalu": "1234",  # You can change or add more
    "demo": "abcd"
}

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# --- Login Form ---
if not st.session_state.logged_in:
    st.title("🔐 Stress Detection App Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in USERS and USERS[username] == password:
            st.session_state.logged_in = True
            st.success("Login successful!")
            st.experimental_rerun()
        else:
            st.error("Invalid username or password")
    st.stop()  # Stop app until logged in
from fpdf import FPDF
import pandas as pd
import base64

def create_pdf(content):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in content.split("\n"):
        pdf.cell(200, 10, txt=line, ln=True)
    pdf.output("stress_report.pdf")
    with open("stress_report.pdf", "rb") as f:
        pdf_bytes = f.read()
    b64_pdf = base64.b64encode(pdf_bytes).decode()
    href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="stress_report.pdf">📄 Download Report as PDF</a>'
    return href

def create_csv():
    df = pd.DataFrame(st.session_state.history)
    csv = df.to_csv(index=False)
    b64_csv = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:fi

if 'history' not in st.session_state:
    st.session_state['history'] = []

# Load model and encoder
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "random_forest_model.pkl")
ENCODER_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "label_encoder.pkl")
model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

st.title("Human Stress Detection")

# Input fields
# User Inputs
with st.form("stress_form"):
    st.subheader("🧠 Enter Your Health Data")

    age = st.slider("Age", 10, 100, 25)
    gender = st.radio("Gender", ("Male", "Female"))

    hr = st.number_input("Heart Rate (bpm)", min_value=40, max_value=180, value=72)
    spo2 = st.number_input("SpO2 (%)", min_value=70, max_value=100, value=98)

    # Use average values for other 6 features (optional)
    temp = 98.6
    bp = 120
    rr = 18
    sugar = 90
    ecg = 80

    submitted = st.form_submit_button("Detect Stress")


# Default average values (you can customize these)
temp_avg = 98.6
bp_sys_avg = 120
bp_dia_avg = 80
resp_rate_avg = 16

# Prepare feature vector
features = [[age, 0 if gender == "Male" else 1, hr, spo2, temp_avg, bp_sys_avg, bp_dia_avg, resp_rate_avg]]


# Predict button
if submitted:
    # Encode gender
    gender_encoded = 1 if gender == "Male" else 0

    # Prepare input vector with all 8 features
    input_features = [[
        age, gender_encoded, hr, spo2,
        temp, bp, rr, sugar, ecg
    ]]

    # Predict stress
    pred_encoded = model.predict(input_features)[0]
    stress_level = label_encoder.inverse_transform([pred_encoded])[0]

    # Show result
    st.subheader(f"🧠 Stress Level: {stress_level}")
    suggestion = get_suggestion(stress_level)
    st.success(suggestion)

    # Save to session history
    st.session_state['history'].append({
        "Age": age, "Gender": gender, "HR":



        # Display result
        st.subheader(f"Stress Level: {stress_level}")
        suggestion = get_suggestion(stress_level)
        st.write(suggestion)

    # -----------------------------
# 📊 Stress Level History Chart
# -----------------------------
if st.session_state['history']:
    st.markdown("### 📈 Stress History Chart")

    history_df = pd.DataFrame(st.session_state['history'])

    # Count how many times each stress level occurred
    stress_counts = history_df['Stress Level'].value_counts()

    # Bar chart
    st.bar_chart(stress_counts)

    # Optional: Show raw data
    with st.expander("Show History Table"):
        st.dataframe(history_df)


    except Exception as e:
        st.error(f"Something went wrong: {e}")

from fpdf import FPDF
import datetime

def export_pdf(stress_level, suggestion, user_info=None):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Stress Detection Report", ln=True, align='C')
    pdf.ln(10)

    if user_info:
        pdf.cell(200, 10, txt=f"User Info: {user_info}", ln=True)

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pdf.cell(200, 10, txt=f"Date: {now}", ln=True)
    pdf.cell(200, 10, txt=f"Stress Level: {stress_level}", ln=True)
    pdf.multi_cell(0, 10, txt=f"Suggestion:\n{suggestion}")

    output_path = "stress_report.pdf"
    pdf.output(output_path)
    return output_path
if st.button("Export as PDF"):
    path = export_pdf(stress_level, suggestion)
    with open(path, "rb") as f:
        st.download_button("Download Report", f, file_name="stress_report.pdf")

if st.session_state['history']:
    st.subheader("📈 Stress Detection History")
    history_df = pd.DataFrame(st.session_state['history'])
    st.dataframe(history_df)
    
# Plot Mood Graph
st.subheader("🧠 Mood Overview")

# Count stress levels
stress_counts = history_df["Stress Level"].value_counts()

# Choose graph type
chart_type = st.radio("Choose Chart Type", ["Bar Chart", "Pie Chart"], horizontal=True)

if chart_type == "Bar Chart":
    st.bar_chart(stress_counts)
else:
    fig, ax = plt.subplots()
    ax.pie(stress_counts, labels=stress_counts.index, autopct="%1.1f%%", startangle=90)
    ax.axis("equal")
    st.pyplot(fig)
# Export History
st.subheader("📥 Export History")

# Download as CSV
csv = history_df.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", csv, "stress_history.csv", "text/csv")

# Download as PDF (basic table format)
from fpdf import FPDF

def convert_df_to_pdf(df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Stress Detection History", ln=True, align="C")
    pdf.ln(10)
    col_width = pdf.w / 4.5
    row_height = pdf.font_size + 2

    # Add column headers
    for col in df.columns:
        pdf.cell(col_width, row_height, txt=str(col), border=1)
    pdf.ln(row_height)

    # Add rows
    for _, row in df.iterrows():
        for item in row:
            pdf.cell(col_width, row_height, txt=str(item), border=1)
        pdf.ln(row_height)

    return pdf.output(dest="S").encode("latin-1")

pdf_data = convert_df_to_pdf(history_df)
st.download_button("Download PDF", pdf_data, file_name="stress_history.pdf")

# --- Export Section ---
st.markdown("---")
st.subheader("📥 Export Your Results")

if st.session_state.history:
    history_text = "\n".join(
        [f"{i+1}. HR: {x['HR']}, SpO2: {x['SpO2']}, Age: {x['Age']}, Gender: {x['Gender']}, Stress: {x['Stress']}" for i, x in enumerate(st.session_state.history)]
    )
    st.markdown(create_pdf(history_text), unsafe_allow_html=True)
    st.markdown(create_csv(), unsafe_allow_html=True)
else:
    st.info("No history to export yet.")


