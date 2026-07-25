import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
from datetime import datetime

from src.components import data_validation
from src.logger import logging
from src.utils import model_exists, load_model, load_scaler, load_encoder, load_explainer
from src.recommendation_engine import RecommendationEngine
from src.report_generator import ReportGenerator
from config.config import *


def show_prediction():
    st.markdown(
        """
        <div style="margin-bottom: 25px;">
            <p style="color: #00F2FE; font-weight: 600; text-transform: uppercase; letter-spacing: 2px; font-size: 0.85rem; margin-bottom: 5px;">
                Prediction Engine
            </p>
            <h2 style="color: white; font-weight: 700; font-size: 1.8rem; margin-top: 0;">
                🔮 Machine Failure Diagnostics
            </h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    if not model_exists():
        st.error("Model file missing")
        return

    try:
        model = load_model()
        scaler = load_scaler()
        encoder = load_encoder()
        explainer = load_explainer()
    except Exception as e:
        logging.error(str(e))
        st.error("Unable to load model artifacts.")
        return

    # Use HTML wrapping for the container layout
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<p style="color: #00F2FE; font-weight: 600; margin-top: 0;">🔧 Machine Operational Telemetry Inputs</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)

    with col1:
        machine_type = st.selectbox("Machine Type", ["L", "M", "H"])
        air_temp = st.number_input("Air Temperature (K)", value=298.0, step=0.1)
        process_temp = st.number_input("Process Temperature (K)", value=308.0, step=0.1)

    with col2:
        rotational_speed = st.number_input("Rotational Speed (RPM)", value=1500, step=10)
        torque = st.number_input("Torque (Nm)", value=40.0, step=0.5)
        tool_wear = st.number_input("Tool Wear (min)", value=100, step=1)

    st.markdown('</div>', unsafe_allow_html=True)

    predict_btn = st.button("🔮 Calculate Machine Health & Diagnostics")

    if predict_btn:
        logging.info("Prediction request received")

        is_valid = data_validation.DataValidator.validate_input(
            air_temp,
            process_temp,
            rotational_speed,
            torque,
            tool_wear
        )

        if not is_valid:
            st.error("Invalid Input Values: Please verify sensor thresholds.")
            return

        machine_type_encoded = encoder.transform([machine_type])[0]

        input_df = pd.DataFrame({
            "Type": [machine_type_encoded],
            "Air_temperature_K": [air_temp],
            "Process_temperature_K": [process_temp],
            "Rotational_speed_rpm": [rotational_speed],
            "Torque_Nm": [torque],
            "Tool_wear_min": [tool_wear]
        })

        scaled_input = scaler.transform(input_df)

        prediction = model.predict(scaled_input)
        probability = model.predict_proba(scaled_input)
        shap_values = explainer.explain(scaled_input)

        failure_probability = probability[0][1] * 100
        confidence = max(probability[0]) * 100

        # Predictions Results Section
        st.markdown(
            """
            <div style="margin-top: 40px; margin-bottom: 20px;">
                <p style="color: #00F2FE; font-weight: 600; text-transform: uppercase; letter-spacing: 2px; font-size: 0.85rem; margin-bottom: 5px;">
                    Analysis Output
                </p>
                <h3 style="color: white; font-weight: 700; font-size: 1.5rem; margin-top: 0;">
                    🔍 Diagnostic Results & Risk Assessment
                </h3>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Risk indicator calculation
        if failure_probability < 30:
            risk = "Low"
            risk_color = "#10B981"  # green
            risk_bg = "rgba(16, 185, 129, 0.15)"
            status_msg = "Machine Healthy ✅"
            status_box_type = "success"
        elif failure_probability < 70:
            risk = "Medium"
            risk_color = "#F59E0B"  # orange
            risk_bg = "rgba(245, 158, 11, 0.15)"
            status_msg = "Maintenance Recommended ⚠️"
            status_box_type = "warning"
        else:
            risk = "High"
            risk_color = "#EF4444"  # red
            risk_bg = "rgba(239, 68, 68, 0.15)"
            status_msg = "Immediate Inspection Required 🚨"
            status_box_type = "error"

        # Custom Metrics and Risk Meter inside glass card
        st.markdown(
            f"""<div class="glass-card">
<div style="display: flex; flex-wrap: wrap; justify-content: space-around; gap: 15px; margin-bottom: 10px;">
<div style="text-align: center; flex: 1; min-width: 150px;">
<span style="color: #94A3B8; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px;">Failure Probability</span>
<div style="font-size: 2.5rem; font-weight: 700; color: #00F2FE; margin-top: 5px;">{failure_probability:.2f}%</div>
</div>
<div style="text-align: center; flex: 1; min-width: 150px;">
<span style="color: #94A3B8; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px;">Model Confidence</span>
<div style="font-size: 2.5rem; font-weight: 700; color: #CBD5E1; margin-top: 5px;">{confidence:.2f}%</div>
</div>
<div style="text-align: center; flex: 1; min-width: 150px; background-color: {risk_bg}; border-radius: 12px; padding: 10px; border: 1px solid {risk_color}33;">
<span style="color: #94A3B8; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px;">Risk Category</span>
<div style="font-size: 2.5rem; font-weight: 700; color: {risk_color}; margin-top: 5px;">{risk}</div>
</div>
</div>
<div style="margin-top: 25px;">
<span style="color: #94A3B8; font-size: 0.85rem;">Risk Meter Level:</span>
<div style="width: 100%; background-color: rgba(255, 255, 255, 0.08); border-radius: 8px; height: 16px; margin-top: 5px; overflow: hidden; border: 1px solid rgba(255, 255, 255, 0.05);">
<div style="width: {max(failure_probability, 3.0)}%; background: linear-gradient(90deg, #10B981 0%, #F59E0B 50%, #EF4444 100%); height: 100%; border-radius: 8px; transition: width 0.8s ease-in-out;"></div>
</div>
</div>
</div>""",
            unsafe_allow_html=True
        )

        recommendation = RecommendationEngine.recommend(failure_probability)
        
        # Display recommendations with premium styling
        if risk == "High":
            st.error(f"**Status:** {status_msg}\n\n**Action Plan:** {recommendation}")
        elif risk == "Medium":
            st.warning(f"**Status:** {status_msg}\n\n**Action Plan:** {recommendation}")
        else:
            st.success(f"**Status:** {status_msg}\n\n**Action Plan:** {recommendation}")

        # Explainability Graph and Table side-by-side
        st.markdown(
            """
            <div style="margin-top: 35px; margin-bottom: 15px;">
                <h4 style="color: white; font-weight: 700;">🤖 Machine Diagnostic Explainer (SHAP)</h4>
                <p style="color: #94A3B8; font-size: 0.9rem;">
                    The chart below highlights which sensors contributed most to this prediction. Values to the right indicate factors driving the machine closer to failure.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        feature_importance = pd.DataFrame({
            "Feature": ["Type", "Air Temp", "Process Temp", "RPM", "Torque", "Tool Wear"],
            "Impact": abs(shap_values[0, :, 1])
        })

        feature_importance = feature_importance.sort_values(
            by="Impact",
            ascending=True
        )

        col_g1, col_g2 = st.columns([3, 2])
        
        with col_g1:
            shap_chart = px.bar(
                feature_importance,
                x="Impact",
                y="Feature",
                orientation="h",
                title="Telemetry Feature Impact"
            )
            shap_chart.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#E2E8F0",
                title_font_color="#00F2FE",
                xaxis=dict(showgrid=True, gridcolor="rgba(255, 255, 255, 0.1)"),
                yaxis=dict(showgrid=False),
                margin=dict(l=20, r=20, t=40, b=20),
                height=300
            )
            st.plotly_chart(shap_chart, use_container_width=True)

        with col_g2:
            st.markdown('<div style="margin-top: 25px;">', unsafe_allow_html=True)
            st.dataframe(
                feature_importance.sort_values(by="Impact", ascending=False),
                use_container_width=True,
                height=260
            )
            st.markdown('</div>', unsafe_allow_html=True)

        # Log to History file
        try:
            history = pd.DataFrame({
                "Timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                "Prediction": [prediction[0]],
                "Probability": [failure_probability],
                "Risk": [risk]
            })

            history.to_csv(
                "data/prediction_history.csv",
                mode="a",
                header=False,
                index=False
            )
        except Exception as write_err:
            logging.warning(f"Could not write prediction history: {write_err}")

        # Download buttons side-by-side inside a card
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<p style="color: #94A3B8; font-size: 0.9rem; font-weight: 600; margin-top: 0;">📥 Export Diagnostics & Records</p>', unsafe_allow_html=True)
        
        col_d1, col_d2 = st.columns(2)
        
        with col_d1:
            st.download_button(
                label="📊 Download Prediction History CSV",
                data=history.to_csv(index=False),
                file_name="prediction_history.csv",
                mime="text/csv"
            )

        with col_d2:
            report = ReportGenerator.generate(
                failure_probability,
                risk,
                recommendation
            )
            st.download_button(
                label="📄 Download Diagnostic Report (.txt)",
                data=report,
                file_name="maintenance_report.txt",
                mime="text/plain"
            )
            
        st.markdown('</div>', unsafe_allow_html=True)