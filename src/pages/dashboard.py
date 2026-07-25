import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from src.utils import load_model


def show_dashboard():
    st.markdown(
        """
        <div style="margin-bottom: 25px;">
            <p style="color: #00F2FE; font-weight: 600; text-transform: uppercase; letter-spacing: 2px; font-size: 0.85rem; margin-bottom: 5px;">
                Diagnostics Hub
            </p>
            <h2 style="color: white; font-weight: 700; font-size: 1.8rem; margin-top: 0;">
                📊 Industrial Analytics Dashboard
            </h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    try:
        df = pd.read_csv("data/cleaned_data.csv")
    except Exception as e:
        st.error(f"Error loading analytics data: {e}")
        return

    total_records = len(df)
    total_failures = df["Machine_failure"].sum()
    failure_rate = (total_failures / total_records) * 100

    # Custom HTML metrics row
    st.markdown(
        f"""
        <div class="kpi-container">
            <div class="kpi-card">
                <div style="font-size: 1.8rem; margin-bottom: 5px;">🏭</div>
                <div class="kpi-value">{total_records:,}</div>
                <div class="kpi-label">Total Monitored Machines</div>
            </div>
            <div class="kpi-card">
                <div style="font-size: 1.8rem; margin-bottom: 5px;">🚨</div>
                <div class="kpi-value">{total_failures:,}</div>
                <div class="kpi-label">Detected Failures</div>
            </div>
            <div class="kpi-card">
                <div style="font-size: 1.8rem; margin-bottom: 5px;">📈</div>
                <div class="kpi-value">{failure_rate:.2f}%</div>
                <div class="kpi-label">System Failure Rate</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    tab1, tab2 = st.tabs(["📊 Operational Analytics", "📜 Diagnostics Log History"])

    with tab1:
        st.markdown(
            """
            <div style="margin-top: 20px; margin-bottom: 20px;">
                <h4 style="color: white; font-weight: 700;">Telemetry Distibution & Data Insights</h4>
                <p style="color: #94A3B8; font-size: 0.95rem;">
                    Exploratory plots showing historical operational status and parameters.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            # Pie Chart
            failure_chart = px.pie(
                df,
                names="Machine_failure",
                title="Overall Failure Distribution (0 = Healthy, 1 = Failed)",
                color_discrete_sequence=["#10B981", "#EF4444"]
            )
            failure_chart.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#E2E8F0",
                title_font_color="#00F2FE",
                margin=dict(l=10, r=10, t=40, b=10),
                height=320
            )
            st.plotly_chart(failure_chart, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            # Scatter Plot
            temp_chart = px.scatter(
                df,
                x="Air_temperature_K",
                y="Process_temperature_K",
                color="Machine_failure",
                title="Air Temp vs Process Temp (K)",
                color_continuous_scale=["#10B981", "#EF4444"]
            )
            temp_chart.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#E2E8F0",
                title_font_color="#00F2FE",
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                margin=dict(l=10, r=10, t=40, b=10),
                height=320
            )
            st.plotly_chart(temp_chart, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            # Histogram
            machine_chart = px.histogram(
                df,
                x="Type",
                title="Machine Class Distribution (H=High, M=Medium, L=Low)",
                color_discrete_sequence=["#4FACFE"]
            )
            machine_chart.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#E2E8F0",
                title_font_color="#00F2FE",
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                margin=dict(l=10, r=10, t=40, b=10),
                height=320
            )
            st.plotly_chart(machine_chart, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            # Model Feature Importance
            try:
                model = load_model()
                importance_df = pd.DataFrame({
                    "Feature": ["Type", "Air Temp", "Process Temp", "RPM", "Torque", "Tool Wear"],
                    "Importance": model.feature_importances_
                })
                importance_df = importance_df.sort_values(by="Importance", ascending=True)

                importance_chart = px.bar(
                    importance_df,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    title="Classifier Global Feature Importance",
                    color_discrete_sequence=["#00F2FE"]
                )
                importance_chart.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#E2E8F0",
                    title_font_color="#00F2FE",
                    xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                    yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                    margin=dict(l=10, r=10, t=40, b=10),
                    height=320
                )
                st.plotly_chart(importance_chart, use_container_width=True)
            except Exception as e:
                st.warning(f"Unable to load model feature importances: {e}")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(
            """
            <div class="glass-card" style="margin-top: 20px;">
                <h4 style="color: #00F2FE; margin-top: 0;">🔮 Model Analysis Summary</h4>
                <p style="color: #CBD5E1; font-size: 0.95rem; line-height: 1.5; margin-bottom: 0;">
                    The Random Forest classifier is deployed because it achieved the highest F1-Score and ROC-AUC during training. 
                    It leverages multi-sensor signals (torque, temperature differentials, and rotation speed dynamics) 
                    to compute failure probabilities in real-time, providing proactive maintenance scheduling alerts before physical machine wear occurs.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with tab2:
        st.markdown(
            """
            <div style="margin-top: 20px; margin-bottom: 20px;">
                <h4 style="color: white; font-weight: 700;">📜 Local Diagnostics History</h4>
                <p style="color: #94A3B8; font-size: 0.95rem;">
                    Chronological records of predictions generated during this session.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        try:
            history_df = pd.read_csv(
                "data/prediction_history.csv",
                names=["Timestamp", "Prediction", "Probability", "Risk"],
                on_bad_lines="skip"
            )
            # Display history in a clean dataframe
            st.dataframe(history_df, use_container_width=True)
        except Exception as e:
            st.info("No prediction history recorded yet. Perform diagnostics in the 'Prediction' tab to log data.")