import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
# --- Streamlit Page Config ---
st.set_page_config(page_title="Power Consumption Forecast", layout="wide")

st.title("⚡ Power Consumption Forecast Dashboard")

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("daily_state_usage.csv")
    df["Dates"] = pd.to_datetime(df["Dates"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Dates"])
    return df

# If user uploaded file, use it, else load default dataset
if "user_df" in st.session_state:
    df = st.session_state["user_df"]
else:
    df = load_data()


# --- User Inputs ---
states = sorted(df["States"].unique())
state_choice = st.selectbox("Select a State", states)
forecast_years = st.slider("Select Forecast Duration (Years)", 1, 5, 2)

# --- Filter Data ---
state_df = df[df["States"] == state_choice]

# --- Create Tabs ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📈 Line Chart", "🥧 Pie Chart", "📊 Bar Chart", "🗺️ Map Visualization", "🔮 Forecast", "📂 Upload Data"])
# --------------------------------------------------------------------
# 📈 TAB 1: Line Chart (Historical + Forecast)
# --------------------------------------------------------------------
with tab1:
    st.subheader(f"📈 Consumption Trend for {state_choice}")

    # Line Chart
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(state_df["Dates"], state_df["Usage"], label="Historical", color="tab:blue")

    # Forecast Calculation
    model = ExponentialSmoothing(state_df["Usage"], trend="add", seasonal=None)
    fit = model.fit()
    future_index = pd.date_range(
        state_df["Dates"].max() + pd.offsets.MonthEnd(1),
        periods=forecast_years * 12,
        freq="M"
    )
    forecast = fit.forecast(len(future_index))
    ax.plot(future_index, forecast, label="Forecast", linestyle="--", color="tab:orange")

    ax.legend()
    ax.set_xlabel("Date")
    ax.set_ylabel("Usage (GWh)")
    st.pyplot(fig)
# --------------------------------------------------------------------
# 🥧 TAB 2: Pie Chart (Average Usage by State - Dynamic by Year)
# --------------------------------------------------------------------
with tab2:
    st.subheader("🥧 State-wise Average Usage Distribution")

    # --- Make pie chart dynamic by selected forecast years ---
    # Filter dataset for the most recent N years chosen by user
    # Filter data based on selected forecast years
    latest_date = df["Dates"].max()
    start_date = latest_date - pd.DateOffset(years=forecast_years)
    filtered_df = df[df["Dates"] >= start_date]


    # Calculate Average Usage by State (for filtered data)
    avg_usage = filtered_df.groupby("States")["Usage"].mean().reset_index()
    avg_usage["Usage"] = avg_usage["Usage"].astype(float)

    # Combine states with average < 3 as "Others"
    low_avg_sum = avg_usage.loc[avg_usage["Usage"] < 3, "Usage"].sum()
    avg_usage = avg_usage.loc[avg_usage["Usage"] >= 3]
    avg_usage = pd.concat([avg_usage, pd.DataFrame({"States": ["Others"], "Usage": [low_avg_sum]})])

    # Pie Chart
    fig2, ax2 = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax2.pie(
        avg_usage["Usage"],
        labels=avg_usage["States"],
        autopct=lambda p: f'{p:.1f}%' if p > 3 else '',
        startangle=90,
        wedgeprops={"edgecolor": "white"},
        pctdistance=0.75,
        textprops={'fontsize': 7, 'fontweight': 'bold', 'color': 'black'}
    )

    plt.setp(autotexts, size=10, weight="bold", color="white")
    ax2.axis("equal")
    st.pyplot(fig2)

    # --------------------------------------------------------------------
# 📊 TAB 3: Bar Chart (State-wise Average Usage)
# --------------------------------------------------------------------
with tab3:
    st.subheader("📊 Average Usage by State (Bar Chart)")
# Filter data based on selected forecast years
    latest_date = df["Dates"].max()
    start_date = latest_date - pd.DateOffset(years=forecast_years)
    filtered_df = df[df["Dates"] >= start_date]

    # Calculate average usage and percentage
    avg_usage = filtered_df.groupby("States")["Usage"].mean().reset_index()
    total_usage = avg_usage["Usage"].sum()
    avg_usage["Percentage"] = (avg_usage["Usage"] / total_usage) * 100

    # Combine states with < 3% usage into "Others"
    others_sum = avg_usage.loc[avg_usage["Percentage"] < 3, "Usage"].sum()
    avg_usage = avg_usage.loc[avg_usage["Percentage"] >= 3]
    avg_usage = pd.concat([
        avg_usage,
        pd.DataFrame({"States": ["Others"], "Usage": [others_sum]})
    ])
    avg_usage["Percentage"] = (avg_usage["Usage"] / avg_usage["Usage"].sum()) * 100

    # Sort by usage descending
    avg_usage = avg_usage.sort_values(by="Usage", ascending=False)

    # Plot bar chart
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    bars = ax3.bar(avg_usage["States"], avg_usage["Usage"], color=plt.cm.tab20.colors)

    ax3.set_xlabel("States")
    ax3.set_ylabel("Average Usage (GWh)")
    ax3.set_xticklabels(avg_usage["States"], rotation=45, ha="right")

    # Add percentage labels **on top** of bars
    for bar, pct in zip(bars, avg_usage["Percentage"]):
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            height + (0.02 * max(avg_usage["Usage"])),
            f"{pct:.1f}%",
            ha="center",
            va="bottom",
            color="black",
            fontsize=7,
            fontweight="normal"
        )

    st.pyplot(fig3)
    # --------------------------------------------------------------------
# 🗺️ TAB 4: Map Visualization (Dynamic India Map)
# --------------------------------------------------------------------
with tab4:
    st.subheader("🗺️ Power Consumption Map")
    st.markdown("Map visualization appears in the Map section below.")

    # --- Dynamic India Map Visualization ---
    import pydeck as pdk

    st.markdown("---")
    st.subheader("🗺️ Power Consumption Across All States")

# Predefined coordinates for all Indian states (approx center points)
    state_coords = {
       "Andhra Pradesh": [15.9129, 79.7400],
       "Arunachal Pradesh": [28.2180, 94.7278],
       "Assam": [26.2006, 92.9376],
       "Bihar": [25.0961, 85.3131],
       "Chhattisgarh": [21.2787, 81.8661],
       "Goa": [15.2993, 74.1240],
       "Gujarat": [22.2587, 71.1924],
       "Haryana": [29.0588, 76.0856],
       "Himachal Pradesh": [31.1048, 77.1734],
       "Jharkhand": [23.6102, 85.2799],
       "Karnataka": [15.3173, 75.7139],
       "Kerala": [10.8505, 76.2711],
       "Madhya Pradesh": [22.9734, 78.6569],
       "Maharashtra": [19.7515, 75.7139],
       "Manipur": [24.6637, 93.9063],
       "Meghalaya": [25.4670, 91.3662],
       "Mizoram": [23.1645, 92.9376],
       "Nagaland": [26.1584, 94.5624],
       "Odisha": [20.9517, 85.0985],
       "Punjab": [31.1471, 75.3412],
       "Rajasthan": [27.0238, 74.2179],
       "Sikkim": [27.5330, 88.5122],
       "Tamil Nadu": [11.1271, 78.6569],
       "Telangana": [17.1232, 79.2088],
       "Tripura": [23.9408, 91.9882],
       "Uttar Pradesh": [26.8467, 80.9462],
       "Uttarakhand": [30.0668, 79.0193],
       "West Bengal": [22.9868, 87.8550],
       "Delhi": [28.7041, 77.1025],
       "Jammu and Kashmir": [33.7782, 76.5762],
       "Ladakh": [34.2996, 78.2932],
       "Puducherry": [11.9416, 79.8083],
       "Chandigarh": [30.7333, 76.7794],
       "Andaman and Nicobar Islands": [11.7401, 92.6586],
       "Lakshadweep": [10.3280, 72.7846]
}

# Compute average usage for each state available in dataset
    # Filter data based on selected forecast years
    latest_date = df["Dates"].max()
    start_date = latest_date - pd.DateOffset(years=forecast_years)
    filtered_df = df[df["Dates"] >= start_date]

# Compute average usage for each state available in filtered dataset
    map_data = pd.DataFrame([
    {
        "State": state,
        "lat": state_coords[state][0],
        "lon": state_coords[state][1],
        "Avg_Usage": filtered_df[filtered_df["States"] == state]["Usage"].mean()
    }
    for state in filtered_df["States"].unique() if state in state_coords
    ])


# Remove missing usage states
    map_data = map_data.dropna(subset=["Avg_Usage"])

# Normalize values for color and circle size
    max_usage = map_data["Avg_Usage"].max()
    map_data["Size"] = (map_data["Avg_Usage"] / max_usage) * 90000
    map_data["Color"] = map_data["Avg_Usage"].apply(
    lambda x: [
        int(255 * (x / max_usage)),         # Red intensity (higher = more usage)
        int(255 * (1 - x / max_usage)),     # Green intensity (lower = more usage)
        120,
        160
    ]
)

# Define PyDeck layer
    layer = pdk.Layer(
       "ScatterplotLayer",
       data=map_data,
       get_position=["lon", "lat"],
       get_color="Color",
       get_radius="Size",
       pickable=True,
)

# Set map view centered on India
    view_state = pdk.ViewState(latitude=22.9734, longitude=78.6569, zoom=4.3, pitch=30)

# Render the map
    st.pydeck_chart(pdk.Deck(
       layers=[layer],
       initial_view_state=view_state,
       tooltip={"text": "{State}\nAverage Usage: {Avg_Usage} GWh"}
))
    
 # --------------------------------------------------------------------
# 🔮 TAB 5: Forecast & Summary Metrics
# --------------------------------------------------------------------
with tab5:
    st.subheader("🔮 Forecast Summary")

    model = ExponentialSmoothing(state_df["Usage"], trend="add", seasonal=None)
    fit = model.fit()
    future_index = pd.date_range(
        state_df["Dates"].max() + pd.offsets.MonthEnd(1),
        periods=forecast_years * 12,
        freq="M"
    )
    forecast = fit.forecast(len(future_index))

    avg_usage_val = state_df["Usage"].mean()
    growth = (forecast.iloc[-1] / state_df["Usage"].iloc[-1] - 1) * 100

    col1, col2 = st.columns(2)
    col1.metric("Average Usage (GWh)", f"{avg_usage_val:.2f}")
    col2.metric(f"Forecasted Growth ({forecast_years} yrs)", f"{growth:.2f}%")

    # --- Show Raw Data ---
    st.markdown("---")
    with st.expander("📋 Show Raw Data"):
        st.dataframe(state_df)

    # --- Download Forecast CSV ---
    forecast_df = pd.DataFrame({"Date": future_index, "Forecast_Usage": forecast})
    csv = forecast_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Forecast Data", csv, "forecast.csv", "text/csv")

    # --------------------------------------------------------------------
# 📂 TAB 6: Upload Your Own Dataset
# --------------------------------------------------------------------
with tab6:
    st.subheader("📂 Upload Your Dataset (CSV / Excel)")

    uploaded_file = st.file_uploader("Upload File", type=["csv", "xlsx"])

    if uploaded_file:
        try:
            # Calculate file size in MB
            file_size_mb = uploaded_file.size / (1024 * 1024)

            if file_size_mb > 500:
                st.warning("⚠️ File exceeds 500 MB. Please upload a smaller file.")
            else:
                # Load file dynamically
                if uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file, engine="openpyxl")

                st.success("✅ File Uploaded Successfully!")
                st.write(f"**File Name:** {uploaded_file.name}")
                st.write(f"**File Size:** {file_size_mb:.2f} MB")

                st.write("### 📊 Preview of Uploaded Data:")
                st.dataframe(df.head())

                st.info("🔁 Dashboard is now using your uploaded dataset.")
        except Exception as e:
            st.error("❌ Upload Failed")
            st.info("""
            Please ensure your file:
            - Is a valid `.csv` or `.xlsx`
            - Is not password-protected
            - Is under 500 MB
            """)
            st.caption(f"**Error details:** {e}")
    else:
        st.warning("📎 Please upload a file to proceed.")
