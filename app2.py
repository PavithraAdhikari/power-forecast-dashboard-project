import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pydeck as pdk
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.set_page_config(page_title="Power Consumption Forecast", layout="wide")

st.title("⚡ Power Consumption Forecast Dashboard")

# ---------------------------------------------------
# 🔧 SMART NORMALIZATION (WORKS FOR ANY DATASET)
# ---------------------------------------------------
def normalize_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert ANY dataset into a standard format:
    Columns: ['States', 'Dates', 'Usage']
    - Tries to auto-detect date column
    - Auto-detects category column
    - Auto-detects numeric column
    """
    df = raw_df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if df.empty or len(df.columns) < 2:
        raise ValueError("Dataset must have at least 2 columns with data.")

    lower_map = {c.lower(): c for c in df.columns}

    # 1) Detect date-like column
    date_col = None
    date_keywords = ["date", "year", "time", "month", "day"]
    for key, original in lower_map.items():
        if any(k in key for k in date_keywords):
            date_col = original
            break
    if date_col is None:
        # fallback: first column
        date_col = df.columns[0]

    # Build Dates column
    dates_raw = df[date_col]

    if np.issubdtype(dates_raw.dtype, np.number) and "year" in date_col.lower():
        # Treat as YEAR
        dates = pd.to_datetime(dates_raw.astype(int).astype(str) + "-01-01", errors="coerce")
    else:
        dates = pd.to_datetime(dates_raw, errors="coerce")

    df["Dates"] = dates

    # 2) Detect category column (for "States")
    cat_col = None
    cat_keywords = ["state", "region", "category", "sector", "name", "type", "area"]
    for key, original in lower_map.items():
        if any(k in key for k in cat_keywords) and original != date_col:
            cat_col = original
            break

    if cat_col is None:
        # fallback: first non-date column
        possible = [c for c in df.columns if c not in [date_col, "Dates"]]
        if not possible:
            possible = [date_col]
        cat_col = possible[0]

    # 3) Detect numeric column (for "Usage")
    numeric_cols = [
        c for c in df.columns
        if c not in ["Dates", date_col, cat_col] and np.issubdtype(df[c].dtype, np.number)
    ]

    # If no numeric dtype, try converting something
    if not numeric_cols:
        for c in df.columns:
            if c in ["Dates", date_col, cat_col]:
                continue
            maybe = pd.to_numeric(df[c], errors="coerce")
            if maybe.notna().sum() > 0:
                df[c] = maybe
                numeric_cols.append(c)
                break

    if not numeric_cols:
        raise ValueError("Could not find any numeric column to use as 'Usage'.")

    usage_col = numeric_cols[-1]  # last numeric col

    out = pd.DataFrame({
        "States": df[cat_col].astype(str),
        "Dates": df["Dates"],
        "Usage": pd.to_numeric(df[usage_col], errors="coerce")
    })

    out = out.dropna(subset=["Dates", "Usage"])
    if out.empty:
        raise ValueError("After cleaning, no valid (Dates, Usage) rows remain.")
    return out


# ---------------------------------------------------
# 📥 DEFAULT DATA LOADER
# ---------------------------------------------------
@st.cache_data
def load_default() -> pd.DataFrame:
    # Your original default dataset
    raw = pd.read_csv("daily_state_usage.csv")
    return normalize_df(raw)


def set_df_in_session(new_raw: pd.DataFrame):
    norm = normalize_df(new_raw)
    st.session_state["df"] = norm


def get_df() -> pd.DataFrame:
    if "df" in st.session_state:
        return st.session_state["df"]
    df = load_default()
    st.session_state["df"] = df
    return df


# ---------------------------------------------------
# 🧊 SIDEBAR: MAIN UPLOADER + RESET
# ---------------------------------------------------
st.sidebar.header("Upload / Control Data")

main_upload = st.sidebar.file_uploader(
    "Upload dataset (CSV/XLSX) — main",
    type=["csv", "xlsx"],
    key="main_upload"
)

if main_upload is not None:
    try:
        if main_upload.name.lower().endswith(".csv"):
            raw = pd.read_csv(main_upload)
        else:
            raw = pd.read_excel(main_upload, engine="openpyxl")
        set_df_in_session(raw)
        st.sidebar.success("✅ New dataset applied to dashboard.")
    except Exception as e:
        st.sidebar.error("❌ Could not use this file.")
        st.sidebar.caption(str(e))

# Reset button → back to default
if st.sidebar.button("🔄 Reset to Default Dataset"):
    if "df" in st.session_state:
        del st.session_state["df"]
    st.sidebar.success("Back to default dataset.")
    st.rerun()

# ---------------------------------------------------
# ✅ ACTIVE DATAFRAME (DEFAULT OR UPLOADED)
# ---------------------------------------------------
df = get_df()

# If somehow still empty
if df.empty:
    st.error("No data available after processing. Check your dataset.")
    st.stop()

# ---------------------------------------------------
# 🎚 GLOBAL CONTROLS
# ---------------------------------------------------
states = sorted(df["States"].unique())
state_choice = st.selectbox("Select a State / Category", states)
forecast_years = st.slider("Select Forecast Duration (Years)", 1, 5, 2)

state_df = df[df["States"] == state_choice].sort_values("Dates").copy()
if state_df.empty:
    st.warning("No data for this selection.")
    st.stop()

# ---------------------------------------------------
# 🧩 TABS
# ---------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Line Chart",
    "🥧 Pie Chart",
    "📊 Bar Chart",
    "🗺️ Map Visualization",
    "🔮 Forecast",
    "📂 Upload Data"
])

# ---------------------------------------------------
# 📈 TAB 1: LINE (HISTORICAL + FORECAST)
# ---------------------------------------------------
with tab1:
    st.subheader(f"📈 Trend for '{state_choice}'")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(state_df["Dates"], state_df["Usage"], label="Historical", linewidth=2)

    # Forecast using Holt-Winters where possible
    if len(state_df) >= 2:
        try:
            model = ExponentialSmoothing(state_df["Usage"], trend="add", seasonal=None)
            fit = model.fit()
            future_index = pd.date_range(
                state_df["Dates"].max() + pd.offsets.MonthEnd(1),
                periods=forecast_years * 12,
                freq="M"
            )
            forecast = fit.forecast(len(future_index))
        except Exception:
            last = state_df["Usage"].iloc[-1]
            future_index = pd.date_range(
                state_df["Dates"].max() + pd.offsets.MonthEnd(1),
                periods=forecast_years * 12,
                freq="M"
            )
            forecast = pd.Series(last, index=future_index)
        ax.plot(future_index, forecast, linestyle="--", label="Forecast")
    else:
        st.info("Not enough data points to build a proper forecast. Showing only historical.")

    ax.set_xlabel("Date")
    ax.set_ylabel("Usage / Value")
    ax.legend()
    st.pyplot(fig)

# ---------------------------------------------------
# 🥧 TAB 2: PIE CHART (DYNAMIC BY YEARS)
# ---------------------------------------------------
with tab2:
    st.subheader("🥧 Average Usage Share by State / Category")

    latest_date = df["Dates"].max()
    start_date = latest_date - pd.DateOffset(years=forecast_years)
    filtered_df = df[df["Dates"] >= start_date]

    avg_usage = filtered_df.groupby("States")["Usage"].mean().reset_index()
    avg_usage["Usage"] = avg_usage["Usage"].astype(float)

    # combine small ones into "Others" if < 3.0
    low_mask = avg_usage["Usage"] < 3.0
    low_sum = avg_usage.loc[low_mask, "Usage"].sum()
    main = avg_usage.loc[~low_mask].copy()
    if low_sum > 0:
        main = pd.concat(
            [main, pd.DataFrame({"States": ["Others"], "Usage": [low_sum]})],
            ignore_index=True
        )

    if main.empty:
        st.warning("Not enough data to build pie chart for this period.")
    else:
        fig2, ax2 = plt.subplots(figsize=(7, 7))
        wedges, texts, autotexts = ax2.pie(
            main["Usage"],
            labels=main["States"],
            autopct=lambda p: f"{p:.1f}%" if p > 3 else "",
            startangle=90,
            wedgeprops={"edgecolor": "white"},
            pctdistance=0.75,
            textprops={'fontsize': 7, 'fontweight': 'bold', 'color': 'black'}
        )
        plt.setp(autotexts, size=10, weight="bold")
        ax2.axis("equal")
        st.pyplot(fig2)

# ---------------------------------------------------
# 📊 TAB 3: BAR CHART (DYNAMIC BY YEARS)
# ---------------------------------------------------
with tab3:
    st.subheader("📊 Average Usage by State / Category")

    latest_date = df["Dates"].max()
    start_date = latest_date - pd.DateOffset(years=forecast_years)
    filtered_df = df[df["Dates"] >= start_date]

    avg_usage = filtered_df.groupby("States")["Usage"].mean().reset_index()
    total_usage = avg_usage["Usage"].sum()

    if total_usage == 0 or avg_usage.empty:
        st.warning("No data in this period to show bar chart.")
    else:
        avg_usage["Percentage"] = (avg_usage["Usage"] / total_usage) * 100

        # merge <3% into Others
        small_mask = avg_usage["Percentage"] < 3
        others_sum = avg_usage.loc[small_mask, "Usage"].sum()
        main = avg_usage.loc[~small_mask].copy()
        if others_sum > 0:
            main = pd.concat([
                main,
                pd.DataFrame({"States": ["Others"], "Usage": [others_sum]})
            ], ignore_index=True)
        main["Percentage"] = (main["Usage"] / main["Usage"].sum()) * 100
        main = main.sort_values("Usage", ascending=False)

        fig3, ax3 = plt.subplots(figsize=(10, 5))
        bars = ax3.bar(main["States"], main["Usage"])

        ax3.set_xlabel("State / Category")
        ax3.set_ylabel("Average Usage / Value")
        ax3.set_xticklabels(main["States"], rotation=45, ha="right")

        max_usage_val = main["Usage"].max()
        for i, (bar, pct) in enumerate(zip(bars, main["Percentage"])):
            height = bar.get_height()
            y = height + (0.02 * max_usage_val)
            ax3.text(
                bar.get_x() + bar.get_width() / 2,
                y,
                f"{pct:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8
            )

        st.pyplot(fig3)

# ---------------------------------------------------
# 🗺️ TAB 4: GEO MAP (ONLY WORKS IF STATES LOOK LIKE INDIA STATES)
# ---------------------------------------------------
with tab4:
    st.subheader("🗺️ Map Visualization (India States Only)")

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
        "Lakshadweep": [10.3280, 72.7846],
    }

    latest_date = df["Dates"].max()
    start_date = latest_date - pd.DateOffset(years=forecast_years)
    filtered_df = df[df["Dates"] >= start_date]

    map_data = pd.DataFrame([
        {
            "State": s,
            "lat": state_coords[s][0],
            "lon": state_coords[s][1],
            "Avg_Usage": filtered_df[filtered_df["States"] == s]["Usage"].mean()
        }
        for s in filtered_df["States"].unique()
        if s in state_coords
    ])

    map_data = map_data.dropna(subset=["Avg_Usage"])

    if map_data.empty:
        st.warning("No matching Indian state names found in this dataset for map.")
    else:
        max_usage = map_data["Avg_Usage"].max()
        map_data["Size"] = (map_data["Avg_Usage"] / max_usage) * 90000
        map_data["Color"] = map_data["Avg_Usage"].apply(
            lambda x: [
                int(255 * (x / max_usage)),
                int(255 * (1 - x / max_usage)),
                120,
                160
            ]
        )

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_data,
            get_position=["lon", "lat"],
            get_color="Color",
            get_radius="Size",
            pickable=True,
        )

        view_state = pdk.ViewState(
            latitude=22.9734,
            longitude=78.6569,
            zoom=4.3,
            pitch=30
        )

        st.pydeck_chart(
            pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip={"text": "{State}\nAverage Usage: {Avg_Usage}"}
            )
        )

# ---------------------------------------------------
# 🔮 TAB 5: FORECAST & METRICS
# ---------------------------------------------------
# ---------------------------------------------------
# 🔮 TAB 5: FORECAST & METRICS
# ---------------------------------------------------
with tab5:
    st.subheader("🔮 Forecast Summary")

    # -----------------------------
    # Build Forecast Model
    # -----------------------------
    if len(state_df) >= 2:
        try:
            model = ExponentialSmoothing(state_df["Usage"], trend="add", seasonal=None)
            fit = model.fit()

            # Forecast future values
            future_index = pd.date_range(
                state_df["Dates"].max() + pd.offsets.MonthEnd(1),
                periods=forecast_years * 12,
                freq="M"
            )
            forecast = fit.forecast(len(future_index))

            # -----------------------------
            # Calculate Accuracy (MAPE + Accuracy Score)
            # -----------------------------
            try:
                fitted_vals = fit.fittedvalues
                real_vals = state_df["Usage"].iloc[-len(fitted_vals):]

                mape = np.mean(np.abs((real_vals - fitted_vals) / real_vals)) * 100
                accuracy_score = 100 - mape
            except:
                mape = None
                accuracy_score = None

        except Exception:
            # Fallback if model fails
            last = state_df["Usage"].iloc[-1]
            future_index = pd.date_range(
                state_df["Dates"].max() + pd.offsets.MonthEnd(1),
                periods=forecast_years * 12,
                freq="M"
            )
            forecast = pd.Series(last, index=future_index)
            mape = None
            accuracy_score = None

    else:
        # Not enough data
        future_index = pd.date_range(pd.Timestamp.today(), periods=forecast_years * 12, freq="M")
        forecast = pd.Series(0, index=future_index)
        mape = None
        accuracy_score = None

    # -----------------------------
    # METRICS
    # -----------------------------
    avg_usage_val = state_df["Usage"].mean()

    if state_df["Usage"].iloc[-1] != 0:
        growth = (forecast.iloc[-1] / state_df["Usage"].iloc[-1] - 1) * 100
    else:
        growth = 0

    col1, col2 = st.columns(2)
    col1.metric("Average Usage / Value", f"{avg_usage_val:.2f}")
    col2.metric(f"Forecasted Growth ({forecast_years} yrs)", f"{growth:.2f}%")

    # -----------------------------
    # ACCURACY STATEMENT
    # -----------------------------
    st.markdown("### 📏 Model Forecast Accuracy")

    if accuracy_score is not None:
        st.success(
            f"The model achieves an estimated accuracy of approximately **{accuracy_score:.2f}% based on historical data.** "
            f"(MAPE: {mape:.2f}%)."
        )
    else:
        st.warning("Accuracy could not be calculated due to insufficient or invalid historical data.")

    # -----------------------------
    # RAW DATA + DOWNLOAD
    # -----------------------------
    st.markdown("---")
    with st.expander("📋 Show Raw Data"):
        st.dataframe(state_df)

    forecast_df = pd.DataFrame({"Date": future_index, "Forecast_Usage": forecast})
    csv = forecast_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Forecast Data", csv, "forecast.csv", "text/csv")

# ---------------------------------------------------
# 📂 TAB 6: EXTRA UPLOAD (INSIDE TAB)
# ---------------------------------------------------
with tab6:
    st.subheader("📂 Upload Another Dataset (Alt Input)")
    tab6_upload = st.file_uploader("Upload File (Tab 6)", type=["csv", "xlsx"], key="tab6_uploader")

    if tab6_upload:
        try:
            size_mb = tab6_upload.size / (1024 * 1024)
            if size_mb > 500:
                st.warning("⚠️ File exceeds 500 MB.")
            else:
                if tab6_upload.name.lower().endswith(".csv"):
                    new_raw = pd.read_csv(tab6_upload)
                else:
                    new_raw = pd.read_excel(tab6_upload, engine="openpyxl")

                set_df_in_session(new_raw)
                st.success("✅ Dataset applied. Dashboard now uses this data.")
                st.write(f"**File Name:** {tab6_upload.name}")
                st.write(f"**File Size:** {size_mb:.2f} MB")
                st.dataframe(st.session_state["df"].head())
                st.info("Go to other tabs to see updated charts.")
        except Exception as e:
            st.error("❌ Upload failed.")
            st.caption(str(e))
    else:
        st.info("You can upload from here or from the sidebar.")
