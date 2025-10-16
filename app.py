import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(
    page_title="Interactive Diagnosis Dashboard",
    page_icon="📊",
    layout="wide"
)

# --- Data Loading ---
@st.cache_data
def load_data(filepath):
    """Loads the base data from a CSV file."""
    try:
        df = pd.read_csv(filepath)
        # Handle potential mixed types or missing values in diagnosis column
        if 'diagnosis' in df.columns:
            df['diagnosis'] = pd.to_numeric(df['diagnosis'], errors='coerce')
        return df
    except FileNotFoundError:
        st.error(f"Error: The file '{filepath}' was not found. Please make sure it's in the same folder as the app.py script.")
        return None

# --- Helper Function for Calculations ---
def posterior_prob(log_odds: pd.Series) -> pd.Series:
    """Converts a pandas Series of log-odds to posterior probabilities."""
    return 1 / (1 + np.exp(-log_odds))

# --- Main Application ---

# Load the base data
df = load_data("dashboard_data.csv")

if df is not None:
    # --- Sidebar for Controls ---
    st.sidebar.header("📊 Dashboard Controls")

    # --- Axis Selection for Plot (in sidebar) ---
    st.sidebar.markdown("#### Plot Controls")
    axis_options = [
        'diag_marker_kids', 'diag_marker_adults', 'diag_marker_no_msa', 'attention_marker', 'impulsivity_marker',
        'pupil_score', 'synchrony_score', 'kids_msa_score', 'adults_msa_score', 'attention_score'
    ]

    # Default thresholds
    thresholds = {
        'diag_marker_kids': {'low': 0.34, 'high': 0.64},
        'diag_marker_adults': {'low': 0.34, 'high': 0.64},
        'attention_marker': {'low': 0.44, 'high': 0.51},
        'impulsivity_marker': {'low': 0.31, 'high': 0.64}
    }

    def get_thresholds_for_marker(marker_key):
        """Returns the specific thresholds for a marker, or default values if not defined."""
        default = {'low': 0.33, 'high': 0.66}
        return thresholds.get(marker_key, default)

    # Function to reset/update all thresholds in session state
    def update_thresholds():
        x_axis_key = st.session_state.get('x_axis_select')
        y_axis_key = st.session_state.get('y_axis_select')
        
        x_thresh = get_thresholds_for_marker(x_axis_key)
        st.session_state.x_low = x_thresh['low']
        st.session_state.x_high = x_thresh['high']
        
        y_thresh = get_thresholds_for_marker(y_axis_key)
        st.session_state.y_low = y_thresh['low']
        st.session_state.y_high = y_thresh['high']

    # --- Individual Reset Functions ---
    def reset_x_thresholds():
        x_axis_key = st.session_state.get('x_axis_select')
        x_thresh = get_thresholds_for_marker(x_axis_key)
        st.session_state.x_low = x_thresh['low']
        st.session_state.x_high = x_thresh['high']

    def reset_y_thresholds():
        y_axis_key = st.session_state.get('y_axis_select')
        y_thresh = get_thresholds_for_marker(y_axis_key)
        st.session_state.y_low = y_thresh['low']
        st.session_state.y_high = y_thresh['high']


    # Select axes, with a callback to update thresholds when the selection changes
    x_axis = st.sidebar.selectbox("Select X-Axis", options=axis_options, index=1, key='x_axis_select', on_change=update_thresholds)
    y_axis = st.sidebar.selectbox("Select Y-Axis", options=axis_options, index=3, key='y_axis_select', on_change=update_thresholds)
    
    # Add Z-axis selector
    z_axis_options = ["None"] + axis_options
    z_axis = st.sidebar.selectbox("Select Z-Axis (Optional)", options=z_axis_options, index=0)
    
    st.sidebar.markdown("---")

    # Sliders for prior probabilities
    prior_diag = st.sidebar.slider(
        "Prior Probability (Diagnosis)", 0.0, 1.0, 0.5, 0.01
    )
    prior_imp = st.sidebar.slider(
        "Prior Probability (Impulsivity)", 0.0, 1.0, 0.5, 0.01
    )

    # --- Gender Selection ---
    st.sidebar.markdown("#### Select Gender")
    selected_gender = st.sidebar.radio(
        "Filter by Gender",
        ["All", "Female", "Male"],
        key="gender_radio"
    )

    # --- Checkboxes for Source Selection ---
    st.sidebar.markdown("#### Select Sources")
    unique_sources = df['source_file'].dropna().unique()
    selected_sources = []
    for source in unique_sources:
        # Use a unique key for each checkbox to maintain state correctly
        if st.sidebar.checkbox(source, value=True, key=f"source_{source}"):
            selected_sources.append(source)

    # --- Checkboxes for Diagnosis Type Selection ---
    st.sidebar.markdown("#### Select Diagnosis Types")
    
    # Map diagnosis codes to descriptive names
    diagnosis_name_map = {
        1: "Unknown",
        2: "No Diagnosis Made Yet",
        3: "Healthy Control",
        4: "Diagnosed Patient",
        5: "Suspicious"
    }

    unique_diagnoses = sorted(df[df['diagnosis'].notna()]['diagnosis'].unique())
    selected_diagnoses = []
    for diag in unique_diagnoses:
        # Get the descriptive name for the checkbox label
        label = diagnosis_name_map.get(int(diag), f"Type {int(diag)}")
        # Use a unique key for each checkbox
        if st.sidebar.checkbox(label, value=True, key=f"diag_{diag}"):
            selected_diagnoses.append(diag)

    # --- Age Range Filter ---
    st.sidebar.markdown("#### Select Age Range")
    min_age = int(df['age'].min())
    max_age = int(df['age'].max())
    selected_age = st.sidebar.slider(
        "Filter by Age",
        min_value=min_age,
        max_value=max_age,
        value=(min_age, max_age)
    )

    # --- Flag Status Filter (Moved) ---
    st.sidebar.markdown("#### Select Flag Status")
    selected_flag = st.sidebar.radio(
        "Filter by Flag",
        ["All", "True", "False"],
        key="flag_radio"
    )
    st.sidebar.caption("Flag is True: No commission errors, trials with fastest reaction times were used.")
    
    # Initialize session state for thresholds if they don't exist
    if 'x_low' not in st.session_state:
        update_thresholds()

    # --- Data Filtering ---
    # Create masks for each filter criteria
    source_mask = df['source_file'].isin(selected_sources)
    diagnosis_mask = df['diagnosis'].isin(selected_diagnoses)
    age_mask = (df['age'] >= selected_age[0]) & (df['age'] <= selected_age[1])

    # Apply gender filter (assuming 1 is Female, 0 is Male)
    if selected_gender == "Female":
        gender_mask = (df['gender'] == 1)
    elif selected_gender == "Male":
        gender_mask = (df['gender'] == 0)
    else: # 'All'
        gender_mask = pd.Series(True, index=df.index)

    # Apply flag filter
    if selected_flag == "True":
        flag_mask = (df['flag'] == True)
    elif selected_flag == "False":
        flag_mask = (df['flag'] == False)
    else: # 'All'
        flag_mask = pd.Series(True, index=df.index)

    # Combine masks and create the filtered dataframe
    filtered_df = df[source_mask & diagnosis_mask & gender_mask & age_mask & flag_mask].copy()


    # --- On-the-Fly Calculations ---
    if not filtered_df.empty:
        # Calculate log priors based on slider inputs
        epsilon = 1e-9
        log_prior_odds_diag = np.log((prior_diag + epsilon) / (1 - prior_diag + epsilon))
        log_prior_odds_imp = np.log((prior_imp + epsilon) / (1 - prior_imp + epsilon))

        # Perform all marker calculations on the filtered data
        logLR_imp = filtered_df['synchrony_logLR']
        
        # Corrected Diagnosis Marker Calculations based on user feedback
        log_prior_pupil_imp = log_prior_odds_diag + filtered_df['pupil_logLR'] + logLR_imp

        log_posterior_diag_kids = log_prior_pupil_imp + filtered_df['kids_msa_logLR']
        log_posterior_diag_adults = log_prior_pupil_imp + filtered_df['adults_msa_logLR']
        # For the 'no msa' marker, the posterior is just the prior with pupil and impulsivity
        log_posterior_diag_without_msa = log_prior_pupil_imp

        filtered_df['diag_marker_kids'] = posterior_prob(log_posterior_diag_kids)
        filtered_df['diag_marker_adults'] = posterior_prob(log_posterior_diag_adults)
        filtered_df['diag_marker_no_msa'] = posterior_prob(log_posterior_diag_without_msa)

        # Attention Marker (un-changed)
        log_posterior_att = log_prior_odds_diag + filtered_df['attention_logLR']
        filtered_df['attention_marker'] = posterior_prob(log_posterior_att)

        # Impulsivity Marker (un-changed)
        log_posterior_imp = log_prior_odds_imp + logLR_imp
        filtered_df['impulsivity_marker'] = posterior_prob(log_posterior_imp)


        # --- Main Page Display ---
        st.title("Analysis of different models performances for Adults")
        
        # --- Create columns for Plot and its Controls ---
        plot_col, control_col = st.columns([3, 1])

        with plot_col:
            
            filtered_df['diagnosis_str'] = filtered_df['diagnosis'].astype(str)
            
            color_map = {
                '4.0': 'red', '3.0': 'green', '5.0': 'pink',
                '1.0': 'lightblue', '2.0': 'cyan'
            }
            
            # Define the columns to show on hover
            hover_info = [
                'file_id', 'source_file', 'diag_marker_kids', 'diag_marker_adults', 
                'diag_marker_no_msa', 'attention_marker', 'impulsivity_marker'
            ]
            
            # --- Conditional Plotting: 2D or 3D ---
            if z_axis == "None":
                st.header(f"{y_axis.replace('_', ' ').title()} vs. {x_axis.replace('_', ' ').title()}")
                fig = px.scatter(
                    filtered_df, x=x_axis, y=y_axis, color='diagnosis_str',
                    hover_data=hover_info, color_discrete_map=color_map,
                )

                # Add Threshold Lines using values from session state
                t_low_x, t_high_x = st.session_state.x_low, st.session_state.x_high
                fig.add_vline(x=t_low_x, line_dash="dash", line_color="grey", annotation_text=f"Low: {t_low_x:.2f}")
                fig.add_vline(x=t_high_x, line_dash="dash", line_color="grey", annotation_text=f"High: {t_high_x:.2f}")
            
                t_low_y, t_high_y = st.session_state.y_low, st.session_state.y_high
                fig.add_hline(y=t_low_y, line_dash="dash", line_color="grey", annotation_text=f"Low: {t_low_y:.2f}")
                fig.add_hline(y=t_high_y, line_dash="dash", line_color="grey", annotation_text=f"High: {t_high_y:.2f}")

                fig.update_xaxes(range=[0, 1])
                fig.update_yaxes(range=[0, 1])
            
            else: # If a Z-axis is selected, create a 3D plot
                st.header(f"{z_axis.replace('_', ' ').title()} vs. {y_axis.replace('_', ' ').title()} vs. {x_axis.replace('_', ' ').title()}")
                fig = px.scatter_3d(
                    filtered_df, x=x_axis, y=y_axis, z=z_axis, color='diagnosis_str',
                    hover_data=hover_info, color_discrete_map=color_map,
                )
                fig.update_layout(scene = dict(
                    xaxis = dict(range=[0,1]),
                    yaxis = dict(range=[0,1]),
                    zaxis = dict(range=[0,1]),)
                )

            st.plotly_chart(fig, use_container_width=True)

        # --- Threshold Controls moved to the main area, next to the plot ---
        # These controls only affect the 2D plot but are always visible for simplicity
        with control_col:
            st.markdown("##### Adjust Thresholds")
            
            st.slider("X-Axis Low", 0.0, 1.0, key='x_low')
            st.slider("X-Axis High", 0.0, 1.0, key='x_high')
            st.button("Reset X", on_click=reset_x_thresholds, key='reset_x')
            
            st.markdown("---")

            st.slider("Y-Axis Low", 0.0, 1.0, key='y_low')
            st.slider("Y-Axis High", 0.0, 1.0, key='y_high')
            st.button("Reset Y", on_click=reset_y_thresholds, key='reset_y')

        # --- Marker Statistics Section ---
        st.header("Marker Statistics")
        diagnoses_in_data = sorted(filtered_df['diagnosis_str'].unique())
        stat_col1, stat_col2 = st.columns(2)

        # Calculate stats for X-axis using session state thresholds
        with stat_col1:
            st.subheader(f"{x_axis.replace('_', ' ').title()}")
            t = {'low': st.session_state.x_low, 'high': st.session_state.x_high}
            stats_list = []
            for diag_str in diagnoses_in_data:
                diag_code = int(float(diag_str))
                diag_name = diagnosis_name_map.get(diag_code, f"Type {diag_code}")
                diag_df = filtered_df[filtered_df['diagnosis_str'] == diag_str]
                below = diag_df[diag_df[x_axis] < t['low']].shape[0]
                between = diag_df[(diag_df[x_axis] >= t['low']) & (diag_df[x_axis] <= t['high'])].shape[0]
                above = diag_df[diag_df[x_axis] > t['high']].shape[0]
                stats_list.append({
                    "Diagnosis": diag_name, "Below (Healthy)": below,
                    "Between (Not Sure)": between, "Above (Patient)": above
                })
            if stats_list:
                stats_df = pd.DataFrame(stats_list).set_index("Diagnosis")
                stats_df.loc['Total'] = stats_df.sum()
                st.dataframe(stats_df)

        # Calculate stats for Y-axis using session state thresholds
        with stat_col2:
            st.subheader(f"{y_axis.replace('_', ' ').title()}")
            t = {'low': st.session_state.y_low, 'high': st.session_state.y_high}
            stats_list = []
            for diag_str in diagnoses_in_data:
                diag_code = int(float(diag_str))
                diag_name = diagnosis_name_map.get(diag_code, f"Type {diag_code}")
                diag_df = filtered_df[filtered_df['diagnosis_str'] == diag_str]
                below = diag_df[diag_df[y_axis] < t['low']].shape[0]
                between = diag_df[(diag_df[y_axis] >= t['low']) & (diag_df[y_axis] <= t['high'])].shape[0]
                above = diag_df[diag_df[y_axis] > t['high']].shape[0]
                stats_list.append({
                    "Diagnosis": diag_name, "Below (Healthy)": below,
                    "Between (Not Sure)": between, "Above (Patient)": above
                })
            if stats_list:
                stats_df = pd.DataFrame(stats_list).set_index("Diagnosis")
                stats_df.loc['Total'] = stats_df.sum()
                st.dataframe(stats_df)

        with st.expander("View Filtered Data and Calculated Markers"):
            st.dataframe(filtered_df.drop(columns=['diagnosis_str']))
    else:
        st.warning("No data matches the current filter settings.")

