# +
import folium
import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from folium.plugins import HeatMap
from datetime import datetime

from IPython.display import display
from streamlit_folium import st_folium
from streamlit_folium import folium_static
from streamlit_navigation_bar import st_navbar
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)

import warnings
warnings.filterwarnings('ignore')
# -



st.set_page_config(layout='wide')

df_cleaned = pd.read_csv('cleaned.csv')

# Set data types for specific columns
df_cleaned = df_cleaned.astype({'sex': 'category', 'start_type': 'category', 'end_type': 'category'})

# Parse columns as dates
df_cleaned['start_date'] = pd.to_datetime(df_cleaned['start_date'])
df_cleaned['end_date'] = pd.to_datetime(df_cleaned['end_date'])

# -

df_cleaned.isnull().sum()

#Clean

df_cleaned = df_cleaned.drop(columns='year')

# Ensure necessary columns are in the correct format
df_cleaned['dob'] = pd.to_datetime(df_cleaned['dob'], errors='coerce')
df_cleaned['start_date'] = pd.to_datetime(df_cleaned['start_date'], errors='coerce')
df_cleaned['end_date'] = pd.to_datetime(df_cleaned['end_date'], errors='coerce')

# -----------------------

df_cleaned.info()

#Add bairro
df_cleaned['bairro'] = df_cleaned['household'].str[0:4]

#Convert to numerica
df_cleaned['latitude'] = pd.to_numeric(df_cleaned['latitude'], errors='coerce')

# +
# Identify records still missing latitude
missing_lat = df_cleaned['latitude'].isnull()

# Fill missing latitude using bairro mean
df_cleaned.loc[missing_lat, 'latitude'] = df_cleaned[missing_lat].groupby('bairro')['latitude'].transform(
    lambda x: x.fillna(df_cleaned[df_cleaned['bairro'] == x.name]['latitude'].mean())
)
# -

df_cleaned['latitude'] = df_cleaned['latitude'].clip(lower=-90, upper=90)

#Convert to numerica
df_cleaned['longitude'] = pd.to_numeric(df_cleaned['longitude'], errors='coerce')

# Check for valid longitude range
df_cleaned['longitude'] = df_cleaned['longitude'].clip(lower=-180, upper=180)

# +
# Identify records still missing longitude
missing_lon = df_cleaned['longitude'].isnull()

# Fill missing longitude using bairro mean
df_cleaned.loc[missing_lon, 'longitude'] = df_cleaned[missing_lon].groupby('bairro')['longitude'].transform(
    lambda x: x.fillna(df_cleaned[df_cleaned['bairro'] == x.name]['longitude'].mean())
)

# +
# Example date (you can replace this with other dates if needed)
date = pd.Timestamp('2020-07-01')

# Filter the DataFrame based on the start_date and end_date conditions
filtered_df = df_cleaned.loc[
    lambda x: (x['start_date'] <= date) &
              ((x['end_date'] > date) | x['end_date'].isna())
]
# -

filtered_df['year'] = filtered_df['start_date'].dt.year


# +
# Create the age_days column by calculating the difference between the date and dob in days
filtered_df = filtered_df.assign(
    age_days=lambda y: (date - y['dob']).dt.days
)

# Create the age column by converting age_days into years
filtered_df = filtered_df.assign(
    age=lambda z: (z['age_days'] // 365)  # Integer division to get full years
)

# Display the resulting DataFrame with the new columns
# filtered_df
# -

# ---------------------

# **Add age_days and age**

# +

# Ensure 'dob' is in datetime format
df_cleaned['dob'] = pd.to_datetime(df_cleaned['dob'])

# Example reference date (you can replace this with other dates if needed)
reference_date = pd.Timestamp('2020-07-01')

# Calculate age_days and age based on the reference date
df_cleaned = df_cleaned.assign(
    age_days=lambda x: (reference_date - x['dob']).dt.days,
    age=lambda x: (x['age_days'] // 365)  # Integer division to get full years
)

# Display the updated DataFrame with the new columns
# print(df_cleaned[['perm_id', 'dob', 'start_date', 'end_date', 'age_days', 'age']].head())

# -

# -------------

# **Calculate Total population**

# +
# Ensure that the 'year' column is correctly populated for each record
df_cleaned['year'] = df_cleaned['start_date'].dt.year

# Calculate the total population for each year by applying the same filtering logic
def calculate_population(year):
    # Create a date for July 1st of the given year
    date = pd.Timestamp(f'{year}-07-01')
    # Filter records for the year based on start_date and end_date conditions
    filtered_df = df_cleaned.loc[
        lambda x: (x['start_date'] <= date) &
                  ((x['end_date'] > date) | x['end_date'].isna())
    ]
    # Return the number of unique perm_id for the filtered records
    return filtered_df['perm_id'].nunique()

# Get the unique years from the DataFrame
years = df_cleaned['year'].unique()

# Calculate total population for each year
total_population_per_year = pd.DataFrame({
    'year': years,
    'total_population': [calculate_population(year) for year in years]
})

# Sort the DataFrame by year (optional)
total_population_per_year = total_population_per_year.sort_values('year').reset_index(drop=True)

# Display the result
# total_population_per_year

# -

# ----------------

# ### &#x1F50E; Demographic Overview

# **Streamlit**

# Create tabs
tab1, tab2 = st.tabs(["Demographic Overview", "Survival Analysis"])

# Sidebar filters
with tab1:
    st.sidebar.image('logoCISM.png', use_column_width=True)
    st.sidebar.title('Population')
    st.sidebar.subheader('Filters')

# Year selector
with tab1:
    year_options = total_population_per_year['year'].tolist()
    selected_year = st.sidebar.selectbox('Select Year', options=year_options)

# **Time interval**

# Time interval selector
with tab1:
    time_intervals = ['January - June', 'July - December']
    selected_interval = st.sidebar.selectbox('Select Time Interval', options=time_intervals)

    # Filter data based on the selected year and time interval
    if selected_interval == 'January - June':
        start_date = pd.Timestamp(f'{selected_year}-01-01')
        end_date = pd.Timestamp(f'{selected_year}-06-30')
    else:
        start_date = pd.Timestamp(f'{selected_year}-07-01')
        end_date = pd.Timestamp(f'{selected_year}-12-31')

    # Filter the DataFrame based on the selected year and interval
    filtered_interval_df = df_cleaned.loc[
        (df_cleaned['start_date'] <= end_date) &
        ((df_cleaned['end_date'] >= start_date) | df_cleaned['end_date'].isna())
    ]

# **Household population**

with tab1:
    # Recalculate household population based on the filtered data
    household_population = filtered_interval_df.groupby('household')['perm_id'].count().reset_index()
    household_population.columns = ['household', 'population']

    total_household_population = household_population['population'].sum()

with tab1:
    # Calculate household size
    household_size = filtered_interval_df.groupby('household')['perm_id'].nunique().reset_index()
    household_size.columns = ['household', 'household_size']

    # Merge household size back to the original df_cleaned DataFrame
    df_cleaned = df_cleaned.merge(household_size, on='household', how='left')


# **Total Births**

with tab1:
    total_births = filtered_interval_df[filtered_interval_df['start_type'] == 'BIR'].shape[0]

# **Calculate totals**

with tab1:
    # Filter data for the selected year
    population_for_year = total_population_per_year[total_population_per_year['year'] == selected_year]['total_population'].values[0]
    filtered_year_df = df_cleaned[df_cleaned['year'] == selected_year]

    # Calculate total population for the selected year and interval
    population_for_interval = filtered_interval_df['perm_id'].nunique()

    # Calculate total immigration, outmigration, and deaths for the selected interval
    total_immigration = filtered_interval_df[filtered_interval_df['start_type'] == 'ENT'].shape[0]
    total_outmigration = filtered_interval_df[filtered_interval_df['end_type'] == 'EXT'].shape[0]
    total_deaths = filtered_interval_df[filtered_interval_df['end_type'] == 'DTH'].shape[0]

    # # Calculate total immigration, outmigration, and deaths for the selected year
    # total_immigration = df_cleaned[(df_cleaned['year'] == selected_year) & (df_cleaned['start_type'] == 'ENT')].shape[0]
    # total_outmigration = df_cleaned[(df_cleaned['year'] == selected_year) & (df_cleaned['end_type'] == 'EXT')].shape[0]
    # total_deaths = df_cleaned[(df_cleaned['year'] == selected_year) & (df_cleaned['end_type'] == 'DTH')].shape[0]

    # Display the total population for the selected year
    # Display total population, immigration, outmigration, and deaths
    st.title(f'Demographic Overview for {selected_year}')

    # Enhanced and styled grid display
    st.markdown(f"""
    <div style="display: flex; flex-wrap: wrap; gap: 20px; justify-content: center;">
        <div style="background-color: #ffcccc; padding: 20px; text-align: center; width: 23%; border-radius: 10px; box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);">
            <h3 style="color: black; margin-bottom: 10px;">Total Population</h3>
            <h1 style="color: black; font-size: 24px;">{population_for_year}</h1>
        </div>
        <div style="background-color: #ccffcc; padding: 20px; text-align: center; width: 23%; border-radius: 10px; box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);">
            <h3 style="color: black; margin-bottom: 10px;">Total Immigration</h3>
            <h1 style="color: black; font-size: 24px;">{total_immigration}</h1>
        </div>
        <div style="background-color: #ccccff; padding: 20px; text-align: center; width: 23%; border-radius: 10px; box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);">
            <h3 style="color: black; margin-bottom: 10px;">Total Outmigration</h3>
            <h1 style="color: black; font-size: 24px;">{total_outmigration}</h1>
        </div>
        <div style="background-color: #ffffcc; padding: 20px; text-align: center; width: 23%; border-radius: 10px; box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);">
            <h3 style="color: black; margin-bottom: 10px;">Total Deaths</h3>
            <h1 style="color: black; font-size: 24px;">{total_deaths}</h1>
        </div>
        <div style="background-color: #e6b0aa; padding: 20px; text-align: center; width: 23%; border-radius: 10px; box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);">
            <h3 style="color: black; margin-bottom: 10px;">Total Births</h3>
            <h1 style="color: black; font-size: 24px;">{total_births}</h1>
        </div>
        <div style="background-color: #f5b041; padding: 20px; text-align: center; width: 23%; border-radius: 10px; box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);">
            <h3 style="color: black; margin-bottom: 10px;">Total Household Population</h3>
            <h1 style="color: black; font-size: 24px;">{total_household_population}</h1>
        </div>
    </div>
    """, unsafe_allow_html=True)

# **Line plot for total population**

with tab1:
    # Plot total population over time
    fig = px.line(total_population_per_year, x='year', y='total_population', 
                  title='Total Population Over Time',
                  labels={'year': 'Year', 'total_population': 'Total Population'})
    st.plotly_chart(fig)

# **Overall conversion rate table by gender and age**

with tab1:
    # Prepare data for the conditional formatting
    formatted_data = pd.DataFrame({
        'Metric': ['Total Population', 'Total Immigration', 'Total Outmigration', 'Total Deaths'],
        'Value': [population_for_year, total_immigration, total_outmigration, total_deaths]
    })

    # Convert columns to numeric where possible, invalid parsing will be set as NaN
    formatted_data['Value'] = pd.to_numeric(formatted_data['Value'], errors='coerce')

    # Highlight the maximum value in each column
    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: lightgreen' if v else '' for v in is_max]

    # Apply formatting only to numeric columns
    numeric_cols = formatted_data.select_dtypes(include='number').columns

    styled_df = formatted_data.style.apply(highlight_max, subset=['Value'])\
        .format({col: "{:.0f}" for col in numeric_cols})\
        .set_properties(**{'text-align': 'center'})\
        .set_table_styles([dict(selector='th', props=[('text-align', 'center')])])

    st.write("### Key Metrics for the Selected Year and Interval")
    st.dataframe(styled_df, use_container_width=True)


# **Gender Distribution**

with tab1:
    # Calculate gender distribution
    gender_distribution = filtered_interval_df['sex'].value_counts().reset_index()
    gender_distribution.columns = ['sex', 'count']

# **Pyramid**

with tab1:
    # Update the population pyramid for the selected interval
    age_bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    filtered_interval_df['age_group'] = pd.cut(filtered_interval_df['age'], bins=age_bins, right=False)

    pyramid_data = filtered_interval_df.groupby(['age_group', 'sex']).size().unstack().fillna(0)

    fig_pyramid = go.Figure()

    fig_pyramid.add_trace(go.Bar(
        y=pyramid_data.index.astype(str),
        x=-pyramid_data['F'],  # Negative values for the left side (Female)
        name='Female',
        orientation='h',
        marker=dict(color='lightpink')
    ))

    fig_pyramid.add_trace(go.Bar(
        y=pyramid_data.index.astype(str),
        x=pyramid_data['M'],  # Positive values for the right side (Male)
        name='Male',
        orientation='h',
        marker=dict(color='lightblue')
    ))

    fig_pyramid.update_layout(
        title=f'Population Pyramid for {selected_year} ({selected_interval})',
        xaxis=dict(title='Population', tickvals=[-200, -100, 0, 100, 200], ticktext=[200, 100, 0, 100, 200]),
        yaxis=dict(title='Age Group'),
        barmode='overlay',
        bargap=0.1,
        bargroupgap=0.1,
        plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent plot background
        paper_bgcolor='rgba(0, 0, 0, 0)'  # Transparent paper background
    )

    #Gender Distribution

    # Create a pie chart for gender distribution
    fig_pie = go.Figure(data=[go.Pie(labels=gender_distribution['sex'], values=gender_distribution['count'],
                                    hole=0.3, 
                                    marker=dict(colors=['lightpink','lightblue' ]))])
    fig_pie.update_layout(
        title=f'Gender Distribution for {selected_year} ({selected_interval})',
        paper_bgcolor='rgba(0, 0, 0, 0)'  # Transparent background
    )

    # Create two columns in Streamlit
    col1, col2 = st.columns(2)

    # Display the plots side by side
    with col1:
        st.plotly_chart(fig_pie) 

    with col2:
        st.plotly_chart(fig_pyramid)

# **In and Out migration trend, house population**

with tab1:
    # Step 1: Calculate annual totals for immigration and outmigration
    immigration_trend = df_cleaned[df_cleaned['start_type'] == 'ENT'].groupby('year').size().reset_index(name='total_immigration')
    outmigration_trend = df_cleaned[df_cleaned['end_type'] == 'EXT'].groupby('year').size().reset_index(name='total_outmigration')

    # Step 2: Merge the two trends into one DataFrame
    migration_trend = pd.merge(immigration_trend, outmigration_trend, on='year', how='outer').fillna(0)

    # Step 3: Filter the data to include only years up to the selected year
    migration_trend_filtered = migration_trend[migration_trend['year'] <= selected_year]

    # Step 4: Create a line plot with both trends
    fig_migration_trend = px.line(migration_trend_filtered, x='year', y=['total_immigration', 'total_outmigration'],
                                  labels={'value': 'Total', 'year': 'Year'},
                                  title='Immigration and Outmigration Trend Over Time')

    # Step 5: Customize the plot (optional)
    fig_migration_trend.update_layout(
        xaxis_title='Year',
        yaxis_title='Number of People',
        legend_title='Migration Type',
        plot_bgcolor='rgba(0,0,0,0)'  # Removes background color
    )

    # Household size plot
    # Sort by household size and select top 10
    top_10_household_size = household_size.sort_values(by='household_size', ascending=False).head(10)

    # Create a bar plot showing the top 10 household sizes
    fig_household_size = px.bar(
        top_10_household_size,
        x='household',
        y='household_size',
        labels={'household': 'Household', 'household_size': 'Household Size'},
        title=f'Top 10 Household Sizes for {selected_year} ({selected_interval})',
        color='household_size'
    )

    # Step 6: Display the plots side by side in Streamlit
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(fig_migration_trend)

    with col2:
        st.plotly_chart(fig_household_size)

# **MAP**

with tab1:
    # Step 1: Aggregate population data by latitude and longitude
    # Group by latitude and longitude and calculate the total population for each location
    population_by_location = df_cleaned.groupby(['latitude', 'longitude']).size().reset_index(name='population')

    # Step 2: Create the scatter mapbox plot
    fig = px.scatter_mapbox(
        population_by_location,
        lat='latitude',
        lon='longitude',
        size='population',
        color='population',
        color_continuous_scale='Viridis',  # Choose a color scale
        size_max=30,  # Maximum size of the markers
        zoom=5,  # Adjust the zoom level
        height=600,
        title="Population Distribution by Latitude and Longitude"
    )

    # Update layout for the map
    fig.update_layout(
        mapbox_style="open-street-map",  # Use OpenStreetMap or any other available styles
        geo=dict(
            showland=True,
            landcolor="white",
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # Display the map in Streamlit
    st.plotly_chart(fig)


# --------------------------------------

# ### &#x1F4BB; Machine Learning

# **Survival Analysis**

with tab2:
    #Library
    from lifelines import KaplanMeierFitter, CoxPHFitter
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    df = df_cleaned.copy()

with tab2:
    # Convert dates to datetime objects
    df['start_date'] = pd.to_datetime(df['start_date'])
    df['end_date'] = pd.to_datetime(df['end_date'])
    df['dob'] = pd.to_datetime(df['dob'])

    # Calculate survival time and event observed
    df['survival_time'] = (df['end_date'] - df['start_date']).dt.days
    df['event_observed'] = df['end_type'].apply(lambda x: 1 if x == 'DTH' else 0)

    #Handling missing values
    df = df.dropna(subset=['survival_time', 'event_observed', 'age'])
# **Streamlit**<br>
# Kaplan-Meier Survival Curve:

#Survival Analysis Tab Content
with tab2:
    #st.sidebar.header('Survival Analysis Settings')
    
    # Kaplan-Meier Curve
    st.title('Kaplan-Meier Survival Curve')
   
    kmf = KaplanMeierFitter()
    kmf.fit(df['survival_time'], event_observed=df['event_observed'])

    # Kaplan-Meier Survival Curve using Plotly
    kmf_fig = go.Figure()
    kmf_fig.add_trace(go.Scatter(
        x=kmf.survival_function_.index,
        y=kmf.survival_function_['KM_estimate'],
        mode='lines',
        name='Kaplan-Meier Estimate'
    ))

    kmf_fig.update_layout(
        title="Kaplan-Meier Survival Curve",
        xaxis_title="Time (days)",
        yaxis_title="Survival Probability"
    )

    # Display the Kaplan-Meier plot
    st.plotly_chart(kmf_fig, use_container_width=True)
    st.write('Kaplan-Meier estimator is a useful tool in survival analysis that helps us understand how likely it is that individuals will "survive" over time in the presence of censored data.\n')


# -----------

# **Population Growth Prediction:**

# +
with tab2:
    # Data Preparation
    df_cleaned['start_date'] = pd.to_datetime(df_cleaned['start_date'])
    df_cleaned['end_date'] = pd.to_datetime(df_cleaned['end_date'])

    # Calculate population per bairro per year
    df_population = df_cleaned.groupby(['bairro', 'year']).size().reset_index(name='population')

    # Feature Engineering
    df_population['year'] = pd.to_datetime(df_population['year'], format='%Y')
    df_population['year'] = df_population['year'].dt.year
    df_population['population_last_year'] = df_population.groupby('bairro')['population'].shift(1)
    df_population = df_population.fillna(0)

#     st.write(df_population)  # Display the processed data


# -

with tab2:
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    st.header("Forecasted the population for future years.")

    # Prepare data for modeling
    X = df_population[['year', 'population_last_year']]
    y = df_population['population']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model training
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

with tab2:
    # Future Predictions
    future_years = np.array([[2024, df_population['population'].iloc[-1]],
                             [2025, df_population['population'].iloc[-1]],
                             [2026, df_population['population'].iloc[-1]],
                             [2027, df_population['population'].iloc[-1]],
                             [2028, df_population['population'].iloc[-1]]])
    
    future_predictions = model.predict(future_years)

# +
with tab2:
    # Evaluation
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5

#     st.write(f"MAE: {mae}")
#     st.write(f"MSE: {mse}")
#     st.write(f"RMSE: {rmse}")

# -

with tab2:
    # Plot actual vs predicted
    comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=comparison.index, y=comparison['Actual'], mode='lines+markers', name='Actual'))
#     fig.add_trace(go.Scatter(x=comparison.index, y=comparison['Predicted'], mode='lines+markers', name='Predicted'))
#     st.plotly_chart(fig)

    # Plot future predictions
    future_df = pd.DataFrame({
        'Year': future_years[:, 0],
        'Predicted Population': future_predictions
    })

with tab2:
    fig_future = go.Figure()
    fig_future.add_trace(go.Scatter(x=future_df['Year'], y=future_df['Predicted Population'], mode='lines+markers', name='Future Prediction'))
    st.plotly_chart(fig_future)
    
    st.write('The model forecasts the population for future years within specific neighborhoods ("bairros"). It begins by preparing the data, including converting dates and calculating the population per "bairro" per year. It then engineers features, such as the population from the previous year.') 

    st.write('A Linear Regression model is trained using this data to predict future population sizes. After training, the model is evaluated using mean absolute error (MAE) and root mean squared error (RMSE) on a test dataset.')

    st.write('Finally, the model makes predictions for the population in future years (2024-2028), and the results are plotted to show both the actual vs. predicted values and the future population predictions.')

# ------------------

# **Churn Prediction**<br>
# Predict whether individuals will migrate out of a bairro or change households.

with tab2:
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.preprocessing import LabelEncoder
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, roc_auc_score


# +
# # Assuming df_cleaned is available and preloaded
# df_churn = df_cleaned.copy()

# def load_data():
#     # Feature Selection
#     features = ['start_date', 'end_date', 'start_type', 'end_type', 'household', 'bairro', 'age', 'sex']
#     df_churn = df_cleaned[features]

#     # Target variable
#     df_churn['churn'] = np.where(df_churn['end_date'].notnull(), 1, 0)

#     # Feature Engineering: Calculate duration of stay
#     df_churn['duration'] = (df_churn['end_date'] - df_churn['start_date']).dt.days
#     df_churn['duration'] = df_churn['duration'].fillna(0)

#     # Handling categorical variables with Label Encoding
#     le = LabelEncoder()
#     df_churn['start_type'] = le.fit_transform(df_churn['start_type'])
#     df_churn['end_type'] = le.fit_transform(df_churn['end_type'])
#     df_churn['household'] = le.fit_transform(df_churn['household'])
#     df_churn['bairro'] = le.fit_transform(df_churn['bairro'])
#     df_churn['sex'] = le.fit_transform(df_churn['sex'])

#     return df_churn

# # Function to define features and target
# def define_features_and_target(df_churn):
#     X = df_churn[['start_type', 'end_type', 'household', 'bairro', 'age', 'sex', 'duration']]
#     y = df_churn['churn']
#     return X, y    

# with tab2:
#     st.title('Churn Prediction App')

#     # Load and prepare data
#     df_churn = load_data()
#     X, y = define_features_and_target(df_churn)

#     # Sidebar for model selection
#     model_type = st.sidebar.selectbox('Select Model', ('Logistic Regression', 'Random Forest'))

#     st.write(f'You selected: {model_type}')

#     # Split the data
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#     # Train the model based on user selection
#     if model_type == 'Logistic Regression':
#         model = LogisticRegression(random_state=42)
#     elif model_type == 'Random Forest':
#         model = RandomForestClassifier(random_state=42)

#     if st.button('Train Model'):
#         with st.spinner('Training the model...'):
#             model.fit(X_train, y_train)
#             y_pred = model.predict(X_test)
#             accuracy = accuracy_score(y_test, y_pred)
#             report = classification_report(y_test, y_pred)
#             roc_auc = roc_auc_score(y_test, y_pred)

#             st.success('Model trained successfully!')
#             st.write(f'**Accuracy:** {accuracy}')
#             st.write(f'**ROC-AUC:** {roc_auc}')
#             st.text('**Classification Report:**')
#             st.text(report)

# +
# '''
# Cross-validation is a technique where the dataset is split into multiple folds. 
# The model is trained on some folds and tested on the remaining fold.
# '''
# with tab2:
#     from sklearn.model_selection import cross_val_score

#     # Define the model (Random Forest for example)
#     model = RandomForestClassifier(random_state=42)

#     # Perform 5-fold cross-validation
#     cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

#     # Output the average accuracy and standard deviation
#     st.write(f'Cross-Validation Accuracy: {cv_scores.mean()}')
#     st.write(f'Standard Deviation: {cv_scores.std()}\n')
# -


# ----------------

# **Age Prediction**<br>
# Predict the age of individuals based on their demographics and location.<br>
# Data Preprocessing:

# +
with tab2:
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    import lightgbm as lgb
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import train_test_split
    
    df_age = df_cleaned.copy()

with tab2:
    # Feature selection
    features = ['dob', 'sex', 'start_date', 'end_date', 'household', 'bairro', 'latitude', 'longitude', 'year']
    X = df_age[features]
    y = df_age['age']

with tab2:
    # Handle missing values and preprocess categorical data
    # Convert 'dob', 'start_date', 'end_date' to datetime
    X['dob'] = pd.to_datetime(X['dob'])
    X['start_date'] = pd.to_datetime(X['start_date'])
    X['end_date'] = pd.to_datetime(X['end_date'])

    # Feature engineering: Calculate age from 'dob' and 'year' using the difference in days and converting to years
#     X['calculated_age'] = X.apply(lambda row: (pd.to_datetime(f'{row["year"]}-12-31') - row['dob']).days // 365, axis=1)
    X['calculated_age'] = X['year'] - X['dob'].dt.year
    
    # Define preprocessing for categorical and numerical data
    categorical_features = ['sex', 'bairro']
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    numerical_features = ['latitude', 'longitude', 'calculated_age']
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean'))])

with tab2:
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)])

    # Preprocess data
    X_preprocessed = preprocessor.fit_transform(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

# **Model Building**
# We'll train a LightGBM model on the preprocessed data:

with tab2:
    # Train a LightGBM model
    model = lgb.LGBMRegressor(n_estimators=50, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f'Mean Absolute Error: {mae:.2f} years')

with tab2:
    import streamlit as st
    from datetime import datetime

    # Streamlit App
    st.title("Age Prediction App")

    # User inputs
    dob = st.date_input("Date of Birth")
    sex = st.selectbox("Sex", ["Male", "Female"])
    start_date = st.date_input("Household Start Date")
    end_date = st.date_input("Household End Date", value=None)
    household = st.text_input("Household ID")
    bairro = st.selectbox("Bairro", ["Bairro A", "Bairro B", "Bairro C"])
    latitude = st.number_input("Latitude")
    longitude = st.number_input("Longitude")
    year = st.number_input("Year", min_value=1900, max_value=2100)

    # Preprocess the inputs
    input_data = pd.DataFrame({
        'dob': [pd.to_datetime(dob)],
        'sex': [sex],
        'start_date': [pd.to_datetime(start_date)],
        'end_date': [pd.to_datetime(end_date) if end_date else None],
        'household': [household],
        'bairro': [bairro],
        'latitude': [latitude],
        'longitude': [longitude],
        'year': [year]
    })

    # Feature engineering: Calculate age from 'dob' and 'year' using the difference in days and converting to years
#     input_data['calculated_age'] = input_data.apply(lambda row: (pd.to_datetime(f'{row["year"]}-12-31') - row['dob']).days // 365, axis=1)
    input_data['calculated_age'] = input_data['year'] - input_data['dob'].dt.year
    
    
    # Preprocess input data
    input_data_preprocessed = preprocessor.transform(input_data)

    # Make prediction
    predicted_age = model.predict(input_data_preprocessed)

    # Display the prediction
    st.write(f"Predicted Age: {predicted_age[0]:.2f} years")
    
    st.write('The code builds and deploys an age prediction model using LightGBM within a Streamlit app. It preprocesses data by handling missing values, encoding categorical features, and calculating age. The model is trained to predict age based on demographic and location data, and its accuracy is evaluated. The Streamlit app allows users to input their details, processes this data, and predicts their age, displaying the result in real-time.')
    st.write('LightGBM (Light Gradient Boosting Machine) is an open-source, high-performance framework for gradient boosting that is designed to be highly efficient and scalable. ')
# -
# --------------------

# **Birth Prediction**<br>
# Objective: Predict the likelihood of births occurring within certain time frames or areas.
#
# Features: sex, dob, household, bairro, latitude, longitude, year.
#
# Target: Birth status (whether a birth occurs in the specified time frame/location).


df_birth = df_cleaned.copy()

with tab2:
    # Feature selection
    features = ['sex', 'dob', 'household', 'bairro', 'latitude', 'longitude', 'year']
    X = df_birth[features]
    y = df_birth['start_type'].apply(lambda x: 1 if x == 'BIR' else 0)

with tab2:
    # Data preprocessing
    X['dob'] = pd.to_datetime(X['dob'])
    
    # Feature engineering: Calculate age from 'dob' and 'year'\
    # Feature engineering: Calculate age from 'dob' and 'year'
#     X['calculated_age'] = X.apply(lambda row: (pd.to_datetime(f'{row["year"]}-12-31') - row['dob']).days // 365, axis=1)
    X['calculated_age'] = X['year'] - X['dob'].dt.year
    
    # Define preprocessing for categorical and numerical data
    categorical_features = ['sex', 'bairro']
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    numerical_features = ['latitude', 'longitude', 'calculated_age']
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean'))])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)])

    # Preprocess data
    X_preprocessed = preprocessor.fit_transform(X)

with tab2:
    # Preprocess data
    X_preprocessed = preprocessor.fit_transform(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

    # Train a LightGBM model
    model = lgb.LGBMClassifier(n_estimators=50, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Model Accuracy: {accuracy:.2f}")

    # Streamlit App
    st.title("Birth Prediction App")

    # User inputs
    # User inputs with unique keys
    dob = st.date_input("Date of Birth", key='birth_dob')
    sex = st.selectbox("Sex", ["Male", "Female"], key='birth_sex')
    household = st.text_input("Household ID", key='birth_household')
    bairro = st.selectbox("Bairro", ["Bairro A", "Bairro B", "Bairro C"], key='birth_bairro')
    latitude = st.number_input("Latitude", key='birth_latitude')
    longitude = st.number_input("Longitude", key='birth_longitude')
    year = st.number_input("Year", min_value=1900, max_value=2100, key='birth_year')

    # Preprocess the inputs
    input_data = pd.DataFrame({
        'dob': [pd.to_datetime(dob)],
        'sex': [sex],
        'household': [household],
        'bairro': [bairro],
        'latitude': [latitude],
        'longitude': [longitude],
        'year': [year]
    })

    # Feature engineering
#     input_data['calculated_age'] = input_data.apply(lambda row: (pd.to_datetime(f'{row["year"]}-12-31') - row['dob']).days // 365, axis=1)
    input_data['calculated_age'] = input_data['year'] - input_data['dob'].dt.year
    
    # Preprocess input data
    input_data_preprocessed = preprocessor.transform(input_data)

    # Make prediction
    birth_likelihood = model.predict_proba(input_data_preprocessed)[:, 1]

    # Display the prediction
    st.write(f"Predicted Birth Likelihood: {birth_likelihood[0]:.2f}")
    
    st.write('The code builds a birth prediction model using LightGBM, which is deployed in a Streamlit app. It begins by selecting relevant features like sex, dob, household, bairro, latitude, longitude, and year. The target variable is whether the event is a birth (start_type), which is converted into a binary outcome.\n')
    
    st.write('The results indicate that the model has high accuracy (96%) overall, primarily due to its excellent performance in predicting the majority class (0). Specifically:')

    st.write('Class 0 (Non-birth events): The model performs exceptionally well with a precision of 96%, recall of 100%, and an F1-score of 98%. This means it correctly identifies almost all non-birth events.')

    st.write('Class 1 (Birth events): The model struggles with the minority class, showing a precision of 67%, but a very low recall of 4% and an F1-score of 8%. This indicates that while the model is fairly precise when it predicts a birth event, it fails to identify most actual birth events.')

    st.write('Confusion Matrix: The model correctly classifies 113,505 non-birth events and 218 birth events but misclassifies 5,190 birth events as non-birth.')

    st.write("Macro Average: The average performance across both classes is moderate, with an F1-score of 53%, reflecting the model's difficulty with the minority class.")

    st.write("Weighted Average: The overall weighted performance is strong, with an F1-score of 94%, heavily influenced by the model's success in predicting the majority class.")

# **Death Prediction**

with tab2:
    from sklearn.metrics import classification_report, confusion_matrix
  
    # Load and prepare the dataset
    df_death = df_cleaned.copy()
    df_death['end_type'] = df_death['end_type'].fillna('EXT')
    df_death['death_event'] = df_death['end_type'].apply(lambda x: 1 if x == 'DTH' else 0)

    # Select features and target
    features = ['age', 'sex', 'latitude', 'longitude', 'household_size', 'bairro']
    X = df_death[features]
    y = df_death['death_event']

    # Fill missing values
    num_cols = ['age', 'latitude', 'longitude', 'household_size']
    for col in num_cols:
        X[col] = X[col].fillna(X[col].median())

    cat_cols = ['sex', 'bairro']
    for col in cat_cols:
        X[col] = X[col].fillna(X[col].mode()[0])

    # Encode categorical variables using OneHotEncoder
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    X = pd.get_dummies(X, columns=['sex', 'bairro'], drop_first=True)

    # Scale features
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)

    # Evaluate the model
    y_pred = log_reg.predict(X_test)
    st.write("Model Performance Metrics")
    st.text(classification_report(y_test, y_pred))
    st.text(confusion_matrix(y_test, y_pred))

    # Streamlit App for Death Prediction
    st.title('Death Prediction App')

    age = st.number_input('Age', min_value=0, key='death_age')
    sex = st.selectbox('Sex', ['Male', 'Female'], key='death_sex')
    latitude = st.number_input('Latitude', key='death_latitude')
    longitude = st.number_input('Longitude', key='death_longitude')
    household_size = st.number_input('Household Size', min_value=1, key='key_household_size')
    unique_bairros = df_death['bairro'].unique()
    bairro = st.selectbox('Bairro', unique_bairros, key='death_bairro')

    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'latitude': [latitude],
        'longitude': [longitude],
        'household_size': [household_size],
        'bairro': [bairro]
    })

    # Preprocess input data using OneHotEncoder
    input_data = pd.get_dummies(input_data, columns=['sex', 'bairro'], drop_first=True)
    input_data = input_data.reindex(columns=X_train.columns, fill_value=0)

    # Make a prediction
    if st.button('Predict'):
        prediction = log_reg.predict(input_data)
        if prediction[0] == 1:
            st.write('Prediction: Death Event')
        else:
            st.write('Prediction: No Death Event')
            
    st.write('The model primarily helps in predicting the risk of death events, using logistic regression to provide binary outcomes (death event or no death event).')



# !streamlit run Dashboard.py

# +
# # Assuming df_cleaned['start_date'] is already in datetime format
# start_date_condition = df_cleaned['start_date'] == pd.Timestamp('2020-01-01')
# start_type_condition = df_cleaned['start_type'] == 'BIR'

# see = df_cleaned[start_date_condition & start_type_condition]
# # see

# +
# df_cleaned[df_cleaned.start_date > '2020-01-01']

# +
# # Filter data for the selected year
# filtered_data = df_cleaned[df_cleaned['year'] == selected_year]
# +
# # Create the Streamlit app
# st.title('Total Population Over Time')

# # Plot the total population per year
# fig = px.line(total_population_per_year, x='year', y='total_population', 
#               title='Total Population Over Time',
#               labels={'year': 'Year', 'total_population': 'Total Population'})

# # Display the plot
# st.plotly_chart(fig)

# +
# # Calculate metrics based on the selected year
# total_population = calculate_population(year_selected)
# total_immigration = calculate_immigration(year_selected)
# total_outmigration = calculate_outmigration(year_selected)
# total_deaths = calculate_deaths(year_selected)

# +
# date = pd.Timestamp('2020-07-01')

# +
# df_cleaned.loc[lambda x: (x['start_date'] <= date) &
#                        ((x['end_date'] > date) | x['end_date'].isna())]
# -



# +
# # Calculate the total population, inmigration, outmigration, and deaths for the selected year
# total_population = filtered_data['perm_id'].nunique()
# total_inmigration = df_cleaned[(df_cleaned['start_date'].dt.year == selected_year) & (df_cleaned['start_type'] == 'ENT')].shape[0]
# total_outmigration = df_cleaned[(df_cleaned['end_date'].dt.year == selected_year) & (df_cleaned['end_type'] == 'EXT')].shape[0]
# total_deaths = df_cleaned[(df_cleaned['end_date'].dt.year == selected_year) & (df_cleaned['end_type'] == 'DTH')].shape[0]
# -



