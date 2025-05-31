import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import io
import base64

# ==============================
# Database Setup
# ==============================
conn = sqlite3.connect("DataA.db")
c = conn.cursor()
c.execute('''
CREATE TABLE IF NOT EXISTS inputs (
    section TEXT,
    input_value TEXT
)
''')
c.execute('''
CREATE TABLE IF NOT EXISTS outputs (
    section TEXT,
    output_value TEXT
)
''')
conn.commit()

# ==============================
# Streamlit App UI
# ==============================
st.set_page_config(page_title="AI Trends App", layout="wide")
st.markdown("""
    <style>
    .main {
        background-color: hsla(169, 53%, 25%, 0.8));
        color: black;
    }
    .stApp {
        background-color: hsla(169, 53%, 34%, 0.4);
        color: black;
    }
     h1, h2, h3, h4, h5, h6, p, span, div, label {
        color: black;
    }
    section[data-testid="stSidebar"] {
        background-color: hsla(169, 29%, 27%, 1);
        color: white;
        border-right: 2px solid #ccc;
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    </style>
""", unsafe_allow_html=True)

st.title(" AI Trends Analysis & Classification & Clustering")
menu = st.sidebar.radio("Select section", ["Data Analysis", "Classification", "Clustering", "Database Viewer"])

# ==============================
# Load Data analysis file
# ==============================
df = pd.read_csv('/content/drive/My Drive/AI project/updated_file.csv')

# ==============================
# Data Analysis Section
# ==============================

if menu == "Data Analysis":
    st.subheader("📊 Custom analytics by category")

    analysis_option = st.selectbox("Select the category you want to analyze:", [
        "📈 AI Trends Over Time",
        "📈 AI_Trend Predictions",
        "📈 Trend Evolution",
        "📊 Domain Trends",
        "📌 Top 5 Domains",
        "📊 Filtered Trends",
        "🏢 Organization Analysis",
        "🌍 Country Analysis"
    ])

    if analysis_option == "📈 AI Trends Over Time":
        df = df[df['AI_Trend'] != 'Other']
        trend_by_year = df.groupby(['Year', 'AI_Trend']).size().reset_index(name='Model_Count')
        AI_Trends_Frequency_df = df['AI_Trend'].value_counts().reset_index()
        AI_Trends_Frequency_df.columns = ['AI_Trend', 'Frequency']
        top_5_trends = AI_Trends_Frequency_df.nlargest(5, columns='Frequency')['AI_Trend'].tolist()
        filtered_top_5 = trend_by_year[trend_by_year['AI_Trend'].isin(top_5_trends)]

        sns.set(style="whitegrid")
        fig, ax = plt.subplots(figsize=(14, 7))
        sns.lineplot(data=filtered_top_5, x='Year', y='Model_Count', hue='AI_Trend', marker='o', ax=ax)
        ax.set_title('Evolution of Top 5 AI Trends Over Years')
        st.pyplot(fig)

    elif analysis_option == "📊 Domain Trends":
        df_clean = df.dropna(subset=['Domain'])
        df_exploded = df_clean.assign(Domain=df_clean['Domain'].str.split(',')).explode('Domain')
        df_exploded['Domain'] = df_exploded['Domain'].str.strip()
        domain_trends = df_exploded.groupby(['Year', 'Domain']).size().unstack(fill_value=0)
        st.dataframe(domain_trends.tail())

    elif analysis_option == "📌 Top 5 Domains":
        df_clean = df.dropna(subset=['Domain'])
        df_exploded = df_clean.assign(Domain=df_clean['Domain'].str.split(',')).explode('Domain')
        df_exploded['Domain'] = df_exploded['Domain'].str.strip()
        domain_trends = df_exploded.groupby(['Year', 'Domain']).size().unstack(fill_value=0)
        top_domains = domain_trends.sum().sort_values(ascending=False).head(5).index

        sns.set(style="whitegrid")
        fig, ax = plt.subplots(figsize=(14, 6))
        domain_trends[top_domains].plot(marker='o', linewidth=1, ax=ax)
        ax.set_title("Trends in the Top 5 Most Popular Domains (2021–2025)")
        st.pyplot(fig)

    elif analysis_option == "📈 Trend Evolution":
        trend_by_year = df.groupby(['Year', 'AI_Trend']).size().reset_index(name='Model_Count')
        all_trends = df['AI_Trend'].unique()
        filtered = trend_by_year[trend_by_year['AI_Trend'].isin(all_trends)]
        for trend in all_trends:
            fig, ax = plt.subplots(figsize=(10, 5))
            data = filtered[filtered['AI_Trend'] == trend]
            sns.lineplot(data=data, x='Year', y='Model_Count', marker='o', ax=ax)
            ax.set_title(f'Evolution of {trend} Over Years')
            st.pyplot(fig)

    elif analysis_option == "📊 Filtered Trends":
        trend_by_year = df.groupby(['Year', 'AI_Trend']).size().reset_index(name='Model_Count')
        AI_Trends_Frequency_df = df['AI_Trend'].value_counts().reset_index()
        AI_Trends_Frequency_df.columns = ['AI_Trend', 'Frequency']
        top_5_trends = AI_Trends_Frequency_df.nlargest(5, columns='Frequency')['AI_Trend'].tolist()
        filtered_trends_2021_2025 = trend_by_year[
            (trend_by_year['AI_Trend'].isin(top_5_trends)) &
            (trend_by_year['Year'] >= 2021) &
            (trend_by_year['Year'] <= 2025)
        ].copy()
        st.dataframe(filtered_trends_2021_2025)

    elif analysis_option == "📈 AI_Trend Predictions":
# Cleaning the two main columns
        df['AI_Trend'] = df['AI_Trend'].str.strip()
        df['Year'] = df['Year'].astype(int)

# Collect the number of models for each trend by year
        trend_by_year = df.groupby(['Year', 'AI_Trend']).size().reset_index(name='Model_Count')

# Extract the 5 most popular trends
        AI_Trends_Frequency_df = df['AI_Trend'].value_counts().reset_index()
        AI_Trends_Frequency_df.columns = ['AI_Trend', 'Frequency']
        top_5_trends = AI_Trends_Frequency_df.nlargest(5, columns='Frequency')['AI_Trend'].tolist()

# Data filtering for the years 2021 to 2025
        filtered_trends_2021_2025 = trend_by_year[
             (trend_by_year['AI_Trend'].isin(top_5_trends)) &
             (trend_by_year['Year'] >= 2021) &
             (trend_by_year['Year'] <= 2025)
        ].copy()

        st.subheader("Predict the number of models for a future year")
        year_to_predict = st.number_input("Enter the year you want to predict(example: 2026)", min_value=2026, step=1)

        if st.button("View forecasts"):
            trend_models = {}
            trend_predictions = {}

            for trend in top_5_trends:
                trend_data = filtered_trends_2021_2025[filtered_trends_2021_2025['AI_Trend'] == trend]

                if not trend_data.empty:
                    X = trend_data['Year'].values.reshape(-1, 1)
                    y = trend_data['Model_Count'].values

                    model = LinearRegression()
                    model.fit(X, y)
                    trend_models[trend] = model

                    prediction = model.predict([[year_to_predict]])
                    trend_predictions[trend] = {
                           year_to_predict: max(0, int(round(prediction[0])))
                    }
                else:
                     trend_predictions[trend] = {year_to_predict: 0}
  
# --- Convert Forecasts to DataFrame ---
            predictions_list = []
            for trend, years_data in trend_predictions.items():
                predictions_list.append({
                    'AI_Trend': trend,
                    f'Predicted_Model_Count_{year_to_predict}': years_data[year_to_predict]
                })

            predictions_df = pd.DataFrame(predictions_list)
            predictions_df = predictions_df.sort_values(by=f'Predicted_Model_Count_{year_to_predict}', ascending=False)

# --- View drawing ---
            st.subheader(f"📊 Predict the number of patterns for the top 5 trends in {year_to_predict}")
            sns.set(style="whitegrid")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=predictions_df,
                    x=f'Predicted_Model_Count_{year_to_predict}',
                    y='AI_Trend',
                    palette='viridis',
                    ax=ax)
            ax.set_title(f'Predicted Number of Models for Top 5 AI Trends in {year_to_predict}')
            ax.set_xlabel('Predicted Number of Models')
            ax.set_ylabel('AI Trend')
            st.pyplot(fig)

    elif analysis_option == "🏢 Organization Analysis":
        org_counts = df['Organization'].value_counts().reset_index()
        org_counts.columns = ['Organization', 'Model_Count']

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(data=org_counts.head(10), x='Model_Count', y='Organization', palette='viridis', ax=ax)
        ax.set_title('Top 10 AI Organization by Number of Models')
        st.pyplot(fig)

    elif analysis_option == "🌍 Country Analysis":
        df_countries = df['Country (of organization)'].str.split(',', expand=True).stack().reset_index(level=1, drop=True).reset_index(name='Country')
        df_countries['Country'] = df_countries['Country'].str.strip()

        country_counts = df_countries['Country'].value_counts().reset_index()
        country_counts.columns = ['Country', 'Model_Count']
        top_10_countries = country_counts.head(10)

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(data=top_10_countries, x='Model_Count', y='Country', palette='viridis', ax=ax)
        ax.set_title('Top 10 Countries by Number of AI Model Organizations')
        st.pyplot(fig)


# ==============================
# Classification Section
# ==============================
if menu == "Classification":   
    df1 = pd.read_csv('/content/drive/My Drive/AI project/updated_file3.csv')
    elite_models = df1['Model'].dropna().unique().tolist()

    X = df1.drop("Confidence", axis=1)
    y = df1["Confidence"]
    X_encoded = pd.get_dummies(X)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.3, random_state=42)

# Training a KNN model
    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(X_train, y_train)

# Predict on test set to calculate metrics
    y_pred = knn.predict(X_test)
 
    st.title("Predict Confidence by Model Name")

# Show performance metrics
    st.subheader("📊 Metrics")
    st.text(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    st.text(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.2f}")
    st.text(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.2f}")
    st.text(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.2f}")

# Unique Model List
    model_names = sorted(df['Model'].dropna().unique())

# Choose a model
    selected_model = st.selectbox("Select Model:", model_names)

    if st.button("Predict Confidence"):
        input_data = pd.DataFrame(np.zeros((1, len(X_encoded.columns))), columns=X_encoded.columns)
        col_name = f"Model_{selected_model}"

        if col_name in X_encoded.columns:
            input_data.at[0, col_name] = 1
        else:
            st.warning(f"Model '{selected_model}' not found in training data columns.")
            st.stop()

    # Prediction of value

        prediction_encoded = knn.predict(input_data)
        prediction_label = le.inverse_transform(prediction_encoded)[0]

# Extract the original confidence value from the data
        original_confidence = df1[df1['Model'] == selected_model]['Confidence'].values
        if len(original_confidence) > 0:
            original_confidence = original_confidence[0]
        else:
            original_confidence = "unavailable"

    # عرض النتائج بشكل منسق
        st.write(f"### Results:")
        st.write(f"- **Confidence of the data:** {original_confidence}")
        st.write(f"- **Predicted value from the model:** {prediction_label}")

        if original_confidence != "unavailable":
            if original_confidence == prediction_label:
                st.success("✅ The prediction is correct and matches the original value.")
            else:
                st.error("❌ The prediction does not match the original value.")

# Save the model name entry in the database
        c.execute("INSERT INTO inputs (section, input_value) VALUES (?, ?)", ("classification", selected_model))
        conn.commit()

        if selected_model in df1['Model'].values:
             original_confidence = df1[df1['Model'] == selected_model]['Confidence'].values[0]
        else:
             original_confidence = "Model not found"

        match_result = "Match" if original_confidence == prediction_label else "Mismatch"
        c.execute("INSERT INTO outputs (section, output_value) VALUES (?, ?)", ("classification", match_result))
        conn.commit()

# ==============================
# Clustering Section
# ==============================

if menu == "Clustering":
        # Handle missing values
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].mean(), inplace=True)

    # Label Encoding for categorical features
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Standardize features
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    st.subheader("🔍KMeans Clustering")

        # PCA for visualization
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df_scaled)

        # KMeans clustering
    k = st.slider("Select number of clusters (k):", 2, 10, 3)
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(df_scaled)

        # Add PCA and cluster labels to DataFrame
    df_pca = pd.DataFrame(pca_result, columns=["PCA1", "PCA2"])
    df_pca["Cluster"] = clusters

        # Plotting
    fig, ax = plt.subplots()
    sns.scatterplot(data=df_pca, x="PCA1", y="PCA2", hue="Cluster", palette="Set2", ax=ax)
    ax.set_title("KMeans Clustering (PCA View)")
    st.pyplot(fig)

#save the inputs,outputs in the data set
    c.execute("INSERT INTO inputs (section, input_value) VALUES (?, ?)", ("clustering", str(k)))
    # حفظ الرسم كـ output (base64)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()

    c.execute("INSERT INTO outputs (section, output_value) VALUES (?, ?)", ("clustering", img_base64))
    conn.commit()

# ==============================
# Database Viewer Section
# ==============================
if menu == "Database Viewer":
    st.subheader("📦 View the contents of the database")
 
    st.subheader("📥 Inputs Table")
    inputs_df = pd.read_sql_query("SELECT * FROM inputs", conn)
    st.dataframe(inputs_df)

    st.subheader("📤 Outputs Table")
    outputs_df = pd.read_sql_query("SELECT * FROM outputs", conn)
    st.dataframe(outputs_df)

    for idx, row in outputs_df.iterrows():
        if row["section"] == "clustering":
            st.write(f"📌 Clustering Output Row {idx + 1}:")
            img_data = base64.b64decode(row["output_value"])
            st.image(img_data, caption="KMeans Clustering Graph")
        else:
            st.write(f"Row {idx + 1} - Section: {row['section']} - Output: {row['output_value']}")



