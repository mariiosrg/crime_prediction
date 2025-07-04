# ------- Library ----------- #
import streamlit as st
import streamlit.components.v1 as components
import time
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
import altair as alt
import io
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import pickle
import joblib
import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from feature_engine.imputation import RandomSampleImputer
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# --------------------------------------- FUNCTION -------------------------------------------------- #
def update_progress_bar(start, end, delay=0.05):
    for progress in range(start, end + 1):
        progress_bar.progress(progress)
        time.sleep(delay)
# ---------------------------------------------------------------------------------- #
def data_load(datas):
  import pandas as pd
  data = pd.read_csv(datas)
  return data
# ---------------------------------------------------------------------------------- #
# Fungsi pengecekkan apakah format yang diupload CSV
def is_csv(filename):
    return filename.lower().endswith('.csv')
# ---------------------------------------------------------------------------------- #
# --- Function untuk random sample imputer --- #
def random_sample_imputer(df, column_name):
    # Get non-null values in the column
    non_null_values = df[column_name].dropna().values

    # Define a function to randomly sample from non-null values
    def impute_value(x):
        if pd.isnull(x):
            return np.random.choice(non_null_values)
        else:
            return x

    # Apply the impute_value function to the column
    df[column_name] = df[column_name].apply(impute_value)
    return df
# ---------------------------------------------------------------------------------- #
# ---- Penyelarasan nilai value data fitur "Premis" ---#
# Mengubah value record
def simple_premis(item):
    item = str(item).upper()  # Convert the item to uppercase once
    if 'MTA' in item:
        return 'Public Transport'
    else:
        return item  # If no conditions match, return the original item

# --- Fungsi Penyederhanaan kategori menjadi "Indoor" dan "Outdoor" pada kolom "Premis" --- #
# Outdoor and Indoor classification
indoor = []
outdoor = []
# ---------------------------------------------------------------------------------- #
# fungsi convert
def classify_location(category):
    category_lower = category.lower()
    
    # Keywords that suggest it's outdoor
    outdoor_keywords = ['cemetary','college','valet','outside','skating','cross','freeway','street', 'sidewalk', 'park', 'drive', 'yard', 'lot', 'stop', 'bus', 'track', 'beach', 'outdoor',
                        'slip', 'bridge', 'field', 'dumpster', 'alley', 'vacant', 'can', 'river', 'rail', 'track', 
                        'playground', 'road', 'track', 'vehicle', 'trail', 'garden', 'station', 'encampment', 
                        'court', 'stadium', 'underpass', 'skateboard', 'parking','atm','residen','public']
    
    # Keywords that suggest it's indoor
    indoor_keywords = ['taxi','rental','truck','otel','hospice','shelter','train','arcade','bank','manufactur','website','dealer','tunnel','store', 'shop', 'mall', 'school', 'hospital', 'office', 'hotel', 'museum', 'club', 'gym', 
                       'center', 'station', 'restaurant', 'building', 'home', 'facility', 'venue', 'market', 'apartment', 
                       'dwelling', 'clinic', 'theatre', 'bar', 'library', 'laundry', 'house', 'condo', 'garage', 
                       'gym', 'studio', 'pharmacy', 'spa', 'salon', 'store', 'mortuary', 'church', 'chapel', 
                       'court', 'police', 'warehouse', 'arena', 'arena', 'theater', 'post', 'elevator']
    
    # Categorize based on known keywords
    if any(kw in category_lower for kw in outdoor_keywords):
        return 'Outdoor'
    elif any(kw in category_lower for kw in indoor_keywords):
        return 'Indoor'
    # Manually handle some specific cases that don't follow the pattern
    elif category_lower in ['car wash','golf course','public transport','train tracks', 'greyhound or interstate bus', 'metro station', 'railroad tracks']:
        return 'Outdoor'
    elif category_lower in ['massage parlor','delivery','amtrak train','hockey','care','pay phone','aircraft','truck','arcade','mini-mart','equipment rental','nightclub', 'jewelry store', 'cell phone store', 'telecommunication facility', 
                            'bar', 'coffee shop', 'medical/dental office', 'movie theater', 'grocery store']:
        return 'Indoor'
    else:
        return 'Unknown'
# ---------------------------------------------------------------------------------- #
# ---- Penyelarasan nilai value data fitur "Weapon" ---- #
# Mengubah value record
def label_items(item):
    item = str(item).upper()  # Convert the item to uppercase once
    if 'KNIFE' in item:
        return 'Knife'
    elif any(keyword in item for keyword in ['RIFLE', 'GUN', 'PISTOL', 'ASSAULT']):
        return 'Gun'
    elif 'STRONG' in item:
        return 'BODY-ARM'
    elif 'UNKNOWN' in item:
        return 'Unknown'
    return item  # If no conditions match, return the original item
# ---------------------------------------------------------------------------------- #
# ---- Penyelarasan nilai value data fitur "Status" ---- #
# Mengubah value record
def label_status(stat):
    stat = str(stat).upper()  # Convert the item to uppercase once
    if 'ARREST' in stat:
        return 'Arrested'
    elif 'INVEST' in stat:
        return 'Investigation'
    return 'Investigation'  # If no conditions match, return the original item

# ---------------------------------------------------------------------------------- #
@st.cache_data
def load_data():
        """
        Load raw dataset untuk mendapatkan nilai unik
        dan encoded dataset untuk reference encoding
        """
        # Load raw dataset untuk display dan nilai unik
        raw_df = pd.read_csv('data/raw_dataset.csv')
        
        # Load encoded dataset untuk reference nilai encoding
        encoded_df = pd.read_csv('data/encoded_dataset.csv')
        
        return raw_df, encoded_df
# ---------------------------------------------------------------------------------- #
@st.cache_resource
def load_models():
        """Load model dan encoder"""
        model = pickle.load(open('models/model.pkl', 'rb'))
        encoder = joblib.load('models/encoder.pkl')
        return model, encoder
# ---------------------------------------------------------------------------------- #
def get_feature_names(model):
        """Get feature names dari model dalam urutan yang benar"""
        try:
            return model.feature_names_in_.tolist()
        except AttributeError:
            try:
                return model.feature_names
            except AttributeError:
                return None
# ---------------------------------------------------------------------------------- #
def get_unique_classes(raw_df):
        """Get unique classes dari kolom Status"""
        return sorted(raw_df['Status'].unique())
# ---------------------------------------------------------------------------------- #
def get_column_types(df, feature_names):
        """Identifikasi tipe kolom dengan urutan yang sesuai feature names"""
        numeric_cols = []
        categorical_cols = []
        
        for col in feature_names:
            if col in df.columns:
                if df[col].dtype in ['int64', 'float64']:
                    numeric_cols.append(col)
                else:
                    categorical_cols.append(col)
                    
        return numeric_cols, categorical_cols
# ---------------------------------------------------------------------------------- #
def get_numeric_ranges(df, numeric_cols):
        """Dapatkan range nilai untuk kolom numerik"""
        ranges = {}
        for col in numeric_cols:
            ranges[col] = {
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'mean': float(df[col].mean())
            }
        return ranges
# ---------------------------------------------------------------------------------- #
def get_categorical_values(df, categorical_cols):
        """Dapatkan nilai unik untuk kolom kategorikal"""
        unique_values = {}
        for col in categorical_cols:
            if col != 'Status':  # Skip Status column
                unique_values[col] = sorted(df[col].unique().tolist())
        return unique_values
# ---------------------------------------------------------------------------------- #
def encode_input_features(features, raw_df, encoded_df, categorical_cols, feature_names):
        """
        Encode input features berdasarkan mapping dari dataset dan urutkan sesuai feature names
        """
        encoded_features = {}
        # Encode categorical features
        for col in categorical_cols:
            if col in features:
                # Dapatkan mapping dari raw ke encoded value
                mapping = dict(zip(raw_df[col], encoded_df[col]))
                encoded_features[col] = mapping[features[col]]
        
        # Tambahkan numeric features
        for col in features:
            if col not in categorical_cols:
                encoded_features[col] = features[col]
        
        # Buat DataFrame dengan urutan yang benar
        features_df = pd.DataFrame([encoded_features])
        features_df = features_df[feature_names]  # Reorder columns
        
        return features_df
# ---------------------------------------------------------------------------------- #
# Inisialisasi session state untuk button
if 'button1_clicked' not in st.session_state:
    st.session_state.button1_clicked = False

if 'button2_clicked' not in st.session_state:
    st.session_state.button2_clicked = False

if 'button3_clicked' not in st.session_state:
    st.session_state.button3_clicked = False

# Fungsi untuk mengubah status button
def click_button1():
    st.session_state.button1_clicked = True

def click_button2():
    st.session_state.button2_clicked = True

def click_button3():
    st.session_state.button3_clicked = True


# Page configuration
st.set_page_config(
    page_title="HPC-UG-AI-CoE",
    page_icon="🏂",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("default")
# CSS styling
st.markdown("""
<style>

[data-testid="block-container"] {
    padding-left: 2rem;
    padding-right: 2rem;
    padding-top: 1rem;
    padding-bottom: 0rem;
    margin-bottom: -7rem;
}

[data-testid="stVerticalBlock"] {
    padding-left: 0rem;
    padding-right: 0rem;
}

[data-testid="stMetric"] {
    text-align: center;
    background-color: #f0f2f6;
    padding: 15px 0;
    color: var(--text-color);
}

# @media (prefers-color-scheme: light) {
#     [data-testid="stMetric"] {
#         background-color: #d3d3d3; /* Warna lebih gelap untuk tema light */
#         text-align: center;
#         padding: 15px 0;
#         color: var(--text-color);
#     }
# }

# @media (prefers-color-scheme: dark) {
#     [data-testid="stMetric"] {
#         background-color: #262730; /* Warna lebih terang untuk tema dark */
#         text-align: center;
#         padding: 15px 0;
#         color: var(--text-color);
#     }
# }

[data-testid="stMetricLabel"] {
  display: flex;
  justify-content: center;
  align-items: center;
}

[data-testid="stMetricDeltaIcon-Up"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}

[data-testid="stMetricDeltaIcon-Down"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}

</style>
""", unsafe_allow_html=True)

footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Created by: Pengelola HPC/DGX (UG-AI-CoE) Universitas Gunadarma
</div>
"""
#<a style='display: block; text-align: center;' href="https://www.heflin.dev/" target="_blank">Heflin Stephen Raj S</a></p>

footer1="""<style>
a:link1 , a:visited1{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover1,  a:active1 {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer1 {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer1">
<p>Created by: Pengelola HPC/DGX (UG-AI-CoE) Universitas Gunadarma
</div>
"""

# ------- Streamlit ----------- #

menu_sidebar = st.sidebar.selectbox('MENU',('Data Mining',"Dashboard","Live Prediction"))

# HALAMAN UTAMA #
#    Pengantar Halaman
if menu_sidebar == 'Data Mining':
    st.header(':blue[Panduan]')
    st.markdown('''1. Gunakan format file CSV \
                    \n2. File dataset Max 200MB \
                        ''')
        
    #    Upload file ke Website
    uploaded_file = st.file_uploader('Upload File CSV')

    if uploaded_file is not None:
    #         Kondisi file harus CSV
        file_name = uploaded_file.name
        
        # Membuat garis horizontal
        st.divider()

        # Judul output Dataframe
        st.header(":memo: Dataset Overview :memo:")

        # Output nama dataframe
        html_str = f"""
        <style>
        p.a {{
        font: {10}px arial;
        text-align: center;
        }}
        </style>
        <p class="a">{file_name}</p>
        """

        # Upload data dan hitung waktu upload
        start_time = time.time()
        if is_csv(file_name):
            df = data_load(uploaded_file)
            st.dataframe(df.head(312849))
            st.markdown(html_str, unsafe_allow_html=True)
            st.write(f"Jumlah Baris data : {df.shape[0]} baris")
            st.write(f"Jumlah Kolom data : {df.shape[1]} kolom")
        end_time = time.time()
        elapsed_time = end_time - start_time
        st.write(f"Waktu load data : {elapsed_time:.4f} detik")


        # Eksplorasi Data Button
        # Button pertama
        with st.container(border=True):
         st.button('Eksplorasi Data', on_click=click_button1)
         if st.session_state.button1_clicked:
            
            # Statistik Deskriptif
            st.write(":chart_with_upwards_trend: Statistik Deskriptif Nilai Numerik :chart_with_upwards_trend:")
            deskripsi_numerik = df.describe()
            st.write(deskripsi_numerik)

            deskripsi_string = df.describe(include="O")
            st.write(":capital_abcd: Statistik Deskriptif Nilai Kategorik:capital_abcd:")
            st.write(deskripsi_string)

            # Cek nilai NULL
            st.write("Pengecekkan Nilai Null")

            # Assuming 'df' is your DataFrame
            null_columns = df.isna().sum()

            # Filter columns that only contain NaN values
            only_null_columns = null_columns[null_columns > 0]

            # Display columns with their NaN counts
            st.write(only_null_columns)


    # Button kedua
            st.button('Prepare Data', on_click=click_button2)
            if st.session_state.button2_clicked:
                progress_text = "Sedang memproses data. Mohon tunggu."
                
                # Split the layout into two columns
                col1, col2 = st.columns(2)
                
                # Add elements in the first column
                with col1:
                    st.header("Before")
                    st.dataframe(df.head(10000))

                # Mengisi data null dengan "Unknown"
                # Kolom "Weapon"
                df['Weapon'] = df['Weapon'].fillna('Unknown')
                
                # Imputasi pada kolom "Victim_sex"
                # Menggunakan Random Sample Imputer
                df_imputed = random_sample_imputer(df, 'Victim_sex')

                # Imputasi pada kolom "Victim_descent"
                # Menggunakan Random Sample Imputer
                df_imputed = random_sample_imputer(df, 'Victim_descent')

                # Imputasi pada kolom "Premis"
                # Menggunakan Random Sample Imputer
                df_imputed = random_sample_imputer(df, 'Premis')

                # Pengaplikasian fungsi "label_items"
                df_imputed['Senjata'] = df_imputed['Weapon'].apply(label_items)

                # Apply fungsi untuk kategori MTA
                df_imputed['Premis2'] = df_imputed['Premis'].apply(simple_premis)

                # Apply the classification function to the dataframe and save in new column 'tempat_kejadian'
                df_imputed['tempat_kejadian'] = df_imputed['Premis2'].apply(classify_location)

                # Apply the classification function to the dataframe and save in new column 'tempat_kejadian'
                df_imputed['Status'] = df_imputed['Status'].apply(label_status)
                # Add elements in the second column
                with col2:
                    st.header("After")
                    st.dataframe(df_imputed.head(10000))
                st.write("Jumlah baris data setelah pembersihan dataset:", len(df_imputed))

    # VIZ : Visualisasi "Weapon" yang digunakan untuk kejahatan tanpa kategori "Unknown"
                st.divider()
                st.header(":bar_chart: Visualisasi Data :bar_chart:")

                top_10_weapons = df_imputed[~df_imputed['Senjata'].isin(['Unknown'])]
                top_5_weapons_selected = top_10_weapons.Senjata.value_counts().head(5)

                fig = px.bar(top_5_weapons_selected, x=top_5_weapons_selected.index, y=top_5_weapons_selected.values,
                    labels={'x': 'Senjata', 'y': 'Jumlah'},
                    title='Top 10 Senjata digunakan dalam kejahatan',
                    color=top_5_weapons_selected.index)
                # fig.update_xaxes(tickangle=20)
                st.plotly_chart(fig,theme=None)


        # VIZ (2) : Visualisasi Daerah yang banyak terjadi kejahatan
                category_counts = df_imputed['Area'].value_counts().reset_index()
                category_counts.columns = ['Area', 'Count']

                # Create the bar plot using Seaborn
                plt.figure(figsize=(10, 6))
                bar_plot = sns.barplot(x='Area', y='Count', data=category_counts, palette="plasma")

                # Customize the plot
                plt.title("Lokasi Kejahatan yang sering terjadi")
                plt.xlabel('Area')
                plt.ylabel('Value')
                plt.xticks(rotation=45)

                # Get the max and min values
                max_value = category_counts['Count'].max()
                min_value = category_counts['Count'].min()
                max_category = category_counts.loc[category_counts['Count'] == max_value, 'Area'].values[0]
                min_category = category_counts.loc[category_counts['Count'] == min_value, 'Area'].values[0]

                # Add annotations for the highest bar (max)
                plt.text(category_counts[category_counts['Area'] == max_category].index[0], max_value + 0.2, 
                        f"Max: {max_value}", color='green', ha='center', va='bottom', fontsize=12, 
                        backgroundcolor='lightgreen')

                # Add annotations for the lowest bar (min)
                plt.text(category_counts[category_counts['Area'] == min_category].index[0], min_value + 0.2, 
                        f"Min: {min_value}", color='red', ha='center', va='bottom', fontsize=12, 
                        backgroundcolor='lightcoral')

                # Show the plot
                plt.tight_layout()
                st.pyplot(plt)

                # Visualize the distribution of the resampled 'Status' column
                status_counts_resampled = df_imputed['Status'].value_counts()
                plt.figure(figsize=(10, 6))
                plt.bar(status_counts_resampled.index, status_counts_resampled.values)
                plt.xlabel('Status')
                plt.ylabel('Count')
                plt.title('Distribusi Status Tersangka')
                plt.xticks(rotation=45)
                st.pyplot(plt)

    # Training Data

                st.divider()
                st.button('Latih Data', on_click=click_button3)
                if st.session_state.button3_clicked:

                    # Filter data dengan status "Investigation" dan "Arrested"
                    investigation_df = df_imputed[df_imputed['Status'] == 'Investigation']
                    arrested_df = df_imputed[df_imputed['Status'] == 'Arrested']

                    # Mengambil semua data 'Arrested'
                    arrested_count = len(arrested_df)


                    # Menghitung jumlah sisa data yang harus diambil dari "Investigation"
                    remaining_investigation_count = 200000 - arrested_count

                    # Melakukan sampling pada data "Investigation" sesuai dengan jumlah yang diperlukan
                    investigation_sample = investigation_df.sample(n=remaining_investigation_count, replace=True, random_state=42)

                    # Menggabungkan data 'Arrested' dan sampel 'Investigation'
                    result_df = pd.concat([arrested_df, investigation_sample])

                    
                    
                    # Initialize LabelEncoder
                    le = LabelEncoder()

                # Loop through all the columns and apply Label Encoding for categorical columns
                    for column in result_df.select_dtypes(include=['object']).columns:
                        result_df[column] = le.fit_transform(result_df[column])
                    
                    # define X dan Y
                    X = result_df.drop('Status', axis=1)
                    y = result_df['Status']
                    smote = SMOTE(random_state=42, k_neighbors=1)
                    try:
                        X_resampled, y_resampled = smote.fit_resample(X, y)
                    except ValueError as e:
                    
                        print(f"SMOTE Error: {e}")
                        print("Check the distribution of your target variable ('Status')."
                            "Some classes may have too few samples for SMOTE to work.")

                    else:
                        # Create a new DataFrame with the resampled data
                        df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
                        df_resampled['Status'] = y_resampled

                        # Visualize the distribution of the resampled 'Status' column
                        status_counts_resampled = df_resampled['Status'].value_counts()
                        plt.figure(figsize=(10, 6))
                        plt.bar(status_counts_resampled.index, status_counts_resampled.values)
                        plt.xlabel('Status')
                        plt.ylabel('Count')
                        plt.title('Distribution of Status after SMOTE')
                        plt.xticks(rotation=45)
                        st.pyplot(plt)
# TRAINING OUTPUT
                        st.header("Training Data:mortar_board:")
                        st.write("Model sedang dilatih ")
                        # -------------------Time Count---------------------------# 
                        start_time = time.time()
                        # -------------------------------------------------------# 

                        # Initialize the progress bar
                        progress_bar = st.progress(0)

                        # Split data into training and testing sets
                        update_progress_bar(0, 20)
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                        # Create a Random Forest Classifier (without hyperparameter tuning)
                        update_progress_bar(20, 40)
                        rf_model = RandomForestClassifier(random_state=42)

                        # Train the model
                        update_progress_bar(40, 85)
                        rf_model.fit(X_train, y_train)
                        

                        # Make predictions on the test data
                        update_progress_bar(85, 100)
                        y_pred = rf_model.predict(X_test)
                        

                        # Evaluate the model
                        accuracy = accuracy_score(y_test, y_pred)
                        st.write("Accuracy: ", accuracy)

                        y_test_original = le.inverse_transform(y_test)
                        y_pred_original = le.inverse_transform(y_pred)

                        # Buat DataFrame untuk membandingkan hasil prediksi dengan nilai asli
                        comparison_df = pd.DataFrame({'Original': y_test_original, 'Prediction': y_pred_original})

                        # Reset index agar rapi
                        comparison_df.reset_index(drop=True, inplace=True)

                        # Tampilkan hasilnya
                        st.header("Komparasi Hasil prediksi dan nilai riil")
                        st.write(comparison_df)

                        # Hitung distribusi kelas prediksi
                        predicted_counts = comparison_df['Prediction'].value_counts()
                        predicted_classes = predicted_counts.index
                        predicted_values = predicted_counts.values

                        # Plot pie chart menggunakan Matplotlib
                        plt.figure(figsize=(8, 6))
                        colors = sns.color_palette("pastel", len(predicted_classes))
                        plt.pie(predicted_values, labels=predicted_classes, autopct='%1.1f%%', startangle=140, colors=colors)
                        plt.title("Distribusi Prediksi untuk Setiap Kelas")

                        # Tampilkan pie chart di Streamlit
                        st.write("Distribusi Hasil Prediksi per Kelas:")
                        st.pyplot(plt)

                        # st.write(classification_report(y_test, y_pred))

                        # -------------------Time Count---------------------------# 
                        end_time=time.time()
                        execution_time = end_time - start_time
                        # st.write(f"Execution time: {execution_time} seconds")

# --------------- HALAMAN DASHBOARD -------------------- #
if menu_sidebar == 'Dashboard':
    st.title("Dashboard Analisis Kejahatan Kriminal")
    st.divider()
    # HTML untuk embed Tableau dashboard
    html_temp = """
        <div class='tableauPlaceholder' id='viz1730214301377' style='position: relative; width: 100%; height: 0; padding-top: 75%; overflow: hidden;'>
            <noscript>
                <a href='#'>
                    <img alt='Geographic Overview' src='https://public.tableau.com/static/images/Cr/CrimeAnalysisDashboard_17302099758820/GeographicOverview/1_rss.png' style='border: none' />
                </a>
            </noscript>
            <object class='tableauViz' style='position: absolute; top: 0; left: 0; width: 100%; height: 100%;'>
                <param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' />
                <param name='embed_code_version' value='3' />
                <param name='site_root' value='' />
                <param name='name' value='CrimeAnalysisDashboard_17302099758820/GeographicOverview' />
                <param name='tabs' value='no' />
                <param name='toolbar' value='yes' />
                <param name='animate_transition' value='yes' />
                <param name='display_static_image' value='yes' />
                <param name='display_spinner' value='yes' />
                <param name='display_overlay' value='yes' />
                <param name='display_count' value='yes' />
                <param name='language' value='en-US' />
                <param name='filter' value='publish=yes' />
            </object>
        </div>
        <script type='text/javascript'>
            var divElement = document.getElementById('viz1730214301377');
            var vizElement = divElement.getElementsByTagName('object')[0];
            if (divElement.offsetWidth > 800) {
                vizElement.style.width = '1366px';
                vizElement.style.height = '768px';
            } else if (divElement.offsetWidth > 500) {
                vizElement.style.width = '100%';
                vizElement.style.height = '768px';
            } else {
                vizElement.style.width = '100%';
                vizElement.style.height = '100%';
            }
            var scriptElement = document.createElement('script');
            scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';
            vizElement.parentNode.insertBefore(scriptElement, vizElement);
        </script>
    """

    # Menggunakan komponen HTML di Streamlit
    components.html(html_temp, width=1366,height=768)
    st.markdown(f'Link to the public dashboard [here](https://public.tableau.com/app/profile/andrian.kharisma.wijaya/viz/CrimeAnalysisDashboard_17302099758820/GeographicOverview?publish=yes)')
    
# --------------- HALAMAN LIVE PREDICTION -------------------- #
if menu_sidebar == 'Live Prediction':
    st.title(":orange[Model Machine Learning untuk Klasifikasi Data Kasus Kriminal]")

    def main():
        try:
            # Load data dan model
            raw_df, encoded_df = load_data()
            model, encoder = load_models()
            
            # Get feature names dari model
            feature_names = get_feature_names(model)
            if feature_names is None:
                st.error("Tidak dapat mendapatkan feature names dari model")
                return
                
            # Get unique classes
            classes = get_unique_classes(raw_df)
                
            # Exclude target variable if present
            feature_names = [f for f in feature_names if f != 'Status']
            
            # Get column types dari raw dataset dengan urutan yang sesuai
            numeric_cols, categorical_cols = get_column_types(raw_df, feature_names)
            
            # Get ranges dan unique values
            numeric_ranges = get_numeric_ranges(raw_df, numeric_cols)
            categorical_values = get_categorical_values(raw_df, categorical_cols)
            
            # Header
            st.title("🔮 Dashboard Prediksi")
            st.write("Masukkan nilai fitur untuk mendapatkan prediksi")
            
            # Layout dengan 2 kolom
            col1, col2 = st.columns(2)
            
            # Dictionary untuk menyimpan input features
            features = {}
            
            with col1:
                # Input fitur kategorikal
                st.subheader("Fitur Kategorikal")
                for col in categorical_cols:
                    if col != 'Status':  # Skip Status column
                        features[col] = st.selectbox(
                            f"{col}",
                            options=categorical_values[col],
                            help=f"Unique values: {len(categorical_values[col])}"
                        )
            with col2:
                 # Input fitur numerik
                st.subheader("Fitur Numerik")
                for col in numeric_cols:
                    if col == 'Time_occured':
                        time_input = st.time_input(f"{col}", value=datetime.time(12, 0), step=60)
                        features[col] = time_input.hour * 100 + time_input.minute
                    else:
                        features[col] = st.number_input(
                            f"{col}",
                            min_value=int(numeric_ranges[col]['min']),
                            max_value=int(numeric_ranges[col]['max']),
                            value=int(numeric_ranges[col]['mean']),
                            help=f"Range: {int(numeric_ranges[col]['min'])} - {int(numeric_ranges[col]['max'])}"
                        )
            # Button untuk prediksi
            if st.button("Prediksi"):
                # Encode dan urutkan input features
                features_df = encode_input_features(
                    features, raw_df, encoded_df, categorical_cols, feature_names
                )
                
                # Predict
                prediction = model.predict(features_df)[0]
                probability = model.predict_proba(features_df)[0]
                categories = ['Arrested', 'Investigation']
                # Map prediction to category
                predicted_category = categories[prediction]
                # Display results
                st.write("---")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Hasil Prediksi", predicted_category)
                
                with col2:
                    prob_text = f"{probability.max()*100:.2f}%"
                    st.metric("Probabilitas", prob_text)
                
                with col3:
                    confidence = "Tinggi" if probability.max() > 0.8 else "Sedang" if probability.max() > 0.6 else "Rendah"
                    st.metric("Tingkat Keyakinan", confidence)
                
                # Tampilkan visualisasi probabilitas
                st.write("### Probabilitas per Kelas")
                prob_df = pd.DataFrame({
                    'Kelas': classes,
                    'Probabilitas': probability
                })
                st.bar_chart(prob_df.set_index('Kelas'))


            # Tampilkan data distributions
            if st.checkbox("Tampilkan Distribusi Data"):
                st.write("### Distribusi Data")
                
                # Numeric distributions
                st.write("#### Distribusi Fitur Numerik")
                for col in numeric_cols:
                    if col != 'Status':
                        fig_col1, fig_col2 = st.columns(2)
                        with fig_col1:
                            st.write(f"Histogram {col}")
                            st.bar_chart(raw_df[col].value_counts())
                        with fig_col2:
                            st.write(f"Statistics {col}")
                            st.write(raw_df[col].describe())
                
                # Categorical distributions
                st.write("#### Distribusi Fitur Kategorikal")
                for col in categorical_cols:
                    if col != 'Status':
                        st.write(f"\nDistribusi {col}")
                        st.bar_chart(raw_df[col].value_counts())
            
        except Exception as e:
            st.error(f"Terjadi error: {str(e)}")
            st.write("Pastikan semua file yang diperlukan tersedia:")
            st.write("- data/raw_dataset.csv")
            st.write("- data/encoded_dataset.csv")
            st.write("- models/model.pkl")
            st.write("- models/encoder.pkl")

    if __name__ == "__main__":
        main()

    st.markdown(footer1,unsafe_allow_html=True)
