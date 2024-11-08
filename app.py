import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

st.set_page_config(page_title="User Behavior Analysis App", layout="wide", page_icon = "page logo.jpg", initial_sidebar_state="expanded")

st.sidebar.title("Dashboard")
st.sidebar.markdown("---")    
page = st.sidebar.selectbox("Choose a section", ["Home", "About"])

dataset_path = "streaming_data.csv"
df = pd.read_csv(dataset_path)

# Provide a download button for this predefined dataset
st.sidebar.markdown("<br><br><br><br><br>", unsafe_allow_html=True)
st.sidebar.markdown("**This is the default dataset. Only this streaming dataset works with this application. Download the file below:**")
csv_data = df.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(
    label="Download Default Dataset",
    data=csv_data,
    file_name="streaming_data.csv",
    mime="text/csv"
)
if page == "Home":
    st.title("User Behavior Analysis and Video Recommendation System 👥")
    st.write("Using K-Means Clustering and RNN/LSTM for user segmentation and recommendations.")

    # Initialize session state for buttons and selected user ID
    if "kmeans_ran" not in st.session_state:
        st.session_state.kmeans_ran = False
    if "n_clusters" not in st.session_state:
        st.session_state.n_clusters = 5    
    if "recommendations_generated" not in st.session_state:
        st.session_state.recommendations_generated = False
    if "lstm_trained" not in st.session_state:
        st.session_state.lstm_trained = False
    if "selected_user_id" not in st.session_state:
        st.session_state.selected_user_id = None
    if "recommend_button_clicked" not in st.session_state:
        st.session_state.recommend_button_clicked = False

    uploaded_file = st.file_uploader("Upload your streaming data CSV file", type=["csv", "xlsx"])
    if uploaded_file is not None:
        st.subheader("Dataset Preview:")
        st.write(df.head())

        # Fill missing values
        df['Duration_Watched (minutes)'].fillna(df['Duration_Watched (minutes)'].mean(), inplace=True)
        df['Ratings'].fillna(df['Ratings'].mode()[0], inplace=True)

        df = pd.get_dummies(df, columns=['Genre', 'Playback_Quality', 'Subscription_Status', 'Device_Type', 'Languages'], drop_first=True)
        st.subheader("Data after Preprocessing:")
        st.write(df.head())

        st.write("## User Clustering with K-Means")

        # Choose number of clusters
        n_clusters = st.slider("Number of clusters", 2, 10, st.session_state.n_clusters, key="n_clusters_slider")
        if n_clusters != st.session_state.n_clusters:
            st.session_state.kmeans_ran = False  # Reset if number of clusters changes
            st.session_state.n_clusters = n_clusters  # Update session state with new number of clusters

        @st.cache_data
        def run_kmeans(data, n_clusters):
            st.session_state.kmeans_ran = False
            
            with st.spinner("Normalizing data..."):

                # Standardize data if columns are present
                scaler = StandardScaler()
                if all(col in df.columns for col in ['Duration_Watched (minutes)', 'Ratings', 'Interaction_Events']):
                    normalized_data = scaler.fit_transform(df[['Duration_Watched (minutes)', 'Ratings', 'Interaction_Events']])

                    # Perform KMeans clustering
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    df['Cluster'] = kmeans.fit_predict(normalized_data)

                    # Visualize clusters
                    st.write("### Cluster Visualization")
                    fig = px.scatter(data, x='Duration_Watched (minutes)', y='Interaction_Events', color=data['Cluster'].astype(str), 
                    title="User Clusters", labels={'color': 'Cluster'})
                    st.plotly_chart(fig, use_container_width=True)

                    # Set the kmeans_ran state to True after clustering
                    st.session_state.kmeans_ran = True
                    st.success("Data normalized successfully")
                else:
                    st.write("Required columns for clustering are missing.")
        if st.button("Run K-Means Clustering") or (st.session_state.kmeans_ran and n_clusters == st.session_state.n_clusters):
            run_kmeans(df, n_clusters)    

        st.write("")

        if "User_ID" in df.columns:
            # Reshape data for LSTM
            X = []
            y = []
            time_step = 5
            features = df[['Duration_Watched (minutes)', 'Ratings', 'Interaction_Events']].values
            target = df['Interaction_Events'].shift(-1).fillna(0).values

            for i in range(len(features) - time_step):
                X.append(features[i:i+time_step])
                y.append(target[i+time_step])
            X = np.array(X)
            y = np.array(y)

            # Split into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Build and train LSTM model
            model = Sequential()
            model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
            model.add(Dropout(0.2))
            model.add(LSTM(50, return_sequences=False))
            model.add(Dropout(0.2))
            model.add(Dense(25))
            model.add(Dense(1))

            model.compile(optimizer='adam', loss='mean_squared_error')

            if st.button("Train LSTM Model"):
                with st.spinner("Training the LSTM Model..."):
                    model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))
                    st.success("LSTM Model trained successfully!")
                    st.session_state.lstm_trained = True

        # Only show user selection and recommendation button after LSTM model is trained
        if st.session_state.lstm_trained:
            st.write("### Generate Video Recommendations")

            user_ids = df['User_ID'].unique()
            selected_user_id = st.selectbox("Select a User ID for Recommendations", user_ids)

            if selected_user_id != st.session_state.selected_user_id:
                st.session_state.selected_user_id = selected_user_id
                st.session_state.recommend_button_clicked = False  # Reset the button state for a new user

            user_item_matrix = df.pivot_table(index='User_ID', columns='Video_ID', values='Ratings').fillna(0)
    
            # Normalize by subtracting each user's mean rating to focus on preferences
            user_mean_ratings = user_item_matrix.mean(axis=1)
            user_item_normalized = user_item_matrix.sub(user_mean_ratings, axis=0)
    
            # Calculate item-item similarity matrix
            item_similarity = cosine_similarity(user_item_normalized.T)
            item_similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)
        
            # Generate recommendations based on LSTM predictions
            if st.button("Generate LSTM-Based Recommendations"):
                st.session_state.recommend_button_clicked = True  # Set the button clicked state

            # Generate recommendations if the button was clicked
            if st.session_state.recommend_button_clicked:
                @st.cache_data
                def lstm_based_recommendations(user_id, num_recommendations=5):
                    user_data = df[df['User_ID'] == user_id][['Duration_Watched (minutes)', 'Ratings', 'Interaction_Events']].values
                    if len(user_data) < time_step:
                        padding = np.zeros((time_step - len(user_data), user_data.shape[1]))
                        user_data = np.vstack([padding, user_data])
                
                    user_sequence = user_data[-time_step:]
                    user_sequence = np.expand_dims(user_sequence, axis=0)
                
                    predicted_interaction = model.predict(user_sequence)[0][0]

                    user_ratings = user_item_normalized.loc[user_id]
                    weighted_ratings = item_similarity_df.dot(user_ratings).div(item_similarity_df.sum(axis=1))
                
                    recommended_videos = weighted_ratings[user_item_matrix.loc[user_id] == 0] * predicted_interaction
                    top_recommendations = recommended_videos.sort_values(ascending=False).head(num_recommendations)

                    return top_recommendations.index.tolist()

                # Display recommendations for the selected user ID
                lstm_recommendations = lstm_based_recommendations(st.session_state.selected_user_id)
                st.write(f"LSTM-Based Recommended Video IDs for User {st.session_state.selected_user_id}:")
                st.write(lstm_recommendations)
    else:
        st.write("Please upload a dataset to begin.")
elif page == "About":
    st.title("About the User Behavior Analysis and Video Recommendation System")
    st.markdown("""
    Welcome to the **User Behavior Analysis and Video Recommendation System**! This application leverages **K-Means Clustering** and **RNN/LSTM** models to analyze user engagement and recommend videos tailored to individual preferences. The tool is designed for video streaming platforms to better understand user behavior and enhance engagement through personalized recommendations.

    ### Key Features
    - **User Segmentation with K-Means Clustering**: Clusters users based on their engagement metrics, such as watch duration, ratings, and interaction events.
    - **Personalized Video Recommendations**: Utilizes a **Recurrent Neural Network (LSTM)** model for sequential data to predict future engagement and generate video recommendations.
    - **Cosine Similarity for Video Recommendations**: Calculates similarity between video content to further refine recommendations based on user interactions.
    - **Data Preprocessing**: Handles missing data, applies one-hot encoding for categorical features, and scales numerical features to optimize model performance.
    """)            
    st.markdown("<br>", unsafe_allow_html=True)  

    st.markdown("""
    ### How to Use
    1. **Upload Dataset**: Start by uploading your streaming data CSV file. Alternatively, use the provided sample dataset for a quick start.
    2. **Run K-Means Clustering**: Select the number of clusters for user segmentation and run the clustering model to view visualizations of user groups.
    3. **Train LSTM Model**: Train the LSTM model on user behavior data to capture sequential patterns in engagement.
    4. **Generate Recommendations**: Choose a user ID to generate and display personalized video recommendations.
    """)
    st.markdown("<br>", unsafe_allow_html=True) 

    st.markdown("""
    ### About the Models
    - **K-Means Clustering**: Groups users based on similar viewing patterns to identify user segments, helping to tailor recommendations and marketing efforts.
    - **LSTM Model**: Predicts user engagement patterns over time, providing a foundation for sequential recommendations based on historical viewing behavior.
    - **Cosine Similarity**: Enhances recommendations by calculating similarity between video features, ensuring users receive relevant content.
    """)
    st.markdown("<br>", unsafe_allow_html=True)             

    st.markdown("""
    ### Technologies Used
    - **Python Libraries**: Streamlit, Pandas, Numpy, Scikit-Learn, TensorFlow, Matplotlib
    - **Machine Learning Algorithms**: K-Means Clustering, LSTM (Long Short-Term Memory Network), Cosine Similarity
    - **Data Preprocessing**: Standardization, Missing Value Handling, One-Hot Encoding
    """)
    st.write("")
    st.markdown("""
    This app was built with **Streamlit** for an interactive experience, that allows you to explore and analyze data effortlessly. We hope you find this tool helpful in understanding and enhancing user engagement!
    """)

st.markdown("""
    <style>
    .stApp {
        padding: 10px;
    }
    .stButton button {
        background-color: #2196F3;  /* Blue background */
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        transition: background-color 0.3s, transform 0.3s;
    }

    .stButton button:hover {
        background-color: #0b7dda;  /* Darker blue when hovered */
        transform: scale(1.05);  /* Slight scale effect when hovered */
    }

    /* LSTM button style */
    .stButton button.st-lstm {
        background-color: #FF5722;  /* Orange background */
    }

    .stButton button.st-lstm:hover {
        background-color: #e64a19;  /* Darker orange when hovered */
    }

    /* File Uploader style */
    .stFileUploader button {
        background-color: #3F51B5;  /* Blue background */
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        transition: background-color 0.3s, transform 0.3s;
    }

    .stFileUploader button:hover {
        background-color: #303F9F;  /* Darker blue when hovered */
        transform: scale(1.05);
    }
            
    .stSidebar {background-color: #f0f4fc;}
    .stDownloadButton button {
        background-color: #008CBA;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
    }
    .stDownloadButton button:hover {
        background-color: #007bb5;
    }
    </style>
""", unsafe_allow_html=True)
