# User-Behavior-Analysis-and-Video-Recommendation-System
This project titled "User Behavior Analysis and Video Recommendation System" aims to build an intelligent video recommendation system using a combination of K-Means Clustering and LSTM (Long Short-Term Memory) models to analyze user behavior and predict personalized video recommendations.
<br>
This system designed to analyze user behavior and provide personalized video recommendations using clustering and sequential modeling techniques. This project utilizes machine learning methods to group users based on behavior patterns and recommend content that aligns with their viewing preferences.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
The **User Behavior Analysis and Video Recommendation System** aims to leverage user viewing data to analyze behavior patterns and generate recommendations. The system is built on machine learning models, including K-Means clustering and sequential models like RNNs or LSTMs, to group users and predict video recommendations.

## Features
- **User Behavior Clustering**: Groups users based on their viewing habits.
- **Recommendation System**: Suggests videos tailored to individual preferences.
- **Data Visualization**: Provides visual insights into user behavior and recommendation outcomes.
- **Interactive Dashboard**: Uses Streamlit to allow interactive exploration of the recommendation system.

## Architecture
1. **Data Preprocessing**: Cleans and prepares the data.
2. **Clustering with K-Means**: Segments users based on behavior.
3. **Sequential Modeling**: Uses RNN/LSTM models to predict recommendations.
4. **Web Application**: An interactive dashboard built with Streamlit.

## Dataset
The dataset used in this project includes user interactions with a streaming platform, such as:
- **User ID**: Unique identifier for each user.
- **Video ID**: Unique identifier for each video.
- **Interaction Data**: Clicks, views, likes, etc.

Ensure the dataset is stored in a `data/` directory as expected by the codebase.

## Installation

1. **Clone the Repository**
    ```bash
    git clone https://github.com/navadeep-05/User-Behavior-Analysis-and-Video-Recommendation-System.git
    cd User-Behavior-Analysis-and-Video-Recommendation-System
    ```

2. **Create and Activate a Virtual Environment**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. **Prepare the Dataset**: Place your dataset in the `data/` folder.
2. **Run the Application**
    ```bash
    streamlit run app.py
    ```
3. **Access the Web Interface**: Go to `http://localhost:8501` to view the dashboard.

## Results
### Clustering
Using K-Means clustering, users are grouped based on interaction frequency, engagement level, and other behavioral metrics.

### Recommendations
The system suggests videos based on user clusters and sequential model predictions, delivering personalized recommendations.

## Technologies Used
- **Programming Language**: Python
- **Libraries**: 
  - **Machine Learning**: scikit-learn, TensorFlow
  - **Web Framework**: Streamlit
  - **Data Manipulation**: pandas, numpy
  - **Visualization**: matplotlib, seaborn

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch.
3. Make your changes.
4. Submit a pull request.

## License
This project is licensed under the Software License. See the [LICENSE](LICENSE) file for details.
