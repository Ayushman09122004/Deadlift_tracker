# Deadlift Tracker and Classifier

This repository contains a set of scripts and models for tracking and classifying deadlift exercises using machine learning techniques. The project includes data collection, training a random forest classifier, and a tracker for monitoring deadlift performance.

## Project Structure

- `deadlift_data.csv`: The dataset containing deadlift data used for training and testing the model.
- `Deadlift_Classifier_Model.ipynb`: Jupyter notebook for creating and training the random forest classifier model.
- `DeadLift_dataCollector.ipynb`: Jupyter notebook for collecting and preprocessing deadlift data.
- `Deadlift_Tracker.ipynb`: Jupyter notebook for tracking deadlift performance using the trained classifier.
- `random_forest_model.pkl`: The saved random forest model trained on the deadlift dataset.

## Description

### Deadlift Data Collection
The `DeadLift_dataCollector.ipynb` notebook is used for collecting and preprocessing deadlift data. It includes steps for data cleaning, feature extraction, and preparation for model training. The final dataset is saved as `deadlift_data.csv`.

### Deadlift Classifier Model
The `Deadlift_Classifier_Model.ipynb` notebook contains the code for creating and training a random forest classifier. The model is trained on the preprocessed deadlift dataset to classify different aspects of deadlift exercises. The trained model is saved as `random_forest_model.pkl`.

### Deadlift Tracker
The `Deadlift_Tracker.ipynb` notebook is used for tracking deadlift performance. It utilizes the trained random forest model to analyze new deadlift data and provide insights into the performance. The tracker can help in monitoring progress and identifying areas for improvement.

## Getting Started

### Prerequisites
- Python 3.x
- Jupyter Notebook
- Required Python libraries: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`

### Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/your-username/deadlift-tracker-and-classifier.git
    cd deadlift-tracker-and-classifier
    ```

2. Install the required Python libraries:
    ```sh
    pip install -r requirements.txt
    ```

### Usage

1. **Data Collection**:
    - Open `DeadLift_dataCollector.ipynb` in Jupyter Notebook.
    - Follow the steps to collect and preprocess deadlift data.
    - Save the preprocessed data as `deadlift_data.csv`.

2. **Model Training**:
    - Open `Deadlift_Classifier_Model.ipynb` in Jupyter Notebook.
    - Load the `deadlift_data.csv` dataset.
    - Train the random forest classifier model.
    - Save the trained model as `random_forest_model.pkl`.

3. **Deadlift Tracking**:
    - Open `Deadlift_Tracker.ipynb` in Jupyter Notebook.
    - Load the saved `random_forest_model.pkl` model.
    - Use the model to track and analyze deadlift performance.

## Contributing
Contributions are welcome! If you have any suggestions or improvements, please create a pull request or open an issue.

## License
This project is created by Ayushman Ranjan

## Acknowledgments
- Special thanks to the open-source community for providing the tools and libraries used in this project.
- Thanks to everyone who contributed to the dataset and model training.

