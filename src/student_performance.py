"""
INTEGRATED CoE PROJECT: Student Performance Analysis and Prediction
Modules covered:
1. Python Programming (Functions, OOP, Data Types, Control Flow)
2. OOP and Database Integration (SQLite)
3. NumPy Arrays (Statistical Operations)
4. Pandas and Data Visualization (Data Cleaning, Aggregation, Matplotlib)
5. Fundamentals of Machine Learning (Random Forest Classifier, Evaluation)

Author: Aman Kumar
Roll Number: 2400320100154
Admission Number: 2024B0101506
Section: CSE 12
"""

import os
import sqlite3
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Determine base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class StudentPerformanceProject:
    """Main class combining data handling, database operations, visualization, and ML."""

    def __init__(
        self,
        csv_file: Optional[str] = None,
        database_file: Optional[str] = None,
        output_dir: Optional[str] = None,
    ):
        """Initialize file paths and attributes."""
        if csv_file is None:
            primary_path = os.path.join(BASE_DIR, "data", "student_data.csv")
            fallback_path = os.path.join(BASE_DIR, "student_data.csv")
            self.csv_file = primary_path if os.path.exists(primary_path) else fallback_path
        else:
            self.csv_file = csv_file

        if database_file is None:
            self.database_file = os.path.join(BASE_DIR, "data", "student_performance.db")
        else:
            self.database_file = database_file

        if output_dir is None:
            self.output_dir = os.path.join(BASE_DIR, "outputs")
        else:
            self.output_dir = output_dir

        os.makedirs(os.path.dirname(self.database_file), exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        self.data: Optional[pd.DataFrame] = None
        self.model: Optional[RandomForestClassifier] = None

    # MODULE 4: Pandas data manipulation
    def load_and_analyse_data(self) -> pd.DataFrame:
        """Load dataset using Pandas and print descriptive summaries."""
        print("=" * 60)
        print("MODULE 4: PANDAS DATA INGESTION & SUMMARY")
        print("=" * 60)
        self.data = pd.read_csv(self.csv_file)
        print(f"Loaded dataset from: {self.csv_file}")
        print("\n--- FIRST FIVE RECORDS ---")
        print(self.data.head())
        print("\n--- DATASET INFORMATION ---")
        print(self.data.info())
        print("\n--- STATISTICAL SUMMARY ---")
        print(self.data.describe())
        print("\n--- MISSING VALUES PER COLUMN ---")
        print(self.data.isnull().sum())
        return self.data

    # MODULE 1 & 3: Python functions and NumPy
    def numpy_analysis(self) -> dict:
        """Convert final scores to a NumPy array and compute statistics."""
        print("\n" + "=" * 60)
        print("MODULE 1 & 3: PYTHON FUNCTIONS & NUMPY ANALYSIS")
        print("=" * 60)
        if self.data is None:
            raise ValueError("Data not loaded. Run load_and_analyse_data() first.")

        marks = self.data["final_score"].to_numpy()
        stats = {
            "mean": round(float(np.mean(marks)), 2),
            "max": float(np.max(marks)),
            "min": float(np.min(marks)),
            "std": round(float(np.std(marks)), 2),
        }
        print(f"Total Student Scores Analyzed: {len(marks)}")
        print(f"Average score        : {stats['mean']}")
        print(f"Highest score        : {stats['max']}")
        print(f"Lowest score         : {stats['min']}")
        print(f"Standard deviation   : {stats['std']}")
        return stats

    # MODULE 2: Database integration using SQLite
    def save_to_database(self) -> pd.DataFrame:
        """Store student dataset in SQLite database and query top records."""
        print("\n" + "=" * 60)
        print("MODULE 2: OOP & SQLITE DATABASE INTEGRATION")
        print("=" * 60)
        if self.data is None:
            raise ValueError("Data not loaded. Run load_and_analyse_data() first.")

        connection = sqlite3.connect(self.database_file)
        self.data.to_sql("students", connection, if_exists="replace", index=False)
        print(f"Records successfully written to SQLite table 'students' at: {self.database_file}")

        saved = pd.read_sql_query("SELECT * FROM students LIMIT 5", connection)
        print("\n--- FIRST FIVE RECORDS RETRIEVED FROM SQLITE ---")
        print(saved)
        connection.close()
        return saved

    # MODULE 4: Data visualization
    def create_visualizations(self, show_plots: bool = False) -> list:
        """Generate exploratory visualizations and save them to output directory."""
        print("\n" + "=" * 60)
        print("MODULE 4: DATA VISUALIZATION WITH MATPLOTLIB")
        print("=" * 60)
        if self.data is None:
            raise ValueError("Data not loaded. Run load_and_analyse_data() first.")

        generated_files = []

        # 1. Performance distribution bar chart
        plt.figure(figsize=(7, 5))
        self.data["performance"].value_counts().plot(
            kind="bar",
            color=["#3498db", "#e74c3c"],
            edgecolor="black",
        )
        plt.title("Student Performance Distribution", fontsize=14, fontweight="bold")
        plt.xlabel("Performance Category", fontsize=12)
        plt.ylabel("Number of Students", fontsize=12)
        plt.xticks(rotation=0)
        plt.tight_layout()
        dist_path = os.path.join(self.output_dir, "performance_distribution.png")
        plt.savefig(dist_path, dpi=300)
        print(f"Saved: {dist_path}")
        generated_files.append(dist_path)
        if show_plots:
            plt.show()
        plt.close()

        # 2. Study hours vs Final score scatter plot
        plt.figure(figsize=(7, 5))
        plt.scatter(
            self.data["study_hours"],
            self.data["final_score"],
            c="#2ecc71",
            edgecolors="#27ae60",
            alpha=0.7,
        )
        plt.title("Study Hours vs Final Score", fontsize=14, fontweight="bold")
        plt.xlabel("Study Hours per Day", fontsize=12)
        plt.ylabel("Final Score", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        scatter_path = os.path.join(self.output_dir, "study_hours_vs_score.png")
        plt.savefig(scatter_path, dpi=300)
        print(f"Saved: {scatter_path}")
        generated_files.append(scatter_path)
        if show_plots:
            plt.show()
        plt.close()

        # 3. Correlation Heatmap
        plt.figure(figsize=(8, 6))
        numeric = self.data.select_dtypes(include="number")
        corr = numeric.corr()
        plt.imshow(corr, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
        plt.colorbar(label="Pearson Correlation")
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
        plt.yticks(range(len(corr.columns)), corr.columns)
        for i in range(len(corr.columns)):
            for j in range(len(corr.columns)):
                plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", color="black", fontsize=9)
        plt.title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
        plt.tight_layout()
        heatmap_path = os.path.join(self.output_dir, "correlation_heatmap.png")
        plt.savefig(heatmap_path, dpi=300)
        print(f"Saved: {heatmap_path}")
        generated_files.append(heatmap_path)
        if show_plots:
            plt.show()
        plt.close()

        return generated_files

    # MODULE 5: Machine Learning
    def train_machine_learning_model(self) -> float:
        """Train and evaluate Random Forest Classifier."""
        print("\n" + "=" * 60)
        print("MODULE 5: MACHINE LEARNING (RANDOM FOREST)")
        print("=" * 60)
        if self.data is None:
            raise ValueError("Data not loaded. Run load_and_analyse_data() first.")

        features = [
            "study_hours",
            "attendance",
            "previous_score",
            "assignments_completed",
            "sleep_hours",
            "participation_level",
        ]
        X = self.data[features]
        y = self.data["performance"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions) * 100
        print(f"Model Training Completed Successfully!")
        print(f"Test Accuracy: {round(accuracy, 2)}%")
        print("\n--- CLASSIFICATION REPORT ---")
        print(classification_report(y_test, predictions))
        print("--- CONFUSION MATRIX ---")
        print(confusion_matrix(y_test, predictions))
        return accuracy

    def predict_new_student(self, sample_data: Optional[dict] = None) -> str:
        """Predict performance category for a sample student."""
        print("\n" + "=" * 60)
        print("SAMPLE INFERENCE")
        print("=" * 60)
        if self.model is None:
            raise ValueError("Model not trained. Run train_machine_learning_model() first.")

        if sample_data is None:
            sample_data = {
                "study_hours": 4.0,
                "attendance": 85.0,
                "previous_score": 72.0,
                "assignments_completed": 8,
                "sleep_hours": 7.0,
                "participation_level": 4,
            }

        new_student = pd.DataFrame([sample_data])
        prediction = self.model.predict(new_student)[0]
        print("Input Features:")
        for k, v in sample_data.items():
            print(f"  - {k}: {v}")
        print(f"\n=> PREDICTED PERFORMANCE CATEGORY: {prediction}")
        return prediction


def main():
    """Execute complete end-to-end data pipeline."""
    project = StudentPerformanceProject()
    project.load_and_analyse_data()
    project.numpy_analysis()
    project.save_to_database()
    project.create_visualizations(show_plots=False)
    project.train_machine_learning_model()
    project.predict_new_student()


if __name__ == "__main__":
    main()
