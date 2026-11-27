"""Run the whole pipeline end to end: data -> train -> score.

    python run_pipeline.py
"""
from src import generate_data, score, train

if __name__ == "__main__":
    print("1/3 Generating synthetic data...")
    generate_data.main()
    print("\n2/3 Training model...")
    train.main()
    print("\n3/3 Scoring active accounts...")
    score.main()
    print("\nDone. Start the app with:  streamlit run app.py")
