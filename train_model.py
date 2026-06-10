import os
import sys
import joblib  # 📦 Required to save .pkl files

# Set up Django environment manually for a standalone script
import django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "health_navigator.settings")
django.setup()

from preprocess_data import load_and_preprocess_dataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def train_predictive_ai():
    # 🛰️ Unpack data arrays AND the preprocessor transformation engine from your script
    X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_dataset()
    
    print("\n🤖 Initializing Random Forest AI Engine...")
    model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
    
    print("🏋️‍♂️ Training the AI model on your patient records...")
    model.fit(X_train, y_train)
    print("🎯 Model training complete!")

    # 💾 THE PKL SAVE TRACK: Freezing files directly to your root folder next to manage.py
    print("\n📦 Exporting trained components to storage...")
    joblib.dump(model, 'length_of_stay_model.pkl')
    joblib.dump(preprocessor, 'data_preprocessor.pkl')
    print("✅ Successfully saved: length_of_stay_model.pkl")
    print("✅ Successfully saved: data_preprocessor.pkl")

    print("\n🔮 Evaluation Metrics:")
    predictions = model.predict(X_test)
    print(f"📐 Mean Absolute Error (MAE): {mean_absolute_error(y_test, predictions):.2f} Days")

if __name__ == "__main__":
    train_predictive_ai()