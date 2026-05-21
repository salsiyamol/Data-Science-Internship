import os
import joblib

# ... Your model training code above ...
# Assuming your trained model variable is named 'model'

print("Training complete. Saving model...")

# 1. Get the main project root folder (CNN_project/)
TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TRAIN_DIR)

# 2. Point directly to the main folder where app.py expects it
save_path = os.path.join(PROJECT_ROOT, "fashion_model_package.joblib")

# 3. Safely delete the old corrupted file if it exists
if os.path.exists(save_path):
    os.remove(save_path)

# 4. Save the new model cleanly
joblib.dump(model, save_path)
print(f"✅ Model successfully saved outside the subfolder at: {save_path}")
import os
import joblib

# ... Your model training code above ...
# Assuming your trained model variable is named 'model'

print("训练完成 / Training complete. Starting file export...")

# 1. Get absolute paths
TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TRAIN_DIR)
save_path = os.path.join(PROJECT_ROOT, "fashion_model_package.joblib")

# 2. Force delete any broken remnants 
if os.path.exists(save_path):
    try:
        os.remove(save_path)
    except OSError:
        pass

# 3. Clean dump with absolute confirmation
try:
    with open(save_path, "wb") as f:
        joblib.dump(model, f)
    
    # Verify file size right after saving to ensure it isn't 0 bytes
    file_size = os.path.getsize(save_path)
    if file_size > 0:
        print(f"✅ SUCCESS: Model saved perfectly! Size: {file_size / 1024:.2f} KB")
        print(f"Location: {save_path}")
    else:
        print("❌ ERROR: File was written but size is still 0 bytes. Check if training completely finished.")
except Exception as e:
    print(f"❌ Failed to save model: {e}")