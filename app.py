from flask import Flask, render_template, request
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import io

app = Flask(__name__)

# Load the trained logistic regression model for numerical detection
numerical_model = joblib.load(r"C:\Users\Admin\Music\project\model\logistic_regression_model.pkl")

# Load the trained deep learning model for image detection
image_model = load_model(r"C:\Users\Admin\Music\project\model\classifier_model.h5")

# Define the desired width and height for the image detection model input
desired_width = 100
desired_height = 100

# Route to the main selection page
@app.route('/')
def index():
    return render_template('index.html')  # This will show options for numerical or image detection

# Route for the first detection page
@app.route('/detection')
def detection():
    return render_template('detection.html')

# Route for the second detection page
@app.route('/detection2')
def detection2():
    return render_template('detection2.html')

# Route for the numerical detection page
@app.route('/numerical', methods=['GET', 'POST'])
def numerical_detection():
    if request.method == 'POST':
        # Get the form data from the HTML form
        weight = float(request.form['weight'])
        hb = float(request.form['hb'])
        cycle_length = float(request.form['cycle_length'])
        marriage_years = float(request.form['marriage_years'])
        pregnant = 1 if request.form['pregnant'].lower() == 'y' else 0
        abortions = float(request.form['abortions'])
        waist = float(request.form['waist'])
        weight_gain = 1 if request.form['weight_gain'].lower() == 'y' else 0
        hair_growth = 1 if request.form['hair_growth'].lower() == 'y' else 0
        skin_darkening = 1 if request.form['skin_darkening'].lower() == 'y' else 0
        hair_loss = 1 if request.form['hair_loss'].lower() == 'y' else 0
        pimples = 1 if request.form['pimples'].lower() == 'y' else 0
        fast_food = 1 if request.form['fast_food'].lower() == 'y' else 0
        exercise = 1 if request.form['exercise'].lower() == 'y' else 0

        # Prepare the data for prediction
        input_data = np.array([[weight, hb, cycle_length, marriage_years, pregnant, abortions, waist,
                                weight_gain, hair_growth, skin_darkening, hair_loss, pimples, fast_food, exercise]])

        # Make a prediction using the logistic regression model
        prediction = numerical_model.predict(input_data)
        result = 'PCOS Detected' if prediction[0] == 1 else 'No PCOS Detected'

        # Render the result to the HTML page
        return render_template('detection2.html', result=result)

    return render_template('detection2.html')

# Route for the image detection page
@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded image file
    file = request.files['image']

    # Read the image file as bytes
    img_bytes = file.read()

    # Create an in-memory stream from the bytes
    img_stream = io.BytesIO(img_bytes)

    # Load and preprocess the image
    img = image.load_img(img_stream, target_size=(desired_width, desired_height))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Make prediction
    pred = image_model.predict(img_array)

    # Convert prediction to human-readable format
    predicted_class = "Infected" if pred[0][0] < 0.5 else "Not Infected"

    # Return predictions
    return render_template('detection.html', predicted_class=predicted_class)
    #return jsonify({'predicted_class': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
