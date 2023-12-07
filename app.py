from flask import Flask, request, render_template, jsonify
import pandas as pd
import pickle
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
import json

app = Flask(__name__)
trained_models = {}


def train_arima_model(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
    model = sm.tsa.statespace.SARIMAX(data, order=order, seasonal_order=seasonal_order).fit()
    return model


@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        # Check if the post request has the file part
        if 'file' not in request.files:
            return jsonify({"error": "No file part"})

        file = request.files['file']

        # Check if the file is not empty
        if file.filename == '':
            return jsonify({"error": "No selected file"})

        # Check if the file is a CSV file
        if file and file.filename.endswith('.csv'):
            # Save the file to a specific folder on the server
            file.save('uploads/loaded_data1.csv')

            return jsonify({"success": "File uploaded successfully"})
        else:
            return jsonify({"error": "Invalid file format. Please upload a CSV file."})
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/monthly_sales', methods=['GET'])
def get_monthly_sales():
    try:
        # Replace 'your_csv_file.csv' with the actual path to your CSV file
        csv_file_path = 'uploads/loaded_data.csv'

        # Read the CSV file into a pandas DataFrame
        monthly_sales_data = pd.read_csv(csv_file_path)
        monthly_sales_data['Date'] = pd.to_datetime(monthly_sales_data['Month'], errors='coerce')
        monthly_sales_data = monthly_sales_data.dropna(subset=['Date'])
        original_column_name = monthly_sales_data.columns[1]
        monthly_sales_data = monthly_sales_data.rename(columns={original_column_name: 'Sales'})

        # Convert Timestamps to strings
        monthly_sales_data['Date'] = monthly_sales_data['Date'].dt.strftime('%Y-%m-%d')

        # Convert the DataFrame to a list of dictionaries
        data_list = monthly_sales_data.to_dict(orient='records')

        # Specify the file path where you want to save the JSON file
        file_path = "output.json"

        # Write the data to the JSON file
        with open(file_path, "w") as json_file:
            json.dump(data_list, json_file, indent=4)

        return jsonify(data_list)
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/gross_sales', methods=['GET'])
def calculate_gross_sales():
    try:
        csv_file_path = 'uploads/loaded_data.csv'
        monthly_sales_data = pd.read_csv(csv_file_path)
        monthly_sales_data['Month'] = pd.to_datetime(monthly_sales_data['Month'], errors='coerce')
        monthly_sales_data = monthly_sales_data.dropna(subset=['Month'])
        original_column_name = monthly_sales_data.columns[1]
        monthly_sales_data = monthly_sales_data.rename(columns={original_column_name: 'Sales'})

        # Group by 'Month' and calculate the sum for each month
        gross_sales_by_month = monthly_sales_data.groupby('Month')['Sales'].sum().reset_index()

        # Calculate the percentage increase from the last month
        gross_sales_by_month['percent_increase'] = gross_sales_by_month['Sales'].pct_change() * 100

        # Replace NaN values in the 'Percentage_Increase' column with a string representation
        gross_sales_by_month['percent_increase'] = gross_sales_by_month['percent_increase'].apply(
            lambda x: f"{round(x, 2)}%" if not pd.isna(x) else 'NaN'
        )

        # Replace NaT values in the 'Month' column with a string representation
        gross_sales_by_month['Month'] = gross_sales_by_month['Month'].fillna('Unknown')

        # Calculate cumulative gross sales by adding sales from the last month to the current month
        gross_sales_by_month['cumulative_gross_sales'] = gross_sales_by_month['Sales'].cumsum()

        # Convert the DataFrame to a list of dictionaries
        data_list = gross_sales_by_month.to_dict(orient='records')

        # Convert the list to JSON and return the response
        return jsonify(data_list)

    except Exception as e:
        return {"error": str(e)}


@app.route('/')
def hello_world():
    return render_template('upload_form.html')


@app.route('/train_model', methods=['GET'])
def train_model_endpoint():
    try:
        df = pd.read_csv('uploads/loaded_data.csv')
        df['Month'] = pd.to_datetime(df['Month'], errors='coerce')
        df = df.dropna(subset=['Month'])
        original_column_name = df.columns[1]
        df = df.rename(columns={original_column_name: 'Sales'})

        # Assuming 'Month' and 'Sales' columns are present in the CSV
        if 'Month' not in df.columns or 'Sales' not in df.columns:
            return {"error": "CSV file must contain 'Month' and 'Sales' columns"}

        df['Month'] = pd.to_datetime(df['Month'])
        df.set_index('Month', inplace=True)

        # Assuming df contains the time series data
        order = (1, 1, 1)  # Set the appropriate order for your ARIMA model
        seasonal_order = (1, 1, 1, 12)
        # Train the ARIMA model
        trained_model = train_arima_model(df['Sales'], order=order, seasonal_order=seasonal_order)

        # Save the trained model to a pickle file
        model_filename = f"trained_model.pkl"
        with open(model_filename, 'wb') as model_file:
            pickle.dump(trained_model, model_file)

        # Store the model filename in the dictionary for later retrieval
        trained_models[model_filename] = order

        return {"message": f"ARIMA Model trained successfully. Model saved as {model_filename}"}

    except Exception as e:
        # Handle any exceptions, e.g., invalid CSV format or missing columns
        return {"error": str(e)}


@app.route('/predict', methods=['GET'])
def predict_endpoint():
    try:
        # Load the trained ARIMA model from the pickle file
        with open('trained_model.pkl', 'rb') as model_file:
            trained_model = pickle.load(model_file)

        # Load actual data for comparison
        actual_data = pd.read_csv('uploads/loaded_data.csv')
        actual_data['Month'] = pd.to_datetime(actual_data['Month'], errors='coerce')
        actual_data = actual_data.dropna(subset=['Month'])
        original_column_name = actual_data.columns[1]
        actual_data = actual_data.rename(columns={original_column_name: 'Sales'})

        # Set the 'Month' column as the index
        actual_data.set_index('Month', inplace=True)

        # Make predictions using the trained ARIMA model
        predictions = trained_model.forecast(steps=len(actual_data) + 1)[-1]

        # Extract the actual value for the next month
        actual_value = actual_data.iloc[-1]['Sales']

        # Calculate Mean Absolute Error (MAE)
        mae = mean_absolute_error([actual_value], [predictions])

        # Calculate Accuracy (assuming a threshold for acceptable difference)
        threshold = 0.1  # Set your own threshold
        absolute_difference = abs(predictions - actual_value)
        accuracy = 100 - (absolute_difference / actual_value) * 100

        # Return the prediction along with accuracy and MAE
        return jsonify({"prediction": round(predictions, 2), "accuracy": round(accuracy, 2), "mae": round(mae, 2),
                        "actual_value": actual_value})

    except Exception as e:
        # Handle any exceptions, e.g., invalid CSV format or missing columns
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    # Run the Flask app on the local network (0.0.0.0) and port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
