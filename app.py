from flask import Flask, request, render_template, jsonify
import pandas as pd
import pickle
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
import json
from flask import Flask, request, jsonify
import os
import uuid

app = Flask(__name__)
trained_models = {}


def train_arima_model(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
    """
        Train an ARIMA model on the given time series data.

        :param data: Time series data
        :param order: ARIMA order parameter
        :param seasonal_order: Seasonal ARIMA order parameter
        :return: Trained ARIMA model
        """
    model = sm.tsa.statespace.SARIMAX(data, order=order, seasonal_order=seasonal_order).fit()
    return model


def save_csv_file(file):
    """
        Save a CSV file to the specified folder with a unique filename.

        :param file: File to save
        :param folder: Destination folder
        :return: Unique filename
        """
    unique_filename = str(uuid.uuid4()) + '.csv'
    file_path = os.path.join('uploads', unique_filename)
    file.save(file_path)
    return unique_filename


@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"})

        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "No selected file"})

        # Check if the file is a CSV file
        if file and file.filename.endswith('.csv'):
            # Save the file to a specific folder on the server
            file.save(os.path.join('uploads', 'loaded_data1.csv'))
            return jsonify({"success": "File uploaded successfully"})
        elif file.filename.endswith(('.xlsx', '.xls')):
            # Save Excel file
            excel_filename = os.path.join('uploads', 'loaded_data1.xlsx')
            file.save(excel_filename)

            # Read Excel file and save as CSV
            read_file = pd.read_excel(excel_filename)
            csv_filename = save_csv_file(file)
            csv_filepath = os.path.join('uploads', csv_filename)
            read_file.to_csv(csv_filepath, index=False)

            return jsonify({"success": "Excel file uploaded successfully", "filename": csv_filename})
        else:
            return jsonify({"error": "Invalid file format. Please upload a CSV file."})
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

        gross_sales_by_month['Month'] = gross_sales_by_month['Month'].dt.strftime('%Y-%m-%d')
        # Convert the DataFrame to a list of dictionaries
        data_list = gross_sales_by_month.to_dict(orient='records')

        file_path = "output.json"

        # Write the data to the JSON file
        with open(file_path, "w") as json_file:
            json.dump(data_list, json_file, default=str, indent=4)

        # Convert the list to JSON and return the response
        return jsonify(data_list)

    except Exception as e:
        return {"error": str(e)}


@app.route('/get_last_item', methods=['GET'])
def get_last_item_from_json(file_path="output.json"):
    try:
        with open(file_path, "r") as json_file:
            data_list = json.load(json_file)

        # Get the last item from the list
        last_item = data_list[-1]

        return last_item
    except Exception as e:
        print(f"Error: {e}")
        return None


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

        # Find the last available month in the dataset
        last_available_month = actual_data.index[-1]

        # Forecast the next two months based on the last available month
        forecast_steps = 3  # Adjust this value to forecast more steps into the future
        future_index = pd.date_range(start=last_available_month + pd.DateOffset(months=1), periods=forecast_steps, freq='M')
        predictions = trained_model.forecast(steps=forecast_steps, index=future_index)

        # Convert predictions to a list
        predictions_list = predictions.tolist()

        # Return the prediction for the next two months
        return jsonify({
            "predictions": predictions_list,
            "future_months": future_index.strftime('%Y-%m').tolist()
        })

    except Exception as e:
        # Handle any exceptions, e.g., invalid CSV format or missing columns
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    # Run the Flask app on the local network (0.0.0.0) and port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
