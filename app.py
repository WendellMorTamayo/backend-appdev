# app.py
import uuid
import re
from flask import Flask, request, render_template, jsonify
from utils.helper import train_arima_model, save_csv_file, get_last_item_from_json, get_sales_performance_history, \
    trained_models, parse_date_xy, detect_date_format, date_parser, process_data
import os
import pickle
import json
import pandas as pd

app = Flask(__name__)


@app.route('/upload_demands')
def upload_demands():
    return render_template('upload_demand_form.html')


@app.route('/upload_demand', methods=['POST'])
def upload_demand_file():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"})

        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "No selected file"})

        # Generate a unique filename
        unique_filename = 'loaded_data_demand.csv'
        file_path = os.path.join('uploads', unique_filename)

        # Check if the file is a CSV or Excel file
        if file and file.filename.endswith('.csv'):
            # Save the file to a specific folder on the server
            file_path = os.path.join('uploads', unique_filename)
            file.save(file_path)

            # Read the file into a DataFrame
            df = pd.read_csv(file_path)

            # Group by product ID and save to separate CSV files
            process_data_result = process_data(df)

            # Call the train_demand_model_endpoint to train the ARIMA model
            train_result = train_demand_model_endpoint()
            predict_result = predict_demand_endpoint()

            return jsonify(
                {"success": "CSV file uploaded successfully",
                 "filename": unique_filename,
                 "result": process_data_result,
                 "train_result": train_result})
        elif file.filename.endswith(('.xlsx', '.xls')):
            # Save Excel file
            excel_filename = os.path.join('uploads', 'loaded_data_demand.xlsx')
            file.save(excel_filename)

            # Read Excel file and save as CSV
            read_file = pd.read_excel(excel_filename)
            csv_filepath = os.path.join('uploads', unique_filename)
            read_file.to_csv(csv_filepath, index=False, header=True)  # Ensure to include header if needed

            df = pd.read_csv(csv_filepath)
            print(df)
            print(csv_filepath)
            # Group by product ID and save to separate CSV files
            process_data_result = process_data(df)

            # Call the train_demand_model_endpoint to train the ARIMA model
            train_result = train_demand_model_endpoint()
            predict_result = predict_demand_endpoint()

            return jsonify(
                {"success": "Excel file uploaded successfully", "filename": unique_filename,
                 "train_result": train_result})
        else:
            return jsonify({"error": "Invalid file format. Please upload a CSV or Excel file."})
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/train_demand_model', methods=['GET'])
def train_demand_model_endpoint():
    try:
        # Specify the output folder for saving trained models
        output_folder = 'models/product_train/'

        # Iterate over each CSV file in the specified folder
        for filename in os.listdir('models/product_csv'):
            if filename.endswith('.csv'):
                # Construct the full file path
                file_path = os.path.join('models/product_csv', filename)

                # Retrieve the DataFrame from the CSV file
                df = pd.read_csv(file_path, parse_dates=[0], date_parser=date_parser)

                # Check if required columns are present in the DataFrame
                required_columns = {'Month', 'ProductID', 'UnitsSold'}
                if not required_columns.issubset(df.columns):
                    return {
                        "error": f"DataFrame in {file_path} must contain 'Month', 'ProductID', and 'UnitsSold' columns"}

                # Group by 'ProductID' and train the ARIMA model for each group
                for product_id, product_group in df.groupby('ProductID'):
                    # Assuming df contains the time series data
                    order = (1, 1, 1)  # Set the appropriate order for your ARIMA model
                    seasonal_order = (1, 1, 1, 12)

                    # Train the ARIMA model on the product-specific data
                    trained_model = train_arima_model(product_group['UnitsSold'], order=order,
                                                      seasonal_order=seasonal_order)

                    # Save the trained model to a pickle file for each product ID
                    model_filename = os.path.join(output_folder, f"product_{product_id}_data.pkl")
                    with open(model_filename, 'wb') as model_file:
                        pickle.dump(trained_model, model_file)

        return {"message": "ARIMA Models trained successfully for each product ID"}

    except Exception as e:
        # Handle any exceptions, e.g., invalid DataFrame format or missing columns
        return {"error": str(e)}

@app.route('/predict_demand', methods=['GET'])
def predict_demand_endpoint():
    try:
        # Replace 'models/product_train/' with the actual path to your trained models
        model_folder = 'models/product_train/'

        # Create a dictionary to store predictions for each product
        product_predictions = {}

        # Load the actual data for comparison
        actual_data = pd.read_csv('uploads/loaded_data_demand.csv')
        actual_data['Month'] = pd.to_datetime(actual_data['Month'], errors='coerce')
        actual_data = actual_data.dropna(subset=['Month'])

        # Set the 'Month' column as the index
        actual_data.set_index('Month', inplace=True)

        # Find the last available month in the dataset
        last_available_month = actual_data.index[-1]

        # Replace 'models/product_train/' with the actual path to your trained models
        model_folder = 'models/product_train/'
        sorted_product_predictions = {}
        # Iterate over each pickle file in the specified folder
        for filename in os.listdir(model_folder):
            if filename.endswith('.pkl'):
                # Construct the full file path for the pickle file
                pickle_file_path = os.path.join(model_folder, filename)

                # Load the trained ARIMA model from the pickle file
                trained_model = pd.read_pickle(pickle_file_path)

                # Extract the product ID from the filename using regex
                match = re.match(r'product_(\d+)_data.pkl', filename)
                if match:
                    product_id = int(match.group(1))  # Convert ProductID to integer
                else:
                    # Handle the case where the filename doesn't match the expected pattern
                    product_id = 'Unknown'

                product_name = actual_data[actual_data['ProductID'] == product_id]['Product'].iloc[0]

                # Forecast the next month based on the last available month
                forecast_steps = 1  # Adjust this value to forecast more steps into the future
                future_index = pd.date_range(start=last_available_month + pd.DateOffset(months=1),
                                             periods=forecast_steps, freq='M')

                # Use the trained ARIMA model to predict the next month's demand
                predictions = trained_model.get_forecast(steps=forecast_steps).predicted_mean

                # Convert predictions to a DataFrame
                forecast_df = pd.DataFrame({
                    'Month': future_index,
                    'ProductID': product_id,
                    'Product': product_name,
                    'UnitsSold': predictions.tolist(),
                })

                # Sort the DataFrame by 'ProductID' and 'Month'
                forecast_df['ProductID'] = forecast_df['ProductID'].astype(int)

                # Convert Timestamp to string for JSON serialization
                future_index_str = forecast_df['Month'].dt.strftime('%Y-%m').tolist()

                # Store forecast_df in the dictionary for each product
                product_predictions[product_id] = {
                    "predictions": forecast_df.to_dict(orient='records'),
                    "future_months": future_index_str
                }
                sorted_product_predictions = dict(sorted(product_predictions.items(), key=lambda x: int(x[0])))

        # Return the product-wise forecasted demand in JSON format
        output_file_path = 'models/product_predict/sorted_product_predictions.json'  # Add .json to the file path
        with open(output_file_path, 'w') as json_file:
            json.dump(sorted_product_predictions, json_file, indent=4, default=str)

        return jsonify(sorted_product_predictions)

    except Exception as e:
        # Handle any exceptions, e.g., invalid DataFrame format or missing columns
        return {"error": str(e)}


@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"})

        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "No selected file"})

        if file and file.filename.endswith(('.csv', '.xlsx', '.xls')):
            df = pd.read_csv(file) if file.filename.endswith('.csv') else pd.read_excel(file)

            # Detect date format and apply the appropriate parsing function
            detected_format = detect_date_format(df['Month'])
            print("Detected Format:", detected_format)
            df['Month'] = pd.to_datetime(df['Month'], errors='coerce')
            df = df.dropna(subset=['Month'])
            original_column_name = df.columns[1]
            df = df.rename(columns={original_column_name: 'Sales'})
            if detected_format == "%Y-%m":
                file_path = os.path.join('uploads', 'loaded_data.csv')
                save_csv_file(df, file_path)
                train_model_endpoint()
                calculate_gross_sales()
                return jsonify({"success": "File uploaded and saved successfully"})
            elif detected_format == "%y-%m":
                # Ensure that the parse_date_xy function is defined and handles parsing correctly
                df['Month'] = df['Month'].apply(parse_date_xy)
                # Save the modified DataFrame to a new CSV file
                file_path = os.path.join('uploads', 'loaded_data.csv')
                save_csv_file(df, file_path)
                train_model_endpoint()
                calculate_gross_sales()
                return jsonify({"success": "File uploaded and saved successfully"})
            else:
                return jsonify({"error": "Invalid date format. Please use either %Y-%m or %y-%m"})

        else:
            return jsonify({"error": "Invalid file format. Please upload a CSV, XLS, or XLSX file."})

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

        gross_sales_by_month = monthly_sales_data.groupby('Month')['Sales'].sum().reset_index()
        gross_sales_by_month['percent_increase'] = gross_sales_by_month['Sales'].pct_change() * 100
        gross_sales_by_month['percent_increase'] = gross_sales_by_month['percent_increase'].apply(
            lambda x: f"{round(x, 2)}%" if not pd.isna(x) else 'NaN'
        )
        gross_sales_by_month['Month'] = gross_sales_by_month['Month'].fillna('Unknown')
        gross_sales_by_month['cumulative_gross_sales'] = gross_sales_by_month['Sales'].cumsum()
        gross_sales_by_month['Month'] = gross_sales_by_month['Month'].dt.strftime('%Y-%m-%d')
        data_list = gross_sales_by_month.to_dict(orient='records')
        file_path = "output.json"

        with open(file_path, "w") as json_file:
            json.dump(data_list, json_file, default=str, indent=4)

        return jsonify(data_list)

    except Exception as e:
        return {"error": str(e)}


@app.route('/get_last_item', methods=['GET'])
def get_last_item():
    return jsonify(get_last_item_from_json())


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

        if 'Month' not in df.columns or 'Sales' not in df.columns:
            return {"error": "CSV file must contain 'Month' and 'Sales' columns"}

        df['Month'] = pd.to_datetime(df['Month'])
        df.set_index('Month', inplace=True)

        order = (1, 1, 1)
        seasonal_order = (1, 1, 1, 12)
        trained_model = train_arima_model(df['Sales'], order=order, seasonal_order=seasonal_order)

        model_filename = f"trained_model.pkl"
        with open(model_filename, 'wb') as model_file:
            pickle.dump(trained_model, model_file)

        trained_models[model_filename] = order

        return {"message": f"ARIMA Model trained successfully. Model saved as {model_filename}"}

    except Exception as e:
        return {"error": str(e)}


@app.route('/predict', methods=['GET'])
def predict_endpoint():
    try:
        with open('trained_model.pkl', 'rb') as model_file:
            trained_model = pickle.load(model_file)

        actual_data = pd.read_csv('uploads/loaded_data.csv')
        actual_data['Month'] = pd.to_datetime(actual_data['Month'], errors='coerce')
        actual_data = actual_data.dropna(subset=['Month'])
        original_column_name = actual_data.columns[1]
        actual_data = actual_data.rename(columns={original_column_name: 'Sales'})

        actual_data.set_index('Month', inplace=True)
        last_available_month = actual_data.index[-1]
        print(last_available_month)
        forecast_steps = 1
        future_index = pd.date_range(start=last_available_month, periods=forecast_steps + 1, freq='M')[1:]

        predictions = round(trained_model.forecast(steps=forecast_steps, index=future_index), 2)
        predictions_list = predictions.tolist()
        last_month_value = actual_data.iloc[-1]['Sales']
        percentage_increase = round((predictions_list[0] - last_month_value) / last_month_value * 100, 2)

        return jsonify({
            "prediction": predictions_list[0],
            "next_month": future_index[0].strftime('%Y-%m'),
            "percentage_increase": percentage_increase
        })

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/get_sales_data', methods=['GET'])
def get_sales_data():
    try:
        actual_data = pd.read_csv('uploads/loaded_data.csv')
        actual_data['Month'] = pd.to_datetime(actual_data['Month'], errors='coerce')
        actual_data = actual_data.dropna(subset=['Month'])
        if actual_data.columns[1] != 'Sales':
            original_column_name = actual_data.columns[1]
            actual_data = actual_data.rename(columns={original_column_name: 'Sales'})

        # Group by 'Month' and calculate the sum for each month
        monthly_sales_data = actual_data.groupby(actual_data['Month'].dt.strftime('%Y-%m'))['Sales'].sum().reset_index()

        # Group by 'Year' and calculate the sum for each year
        yearly_sales_data = actual_data.groupby(actual_data['Month'].dt.year)['Sales'].sum().reset_index()

        return jsonify({
            'monthly_sales_data': monthly_sales_data.to_dict(orient='records'),
            'yearly_sales_data': yearly_sales_data.to_dict(orient='records')
        })
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/line_graph_data', methods=['GET'])
def line_graph_data():
    try:
        # Fetch actual sales data
        actual_data = pd.read_json('output.json')
        actual_data['Month'] = pd.to_datetime(actual_data['Month'], errors='coerce')
        actual_data = actual_data.dropna(subset=['Month'])
        original_column_name = actual_data.columns[1]
        actual_data = actual_data.rename(columns={original_column_name: 'Sales'})

        # Get the last 3 items from sales data
        last_3_items = actual_data.tail(3)

        # Fetch prediction data
        with open('trained_model.pkl', 'rb') as model_file:
            trained_model = pickle.load(model_file)

        last_available_month = last_3_items['Month'].iloc[-1]
        print(last_available_month)
        forecast_steps = 1  # Assuming you want predictions for the next 3 months
        future_index = pd.date_range(start=last_available_month, periods=forecast_steps + 1, freq='M')[1:]
        predictions = trained_model.forecast(steps=forecast_steps, index=future_index)

        # Format the data for the LineGraph
        line_graph_data = [
            {"Month": item['Month'].strftime('%Y-%m'), "Sales": item['Sales']} for _, item in last_3_items.iterrows()
        ]

        line_graph_data.extend(
            [{"Month": month.strftime('%Y-%m'), "Sales": round(prediction, 2)} for month, prediction in
             zip(future_index, predictions)]
        )

        # Calculate the maximum value
        max_value = max(item['Sales'] for item in line_graph_data)
        max_value = float(max_value)  # Convert to int if needed

        # Create a dictionary to be passed to jsonify
        response_data = {"line_graph_data": line_graph_data, "max_value": max_value}

        # Return the LineGraph data and max value as JSON
        return jsonify(response_data)

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/sales_performance_history', methods=['GET'])
def sales_performance_history():
    return get_sales_performance_history()


@app.route('/get_json_data', methods=['GET'])
def get_json_data():
    try:
        # Open the JSON file and load data
        with open('output.json', 'r') as json_file:
            data = json.load(json_file)

        # Return the JSON data
        return jsonify(data[-1])
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
