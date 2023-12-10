# app.py
from flask import Flask, request, render_template, jsonify
from utils.helper import train_arima_model, save_csv_file, get_last_item_from_json, get_sales_performance_history, \
    trained_models
import os
import pickle
from sklearn.metrics import mean_absolute_error
import json
from flask import Flask, request, jsonify
import pandas as pd
from flask import jsonify

app = Flask(__name__)


@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"})

        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "No selected file"})

        if file and file.filename.endswith('.csv'):
            file.save(os.path.join('uploads', 'loaded_data1.csv'))
            return jsonify({"success": "File uploaded successfully"})
        elif file.filename.endswith(('.xlsx', '.xls')):
            excel_filename = os.path.join('uploads', 'loaded_data1.xlsx')
            file.save(excel_filename)
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
        forecast_steps = 1
        future_index = pd.date_range(start=last_available_month + pd.DateOffset(months=1), periods=forecast_steps,
                                     freq='M')
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



@app.route('/sales_performance_history', methods=['GET'])
def sales_performance_history():
    return get_sales_performance_history()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
