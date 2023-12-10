# helpers.py
from datetime import timedelta
import os
import pickle
import json
import pandas as pd
import statsmodels.api as sm
from flask import jsonify
import uuid

trained_models = {}


def train_arima_model(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
    model = sm.tsa.statespace.SARIMAX(data, order=order, seasonal_order=seasonal_order).fit()
    return model


def save_csv_file(file):
    unique_filename = str(uuid.uuid4()) + '.csv'
    file_path = os.path.join('uploads', unique_filename)
    file.save(file_path)
    return unique_filename


def get_last_item_from_json(file_path="output.json"):
    try:
        with open(file_path, "r") as json_file:
            data_list = json.load(json_file)
        last_item = data_list[-1]
        return last_item
    except Exception as e:
        print(f"Error: {e}")
        return None


def get_sales_performance_history():
    try:
        sales_data = pd.read_csv('uploads/loaded_data.csv')
        sales_data['Month'] = pd.to_datetime(sales_data['Month'], errors='coerce')
        df = sales_data.dropna(subset=['Month'])
        original_column_name = df.columns[1]
        df = df.rename(columns={original_column_name: 'Sales'})

        if 'Sales' not in df.columns:
            return jsonify({"error": "Column 'Sales' not found in the DataFrame"})

        df.sort_values(by='Month', inplace=True)

        sales_history_list = [
            {
                'Month': month.strftime('%b %Y'),
                'Sales': round(sales, 2)
            }
            for month, sales in zip(df['Month'], df['Sales'])
        ]

        percentage_increase = [round(((sales_history_list[i]['Sales'] - sales_history_list[i - 1]['Sales']) /
                                      sales_history_list[i - 1]['Sales'] * 100), 2) if i > 0 else None
                               for i in range(len(sales_history_list))]
        sales_increase = [round((sales_history_list[i]['Sales'] - sales_history_list[i - 1]['Sales']), 2)
                          if i > 0 else None
                          for i in range(len(sales_history_list))]

        combined_data = [
            {
                'Month': sales_history_list[i]['Month'],
                'Sales': sales_history_list[i]['Sales'],
                'PercentageIncrease': percentage_increase[i],
                'DifferenceInSales': sales_increase[i]
            }
            for i in range(1, len(sales_history_list))
        ]

        return jsonify(combined_data)

    except Exception as e:
        return jsonify({"error": str(e)})
