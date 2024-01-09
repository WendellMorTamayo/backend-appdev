# helpers.py
import os
from datetime import datetime
import json
import pandas as pd
import statsmodels.api as sm
from flask import jsonify

trained_models = {}


def train_arima_model(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
    model = sm.tsa.statespace.SARIMAX(data, order=order, seasonal_order=seasonal_order).fit()
    return model


def get_last_item_from_json(file_path="output.json"):
    try:
        with open(file_path, "r") as json_file:
            data_list = json.load(json_file)
        last_item = data_list[-1]

        if 'cumulative_gross_sales' in last_item:
            last_item['cumulative_gross_sales'] = round(last_item['cumulative_gross_sales'], 2)

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


def parse_date_xy(input_date):
    # Split the input date into parts
    parts = input_date.split('-')

    # Check the length of the first part (year)
    if len(parts[0]) == 1:
        # Add a leading zero to the year if it's a single digit
        input_datetime = datetime.strptime(f"0{input_date}", "%y-%m")
    else:
        input_datetime = datetime.strptime(f"{input_date}", "%y-%m")

    # Format the datetime object to the desired output format "YYYY-Mon"
    output_date = input_datetime.strftime("%Y-%m")

    return output_date


def detect_date_format(date_column):
    parsed_dates_Ym = pd.to_datetime(date_column, format='%Y-%m', errors='coerce')
    parsed_dates_ym = pd.to_datetime(date_column, format='%y-%m', errors='coerce')

    count_Ym = pd.notna(parsed_dates_Ym).sum()
    count_ym = pd.notna(parsed_dates_ym).sum()

    if count_Ym > count_ym:
        return "%Y-%m"
    else:
        return "%y-%m"


def save_csv_file(df, file_path):
    # Save the DataFrame to a new CSV file
    df.to_csv(file_path, index=False)


def process_data(df):
    try:
        # Convert 'Date' column to datetime format
        df['Month'] = pd.to_datetime(df['Month'], errors='coerce')
        df = df.dropna(subset=['Month'])

        # Extract month and year from the 'Date' column
        df['Month'] = df['Month'].dt.to_period('M')

        # Sort DataFrame by 'Date' within each 'ProductID'
        df = df.sort_values(['ProductID', 'Month'])

        # Save the sorted DataFrame to a single CSV file for each ProductID
        for product_id, product_group in df.groupby('ProductID'):
            filename = f"product_{product_id}_data.csv"
            filepath = os.path.join("models", "product_csv", filename)

            # Save the group to CSV, overwriting the file if it already exists
            product_group[['Month', 'ProductID', 'Product', 'UnitsSold']].to_csv(filepath, index=False, mode='w', header=True)

        return "Grouping and saving completed successfully"

    except Exception as e:
        return {"error": str(e)}


def date_parser(x):
    return datetime.strptime(x, '%Y-%m')