from datetime import datetime
import pandas as pd


def parse_date(input_date):
    # Convert the input date string to a datetime object
    if len(input_date.split('-')[0]) == 1:
        input_datetime = datetime.strptime(f"0{input_date}", "%Y-%m")
    else:
        input_datetime = datetime.strptime(f"{input_date}", "%Y-%m")

    # Format the datetime object to the desired output format "YYYY-Mon"
    output_date = input_datetime.strftime("%Y-%m")

    return output_date


def detect_date_format(date_column):
    if pd.to_datetime(date_column, errors='coerce').notnull().all():
        return "%Y-%m"
    else:
        return "%y-%m"


# Example usage:
input_date = '1964-01'
parsed_date_format = detect_date_format([input_date])
print(parsed_date_format)
