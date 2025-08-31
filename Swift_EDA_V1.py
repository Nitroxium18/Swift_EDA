import os
from datetime import datetime
import statistics
from statistics import StatisticsError
from collections import Counter

# Wrapper
def clean_df(filepath, skiprows=None, nrows=None, infer_types=True, drop_threshold=0.5, impute_strategy="mean",display = True, report_missing = False):

    header, data, types, infer_column_types = read_csv(filepath, skiprows, nrows, infer_types)

    # Drop columns
    header, data = drop_high_missing_cols(data, header, threshold=drop_threshold)

    # Re-infer types using the local function returned by read_csv
    if infer_types:
        types = infer_column_types(data)

    # Impute missing
    impute_cols(data, header, types, strategy=impute_strategy)

    # Typecasting
    data = [cast_row_types(row, types) for row in data]

    # Report missing
    if report_missing:
        print("\nMissing values report AFTER cleaning:")
        report_missing_values(data, header)

    # Tabular display
    if display:
        display_tabular_output(header, data)

    return header, data, types



def read_csv(filepath, skiprows=None, nrows=None, infer_types=True):
    if not filepath.endswith(".csv"):
        raise ValueError("Invalid file format. Function read_csv supports only csv files.")
    data = []

    try:
        with open(filepath, "r", encoding="utf-8") as file:
            print("File opened successfully!")
            if skiprows and skiprows > 0:
                for _ in range(skiprows):
                    file.readline()

            header = file.readline().strip().split(",")
            counter = 0
            for line in file:
                if nrows is not None and counter >= nrows:
                    break
                row = line.strip().split(",")
                if len(row) > len(header):
                    row = row[:len(header)]
                elif len(row) < len(header):
                    row += [""] * (len(header) - len(row))

                data.append(row)
                counter += 1

            if not data:
                raise ValueError("CSV file is empty or has no valid data rows.")

    except FileNotFoundError:
        raise Exception(f"FileNotFound: The path {filepath} was not found. Double-check the file path provided.")
    except UnicodeDecodeError:
        raise Exception("File could not be read as UTF-8. Ensure it's a valid CSV text file with proper encoding.")
    except Exception as e:
        print(f"Error processing the file: {str(e)}")
    finally:
        print("read_csv was executed successfully.")

    def infer_column_types(data):
        columns = list(zip(*data))
        column_types = []

        for col in columns:
            is_int = True
            is_float = True
            is_datetime = True
            is_bool = True

            for val in col:
                val = val.strip()
                if val.lower() in ("", "nan", "null"):
                    continue
                if val.lower() in ("true", "false", "1", "0", "yes", "no"):
                    pass
                else:
                    is_bool = False

                try:
                    datetime.strptime(val, "%Y-%m-%d")
                except ValueError:
                    is_datetime = False
                try:
                    int(val)
                except ValueError:
                    is_int = False
                    try:
                        float(val)
                    except ValueError:
                        is_float = False

            if is_bool:
                column_types.append("bool")
            elif is_datetime:
                column_types.append("datetime")
            elif is_int:
                column_types.append("int")
            elif is_float:
                column_types.append("float")
            else:
                print(f"Column {len(column_types)} inferred as 'str' due to mixed or unrecognized types.")
                column_types.append("str")

        return column_types

    types = None
    if infer_types:
        types = infer_column_types(data)

    return header, data, types, infer_column_types


def display_tabular_output(header, data):
    # Compute column widths
    column_widths = [len(col) for col in header]
    for row in data:
        for i, val in enumerate(row):
            val_str = str(val)  # Convert to string
            if i < len(column_widths):
                column_widths[i] = max(column_widths[i], len(val_str))

    # Create row formatter
    def format_row(row):
        safe_row = row + [""] * (len(header) - len(row))
        return "| " + " | ".join(str(val).ljust(column_widths[i]) for i, val in enumerate(safe_row[:len(header)])) + " |"

    # Separator line
    separator = "+-" + "-+-".join("-" * w for w in column_widths) + "-+"

    # Print table
    print(separator)
    print(format_row(header))
    print(separator)
    for row in data:
        print(format_row(row))
    print(separator)

def report_missing_values(data, header):
    if not data:
        print("No data found. Skipping missing value report.\n")
        return
    print("\nMissing Values Report:")

    total_rows = len(data)
    columns = list(zip(*data))

    print("+----------------+--------+-----------+")
    print("| Column         | Count  | Percent   |")
    print("+----------------+--------+-----------+")

    for i, col in enumerate(columns):
        missing_count = 0
        for val in col:
            if val.strip().lower() in ("", "nan", "null", "none"):
                missing_count += 1
        percent = (missing_count / total_rows) * 100 if total_rows > 0 else 0
        print(f"| {header[i].ljust(14)} | {str(missing_count).rjust(6)} | {str(round(percent, 2)).rjust(8)}% |")

    print("+----------------+--------+-----------+\n")


def drop_high_missing_cols(data, header, threshold=None):
    total_rows = len(data)

    if threshold is None:
        t = float(input("Enter threshold % for dropping columns (0-100): "))
        if t < 0 or t > 100:
            print("Invalid input. Enter a number between 0 and 100.")
            return header, data
        else:
            threshold = t / 100

    columns = list(zip(*data))
    keep_indices = []
    dropped_columns = []

    for i, col in enumerate(columns):
        missing_count = 0
        for val in col:
            if val.strip().lower() in ("", "nan", "null", "none"):
                missing_count += 1
        percent = (missing_count / total_rows) if total_rows > 0 else 0

        if percent < threshold:
            keep_indices.append(i)
        else:
            dropped_columns.append(header[i])

    new_header = [header[i] for i in keep_indices]
    new_data = [[row[i] for i in keep_indices] for row in data]

    if dropped_columns:
        print(f"\nDropped columns due to high missing values (> {threshold*100}%):")
        for col_name in dropped_columns:
            print(f"- {col_name}")
    else:
        print("\nNo columns dropped.")

    return new_header, new_data


def impute_cols(data, header, types, strategy="mean"):
    if strategy not in ("mean", "median", "mode"):
        raise ValueError("Invalid measure of central tendency chosen. Choose from: mean, median, or mode.")

    columns = list(zip(*data))

    for i, col in enumerate(columns):
        col_type = types[i]
        cleaned_values = []

        # Gather valid (non-missing) values for this column
        for val in col:
            val_clean = val.strip().lower()
            if val_clean not in ("", "nan", "null", "none"):
                if col_type in ("int", "float"):
                    try:
                        cleaned_values.append(float(val))
                    except ValueError:
                        continue
                else:
                    cleaned_values.append(val.strip())

        if not cleaned_values:
            print(f"Skipping column '{header[i]}' - no valid data to impute.")
            continue

        # Determine actual imputation method to use for this column
        if col_type in ("int", "float"):
            # Numeric columns use chosen strategy (mean, median, mode)
            actual_strategy = strategy
        else:
            # Non-numeric columns must use mode only
            actual_strategy = "mode"

        # Calculate imputed value with error handling for mode
        try:
            if actual_strategy == "mean":
                imputed_value = statistics.mean(cleaned_values)
            elif actual_strategy == "median":
                imputed_value = statistics.median(cleaned_values)
            else:  # mode
                imputed_value = statistics.mode(cleaned_values)
        except StatisticsError:
            # fallback: manually find most common value
            counter = Counter(cleaned_values)
            imputed_value = counter.most_common(1)[0][0]

        # Format print output nicely
        if isinstance(imputed_value, float):
            print_value = f"{imputed_value:.2f}"
        elif isinstance(imputed_value, int):
            print_value = str(imputed_value)
        else:
            print_value = str(imputed_value)

        print(f"Imputed column '{header[i]}' with {actual_strategy}: {print_value}")

        # Replace missing with imputed value (always as string)
        for row in data:
            val_clean = row[i].strip().lower()
            if val_clean in ("", "nan", "null", "none"):
                if col_type == "int":
                    row[i] = str(int(round(imputed_value)))
                elif col_type == "float":
                    row[i] = f"{float(imputed_value):.2f}"
                else:
                    row[i] = str(imputed_value)

def cast_row_types(row, types):
    casted_row = []
    for val, typ in zip(row, types):
        val = val.strip()
        if typ == "int":
            try:
                casted_row.append(int(float(val)))  # Handle floats like '3.0'
            except ValueError:
                casted_row.append(None)  # or val if you want to keep original
        elif typ == "float":
            try:
                casted_row.append(float(val))
            except ValueError:
                casted_row.append(None)
        elif typ == "bool":
            casted_row.append(val.lower() in ("true", "1", "yes"))
        elif typ == "datetime":
            try:
                casted_row.append(datetime.strptime(val, "%Y-%m-%d"))
            except ValueError:
                casted_row.append(None)
        else:
            casted_row.append(val)
    return casted_row







"""# Test read_csv()
print("=== Test read_csv ===")
header, data, types, infer_column_types = read_csv("test_df.csv", infer_types=True)
print("Header:", header)
print("First 3 rows of data:", data[:3])
print("Inferred types:", types)
print("\n")


# Test display_tabular_output()
print("=== Test display_tabular_output ===")
display_tabular_output(header, data)
print("\n")


# Test report_missing_values()
print("=== Test report_missing_values ===")
report_missing_values(data, header)
print("\n")


# Test drop_high_missing_cols()
print("=== Test drop_high_missing_cols ===")
# Use threshold 0.3 (30%)
new_header, new_data = drop_high_missing_cols(data, header, threshold=0.3)
print("Columns after dropping:", new_header)
print("First 3 rows after dropping:", new_data[:3])
print("\n")


# Test impute_cols()
print("=== Test impute_cols ===")
# For this, we need types. Use types from read_csv with original data.
impute_cols(data, header, types, strategy="mean")
print("Data after imputation (first 3 rows):", data[:3])
print("\n")

"""
# Test clean_df wrapper()
print("=== Test clean_df wrapper ===")
header_w, data_w, types_w = clean_df("Titanic-Dataset.csv", skiprows=None, nrows=None, infer_types=True, drop_threshold=0.5, impute_strategy="mean")
print("Wrapper output header:", header_w)
print("Wrapper output rows:", data_w)
print("Wrapper output types:", types_w)
