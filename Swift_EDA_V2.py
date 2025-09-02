# Section 1: imports & custom errors
import os
from datetime import datetime
import statistics
from statistics import StatisticsError
from collections import Counter
import json

# Section 2: Type Handling
def infer_column_types(data):
        columns = list(zip(*data))
        column_types = []

        for col in columns:
            is_int = True
            is_float = True
            is_datetime = True
            is_bool = True

            for val in col:
                val = str(val).strip()
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
                print(f"Column {len(column_types)} inferred as 'str' (no numeric/bool/date pattern found).")
                column_types.append("str")

        return column_types



def cast_row_types(row, types):
    casted_row = []
    for val, col_type in zip(row, types):
        val = str(val).strip()
        if col_type == "int":
            try:
                casted_row.append(int(float(val)))
            except ValueError:
                casted_row.append(None)
        elif col_type == "float":
            try:
                casted_row.append(float(val))
            except ValueError:
                casted_row.append(None)
        elif col_type == "bool":
            casted_row.append(val.lower() in ("true", "1", "yes"))
        elif col_type == "datetime":
            try:
                casted_row.append(datetime.strptime(val, "%Y-%m-%d"))
            except ValueError:
                casted_row.append(None)
        else:
            casted_row.append(val)
    return casted_row

# Section 3: Readers
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

        print("read_csv was executed successfully.")

        types = infer_column_types(data) if infer_types else None
        return header, data, types, infer_column_types

    except FileNotFoundError:
        raise Exception(f"FileNotFound: The path {filepath} was not found. Double-check the file path provided.")
    except UnicodeDecodeError:
        raise Exception("File could not be read as UTF-8. Ensure it's a valid CSV text file with proper encoding.")
    except Exception as e:
        raise Exception(f"Error processing the file: {str(e)}")

def read_json(filepath, skiprows=None, nrows=None, infer_types=True):
    if not filepath.endswith(".json"):
        raise ValueError("Invalid file format. Function read_json supports only JSON files.")

    try:
        with open(filepath, "r", encoding="utf-8") as file:
            data_json = json.load(file)

        # Support records shape only (list of dicts)
        if not isinstance(data_json, list) or not all(isinstance(item, dict) for item in data_json):
            raise ValueError("Unsupported JSON format. Expected a list of dictionaries (records).")

        # Row slicing first
        if skiprows and skiprows > 0:
            data_json = data_json[skiprows:]
        if nrows is not None:
            data_json = data_json[:nrows]

        if not data_json:
            raise ValueError("JSON file is empty or has no valid data rows after filtering.")

        # Build header in first-seen key order
        header = []
        for record in data_json:
            for key in record.keys():
                if key not in header:
                    header.append(key)

        # Build data with string normalization
        data = []
        for record in data_json:
            row = []
            for col in header:
                v = record.get(col, None)
                if v is None:
                    row.append("")
                elif isinstance(v, bool):
                    row.append("true" if v else "false")
                elif isinstance(v, list):
                    row.append(", ".join(str(x) for x in v))
                else:
                    row.append(str(v))
            data.append(row)

        types = infer_column_types(data) if infer_types else None
        return header, data, types, infer_column_types

    except FileNotFoundError:
        raise Exception(f"FileNotFound: The path {filepath} was not found. Double-check the file path provided.")
    except UnicodeDecodeError:
        raise Exception("File could not be read as UTF-8. Ensure it's a valid JSON text file with proper encoding.")
    except json.JSONDecodeError:
        raise Exception("Invalid JSON structure. Ensure it's a valid JSON file.")
    except Exception as e:
        raise Exception(f"Error processing the file: {str(e)}")
   
# Section 4: I/O Dispatcher
def read_any(filepath, format=None, reader_kwargs=None, infer_types=False):
    if not os.path.isfile(filepath):
        raise Exception(f"FileNotFound: The path {filepath} was not found. Double-check the file path provided.")

    # Guess file format from extension
    if format is None:
        _, ext = os.path.splitext(filepath.lower())
        if ext == ".csv":
            format = "csv"
        elif ext == ".json":
            format = "json"
        else:
            raise Exception("UnsupportedFormat: Use format='csv'|'json'|.")

    reader_kwargs = reader_kwargs or {}

    if format == "csv":
        return read_csv(filepath, infer_types=infer_types, **reader_kwargs)
    if format == "json":
        return read_json(filepath, infer_types=infer_types, **reader_kwargs)

    raise Exception("UnsupportedFormat: Use format='csv'|'json'|.")

# Section 5: Cleaning Pipeline
def clean_df(filepath, format=None, reader_kwargs=None, infer_types=True, drop_threshold=0.5, impute_strategy="mean", display = True, 
             report_missing=False, display_kwargs=None, summary=False, summary_percentiles=[25,50,75]):
    # Read without inferring types in readers to avoid duplicate logs
    header, data, types, infer_types_fn = read_any(
        filepath, format=format, reader_kwargs=reader_kwargs, infer_types=False
    )

    # Drop columns with many missings
    header, data = drop_high_missing_cols(data, header, threshold=drop_threshold)

    # Single, authoritative inference pass after structural changes
    if infer_types:
        types = infer_types_fn(data)

    # Impute and cast
    impute_cols(data, header, types, strategy=impute_strategy)
    data = [cast_row_types(row, types) for row in data]

    if report_missing:
        report_missing_values(data, header)

    if display:
        display_tabular_output(header, data, **(display_kwargs or {}))

    if summary:
        summarize_columns(data, header, types, percentiles=summary_percentiles, print_table=True)

    return header, data, types

# Section 6: Display
def display_tabular_output(header, data, **display_kwargs):
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

# Section 7: Missing Values Report, Dropping & Imputing
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
            if str(val).strip().lower() in ("", "nan", "null", "none"):
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
            if str(val).strip().lower() in ("", "nan", "null", "none"):
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

    # Transpose once for efficiency
    columns = list(zip(*data))
    total_rows = len(data)

    for i, col in enumerate(columns):
        col_type = types[i]

        # Identify missing positions first
        missing_positions = []
        for r_idx, val in enumerate(col):
            val_clean = str(val).strip().lower()
            if val_clean in ("", "nan", "null", "none"):
                missing_positions.append(r_idx)

        # If nothing to fill, skip printing heavy messages
        if not missing_positions:
            # Optional: uncomment to log skips
            # print(f"No missing values in '{header[i]}' — skipping imputation.")
            continue

        # Build cleaned_values for computing the statistic
        cleaned_values = []
        for val in col:
            val_clean = str(val).strip().lower()
            if val_clean not in ("", "nan", "null", "none"):
                if col_type in ("int", "float"):
                    try:
                        cleaned_values.append(float(val))
                    except ValueError:
                        continue
                else:
                    cleaned_values.append(str(val).strip())

        if not cleaned_values:
            # Nothing valid to compute from
            print(f"Skipping column '{header[i]}' - no valid data to impute.")
            continue

        # Determine strategy by type
        if col_type in ("int", "float"):
            actual_strategy = strategy  # numeric uses requested strategy
        else:
            actual_strategy = "mode"    # non-numeric forced to mode

        # Compute imputed value
        try:
            if actual_strategy == "mean":
                imputed_value = statistics.mean(cleaned_values)
            elif actual_strategy == "median":
                imputed_value = statistics.median(cleaned_values)
            else:
                imputed_value = statistics.mode(cleaned_values)
        except StatisticsError:
            # fallback: manually find most common value
            counter = Counter(cleaned_values)
            imputed_value = counter.most_common(1)[0][0]

        # Pretty print value (align bool casing with display)
        if col_type == "bool":
            # cleaned_values were strings; normalize to Python bool for message
            if isinstance(imputed_value, str):
                print_value = "True" if imputed_value.strip().lower() in ("true", "1", "yes") else "False"
            else:
                print_value = str(bool(imputed_value))
        elif isinstance(imputed_value, float):
            print_value = f"{imputed_value:.2f}"
        else:
            print_value = str(imputed_value)

        print(f"Imputed column '{header[i]}' with {actual_strategy}: {print_value}")

        # Replace only the missing positions (always write back strings)
        for r_idx in missing_positions:
            if col_type == "int":
                try:
                    data[r_idx][i] = str(int(round(float(imputed_value))))
                except Exception:
                    data[r_idx][i] = "0"
            elif col_type == "float":
                try:
                    data[r_idx][i] = f"{float(imputed_value):.2f}"
                except Exception:
                    data[r_idx][i] = "0.00"
            elif col_type == "bool":
                # keep lower-case string for later casting
                if isinstance(imputed_value, str):
                    data[r_idx][i] = "true" if imputed_value.strip().lower() in ("true", "1", "yes") else "false"
                else:
                    data[r_idx][i] = "true" if bool(imputed_value) else "false"
            else:
                data[r_idx][i] = str(imputed_value)


# Section 8: Summary Stats
def summarize_columns(data, header, types, percentiles=[25,50,75], print_table=True):
    def _is_missing(s):
        return str(s).strip().lower() in ("", "nan", "null", "none")

    def _percentile_linear(sorted_vals, p):
        if not sorted_vals:
            return None
        if p <= 0:
            return sorted_vals[0]
        if p >= 100:
            return sorted_vals[-1]
        pos = (p / 100.0) * (len(sorted_vals) - 1)
        f = int(pos)
        c = f + 1
        if c >= len(sorted_vals):
            return sorted_vals[-1]
        d = pos - f
        return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * d

    def _fmt_out(v):
        from datetime import datetime as _dt
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.2f}"
        if isinstance(v, _dt):
            return v.strftime("%Y-%m-%d")
        return str(v)

    # fallback types if None provided
    effective_types = types if types is not None else ["str"] * len(header)

    summary = {}

    # Build stats
    for i, col_name in enumerate(header):
        col_type = effective_types[i] if i < len(effective_types) else "str"
        col_vals = [row[i] if i < len(row) else "" for row in data]
        total = len(col_vals)
        missing = sum(1 for v in col_vals if _is_missing(v))

        col_summary = {
            "type": col_type,
            "count": total,
            "missing": missing
        }

        if col_type in ("int", "float"):
            numeric_vals = []
            for v in col_vals:
                if _is_missing(v):
                    continue
                try:
                    numeric_vals.append(float(v))
                except Exception:
                    pass

            if numeric_vals:
                numeric_vals.sort()
                col_summary["min"] = numeric_vals[0]
                col_summary["max"] = numeric_vals[-1]
                try:
                    col_summary["mean"] = statistics.mean(numeric_vals)
                except statistics.StatisticsError:
                    col_summary["mean"] = None
                try:
                    col_summary["median"] = statistics.median(numeric_vals)
                except statistics.StatisticsError:
                    col_summary["median"] = None
                try:
                    col_summary["mode"] = statistics.mode(numeric_vals)
                except statistics.StatisticsError:
                    cnt = Counter(numeric_vals)
                    col_summary["mode"] = cnt.most_common(1)[0][0]

                # percentiles
                for p in percentiles:
                    col_summary[f"p{int(p)}"] = _percentile_linear(numeric_vals, p)

                # IQR
                p25 = col_summary.get("p25")
                p75 = col_summary.get("p75")
                col_summary["iqr"] = (p75 - p25) if (p25 is not None and p75 is not None) else None
            else:
                pass

        elif col_type == "datetime":
            dt_vals = []
            for v in col_vals:
                if _is_missing(v):
                    continue
                if isinstance(v, datetime):
                    dt_vals.append(v)
                else:
                    try:
                        dt_vals.append(datetime.strptime(str(v).strip(), "%Y-%m-%d"))
                    except Exception:
                        pass
            if dt_vals:
                dt_vals.sort()
                col_summary["min"] = dt_vals[0]
                col_summary["max"] = dt_vals[-1]

        elif col_type == "bool":
            norm = []
            for v in col_vals:
                if _is_missing(v):
                    continue
                s = str(v).strip().lower()
                norm.append("true" if s in ("true", "1", "yes") else "false")
            if norm:
                cnt = Counter(norm)
                mode_val, _ = cnt.most_common(1)[0]
                col_summary["unique"] = len(cnt)
                col_summary["mode"] = "True" if mode_val == "true" else "False"
            else:
                col_summary["unique"] = 0
                col_summary["mode"] = None

        else:
            strings = [str(v).strip() for v in col_vals if not _is_missing(v)]
            if strings:
                cnt = Counter(strings)
                mode_val, _ = cnt.most_common(1)[0]
                col_summary["unique"] = len(cnt)
                col_summary["mode"] = mode_val
            else:
                col_summary["unique"] = 0
                col_summary["mode"] = None

        summary[col_name] = col_summary

    if print_table:
        percent_headers = [f"P{int(p)}" for p in percentiles]
        table_header = ["column", "type", "count", "missing", "unique", "min"] + percent_headers + ["median", "max", "mean", "mode", "IQR"]

        table_rows = []
        for col_name in header:
            s = summary[col_name]
            row = [
                col_name,
                s.get("type", ""),
                str(s.get("count", "")),
                str(s.get("missing", "")),
                str(s.get("unique", "")) if "unique" in s else "",
                _fmt_out(s.get("min", "")),
            ]
            for p in percentiles:
                row.append(_fmt_out(s.get(f"p{int(p)}", "")))
            row.append(_fmt_out(s.get("median", "")))
            row.append(_fmt_out(s.get("max", "")))
            row.append(_fmt_out(s.get("mean", "")))
            row.append(_fmt_out(s.get("mode", "")))
            row.append(_fmt_out(s.get("iqr", "")))
            table_rows.append(row)

        display_tabular_output(table_header, table_rows)
    return summary

    
# Section 9: Help Function
def help(name=None):
    if name is None:
        rows = [
            ["clean_df",               "Run end-to-end pipeline (read→drop→infer→impute→cast→display)."],
            ["summarize_columns",      "Compute per-column summary statistics."],
            ["read_any",               "Auto-select reader by format/extension (CSV/JSON; Excel soon)."],
            ["read_csv",               "Read a CSV file into (header, data, types, infer_types_fn)."],
            ["read_json",              "Read a JSON (records) file into (header, data, types, infer_types_fn)."],
            ["report_missing_values",  "Print missing counts/percent per column."],
            ["drop_high_missing_cols", "Drop columns whose missing % exceeds a threshold."],
            ["impute_cols",            "Fill missing values (mean/median/mode for numeric, mode for others)."],
            ["display_tabular_output", "Render a simple text table for small previews."],
        ]
        try:
            display_tabular_output(["function", "what it does"], rows)
        except Exception:
            print("\nFunctions:\n---------")
            for fn, desc in rows:
                print(f"- {fn}: {desc}")
        return

    if callable(name):
        key = getattr(name, "__name__", "").lower()
    else:
        key = str(name).strip().lower()

    alias_map = {
        # primary APIs
        "run": "clean_df",
        "pipeline": "clean_df",
        "summary": "summarize_columns",
        "describe": "summarize_columns",
        # readers
        "read": "read_any",
        "load": "read_any",
        # cleaning utils
        "missing_report": "report_missing_values",
        "drop_missing_cols": "drop_high_missing_cols",
        "impute": "impute_cols",
        # display
        "print_table": "display_tabular_output",
    }
    key = alias_map.get(key, key)

    # help texts
    if key == "clean_df":
        print(
            "clean_df(filepath, format=None, reader_kwargs=None, infer_types=True, drop_threshold=0.5,\n"
            "         impute_strategy='mean', display=True, report_missing=False, display_kwargs=None,\n"
            "         summary=False, summary_percentiles=[25,50,75])\n\n"
            "What it does:\n"
            "  Runs the full pipeline: read -> drop high-missing cols -> infer types -> impute -> cast\n"
            "  Optionally prints a table and a summary\n\n"
            "Returns:\n"
            "  (header, data, types)\n\n"
            "Tips:\n"
            "  - Set format='json' for JSON files (CSV is auto-detected)\n"
            "  - Use reader_kwargs={'skiprows':1,'nrows':100} to read a slice\n"
            "  - Turn on summary with summary=True\n"
        )
        return

    if key == "summarize_columns":
        print(
            "summarize_columns(data, header, types, percentiles=[25,50,75], print_table=True)\n\n"
            "What it does:\n"
            "  Computes per-column stats\n"
            "  - Numeric: count, missing, min, max, mean, median, mode, percentiles, IQR\n"
            "  - Datetime: min, max, count, missing\n"
            "  - Bool: count, missing, unique, mode\n"
            "  - Str: count, missing, unique, mode\n\n"
            "Returns:\n"
            "  dict keyed by column name with the stats\n\n"
            "Tips:\n"
            "  - Call after clean_df so types are accurate\n"
            "  - Set print_table=False to just get the dict\n"
        )
        return

    if key == "read_any":
        print(
            "read_any(filepath, format=None, reader_kwargs=None, infer_types=True)\n\n"
            "What it does:\n"
            "  Chooses the correct reader based on 'format' or file extension\n"
            "  Supports: CSV & JSON (records).\n\n"
            "Returns:\n"
            "  (header, data, types, infer_types_fn)\n\n"
            "Tips:\n"
            "  - Usually you won't call this directly; clean_df uses it\n"
            "  - Provide reader_kwargs like {'skiprows': 1, 'nrows': 100}\n"
        )
        return

    if key == "read_csv":
        print(
            "read_csv(filepath, skiprows=None, nrows=None, infer_types=True)\n\n"
            "What it does:\n"
            "  Reads a CSV file, pads & truncates rows to header length, returns normalized tables\n\n"
            "Returns:\n"
            "  (header, data, types, infer_column_types)\n\n"
            "Tips:\n"
            "  - Uses simple comma-split; for complex CSVs use the csv module later\n"
        )
        return

    if key == "read_json":
        print(
            "read_json(filepath, skiprows=None, nrows=None, infer_types=True)\n\n"
            "What it does:\n"
            "  Reads a JSON file in 'records' shape (list of dicts), builds header in first-seen order,\n"
            "  and normalizes values to strings (lists are joined with ', ').\n\n"
            "Returns:\n"
            "  (header, data, types, infer_column_types)\n\n"
            "Tips:\n"
            "  - Only 'records' shape supported in V2\n"
            "  - For dict of lists or nested JSON files, flatten or convert first\n"
        )
        return
    
    if key == "report_missing_values":
        print(
            "report_missing_values(data, header)\n\n"
            "What it does:\n"
            "  Prints a table with count and percent of missing values per column.\n\n"
            "Tips:\n"
            "  - Missing is defined as '', 'nan', 'null', 'none' (case-insensitive).\n"
        )
        return

    if key == "drop_high_missing_cols":
        print(
            "drop_high_missing_cols(data, header, threshold=None)\n\n"
            "What it does:\n"
            "  Drops columns whose missing fraction >= threshold.\n\n"
            "Returns:\n"
            "  (new_header, new_data)\n\n"
            "Tips:\n"
            "  - Pass a float between 0 and 1 (e.g., 0.5 for 50%).\n"
            "  - If threshold=None, it will prompt for input.\n"
        )
        return

    if key == "impute_cols":
        print(
            "impute_cols(data, header, types, strategy='mean')\n\n"
            "What it does:\n"
            "  Fills missing values in-place. Numeric uses mean/median/mode; non-numeric uses mode.\n\n"
            "Tips:\n"
            "  - Only prints a message when replacements actually happen.\n"
            "  - Writes back strings; casting happens afterwards.\n"
        )
        return

    if key == "display_tabular_output":
        print(
            "display_tabular_output(header, data, **display_kwargs)\n\n"
            "What it does:\n"
            "  Renders a simple text table for small previews.\n\n"
            "Tips:\n"
            "  - Later we can add row/col limits and max_col_width controls.\n"
        )
        return

    print(f"No help available for '{name}'. Try help() to see the index or use an alias like 'run' or 'summary'.")

# Tests
# Test clean_df wrapper()
"""
print("=== Test clean_df wrapper ===")
header_w, data_w, types_w = clean_df("test_df.csv", format=None, reader_kwargs=None, infer_types=True, drop_threshold=0.5, impute_strategy="mean", display = True, 
             report_missing=False, display_kwargs=None)

print("Wrapper output header:", header_w)
print("Wrapper output rows:", data_w)
print("Wrapper output types:", types_w)
"""
# Test clean_df with json 
"""clean_df("test_json.json", format=None, reader_kwargs=None, infer_types=True, drop_threshold=0.5, impute_strategy="mean", display = True, 
             report_missing=False, display_kwargs=None)"""

# Summary stats
"""
header, data, types = clean_df("test_df.csv", format="csv", summary=True)
"""
help()