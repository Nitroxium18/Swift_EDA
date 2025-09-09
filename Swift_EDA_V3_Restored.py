# Section 1: imports & custom errors
import os
from datetime import datetime
import statistics
from statistics import StatisticsError
from collections import Counter
import json
import re as re_csvsmart
import math as _math
import base64
from io import BytesIO
import csv as _csv

# Logging
# ---------- V3: Logging ----------
from datetime import datetime as _dt_v3

LOGS = []

def log(message: str):
    ts = _dt_v3.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {message}"
    print(line)
    LOGS.append(line)

# ---------- V3: Numeric cleaning helpers ----------
import re as _re

# remove common currency symbols & commas, normalize weird minus, trim junk units
_CURRENCY_PUNCT = _re.compile(r"[₹$€£,]")
_TRAIL_JUNK    = _re.compile(r"[^\d\.\-eE]+$")   # trailing non-numeric like 'cm', 'kg', '%'
_LEAD_JUNK     = _re.compile(r"^[^\d\-]+")       # leading junk before first digit or '-'

def _normalize_minus(s: str) -> str:
    # Replace common Unicode minus/dash variants with ASCII hyphen
    return s.replace("−", "-").replace("–", "-").replace("—", "-")

def clean_numeric_token(x):
    """
    Convert a raw token (string/scalar) into float or None.
    Handles $, ₹, €, £, thousands-commas, parentheses negatives, unit suffixes, unicode minus, etc.
    """
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() in ("nan", "null", "none"):
        return None

    s = _normalize_minus(s)
    s = _CURRENCY_PUNCT.sub("", s)          # strip currency & commas

    # parentheses negative: (123.45) -> -123.45
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]

    # strip leading junk (e.g., "approx 12.3kg" -> "12.3kg")
    s = _LEAD_JUNK.sub("", s)
    # strip trailing junk (e.g., "12.3kg" -> "12.3")
    s = _TRAIL_JUNK.sub("", s)

    if s == "" or s in ("-", ".", "-.", "+", "+."):
        return None

    try:
        return float(s)
    except Exception:
        return None

def to_float_or_none(x):
    """Tiny convenience wrapper around clean_numeric_token for readability."""
    return clean_numeric_token(x)

# Section 2: Type Handling
def _parse_dt_try(s: str):
    s = str(s).strip()
    if s == "" or s.lower() in ("nan", "null", "none"):
        return None
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            pass
    return None

def infer_column_types(data, *, numeric_min_frac=0.7, int_min_frac=0.7, dt_min_frac=0.7, bool_min_frac=0.9):
    """
    Robust inference:
      - numeric: uses clean_numeric_token; must parse in >= numeric_min_frac of non-missing
      - int: numeric & integers in >= int_min_frac of numeric parses
      - datetime: allow multiple common formats; >= dt_min_frac of non-missing
      - bool: strict tokens; >= bool_min_frac of non-missing
    Order of precedence: bool -> datetime -> int -> float -> str
    """
    cols = list(zip(*data))
    types = []

    for col in cols:
        vals = [v for v in col]
        n = len(vals)
        non_missing = [v for v in vals if str(v).strip().lower() not in ("", "nan", "null", "none")]
        nm = len(non_missing)

        if nm == 0:
            types.append("str")
            continue

        # BOOL check
        bool_ok = 0
        for v in non_missing:
            s = str(v).strip().lower()
            if s in ("true", "false", "1", "0", "yes", "no"):
                bool_ok += 1
        if bool_ok / nm >= bool_min_frac:
            types.append("bool")
            continue

        # DATETIME check
        dt_ok = sum(1 for v in non_missing if _parse_dt_try(v) is not None)
        if dt_ok / nm >= dt_min_frac:
            types.append("datetime")
            continue

        # NUMERIC check (via cleaner)
        parsed = [clean_numeric_token(v) for v in non_missing]
        num_ok = [p for p in parsed if p is not None]
        if len(num_ok) / nm >= numeric_min_frac:
            # INT vs FLOAT
            ints = sum(1 for p in num_ok if float(p).is_integer())
            if ints / len(num_ok) >= int_min_frac:
                types.append("int")
            else:
                types.append("float")
            continue

        # fallback
        types.append("str")

    return types

def cast_row_types(row, types, prot_idx=None):
    # protected indices must remain str & numeric + datetime stay as is
    prot_idx = prot_idx or set()
    casted_row = []
    for i, (val, col_type) in enumerate(zip(row, types)):
        # Force protected columns to string as-is
        if i in prot_idx:
            casted_row.append(str(val))
            continue

        sval = str(val).strip()
        if col_type == "int":
            try:
                # allow float-like ints (e.g., "123.0")
                casted_row.append(int(float(sval)))
            except Exception:
                casted_row.append(None)
        elif col_type == "float":
            try:
                casted_row.append(float(sval))
            except Exception:
                casted_row.append(None)
        elif col_type == "bool":
            casted_row.append(sval.lower() in ("true", "1", "yes"))
        elif col_type == "datetime":
            try:
                casted_row.append(datetime.strptime(sval, "%Y-%m-%d"))
            except Exception:
                casted_row.append(None)
        else:
            # str fallback
            casted_row.append(sval)
    return casted_row

def recheck_types(data, header, col_types, protected_cols, log_func=print):
    """
    Re-run type inference after imputation/casting for non-protected columns.
    Updates col_types in place and logs any type upgrades.
    """
    for j, col in enumerate(header):
        if col in protected_cols:
            continue  # skip protected columns

        values = [row[j] for row in data if row[j] is not None]
        if not values:
            continue

        inferred_type = infer_column_types(values)

        if inferred_type != col_types[col]:
            log_func(f"[INFO] Column '{col}' type changed from {col_types[col]} → {inferred_type} after cleaning.")
            col_types[col] = inferred_type

    return col_types

# Section 3: Readers
def read_csv(filepath, skiprows=None, nrows=None, infer_types=True):
    if not filepath.endswith(".csv"):
        raise ValueError("Invalid file format. Function read_csv supports only csv files.")

    rows = []
    try:
        with open(filepath, "r", encoding="utf-8-sig", newline="") as f:
            rdr = _csv.reader(f, delimiter=",", quotechar='"', escapechar="\\")
            # Skip header + optional leading rows
            header = next(rdr, None)
            if header is None:
                raise ValueError("CSV file is empty (no header row found).")

            # Normalize header length (trim trailing empties caused by stray commas)
            while len(header) and header[-1] == "":
                header.pop()

            # Skip user-requested rows
            if skiprows and skiprows > 0:
                for _ in range(skiprows):
                    next(rdr, None)

            # Collect data with proper padding/truncation to header length
            count = 0
            for row in rdr:
                if nrows is not None and count >= nrows:
                    break
                # Trim/pad to header length
                if len(row) > len(header):
                    row = row[:len(header)]
                elif len(row) < len(header):
                    row = row + [""] * (len(header) - len(row))
                rows.append(row)
                count += 1

        if not rows:
            raise ValueError("CSV file has no data rows.")

        # Type inference (on properly parsed rows)
        types = infer_column_types(rows) if infer_types else None
        print("read_csv was executed successfully.")
        return header, rows, types, infer_column_types

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

def sanitize_headers(header):
    seen = set()
    for i in range(len(header)):
        name = str(header[i]).strip() if header[i] else f"col_{i}"
        base = name
        k = 1
        while name in seen:
            k += 1
            name = f"{base}__{k}"
        header[i] = name
        seen.add(name)
    return header

def protected_indices(header, name_patterns=("zip","postal","pincode","pin","ssn","nid","uin","id","code","phone","date")):
    S = set()
    for i, h in enumerate(header):
        hlow = str(h).lower()
        if any(p in hlow for p in name_patterns):
            S.add(i)
    return S

# Display missing values pre computing
def compute_missing_table(data, header):
    if not data:
        return []
    total = len(data)
    out = []
    for i, col in enumerate(zip(*data)):
        miss = 0
        for v in col:
            s = str(v).strip().lower()
            if s in ("", "nan", "null", "none"):
                miss += 1
        pct = (miss / total * 100.0) if total else 0.0
        out.append({"column": header[i], "count": miss, "percent": pct})
    # sort: non-zero first by percent desc, then zeros
    nz = [r for r in out if r["count"] > 0]
    z  = [r for r in out if r["count"] == 0]
    nz.sort(key=lambda r: r["percent"], reverse=True)
    return nz + z
def clone_table(data):
    """Shallow row-wise copy for before/after snapshots."""
    return [row[:] for row in data]

def coerce_protected_types(types, prot_idx):
    """Force protected columns to 'str' in the types list."""
    if types is None: 
        return types
    out = types[:]
    for i in prot_idx:
        if i < len(out):
            out[i] = "str"
    return out

def report_missing_values_compare(before_data, after_data, header, protected_idx=None, title_before="Missing (raw)", title_after="Missing (after impute/cast)"):
    """
    Show missing counts BEFORE vs AFTER imputation/casting.
    Marks protected columns so expectations are clear.
    """
    protected_idx = protected_idx or set()

    def _is_missing(v):
        s = str(v).strip().lower()
        return s in ("", "nan", "null", "none")

    def _count_missing(tbl):
        total_rows = len(tbl)
        cols = list(zip(*tbl)) if tbl else [[] for _ in header]
        miss = []
        for i, col in enumerate(cols):
            cnt = sum(1 for v in col if _is_missing(v))
            miss.append((header[i], cnt, (cnt / total_rows * 100 if total_rows else 0.0)))
        return miss

    m_before = _count_missing(before_data)
    m_after  = _count_missing(after_data)

    # Print BEFORE
    print(f"\n{title_before}:")
    print("+---------------------------+--------+-----------+")
    print("| Column                    | Count  | Percent   |")
    print("+---------------------------+--------+-----------+")
    for name, cnt, pct in m_before:
        print(f"| {name.ljust(25)} | {str(cnt).rjust(6)} | {pct:7.2f}% |")
    print("+---------------------------+--------+-----------+")

    # Print AFTER (with protected tag)
    print(f"\n{title_after}:")
    print("+---------------------------+--------+-----------+-----------+")
    print("| Column                    | Count  | Percent   | Protected |")
    print("+---------------------------+--------+-----------+-----------+")
    name_to_idx = {h:i for i,h in enumerate(header)}
    for name, cnt, pct in m_after:
        idx = name_to_idx.get(name, -1)
        prot = "Yes" if idx in protected_idx else "No"
        print(f"| {name.ljust(25)} | {str(cnt).rjust(6)} | {pct:7.2f}% | {prot:^9} |")
    print("+---------------------------+--------+-----------+-----------+\n")


# Section 5: Cleaning Pipeline
def clean_df(
    filepath,
    format=None,
    reader_kwargs=None,
    infer_types=True,
    drop_threshold=0.5,
    impute_strategy="mean",
    display=True,
    report_missing=False,
    display_kwargs=None,
    summary=False,
    summary_percentiles=[25,50,75],
    # --- V3 additions ---
    coerce_numeric_strings=True,
    protect_identifiers=True,
    protected_name_patterns=("zip","postal","pincode","pin","ssn","nid","uin","id","code","phone","date"),
    outlier_method=None,              # None | 'iqr' | 'zscore'
    outlier_action="flag",            # 'flag' | 'remove'
    outlier_cols=None,                # list[str] or None (all numeric)
    iqr_k=1.5,
    zscore_threshold=3.0,
    visualize=False,
    plots=None,                       # e.g., [("hist","salary"), ("scatter","age","gpa"), ("box", ["gpa","salary"]), ("heatmap", None)]
    export_html_path=None,
    print_outlier_summary=True,
    recheck_types = False
):
    log("clean_df: start")

    # 1) Read
    header, data, _types_ignored, infer_types_fn = read_any(
        filepath, format=format, reader_kwargs=reader_kwargs, infer_types=False
    )
    log(f"clean_df: read_any -> rows={len(data)}, cols={len(header)}")

    # 2) Sanitize headers
    header = sanitize_headers(header)
    log("clean_df: headers sanitized")

    # 3) Protected set
    prot_idx = protected_indices(header, protected_name_patterns) if protect_identifiers else set()
    log(f"clean_df: protected indices -> {sorted(list(prot_idx))}")

    # 4) Drop high-missing with protection
    header, data = drop_high_missing_cols_protected(
        data, header, threshold=drop_threshold, protected_idx=prot_idx
    )
    log(f"clean_df: drop_high_missing_cols_protected -> cols={len(header)}")

    # Recompute protected indices after structure change
    prot_idx = protected_indices(header, protected_name_patterns) if protect_identifiers else set()

    # 5) Numeric coercion BEFORE inference (skips protected columns)
    if coerce_numeric_strings:
        changed = 0
        for r, row in enumerate(data):
            for c in range(len(header)):
                if c in prot_idx:
                    continue
                v = row[c] if c < len(row) else None
                if isinstance(v, (int, float)):
                    continue
                num = clean_numeric_token(v)
                if num is not None:
                    row[c] = num
                    changed += 1
        log(f"clean_df: numeric-string coercion complete (cells coerced={changed})")

    # 6) Infer types (after coercion so numeric cols infer correctly)
    if infer_types:
        types = infer_types_fn(data)
        # Keep identifiers/ZIP/phone/date as strings so they never get cast to None
        types = coerce_protected_types(types, prot_idx)
        log("clean_df: type inference complete (protected coerced to str)")
    else:
        types = ["str"] * len(header)
        log("clean_df: infer_types=False; defaulting all types to 'str'")

    # Make types visible to heatmap helper
    global types_global_placeholder
    types_global_placeholder = types

    # 7) Snapshot missing before impute/cast (post-coercion, post-inference)
    missing_before = compute_missing_table(data, header)

    # Keep a deep copy for debugging/experiments
    before_impute_cast = clone_table(data)

    # 8) Impute using the correct types
    impute_cols(data, header, types, strategy=impute_strategy)
    log("clean_df: impute_cols complete")

    # 9) Cast rows to the inferred types (protect identifiers from numeric casting)
    data = [cast_row_types(row, types, prot_idx=prot_idx) for row in data]
    log("clean_df: cast_row_types complete")

    def recheck_column_types(data, header, col_types, protected_cols, log_func=print):
        for j, col in enumerate(header):
            if col in protected_cols:
                continue  # skip protected columns

            values = [row[j] for row in data if row[j] is not None]
            if not values:
                continue

            inferred_type = infer_column_types(values)

            if inferred_type != col_types[col]:
                log_func(f"[INFO] Column '{col}' type changed from {col_types[col]} → {inferred_type} after cleaning.")
                col_types[col] = inferred_type

        return col_types

    # 10) Re-infer types after cast; keep protected as 'str'
    if infer_types:
        try:
            types = infer_types_fn(data)
            types = coerce_protected_types(types, prot_idx)
            types_global_placeholder = types
            log("clean_df: re-inferred types after cast (protected coerced to str)")
        except Exception as e:
            log(f"clean_df: post-cast inference FAILED: {e}")

    # 11) Snapshot missing AFTER impute/cast
    missing_after = compute_missing_table(data, header)

    # 12) Missing report (BEFORE vs AFTER)
    if report_missing:
        print("\n--- Missing Values (BEFORE imputation) ---")
        print("+-------------------------------+--------+-----------+")
        print("| Column                        | Count  | Percent   |")
        print("+-------------------------------+--------+-----------+")
        for r in missing_before:
            print(f"| {r['column'][:29].ljust(29)} | {str(r['count']).rjust(6)} | {r['percent']:>7.2f}% |")
        print("+-------------------------------+--------+-----------+")

        print("\n--- Missing Values (AFTER imputation) ---")
        print("+-------------------------------+--------+-----------+")
        print("| Column                        | Count  | Percent   |")
        print("+-------------------------------+--------+-----------+")
        for r in missing_after:
            print(f"| {r['column'][:29].ljust(29)} | {str(r['count']).rjust(6)} | {r['percent']:>7.2f}% |")
        print("+-------------------------------+--------+-----------+")
        log("clean_df: report_missing_values printed (before & after)")

    # 13) Outliers
    outlier_summary = None
    if outlier_method:
        method = outlier_method.lower()
        if method not in ("iqr","zscore","z"):
            raise ValueError("outlier_method must be None, 'iqr', or 'zscore'")
        params = {"k": iqr_k} if method == "iqr" else {"threshold": zscore_threshold}
        new_header, new_data, new_types, outlier_summary = handle_outliers(
            data, header, types, method=method, cols=outlier_cols, action=outlier_action, **params
        )
        header, data, types = new_header, new_data, new_types
        types_global_placeholder = types  # keep heatmap in sync
        if print_outlier_summary and outlier_summary:
            log("clean_df: outlier summary ->")
            for kcol, meta in outlier_summary.items():
                print(f"[Outlier] {kcol}: {meta}")

    # 14) Visualizations
    figs = []
    if visualize:
        for spec in (plots or []):
            kind = spec[0]
            try:
                if kind == "hist":
                    fig = plot_hist(data, header, spec[1])
                elif kind == "scatter":
                    fig = plot_scatter(data, header, spec[1], spec[2])
                elif kind == "box":
                    cols = spec[1] if isinstance(spec[1], (list,tuple)) else [spec[1]]
                    fig = plot_box(data, header, cols)
                elif kind == "heatmap":
                    fig = plot_heatmap(data, header)
                elif kind == "bar_counts":
                    col = spec[1]
                    top_n = spec[2] if len(spec) > 2 else None
                    normalize = spec[3] if len(spec) > 3 else False
                    fig = plot_bar_counts(data, header, col, top_n=top_n, normalize=normalize)
                elif kind == "bar_agg":
                    cat_col = spec[1]; num_col = spec[2]
                    agg = spec[3] if len(spec) > 3 else "mean"
                    top_n = spec[4] if len(spec) > 4 else None
                    fig = plot_bar_agg(data, header, cat_col, num_col, agg=agg, top_n=top_n)
                elif kind == "line":
                    fig = plot_line(data, header, spec[1], spec[2])
                else:
                    fig = None
                if fig is not None:
                    figs.append(fig)
            except Exception as e:
                log(f"clean_df: visualize '{kind}' FAILED -> {e}")

    # 15) Display preview
    if display:
        display_tabular_output(header, data, **(display_kwargs or {}))
        log("clean_df: display_tabular_output done")

    # 16) Summary
    summary_dict = None
    if summary:
        summary_dict = summarize_columns(data, header, types, percentiles=summary_percentiles, print_table=True)
        log("clean_df: summarize_columns done")

    # 17) HTML report export
    if export_html_path is not None:
        try:
            report = generate_report(
                header, data, types,
                missing_report=True,
                summary_dict=summary_dict,
                outlier_summary=outlier_summary,
                figures=figs,
                missing_before=missing_before,
                missing_after=missing_after
            )
            write_report_html(report, export_html_path)
            log(f"clean_df: report exported -> {export_html_path}")
        except Exception as e:
            log(f"clean_df: write_report_html FAILED: {e}")

    log("clean_df: end")
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
def report_missing_values(data,header,*,sort_by="percent",descending=True,hide_zero=True,top=None,bar_width=20):
    if not data:
        print("No data found. Skipping missing value report.\n")
        return

    total_rows = len(data)

    # Per-column stats
    rows = []
    for i, col in enumerate(zip(*data)):
        miss = 0
        for v in col:
            s = str(v).strip().lower()
            if s in ("", "nan", "null", "none"):
                miss += 1
        pct = (miss / total_rows * 100.0) if total_rows else 0.0
        rows.append({
            "column": header[i],
            "count": miss,
            "percent": pct
        })

    # Filter zeros
    if hide_zero:
        rows = [r for r in rows if r["count"] > 0]

    # Sort
    if sort_by in ("percent", "count"):
        rows.sort(key=lambda r: r[sort_by], reverse=bool(descending))

    # Limit
    if top is not None and top > 0:
        rows = rows[:top]

    # If everything was zero and zeroes are hid:
    if not rows:
        print("\nMissing Values (non-zero only): none \n")
        return

    # Tabular Print
    print("\nMissing Values Report:")
    print("+-------------------------------+--------+-----------+----------------------+")
    print("| Column                        | Count  | Percent   | Bar                  |")
    print("+-------------------------------+--------+-----------+----------------------+")
    for r in rows:
        bar_n = 0 if total_rows == 0 else int(round((r["percent"] / 100.0) * bar_width))
        bar = "█" * bar_n + " " * (bar_width - bar_n)
        print(f"| {r['column'][:29].ljust(29)} | {str(r['count']).rjust(6)} | {r['percent']:>7.2f}% | {bar} |")
    print("+-------------------------------+--------+-----------+----------------------+")
    print(f"Total rows: {total_rows}\n")


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

def drop_high_missing_cols_protected(data, header, threshold, protected_idx):
    """
    Same as drop_high_missing_cols, but never drops columns whose index is in protected_idx.
    """
    total_rows = len(data)
    keep_indices = []
    dropped_columns = []

    def _is_missing(v):
        s = str(v).strip().lower()
        return s in ("", "nan", "null", "none")

    for j in range(len(header)):
        if j in protected_idx:
            keep_indices.append(j)
            continue
        miss = sum(1 for row in data if j >= len(row) or _is_missing(row[j]))
        frac = (miss / total_rows) if total_rows else 1.0
        if frac < float(threshold):
            keep_indices.append(j)
        else:
            dropped_columns.append(header[j])

    new_header = [header[i] for i in keep_indices]
    new_data   = [[row[i] if i < len(row) else "" for i in keep_indices] for row in data]

    if dropped_columns:
        print(f"\nDropped columns due to high missing values (> {float(threshold)*100}%):")
        for col_name in dropped_columns:
            print(f"- {col_name}")
    else:
        print("\nNo columns dropped.")

    return new_header, new_data


def impute_cols(data, header, types, strategy="mean"):
    if strategy not in ("mean", "median", "mode"):
        raise ValueError("Invalid measure of central tendency chosen. Choose from: mean, median, or mode.")

    columns = list(zip(*data))
    total_rows = len(data)

    for i, col in enumerate(columns):
        col_type = types[i]

        # positions to fill
        miss_pos = []
        nonmiss_vals = []
        for r_idx, v in enumerate(col):
            s = str(v).strip().lower()
            if s in ("", "nan", "null", "none"):
                miss_pos.append(r_idx)
            else:
                nonmiss_vals.append(v)

        if not miss_pos:
            continue
        if not nonmiss_vals:
            print(f"Skipping column '{header[i]}' - no valid data to impute.")
            continue

        if col_type in ("int", "float"):
            # parse with cleaner; keep numeric list only
            numeric = [clean_numeric_token(v) for v in nonmiss_vals]
            numeric = [x for x in numeric if x is not None]
            if not numeric:
                print(f"Skipping column '{header[i]}' - no numeric values to impute.")
                continue

            if strategy == "mean":
                imp = statistics.mean(numeric)
            elif strategy == "median":
                imp = statistics.median(numeric)
            else:
                try:
                    imp = statistics.mode(numeric)
                except StatisticsError:
                    imp = Counter(numeric).most_common(1)[0][0]

            # cast to exact type and WRITE BACK AS NUMBERS
            if col_type == "int":
                try:
                    imp = int(round(float(imp)))
                except Exception:
                    imp = 0
            else:
                imp = float(imp)

            print_value = f"{imp:.2f}" if isinstance(imp, float) else str(imp)
            print(f"Imputed column '{header[i]}' with {strategy}: {print_value}")

            for r in miss_pos:
                data[r][i] = imp

        elif col_type == "bool":
            # normalize to bool
            canon = []
            for v in nonmiss_vals:
                s = str(v).strip().lower()
                canon.append(s in ("true", "1", "yes"))
            # mode of booleans
            try:
                imp = statistics.mode(canon)
            except StatisticsError:
                imp = Counter(canon).most_common(1)[0][0]
            print(f"Imputed column '{header[i]}' with mode: {'True' if imp else 'False'}")
            for r in miss_pos:
                data[r][i] = bool(imp)

        elif col_type == "datetime":
            # use the most common parsed date; write back as datetime
            parsed = [_parse_dt_try(v) for v in nonmiss_vals]
            parsed = [d for d in parsed if d is not None]
            if not parsed:
                print(f"Skipping column '{header[i]}' - no valid dates to impute.")
                continue
            # mode by date value
            try:
                imp = statistics.mode(parsed)
            except StatisticsError:
                imp = Counter(parsed).most_common(1)[0][0]
            print(f"Imputed column '{header[i]}' with mode: {imp.strftime('%Y-%m-%d')}")
            for r in miss_pos:
                data[r][i] = imp

        else:
            # strings: mode as before
            strings = [str(v).strip() for v in nonmiss_vals]
            try:
                imp = statistics.mode(strings)
            except StatisticsError:
                imp = Counter(strings).most_common(1)[0][0]
            print(f"Imputed column '{header[i]}' with mode: {imp}")
            for r in miss_pos:
                data[r][i] = imp

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

    effective_types = types if types is not None else ["str"] * len(header)
    summary = {}

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
                if isinstance(v, (int, float)):
                    numeric_vals.append(float(v))
                else:
                    x = clean_numeric_token(v)
                    if x is not None:
                        numeric_vals.append(float(x))

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

                for p in percentiles:
                    col_summary[f"p{int(p)}"] = _percentile_linear(numeric_vals, p)

                p25 = col_summary.get("p25")
                p75 = col_summary.get("p75")
                col_summary["iqr"] = (p75 - p25) if (p25 is not None and p75 is not None) else None

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

# ---------- V3: Outliers ----------
def _percentile_linear(sorted_vals, p):
    if not sorted_vals: return None
    if p <= 0: return sorted_vals[0]
    if p >= 100: return sorted_vals[-1]
    pos = (p/100.0)*(len(sorted_vals)-1)
    f = int(pos)
    c = f + 1
    if c >= len(sorted_vals): return sorted_vals[-1]
    d = pos - f
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * d

def outliers_iqr(data, header, types, cols=None, k=1.5, action="flag", suffix="_outlier_iqr"):
    assert action in ("flag","remove"), "action must be 'flag' or 'remove'"
    num_idx = [i for i,t in enumerate(types or []) if t in ("int","float")]
    if cols:
        n2i = {h:i for i,h in enumerate(header)}
        target = [n2i[c] for c in cols if c in n2i and n2i[c] in num_idx]
    else:
        target = num_idx

    thresholds = {}
    for i in target:
        vals = []
        for row in data:
            v = to_float_or_none(row[i] if i < len(row) else None)
            if v is not None: vals.append(v)
        if len(vals) < 4: 
            log(f"Skipping IQR for '{header[i]}' (insufficient numeric data).")
            continue
        vals.sort()
        q1, q3 = _percentile_linear(vals,25), _percentile_linear(vals,75)
        if q1 is None or q3 is None or (q3 - q1) == 0:
            log(f"IQR zero/None for '{header[i]}', skipping.")
            continue
        thresholds[i] = (q1 - k*(q3-q1), q3 + k*(q3-q1))

    summary = {}
    if action == "flag":
        new_header = header[:]
        for i in thresholds: new_header.append(f"{header[i]}{suffix}")
        new_types = (types or []) + ["bool"] * len(thresholds)
        new_data = []
        for row in data:
            flags = []
            for i,(lo,hi) in thresholds.items():
                v = to_float_or_none(row[i] if i < len(row) else None)
                flags.append(v is not None and (v < lo or v > hi))
            new_data.append(row + flags)
        for i,(lo,hi) in thresholds.items():
            cnt = sum(1 for row in data if ((v:=to_float_or_none(row[i] if i < len(row) else None)) is not None and (v<lo or v>hi)))
            summary[header[i]] = {"method":"IQR","k":k,"low":lo,"high":hi,"flagged":cnt}
        log(f"IQR flagging complete on {len(thresholds)} column(s).")
        return new_header, new_data, new_types, summary

    # remove
    keep = []
    removed = set()
    for r,row in enumerate(data):
        drop = False
        for i,(lo,hi) in thresholds.items():
            v = to_float_or_none(row[i] if i < len(row) else None)
            if v is not None and (v<lo or v>hi):
                drop = True
                removed.add(r)
                break
        if not drop: keep.append(row)
    for i,(lo,hi) in thresholds.items():
        cnt = sum(1 for r,row in enumerate(data) if r in removed and ((v:=to_float_or_none(row[i] if i < len(row) else None)) is not None and (v<lo or v>hi)))
        summary[header[i]] = {"method":"IQR","k":k,"low":lo,"high":hi,"removed":cnt}
    log(f"IQR removal complete: dropped {len(removed)} row(s).")
    return header, keep, types, summary

def _mean_std(vals):
    if not vals: return None,None
    mu = sum(vals)/len(vals)
    var = sum((x-mu)**2 for x in vals)/len(vals)
    return mu, _math.sqrt(var)

def outliers_zscore(data, header, types, cols=None, threshold=3.0, action="flag", suffix="_outlier_z"):
    assert action in ("flag","remove"), "action must be 'flag' or 'remove'"
    num_idx = [i for i,t in enumerate(types or []) if t in ("int","float")]
    if cols:
        n2i = {h:i for i,h in enumerate(header)}
        target = [n2i[c] for c in cols if c in n2i and n2i[c] in num_idx]
    else:
        target = num_idx

    params = {}
    for i in target:
        vals = []
        for row in data:
            v = to_float_or_none(row[i] if i < len(row) else None)
            if v is not None: vals.append(v)
        if len(vals) < 2:
            log(f"Skipping Z-score for '{header[i]}' (insufficient numeric data).")
            continue
        mu, sd = _mean_std(vals)
        if not sd:
            log(f"Std dev zero/None for '{header[i]}', skipping.")
            continue
        params[i] = (mu, sd)

    summary = {}
    if action == "flag":
        new_header = header[:]
        for i in params: new_header.append(f"{header[i]}{suffix}")
        new_types = (types or []) + ["bool"] * len(params)
        new_data = []
        for row in data:
            flags = []
            for i,(mu,sd) in params.items():
                v = to_float_or_none(row[i] if i < len(row) else None)
                flags.append(v is not None and abs((v-mu)/sd) > threshold)
            new_data.append(row + flags)
        for i,(mu,sd) in params.items():
            cnt=0
            for row in data:
                v = to_float_or_none(row[i] if i < len(row) else None)
                if v is not None and abs((v-mu)/sd) > threshold: cnt+=1
            summary[header[i]] = {"method":"Z-score","threshold":threshold,"mean":mu,"std":sd,"flagged":cnt}
        log(f"Z-score flagging complete on {len(params)} column(s).")
        return new_header, new_data, new_types, summary

    # remove
    keep = []
    removed = set()
    for r,row in enumerate(data):
        drop = False
        for i,(mu,sd) in params.items():
            v = to_float_or_none(row[i] if i < len(row) else None)
            if v is not None and abs((v-mu)/sd) > threshold:
                drop=True; removed.add(r); break
        if not drop: keep.append(row)
    for i,(mu,sd) in params.items():
        cnt=0
        for r,row in enumerate(data):
            if r not in removed: continue
            v = to_float_or_none(row[i] if i < len(row) else None)
            if v is not None and abs((v-mu)/sd) > threshold: cnt+=1
        summary[header[i]]={"method":"Z-score","threshold":threshold,"mean":mu,"std":sd,"removed":cnt}
    log(f"Z-score removal complete: dropped {len(removed)} row(s).")
    return header, keep, types, summary

def handle_outliers(data, header, types, method="iqr", cols=None, action="flag", **kwargs):
    method = (method or "iqr").lower()
    if method == "iqr":
        return outliers_iqr(data, header, types, cols=cols, action=action, **kwargs)
    elif method in ("zscore","z"):
        return outliers_zscore(data, header, types, cols=cols, action=action, **kwargs)
    else:
        raise ValueError("Unknown outlier method. Use 'iqr' or 'zscore'.")


# ---------- V3: Visualization wrappers ----------

def _get_plt(need_sns=False, need_np=False):
    """
    Lazy import plotting stack and force a headless backend.
    Returns: (plt, extras_dict) or (None, {}) if imports fail.
    """
    try:
        import matplotlib as _mpl
        try:
            # Force headless backend early (before pyplot import)
            _mpl.use("Agg", force=True)
        except Exception as be:
            log(f"plot: backend select warning -> {be}")
        import matplotlib.pyplot as plt
    except Exception as e:
        log(f"plot: matplotlib import/backend error -> {e}")
        return None, {}

    extras = {}
    if need_np:
        try:
            import numpy as _np
            extras["np"] = _np
        except Exception as e:
            log(f"plot: numpy import error -> {e}")
    if need_sns:
        try:
            import seaborn as _sns
            extras["sns"] = _sns
        except Exception as e:
            log(f"plot: seaborn import error -> {e}")
    return plt, extras


def plot_hist(data, header, column, bins=30, **kwargs):
    plt, _ = _get_plt()
    if plt is None:
        return None
    try:
        idx = header.index(column)
    except ValueError:
        log(f"plot_hist: column '{column}' not found.")
        return None
    vals = [to_float_or_none(row[idx] if idx < len(row) else None) for row in data]
    vals = [v for v in vals if v is not None]
    if not vals:
        log(f"plot_hist: no numeric data for '{column}'.")
        return None
    fig = plt.figure()
    plt.hist(vals, bins=bins, **kwargs)
    plt.title(f"Histogram: {column}")
    plt.xlabel(column); plt.ylabel("Count")
    return fig


def plot_scatter(data, header, xcol, ycol, **kwargs):
    plt, _ = _get_plt()
    if plt is None:
        return None
    try:
        ix, iy = header.index(xcol), header.index(ycol)
    except ValueError as e:
        log(f"plot_scatter: {e}")
        return None
    X = [to_float_or_none(row[ix] if ix < len(row) else None) for row in data]
    Y = [to_float_or_none(row[iy] if iy < len(row) else None) for row in data]
    pts = [(x, y) for x, y in zip(X, Y) if x is not None and y is not None]
    if not pts:
        log(f"plot_scatter: no numeric data for {xcol}/{ycol}")
        return None
    x, y = zip(*pts)
    fig = plt.figure()
    plt.scatter(x, y, **kwargs)
    plt.xlabel(xcol); plt.ylabel(ycol); plt.title(f"{ycol} vs {xcol}")
    return fig


def plot_box(data, header, columns, **kwargs):
    plt, _ = _get_plt()
    if plt is None:
        return None
    try:
        idxs = [header.index(c) for c in columns]
    except ValueError as e:
        log(f"plot_box: {e}")
        return None
    mats = []
    labels = []
    for i, c in zip(idxs, columns):
        col = [to_float_or_none(row[i] if i < len(row) else None) for row in data]
        series = [v for v in col if v is not None]
        if series:
            mats.append(series)
            labels.append(c)
    if not mats:
        log("plot_box: no numeric data in requested columns.")
        return None
    fig = plt.figure()
    plt.boxplot(mats, labels=labels, **kwargs)
    plt.title("Box plot")
    return fig


def plot_heatmap(data, header, numeric_only=True, **kwargs):
    plt, ex = _get_plt(need_np=True, need_sns=True)
    if plt is None:
        return None
    _np = ex.get("np")
    _sns = ex.get("sns")
    if _np is None or _sns is None:
        log("plot_heatmap: numpy/seaborn not available.")
        return None

    # choose numeric columns by types if available
    idxs = [i for i, t in enumerate(globals().get('types_global_placeholder') or []) if t in ("int", "float")]
    if not idxs:
        # fallback: any column with >= 2 numeric values
        for j in range(len(header)):
            col = [to_float_or_none(row[j] if j < len(row) else None) for row in data]
            if sum(1 for v in col if v is not None) >= 2:
                idxs.append(j)
    if not idxs:
        log("plot_heatmap: no numeric columns.")
        return None

    cols = [header[i] for i in idxs]

    # build dense matrix with mean-fill for None
    cols_vals = []
    for i in idxs:
        col = [to_float_or_none(row[i] if i < len(row) else None) for row in data]
        arr = [v for v in col if v is not None]
        mu = sum(arr) / len(arr) if arr else 0.0
        cols_vals.append([mu if v is None else v for v in col])
    M = _np.array(cols_vals, dtype=float)
    if M.ndim != 2 or M.shape[0] < 2:
        log("plot_heatmap: not enough columns for correlation.")
        return None
    corr = _np.corrcoef(M)

    fig = plt.figure()
    ax = _sns.heatmap(corr, xticklabels=cols, yticklabels=cols, **kwargs)
    ax.set_title("Correlation heatmap")
    return fig

# ---------- V3: Extra Visualization wrappers (add-only; no changes to existing code) ----------

def plot_bar_counts(data, header, column, top_n=None, normalize=False, rotate=45):
    """
    Bar chart of category frequencies for a single column.
    - top_n: keep only top N categories by count
    - normalize: plot percentages instead of raw counts
    """
    try:
        import matplotlib as _mpl
        _mpl.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as e:
        log(f"plot_bar_counts: matplotlib import/backend error -> {e}")
        return None

    idx = header.index(column)
    counts = {}
    for row in data:
        v = row[idx] if idx < len(row) else None
        s = str(v).strip()
        if s == "" or s.lower() in ("nan", "null", "none"):
            continue
        counts[s] = counts.get(s, 0) + 1

    if not counts:
        log(f"plot_bar_counts: no non-missing values in '{column}'")
        return None

    items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    if isinstance(top_n, int) and top_n > 0:
        items = items[:top_n]

    labels = [k for k,_ in items]
    vals = [v for _,v in items]

    if normalize:
        total = sum(vals)
        vals = [(v/total)*100.0 for v in vals]

    fig = plt.figure()
    plt.bar(range(len(labels)), vals)
    plt.xticks(range(len(labels)), labels, rotation=rotate, ha="right")
    plt.ylabel("Percent" if normalize else "Count")
    tbits = []
    if top_n: tbits.append(f"top {top_n}")
    if normalize: tbits.append("percent")
    subtitle = f" ({', '.join(tbits)})" if tbits else ""
    plt.title(f"Category frequencies: {column}{subtitle}")
    plt.tight_layout()
    return fig


def plot_bar_agg(data, header, cat_col, num_col, agg="mean", top_n=None, rotate=45):
    """
    Bar chart of an aggregate (mean/median/sum/count) of a numeric column grouped by a categorical column.
    - agg: 'mean' | 'median' | 'sum' | 'count'
    - top_n: keep only top N groups by the aggregated value
    """
    try:
        import matplotlib as _mpl
        _mpl.use("Agg", force=True)
        import matplotlib.pyplot as plt
        import statistics as _stats
    except Exception as e:
        log(f"plot_bar_agg: matplotlib import/backend error -> {e}")
        return None

    ix_cat = header.index(cat_col)
    ix_num = header.index(num_col)

    buckets = {}
    for row in data:
        k = row[ix_cat] if ix_cat < len(row) else None
        k = str(k).strip()
        if k == "" or k.lower() in ("nan", "null", "none"):
            continue
        v = row[ix_num] if ix_num < len(row) else None
        x = to_float_or_none(v)
        if x is None:
            continue
        buckets.setdefault(k, []).append(x)

    if not buckets:
        log(f"plot_bar_agg: no usable data for '{cat_col}' x '{num_col}'")
        return None

    agg_vals = []
    for k, arr in buckets.items():
        if not arr:
            continue
        if agg == "mean":
            val = sum(arr)/len(arr)
        elif agg == "median":
            try:
                val = _stats.median(arr)
            except Exception:
                arr_sorted = sorted(arr)
                m = len(arr_sorted)//2
                val = (arr_sorted[m] if len(arr_sorted)%2==1 else (arr_sorted[m-1]+arr_sorted[m])/2)
        elif agg == "sum":
            val = sum(arr)
        elif agg == "count":
            val = len(arr)
        else:
            log(f"plot_bar_agg: unknown agg '{agg}', using mean")
            val = sum(arr)/len(arr)
        agg_vals.append((k, val))

    agg_vals.sort(key=lambda kv: kv[1], reverse=True)
    if isinstance(top_n, int) and top_n > 0:
        agg_vals = agg_vals[:top_n]

    labels = [k for k,_ in agg_vals]
    vals = [v for _,v in agg_vals]

    fig = plt.figure()
    plt.bar(range(len(labels)), vals)
    plt.xticks(range(len(labels)), labels, rotation=rotate, ha="right")
    plt.ylabel(f"{agg}({num_col})")
    subtitle = f" (top {top_n})" if top_n else ""
    plt.title(f"{agg.capitalize()} of {num_col} by {cat_col}{subtitle}")
    plt.tight_layout()
    return fig


def plot_line(data, header, xcol, ycol, sort_x=True):
    """
    Line plot with a (date-like) x-axis and numeric y-axis.
    - Accepts datetime objects or strings in ISO YYYY-MM-DD; attempts best-effort parse.
    - sort_x: sort by x before plotting.
    """
    try:
        import matplotlib as _mpl
        _mpl.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from datetime import datetime as _dt
    except Exception as e:
        log(f"plot_line: matplotlib import/backend error -> {e}")
        return None

    ix = header.index(xcol)
    iy = header.index(ycol)

    points = []
    for row in data:
        xv = row[ix] if ix < len(row) else None
        yv = to_float_or_none(row[iy] if iy < len(row) else None)
        if yv is None:
            continue
        # parse x
        if isinstance(xv, _dt):
            xd = xv
        else:
            s = str(xv).strip()
            if s == "" or s.lower() in ("nan", "null", "none"):
                continue
            # try a couple formats; fall back to ISO
            parsed = None
            for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%Y/%m/%d"):
                try:
                    parsed = _dt.strptime(s, fmt)
                    break
                except Exception:
                    continue
            if parsed is None:
                try:
                    parsed = _dt.fromisoformat(s)
                except Exception:
                    # not a date, skip
                    continue
            xd = parsed
        points.append((xd, yv))

    if not points:
        log(f"plot_line: no usable date/numeric pairs for {xcol}/{ycol}")
        return None

    if sort_x:
        points.sort(key=lambda p: p[0])

    xs, ys = zip(*points)
    fig = plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.xlabel(xcol); plt.ylabel(ycol)
    plt.title(f"{ycol} over {xcol}")
    fig.autofmt_xdate()
    plt.tight_layout()
    return fig

# ---------- V3: Report & HTML ----------
def _fig_to_data_uri(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"

def generate_report(header, data, types, missing_report=True,
                    summary_dict=None, outlier_summary=None, figures=None,
                    *, missing_before=None, missing_after=None):
    info = {
        "rows": len(data),
        "cols": len(header),
        "columns": header[:],
    }

    miss_before_tbl = None
    miss_after_tbl  = None
    if missing_report:
        # If caller passed snapshots, use them; otherwise compute on the fly
        miss_before_tbl = missing_before if missing_before is not None else compute_missing_table(data, header)
        miss_after_tbl  = missing_after  if missing_after  is not None else compute_missing_table(data, header)

    if summary_dict is None:
        summary_dict = summarize_columns(data, header, types, print_table=False)

    figs_uris = []
    if figures:
        for fig in figures:
            try:
                figs_uris.append(_fig_to_data_uri(fig))
            except Exception:
                pass

    return {
        "dataset_info": info,
        "missing_before": miss_before_tbl,
        "missing_after":  miss_after_tbl,
        "summary_stats": summary_dict,
        "outliers": outlier_summary,
        "figures": figs_uris,
    }

def write_report_html(report, path):
    def esc(s):
        return (str(s)
                .replace("&","&amp;").replace("<","&lt;").replace(">","&gt;"))
    html = []
    html.append("<html><head><meta charset='utf-8'><title>EDA Report</title>")
    html.append("<style>body{font-family:Arial,Helvetica,sans-serif;padding:16px;} table{border-collapse:collapse;margin:12px 0;} th,td{border:1px solid #ddd;padding:6px 10px;font-size:13px;} th{background:#f6f6f6;text-align:left;} h2{margin-top:28px;} .small{color:#666;font-size:12px;}</style>")
    html.append("</head><body>")
    di = report.get("dataset_info",{})
    html.append("<h1>EDA Report</h1>")
    html.append(f"<p class='small'>Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
    html.append("<h2>Dataset Info</h2>")
    html.append(f"<p>Rows: {di.get('rows',0)} &nbsp; Columns: {di.get('cols',0)}</p>")
    if di.get("columns"):
        html.append("<table><tr><th>#</th><th>Column</th></tr>")
        for i,c in enumerate(di["columns"]):
            html.append(f"<tr><td>{i}</td><td>{esc(c)}</td></tr>")
        html.append("</table>")

    # Missing (before/after)
    miss_before = report.get("missing_before") or []
    miss_after  = report.get("missing_after") or []
    if miss_before or miss_after:
        html.append("""
<style>
.mvbar { height:10px; background:#eee; border-radius:4px; overflow:hidden; }
.mvbar > span { display:block; height:100%; background:#f39c12; }
.mvtab th, .mvtab td { white-space:nowrap; }
</style>
""")
        if miss_before:
            html.append("<h2>Missing Values (Before Imputation)</h2>")
            html.append("<table class='mvtab'><tr><th>Column</th><th>Count</th><th>Percent</th><th style='width:180px'>Bar</th></tr>")
            for row in miss_before:
                pct = float(row.get("percent", 0.0))
                w = max(0.0, min(100.0, pct))
                html.append(
                    f"<tr><td>{esc(row['column'])}</td><td>{row['count']}</td><td>{pct:.2f}%</td>"
                    f"<td><div class='mvbar'><span style='width:{w}%;'></span></div></td></tr>"
                )
            html.append("</table>")
        if miss_after:
            html.append("<h2>Missing Values (After Imputation)</h2>")
            html.append("<table class='mvtab'><tr><th>Column</th><th>Count</th><th>Percent</th><th style='width:180px'>Bar</th></tr>")
            for row in miss_after:
                pct = float(row.get("percent", 0.0))
                w = max(0.0, min(100.0, pct))
                html.append(
                    f"<tr><td>{esc(row['column'])}</td><td>{row['count']}</td><td>{pct:.2f}%</td>"
                    f"<td><div class='mvbar'><span style='width:{w}%;'></span></div></td></tr>"
                )
            html.append("</table>")

    # Summary
    summ = report.get("summary_stats") or {}
    if summ:
        html.append("<h2>Summary Statistics</h2>")
        keys = ["type","count","missing","unique","min","p25","p50","p75","median","max","mean","mode","iqr"]
        html.append("<table><tr><th>Column</th>" + "".join(f"<th>{k.upper()}</th>" for k in keys) + "</tr>")
        for col, s in summ.items():
            html.append("<tr><td>"+esc(col)+"</td>" + "".join(f"<td>{esc(s.get(k,''))}</td>" for k in keys) + "</tr>")
        html.append("</table>")

    # Outliers
    outs = report.get("outliers")
    if outs:
        html.append("<h2>Outlier Summary</h2>")
        all_keys = set()
        for v in outs.values():
            all_keys |= set(v.keys())
        all_keys = [k for k in ["method","k","threshold","low","high","mean","std","flagged","removed"] if k in all_keys] + \
                   [k for k in sorted(all_keys) if k not in {"method","k","threshold","low","high","mean","std","flagged","removed"}]
        html.append("<table><tr><th>Column</th>" + "".join(f"<th>{esc(k)}</th>" for k in all_keys) + "</tr>")
        for col,meta in outs.items():
            html.append("<tr><td>"+esc(col)+"</td>" + "".join(f"<td>{esc(meta.get(k,''))}</td>" for k in all_keys) + "</tr>")
        html.append("</table>")

    # Figures
    figs = report.get("figures") or []
    if figs:
        html.append("<h2>Visualizations</h2>")
        for uri in figs:
            html.append(f"<div><img src='{uri}' style='max-width:900px;width:100%;height:auto;border:1px solid #eee;margin:10px 0;'/></div>")

    html.append("</body></html>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(html))

    
# Section 9: Help Function
def help(name=None):
    """
    Swift EDA V3 Help.
    Call help() to see all available functions.
    Call help('clean_df') or help('plots') for details on one function.
    """

    if name is None:
        rows = [
            ["clean_df",               "Main pipeline: read, clean, handle missing values, detect outliers, show results."],
            ["summarize_columns",      "Summary statistics per column (numbers, text, dates, booleans)."],
            ["read_any",               "Automatically pick the right reader for CSV or JSON files."],
            ["read_csv / read_json",   "Manually read CSV or JSON into a header + rows."],
            ["display_tabular_output", "Show a table preview of the dataset."],
            ["report_missing_values",  "Print missing values count and percent per column."],
            ["compute_missing_table",  "Return missing value counts and percents in code form."],
            ["sanitize_headers",       "Fix blank or duplicate column names."],
            ["protected_indices",      "Mark identifier columns (IDs, ZIP, phone, dates) to avoid wrong type conversion."],
            ["drop_high_missing_cols_protected","Drop columns with too many missing values (never drop protected ones)."],
            ["impute_cols",            "Fill missing values (numbers -> mean/median/mode; text -> mode)."],
            ["handle_outliers",        "Detect unusual values (too high/low). Options to flag or remove them."],
            ["plots",                  "How to request charts inside clean_df."],
            ["plot_hist / plot_box",   "Basic plots for one or more numeric columns."],
            ["plot_scatter / plot_line","Relationship plots between two numeric columns."],
            ["plot_heatmap",           "Correlation between numeric columns."],
            ["plot_bar_counts",        "Bar chart of most frequent categories in a column."],
            ["generate_report",        "Collects dataset info, missing values, stats, outliers, and charts."],
            ["write_report_html",      "Save everything into one standalone HTML report."],
            ["recheck_types", "Re-runs type inference after imputing & casting."],
        ]
        try:
            display_tabular_output(["function", "description"], rows)
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
        "run": "clean_df", "pipeline": "clean_df",
        "summary": "summarize_columns", "describe": "summarize_columns",
        "read": "read_any", "load": "read_any",
        "missing_report": "report_missing_values",
        "drop_missing_cols": "drop_high_missing_cols_protected",
        "impute": "impute_cols",
        "viz": "plots", "plotspecs": "plots", "charts": "plots",
        "outlier": "outliers", "outlier_help": "outliers",
        "report": "report_help", "html": "report_help",
    }
    key = alias_map.get(key, key)

    if key == "clean_df":
        doc = """clean_df(
        filepath,
        format=None,
        reader_kwargs=None,
        infer_types=True,
        drop_threshold=0.5,
        impute_strategy='mean',
        display=True,
        report_missing=False,
        display_kwargs=None,
        summary=False,
        summary_percentiles=[25, 50, 75],
        # V3 extras
        coerce_numeric_strings=True,
        protect_identifiers=True,
        protected_name_patterns=("zip","postal","pincode","pin","ssn","nid","uin","id","code","phone","date"),
        outlier_method=None,          # 'iqr' or 'zscore'
        outlier_action='flag',        # 'flag' or 'remove'
        outlier_cols=None,            # list of column names or None (means all numeric)
        iqr_k=1.5,                    # IQR multiplier
        zscore_threshold=3.0,         # Z-score threshold
        visualize=False,
        plots=None,                   # e.g. [('hist','salary'), ('scatter','age','gpa'), ('box',['gpa','salary']), ('heatmap', None)]
        export_html_path=None,
        print_outlier_summary=True,
        recheck_types=True            # re-runs type inference after imputation/casting (skips protected)
    )

    What it does (in order):
    1) Reads your file (CSV or JSON).
    2) Fixes column names (fills blanks and removes duplicates).
    3) Drops columns with too many missing values (keeps ID-like columns safe).
    4) Optionally converts number-looking text (like "$1,200" or "(123)") into real numbers.
    5) Detects column types (number, text, date, True/False).
    6) Fills missing values (mean/median/mode for numbers; mode for others).
    7) Casts rows to inferred types (protects identifiers).
    8) (Optional) Re-runs type inference after imputation/casting (recheck_types).
    9) (Optional) Detects outliers (IQR or Z-score) and either flags or removes them.
    10) (Optional) Shows a preview table and summary stats.
    11) (Optional) Makes charts and saves a full HTML report.

    Returns:
    (header, data, types)

    Key options you may want to change:
    - drop_threshold: 0.5 means drop a column if 50% or more values are missing.
    - impute_strategy: 'mean' | 'median' | 'mode' (used for numeric columns).
    - coerce_numeric_strings: True converts values like '$1,000' -> 1000.0 safely.
    - protect_identifiers: True prevents ID/ZIP/phone-like columns from being altered.
    - recheck_types: True re-infers column types after imputation/casting (skips protected).
    - outlier_method: 'iqr' or 'zscore' to find outliers; set to None to skip.
    - outlier_action: 'flag' to add extra True/False columns, or 'remove' to drop rows.
    - plots: list of plot specs, e.g. [('hist','gpa'), ('scatter','age','salary'), ('heatmap', None)].
    - export_html_path: set to a filename (e.g. 'eda_report.html') to save a report.

    Tips:
    - Set summary=True to print summary statistics for each column.
    - Set visualize=True and provide plots=[...] to generate charts.
    - Use reader_kwargs like {'skiprows': 1, 'nrows': 100} to read a slice.
    - If you see IDs or ZIP codes getting changed, keep protect_identifiers=True.
    - If numeric columns look wrong after imputation, set recheck_types=True.
    """
        print(doc)
        return


    if key == "plots":
        print(
            "Charts inside clean_df:\n"
            "  plots=[('hist','age'), ('scatter','age','fare'), ('box',['age','fare']), ('heatmap',None)]\n\n"
            "Available chart types:\n"
            "  - 'hist' -> histogram (distribution of one numeric column)\n"
            "  - 'box' -> box plot (spread and outliers for numeric columns)\n"
            "  - 'scatter' -> scatter plot (two numeric columns)\n"
            "  - 'line' -> line chart (two numeric columns)\n"
            "  - 'heatmap' -> correlation matrix of numeric columns\n"
            "  - 'bar_counts' -> frequency of categories in a text column"
        )
        return

    if key == "outliers":
        print(
            "Outliers = unusual values that don’t fit the rest.\n"
            "Options in clean_df:\n"
            "  outlier_method='iqr' -> Interquartile Range\n"
            "  outlier_method='zscore' -> Z-score\n"
            "  outlier_action='flag' (mark them) or 'remove' (drop rows)\n"
            "  outlier_cols=['col1','col2'] -> limit to specific columns"
        )
        return

    if key == "report_help":
        print(
            "HTML Report:\n"
            "Set export_html_path='eda_report.html' in clean_df.\n"
            "The report includes:\n"
            "  - Missing values (before and after filling)\n"
            "  - Summary statistics for each column\n"
            "  - Outlier summary (if enabled)\n"
            "  - All requested plots"
        )
        return

    print(f"No help available for '{name}'. Try help() with no args to see the list.")


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
"""
# Basic, safe run
clean_df("test_df.csv", format=None, reader_kwargs=None, drop_threshold=0.5, impute_strategy="mean", display = True, 
             report_missing=False, display_kwargs=None, visualize=True, summary=True)

# With coercion + outlier flagging + visuals + HTML
header, data, types = clean_df(
    "test_df.csv",
    coerce_numeric_strings=True,
    protect_identifiers=True,
    outlier_method="iqr",
    outlier_action="flag",
    visualize=True,
   plots=[
    ("bar_counts", "city"),                          
    ("bar_agg", "city", "salary", "mean", 10),       
    ("line", "date_of_birth", "gpa"),
    ("hist","salary"),                               
    ("heatmap", None),],
    summary=True,
    export_html_path="eda_report.html"
)"""


clean_df(
    "test_df.csv",
    visualize=True,
    plots=[
        ("bar_counts", "city"),
        ("bar_agg", "city", "salary", "mean", 10),
        ("hist", "salary"),
        ("line", "date_of_birth", "gpa"),
        ("heatmap", None),
    ],
    summary=True,
    export_html_path="eda_report.html"
)