# SwiftEDA

SwiftEDA is a lightweight Python library that automates common data cleaning and exploratory data analysis (EDA) tasks for ML pipelines.
It simplifies missing-value handling, imputation, outlier detection, visualization, and dataset reporting into just a few lines of code.

This is my first personal project and my contribution to the data science and ML community.

Features:
## Phase 1 (V1)
Custom CSV reader (read_csv) with:
Header parsing
Row alignment (handles uneven rows)
Data type inference (int, float, datetime, bool, str)
Missing value report generator
Drop columns with high missing values
Impute missing values (mean, median, mode)
Tabular display of cleaned dataset
One-line wrapper: clean_df() for custom control over feature usage

## Phase 2 (V2)

Restructured wrapper: Simple wrapper that encapsulates all helpers with kwargs for modular use.
Multi-format support: Added read_json() alongside read_csv.
(Excel deliberately excluded to avoid dependency bloat — save as CSV/JSON instead).

Summary statistics upgrade:
Mean, median, mode, min, max, 25th percentile, 75th percentile, and IQR.
Limiter function: Optional row limiter (limit=n) to prevent terminal flooding with large datasets.
Wrapper help(): Callable help function that explains wrapper usage, parameters, and defaults.
Improved logging: Cleaner and more descriptive status reporting.

## Phase 3 (V3)
Outlier detection & handling:
IQR and Z-score methods.
Flexible options to flag or remove outliers.

Edge-case refinement:
Numeric coercion for strings like "$1,200" or "3.5%".
Protection for identifiers (ZIP codes, IDs, phone numbers).

Visualization helpers:
Histograms, boxplots, scatter plots, line charts.
Correlation heatmaps and category frequency plots.
Flexible kwargs for matplotlib/seaborn under the hood.

Comprehensive HTML report:
Dataset info, missing value summary (before & after).
Summary statistics, outlier summary, and selected plots.
Exportable with export_html_path="report.html".

Type re-check system:
recheck_types=True automatically re-infers column types after imputation/casting.
Skips protected identifier columns.
Logs upgrades (e.g., "Age" str → float).


SwiftEDA is developed as a learning project and personal contribution to simplify EDA workflows.
The Devlog contains detailed patch notes for each version, implementation details, and real-world test case reports.
Pull requests are welcome! For major changes, please open an issue first to discuss your ideas.

Example Usage (V3)
from Swift_EDA_V3_Final import clean_df

### Clean a dataset with outlier handling, visualization, and HTML report
header, data, types = clean_df(
    "Titanic-Dataset.csv",
    drop_threshold=0.3,
    impute_strategy="median",
    outlier_method="iqr",
    outlier_action="flag",
    visualize=True,
    plots=[("hist", "Age"), ("scatter", "Age", "Fare"), ("heatmap", None)],
    summary=True,
    export_html_path="eda_report.html"
)

print("Header:", header)
print("Types:", types)
