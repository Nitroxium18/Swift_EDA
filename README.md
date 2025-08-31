# SwiftEDA

SwiftEDA is a lightweight Python library that automates common data cleaning tasks for EDA and ML pipelines.  
It simplifies missing-value handling, imputation, and dataset inspection into a few lines of code.
It is my first personal project & my contribution to data scientists & ML practioners

---

## Features (Phase 1)
- Custom CSV reader (`read_csv`) with:
  - Header parsing
  - Row alignment (handles uneven rows)
  - Data type inference (int, float, datetime, bool, str)
- Missing value report generator
- Drop columns with high missing values
- Impute missing values (mean, median, mode)
- Tabular display of cleaned dataset
- One-line wrapper: `clean_df()` for custom control over feature usage
---
## Example Usage
header, data, types = clean_df("Titanic-Dataset.csv", skiprows=None, nrows=None, infer_types=True, drop_threshold=0.5, impute_strategy="mean")
print("Header:", header)
print("Types:", types)
---
## Devlog & Contributing
- I am a student developer building SwiftEDA to simplify the everyday workflows of Analysts and ML practitioners by automating the most repetitive and time-consuming steps in data preparation.
- Read the full devlog for V1 and stay tuned for future updates
- PRs are welcome. For major changes, open an issue to discuss your ideas
## Installation
Clone the repo:
git clone https://github.com/YOUR_USERNAME/SwiftEDA.git
cd SwiftEDA
