# SwiftEDA

SwiftEDA is a lightweight Python library that automates common data cleaning tasks for EDA and ML pipelines.  
It simplifies missing-value handling, imputation, and dataset inspection into just a few lines of code.  

This is my first personal project and my contribution to the data science and ML community.  

---

## Features

### Phase 1 (V1)
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

### Phase 2 (V2)
- **Restructured wrapper**: Simple wrapper that encapsulates all helpers with kwargs for modular use.
- **Multi-format support**: Added `read_json()` alongside `read_csv`.  
  *(Excel deliberately excluded to avoid dependency bloat — save as CSV/JSON instead).*  
- **Summary statistics upgrade**:  
  - Mean, median, mode, min, max, 25th percentile, 75th percentile, and IQR.  
- **Limiter function**: Optional row limiter (`limit=n`) to prevent terminal flooding with large datasets.  
- **Wrapper help()**: Callable help function that explains wrapper usage, parameters, and defaults.  
- **Improved logging**: Cleaner and more descriptive status reporting.  

---
SwiftEDA is developed as a learning project and personal contribution to simplify EDA workflows.

The Devlog contains detailed patch notes for each version, implementation details, and real-world test case reports.

Pull requests are welcome! For major changes, please open an issue first to discuss your ideas.
---
## Example Usage (V2)

```python
from Swift_EDA_V2 import clean_df

# Clean a JSON file with row limit and summary stats
header, data, types = clean_df("data.json",file_type="json",drop_threshold=0.4,impute_strategy="median",limit=10,summary=True)

print("Header:", header)
print("Types:", types)
