# Brain Structural Classification Workflow

This experimental project processes volumes and centroids of various brain structures using pretrained machine learning algorithms.

## Requirements

- FreeSurfer
- Python 3.x
- Required Python packages (see requirements.txt)

## Installation

### Python and Required Packages

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Usage

To run the script, you need to provide the path to the CSV file with structural data. Example command:

```bash
python classify.py --csv_file="data/sample.csv"
```

## Command-Line Arguments

```bash
    --csv_file: Path to the CSV file with structural data.
    --use_larger: Use larger model (trained including validation set).
```

## Output

The script classifies the data into one of the groups (CONTROL, PRODROMAL, PARKINSON).

## License

This project is licensed under the MIT License.

## Data, models, and reuse

- **Original Python code** (`classify.py`): MIT License, as stated in `LICENSE`.
- **Example input** (`data/sample.csv`): contains real research-derived observations shared with the author's permission. No separate open-data license is granted here; do not assume permission to redistribute or link records to individuals. The file is an example of the expected input format, not a clinical validation dataset.
- **Saved models** (`model/*.pkl`): trained by the project author. They are distributed as research artifacts; the MIT license for the Python source should not be treated as a separate model license or as a grant of rights to any underlying training data.
- **FreeSurfer**: an external dependency governed by its own terms.

This is an experimental research workflow, not a medical diagnostic tool. Only load `.pkl` files from sources you trust, because Python pickle deserialization can execute code.
