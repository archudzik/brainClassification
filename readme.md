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
