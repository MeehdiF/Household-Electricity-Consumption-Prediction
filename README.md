# Household Electricity Consumption Prediction

Build a baseline model for predicting Global Active Power from household electricity measurements.

## Why this project

This repository is part of my practical machine-learning portfolio. It focuses on a complete, understandable workflow rather than claiming production readiness.

## Dataset

The repository includes `household_power_consumption.zip`. The README identifies the UCI Household Electric Power Consumption dataset; retain the original attribution and license.

## Approach

Data cleaning, numeric conversion, an 80/20 split, and Linear Regression.

### Features

Global Reactive Power, Voltage, Global Intensity, and sub-metering values. Date and time are currently excluded from the model.

## Evaluation and current result

Residual Sum of Squares and explained variance are calculated in the notebook. Add a clearly named test-set score and units to this README after validating the split.

## Run locally

```bash
git clone https://github.com/MeehdiF/Household-Electricity-Consumption-Prediction.git
cd Household-Electricity-Consumption-Prediction
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python untitled.py
```

For notebook exploration, open the `.ipynb` file with Jupyter after installing the same dependencies.

## Limitations and next steps

Removing date and time prevents the model from learning temporal structure. Replacing missing values with zero may distort measurements; future work should use time-aware validation and compare lag-based or dedicated time-series models.

## Repository structure

- `README.md` — project context and reproducibility notes
- `requirements.txt` — Python dependencies used by the scripts
- `.ipynb` / `.py` files — analysis and model experiments

## License

See [`LICENSE`](LICENSE). Check the dataset's own terms separately; repository code licensing does not automatically license bundled data.
