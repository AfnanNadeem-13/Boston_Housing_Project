# Boston Housing Price Prediction

## Project Overview
This project predicts house prices using the **Boston Housing Dataset**. It includes custom implementations of **Linear Regression and Random Forest** models.

## Dataset
- The dataset is stored in `HousingData.csv`
- Features include **CRIM, RM, AGE, TAX, LSTAT, etc.**
- Target variable: **MEDV (Median house price in $1000s)**

## Steps in the Project
1. **Data Preprocessing**
   - Checked for missing values and handled them.
   - Standardized numerical features.

2. **Model Implementation**
   - Implemented **Linear Regression and Random Forest** from scratch.

3. **Performance Evaluation**
   - Used **RMSE (Root Mean Squared Error) and R² Score** for model evaluation.

## Results
| Model            | RMSE  | R² Score |
|-----------------|--------|-----------|
| Linear Regression | **3.8670** | **0.7961** |
| Random Forest    | **3.4678** | **0.8360** |

## How to Run the Code
1. Clone the repository:
   ```bash
   git clone https://github.com/AfnanNadeem-13/Boston_Housing_Project.git
   cd Boston_Housing_Project
