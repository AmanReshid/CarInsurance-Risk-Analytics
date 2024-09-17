import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, ttest_ind
import os

def generate_insurance_data(num_records=1000):
    """Generate random insurance data with provinces and gender, ensuring significant differences."""
    np.random.seed(42)  # For reproducibility
    provinces = ['Province_A', 'Province_B', 'Province_C']
    genders = ['Male', 'Female']
    zip_codes = ['12345', '67890', '11223'] 

    # Define claim rates for each province to create a significant difference
    claim_rates_province = {
        'Province_A': 0.1,  # 10% claim rate
        'Province_B': 0.9,   # 90% claim rate
        'Province_C': 0.2   # 20% claim rate
    }
    
    # Define claim rates for gender
    claim_rates_gender = {
        'Male': 0.6,   # 60% claim rate for males
        'Female': 0.2  # 20% claim rate for females
    }
    
    # Generate province and gender
    province_choices = np.random.choice(provinces, num_records)
    gender_choices = np.random.choice(genders, num_records)
    zip_code_choices = np.random.choice(zip_codes, num_records) 

    # Generate claims based on gender
    claims = [1 if np.random.rand() < claim_rates_gender[gender] else 0 for gender in gender_choices]

   # Generate random margins (profits) with a normal distribution
    margins = np.random.normal(500, 50, num_records)  # Example margin values around 500 with a standard deviation of 50
     
    data = {
        'Province': province_choices,
        'Gender': gender_choices,
        'ZipCode': zip_code_choices,
        'Claimed': claims,
        'Margin': margins
    }
    
    return pd.DataFrame(data)


def save_data_to_csv(data, filename='insurance_data.csv'):
    
    """Save the generated data to a CSV file in the 'data' folder one level up."""
    # Create the path for the 'data' folder one level up
    data_folder = os.path.abspath(os.path.join(os.getcwd(), '..', 'data'))
    os.makedirs(data_folder, exist_ok=True)  # Create 'data' folder if it doesn't exist
    data.to_csv(os.path.join(data_folder, filename), index=False)

def ab_test(data):
    """Perform A/B testing using Chi-squared test for provinces."""
    contingency_table = pd.crosstab(data['Province'], data['Claimed'])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    return chi2, p, contingency_table

def ab_test_zip_code(data):
    """Perform A/B testing using Chi-squared test for zip codes."""
    contingency_table_zip = pd.crosstab(data['ZipCode'], data['Claimed'])
    chi2_zip, p_zip, dof_zip, expected_zip = chi2_contingency(contingency_table_zip)
    return chi2_zip, p_zip, contingency_table_zip

def margin_analysis_zip_codes(data):
    """Perform t-test analysis based on margins (profits) between zip codes."""
    zip_code_1 = data[data['ZipCode'] == '12345']['Margin']
    zip_code_2 = data[data['ZipCode'] == '67890']['Margin']
    
    t_stat_zip, p_value_zip = ttest_ind(zip_code_1, zip_code_2, equal_var=False)  # Two-sample t-test
    return t_stat_zip, p_value_zip

def gender_analysis(data):
    """Perform t-test analysis based on gender."""
    male_claims = data[data['Gender'] == 'Male']['Claimed']
    female_claims = data[data['Gender'] == 'Female']['Claimed']
    
    t_stat, p_value = ttest_ind(male_claims, female_claims, equal_var=False)
    return t_stat, p_value

def save_insurance_data_to_csv(text_file_path, csv_file_name='insurance_text_data.csv'):
    """Load insurance data from a text file and save it as a CSV."""
    # Read the text file into a DataFrame
    df = pd.read_csv(text_file_path, delimiter='|')
    
    # Create the path for the 'data' folder one level up
    data_folder = os.path.abspath(os.path.join(os.getcwd(), '..', 'data'))
    os.makedirs(data_folder, exist_ok=True)  # Create 'data' folder if it doesn't exist

    # Define the full path for the CSV file
    csv_file_path = os.path.join(data_folder, csv_file_name)
    
    # Save the DataFrame to a CSV file
    df.to_csv(csv_file_path, index=False)

    print(f"Data saved to {csv_file_path}")

# Example usage:
if __name__ == "__main__":
    # Generate the data
    insurance_data = generate_insurance_data()

    # Save the generated data
    save_data_to_csv(insurance_data)

    # Run the A/B test for provinces
    chi2_stat, p_value, contingency_table = ab_test(insurance_data)
    print("Chi-squared Statistic:", chi2_stat)
    print("P-value (Provinces):", p_value)
    print("Contingency Table (Provinces):\n", contingency_table)

    # Run gender analysis
    t_stat, gender_p_value = gender_analysis(insurance_data)
    print("T-statistic (Gender):", t_stat)
    print("P-value (Gender):", gender_p_value)
