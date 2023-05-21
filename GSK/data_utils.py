import pandas as pd
data = pd.read_excel("data/clinical-study-ae-taskB.xlsx")
data["BMI"] = data["weight"]/data["height"]**2

def results(age,height,weight,sex):
    height = height/100
    bmi = height/weight**2
    similar_patients = data[(data["BMI"]> bmi-1) & (data["BMI"]<bmi+1) & (data["age"] > age-5) & (data["age"] < age+5) & (data["sex"]==sex)]
    similar_control = similar_patients[similar_patients["treatment_group"] == "CONTROL"]
    control_found = similar_control.shape[0]
    similar_drug = similar_patients[similar_patients["treatment_group"] == "DRUG"]
    drug_found = similar_drug.shape[0]
    return similar_drug, drug_found, similar_control, control_found

