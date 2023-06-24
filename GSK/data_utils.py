import pandas as pd
from collections import defaultdict
data = pd.read_excel("data/clinical-study-ae-taskB.xlsx")
data["BMI"] = data["weight"]/data["height"]**2

def results(age,height,weight,sex):
    height = height/100
    bmi = weight/height**2
    similar_patients = data[(data["BMI"]> bmi-1) & (data["BMI"]<bmi+1) & (data["age"] > age-5) & (data["age"] < age+5) & (data["sex"]==sex)]
    similar_found = similar_patients.shape[0]
    summary_similar = summary(similar_patients, similar_found)
    similar_control = similar_patients[similar_patients["treatment_group"] == "CONTROL"]
    control_found = similar_control.shape[0]
    summary_control = summary(similar_control, control_found)
    similar_drug = similar_patients[similar_patients["treatment_group"] == "DRUG"]
    drug_found = similar_drug.shape[0]
    summary_drug  = summary(similar_drug, drug_found)
    return  control_found, summary_control, drug_found, summary_drug

def summary(df, n):
    df = df.groupby(["response", "ae"])["participant_id"].count().reset_index().sort_values(["response", "ae"])
    df["group"] = df["response"]+df["ae"]
    df.drop(columns=["response", "ae"], inplace=True)
    dictionary = defaultdict(float, dict(zip(df["group"],  100*df["participant_id"]/n))) if n>0 else {}
    return dictionary
    


