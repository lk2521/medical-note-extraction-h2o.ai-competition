import json
import ast
import pandas as pd

# ----------------------------
# Load model outputs (parts)
# ----------------------------

# Four parts generated from the extraction pipeline
df1 = pd.read_csv("final_output_fewshot_part1.csv")
df2 = pd.read_csv("final_output_fewshot_part2.csv")
df3 = pd.read_csv("final_output_fewshot_part3.csv")
df4 = pd.read_csv("final_output_fewshot_part4.csv")

# Remaining outputs that need postprocessing/corrections
# df_rem = pd.read_csv("final_output_fewshot_remaining.csv")
# df_rem2 = pd.read_csv("final_output_fewshot_remaining_more.csv")

# Optional inspection (kept same as notebook display call)
# df_rem2

# ----------------------------
# Combine primary parts
# ----------------------------
sub = pd.concat([df1, df2, df3, df4], axis=0).reset_index(drop=True)

# Ensure the json column is string-typed before validation/transforms
sub["json"] = sub["json"].astype(str)

# ----------------------------
# JSON validation helper
# ----------------------------
def is_valid_json_dict(x):
    """
    Check if input is a JSON dict (either already a dict or a string that parses to dict).
    """
    if isinstance(x, dict):
        return True
    if not isinstance(x, str):
        return False
    try:
        parsed = json.loads(x)
        return isinstance(parsed, dict)
    except Exception:
        return False

# Validate JSON for combined results
sub["is_valid_dict"] = sub["json"].apply(is_valid_json_dict)

# Filter invalid rows for review (mirrors notebook inspection)
rows = sub[~sub["is_valid_dict"]]

# Validate JSON for the 'remaining' CSVs as well
# df_rem["is_valid_dict"] = df_rem["json"].apply(is_valid_json_dict)
# rows_rem = df_rem[~df_rem["is_valid_dict"]]

# df_rem2["is_valid_dict"] = df_rem2["json"].apply(is_valid_json_dict)
# rows_rem2 = df_rem2[~df_rem2["is_valid_dict"]]

# ----------------------------
# Manual JSON extraction from full_response (when needed)
# ----------------------------
def output_extract_text(text: str) -> str:
    """
    Extract the JSON string by splitting on the 'Assistant:' delimiter.
    Uses the same index as the original pipeline for consistency.
    """
    try:
        return text.split("Assistant:")[6].strip()
    except Exception:
        # If extraction fails, return original text for later checks
        return text

# Try to recover JSON from full_response where df_rem2 json is invalid
# df_rem2["json_manual"] = df_rem2["full_response"].apply(output_extract_text)
# df_rem2["is_valid_dict"] = df_rem2["json_manual"].apply(is_valid_json_dict)

# Show invalid rows after manual extraction (mirrors notebook inspection)
# invalid_rows_rem2 = df_rem2[~df_rem2["is_valid_dict"]]

# ----------------------------
# Update main 'sub' JSON with corrections from df_rem and df_rem2
# ----------------------------

# Update overlaps using df_rem row indices (as in notebook)
# Note: This assumes df_rem rows align with indices to patch in 'sub'
# update_indices = df_rem.index
# sub.loc[update_indices, "json"] = df_rem.loc[update_indices, "json"]

# # Update by ID using df_rem2 manual extraction where available
# df_rem2_map = df_rem2.set_index("ID")["json_manual"]
# mask = sub["ID"].isin(df_rem2_map.index)
# sub.loc[mask, "json"] = sub.loc[mask, "ID"].map(df_rem2_map)

# Optional checks preserved from notebook
# _ = sub.isna().sum()
# sub["cleaned_json"] = sub["json"].astype(str)

# ----------------------------
# Remove nulls from nested JSON objects
# ----------------------------
def remove_nulls(d):
    """
    Recursively remove keys with None values from dicts and lists.
    """
    if isinstance(d, dict):
        return {k: remove_nulls(v) for k, v in d.items() if v is not None}
    if isinstance(d, list):
        return [remove_nulls(v) for v in d if v is not None]
    return d

def clean_json_string(json_string: str):
    """
    Parse, remove nulls, and re-serialize; return None on failure.
    """
    try:
        data = json.loads(json_string)
        cleaned = remove_nulls(data)
        return json.dumps(cleaned, ensure_ascii=False)
    except Exception:
        return None

# Apply cleaning
sub["cleaned_json"] = sub["json"].apply(clean_json_string)

# Replace original 'json' with the cleaned version
sub = sub.drop(columns=["json"])
sub.rename(columns={"cleaned_json": "json"}, inplace=True)

# Optional inspection of empties (preserved)
# sub[sub["json"].isna()]

# ----------------------------
# Symptom normalization/deduplication
# ----------------------------

train = pd.read_csv('data/train.csv')

symptoms = set()
for i, row in sub.iterrows():
    #j = row['json'].strip("'\\'")
    try:
        js = json.loads(row['json'])
        if 'symptoms' in js:
            for sym in js['symptoms']:
                symptoms.add(sym)
    except Exception as e:
        print(i)
        print(e, "="*100)

symptoms_og = set()
for i in range(len(train)):
    for sym in ast.literal_eval(train['json'][i])['symptoms']:
        symptoms_og.add(sym)

sym_mapping = {
    'abdominal_pain': 'abdominal_pain',
    'anxiety': 'anxiety',
    'blurred_vision': 'blurred_vision',
    #'body_aches': 'joint_pain',  # Mapping general aches to the closest pain category     ----------
    'chest_pain': 'chest_pain',
    #'chills': 'fever',  # Chills are a common symptom accompanying fever      ------------------
    'chronic_cough': 'cough',  # Specific type of cough
    'congestion': 'runny_nose',  # Nasal congestion is closely related
    'cough': 'cough',
    #'decreased_appetite': 'nausea',  # Related GI symptom                      ---------------
    'diarrhea': 'diarrhea',
    'difficulty_breathing': 'difficulty_breathing',
    'difficulty_concentrating': 'difficulty_concentrating',
    'difficulty_swallowing': 'sore_throat',  # Often caused by a sore throat
    'dizziness': 'dizziness',
    'dry_skin': 'dry_skin',
    'ear_pain': 'ear_pain',
    'eczema': 'rash',  # Eczema is a specific type of rash
    'facial_pain': 'facial_pain',
    'fatigue': 'fatigue',
    'fever': 'fever',
    'frequent_urination': 'frequent_urination',
    'headache': 'headache',
    'headaches': 'headache',  # Plural
    'heartburn': 'heartburn',
    'increased_thirst': 'increased_thirst',
    'increased_urination': 'frequent_urination',  # Closely related urinary symptom
    'itching': 'rash',  # Itching is the primary symptom of many rashes
    'itchy_eyes': 'itchy_eyes',
    'itchy_skin': 'rash',  # Symptom associated with rash or dry skin
    'joint_pain': 'joint_pain',
    #'loss_of_appetite': 'nausea',  # Related GI symptom
    #'loss_of_interest_in_activities': 'sadness',  # A key component of sadness/depression
    'loss_of_taste_smell': 'loss_of_taste_smell',
    'lower_abdominal_pain': 'abdominal_pain',  # Specific type of abdominal pain
    'nasal_congestion': 'runny_nose',  # Both are symptoms of rhinitis
    'nausea': 'nausea',
    'night_sweats': 'night_sweats',
    #'pain': 'joint_pain',  # Mapping general pain to the closest available pain category      ---------------
    'painful_urination': 'painful_urination',
    'pale_skin': 'pale_skin',
    'paleness': 'pale_skin',  # Synonym
    #'palpitations': 'anxiety',  # Palpitations are a common physical symptom of anxiety        ----------------
    'persistent_cough': 'cough',  # Specific type of cough
    'rash': 'rash',
    'regurgitation': 'heartburn',  # Both are key symptoms of GERD
    'restlessness': 'restlessness',
    'runny_nose': 'runny_nose',
    'sadness': 'sadness',
    'shortness_of_breath': 'difficulty_breathing',  # Synonym
    'sneezing': 'sneezing',
    'sore_throat': 'sore_throat',
    'swollen_lymph_nodes': 'swollen_lymph_nodes',
    'throat_pain': 'sore_throat',  # Synonym
    'vomiting': 'vomiting',
    #'weakness': 'fatigue',  # Closely related symptoms           -----------------
    'weight_loss': 'weight_loss',
    'wheezing': 'wheezing'
}

def map_and_deduplicate_symptoms(json_string: str):
    """
    Map symptoms using sym_mapping and deduplicate while preserving JSON structure.
    """
    try:
        data = json.loads(json_string)
        original = data.get("symptoms", [])
        mapped = {sym_mapping[s] if s in sym_mapping else s for s in original}
        data["symptoms"] = list(mapped)
        return json.dumps(data, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        return json_string

sub["json"] = sub["json"].apply(map_and_deduplicate_symptoms)

# ----------------------------
# Visit motivation normalization
visit_motivation = set()
for i, row in sub.iterrows():
    j = row['json'].strip("'\\'")
    try:
        js = json.loads(j)
        if 'visit_motivation' in js:
            visit_motivation.add(js['visit_motivation'])
    except Exception as e:
        print(i)
        print(e, "="*100)

visit_motivation_og = set()
for i in range(len(train)):
    visit_motivation_og.add(ast.literal_eval(train['json'][i])['visit_motivation'])


vm_mapping = {
    'Acute Coronary Syndrome': 'Heart Disease (Coronary Artery Disease)',
    'Allergies': 'Allergies',
    'Anemia': 'Anemia',
    'Anxiety': 'Anxiety Disorders',
    'Anxiety Disorders': 'Anxiety Disorders',
    'Asthma': 'Asthma',
    'Asthma (Exacerbation)': 'Asthma',
    'COVID-19': 'COVID-19',
    'Chronic Obstructive Pulmonary Disease (COPD)': 'Chronic Obstructive Pulmonary Disease (COPD)',
    'Common Cold': 'Common Cold',
    'Community-Acquired Pneumonia (CAP)': 'Pneumonia',
    'Coronary Artery Disease': 'Heart Disease (Coronary Artery Disease)',
    'Coronary Artery Disease (CAD)': 'Heart Disease (Coronary Artery Disease)',
    'Depression': 'Depression',
    'Diabetes (Type 2)': 'Diabetes (Type 2)',
    'Ear Infection (Otitis Media)': 'Ear Infection (Otitis Media)',
    'Eczema (Atopic Dermatitis)': 'Eczema (Atopic Dermatitis)',
    'Exacerbation of Asthma': 'Asthma',
    'Gastroesophageal Reflux Disease (GERD)': 'Gastroesophageal Reflux Disease (GERD)',
    'Heart Disease (Coronary Artery Disease)': 'Heart Disease (Coronary Artery Disease)',
    'Hypertension': 'Hypertension (High Blood Pressure)',
    'Hypertension (High Blood Pressure)': 'Hypertension (High Blood Pressure)',
    'Influenza': 'Influenza (Flu)',
    'Influenza (Flu)': 'Influenza (Flu)',
    'Major Depressive Disorder (MDD)': 'Depression',
    'Pneumonia': 'Pneumonia',
    'Sinusitis': 'Sinusitis',
    'Strep Throat': 'Strep Throat',
    'Tuberculosis (TB)': 'Tuberculosis (TB)',
    'Urinary Tract Infection': 'Urinary Tract Infection (UTI)',
    'Urinary Tract Infection (UTI)': 'Urinary Tract Infection (UTI)'
}

def map_visit_motivation(json_string: str):
    """
    Map visit_motivation using vm_mapping without altering other fields.
    """
    try:
        data = json.loads(json_string)
        orig = data.get("visit_motivation")
        data["visit_motivation"] = vm_mapping.get(orig, orig)
        return json.dumps(data, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        return json_string

sub["json"] = sub["json"].apply(map_visit_motivation)

# Replace literal 'None' substrings if they appear inside the JSON string
sub["json"] = sub["json"].astype(str).str.replace("None", "")

# Final validity check (mirrors notebook)
sub["is_valid_dict"] = sub["json"].apply(is_valid_json_dict)

# ----------------------------
# Save final submission file
# ----------------------------
# Keep the Kaggle working path consistent with typical submissions
sub[["ID", "json"]].to_csv("submission_llm.csv", index=False)
