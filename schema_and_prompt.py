import ast
import pandas as pd
from typing import Annotated, Literal, Optional, List

from pydantic import BaseModel, Field, ConfigDict
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from config import TRAIN_CSV

# Read training data to derive allowed literal values for visit_motivation and symptoms.
train = pd.read_csv(TRAIN_CSV)

visit_motivation = set()
for i in range(len(train)):
    visit_motivation.add(ast.literal_eval(train["json"][i])["visit_motivation"])

symptoms = set()
for i in range(len(train)):
    for sym in ast.literal_eval(train["json"][i])["symptoms"]:
        symptoms.add(sym)

# ----------------------------
# Pydantic schema definitions
# ----------------------------

class PatientInfo(BaseModel):
    age: Annotated[int, Field(gt=0, description="Age of patient in medical note (must be greater than 0)")]
    gender: Annotated[
        Literal["Male", "Female"],
        Field(description="Gender of patient in medical note")
    ]

class CholesterolLevel(BaseModel):
    value: Annotated[float, Field(descriptiom="Cholesterol Level of patient in medical note")]
    unit: Annotated[str, Field(description="Unit of cholesterol level - mg/dL", default="mg/dL")]

class GlucoseLevel(BaseModel):
    value: Annotated[float, Field(descriptiom="Glucose Level of patient in medical note")]
    unit: Annotated[str, Field(description="Unit of glucose level - mg/dL", default="mg/dL")]

class Temperature(BaseModel):
    value: Annotated[float, Field(descriptiom="Body temperature of patient in medical note")]
    unit: Annotated[str, Field(description="Unit of temperature - °C", default="°C")]

class OxygenSaturation(BaseModel):
    value: Annotated[float, Field(descriptiom="Oxygen Saturation (in %) of patient in medical note")]
    unit: Annotated[str, Field(description="Unit of Oxygen Saturation - %", default="%")]

class SystolicBP(BaseModel):
    value: Annotated[int, Field(descriptiom="Systolic Blood Pressure of patient in medical note")]
    unit: Annotated[str, Field(description="Unit of systolic blood pressure - mmHg", default="mmHg")]
    # model_config = ConfigDict(extra='forbid')

class DiastolicBP(BaseModel):
    value: Annotated[int, Field(descriptiom="Diastolic Blood Pressure of patient in medical note")]
    unit: Annotated[str, Field(description="Unit of diastolic blood pressure - mmHg", default="mmHg")]
    # model_config = ConfigDict(extra='forbid')

class BloodPressure(BaseModel):
    systolic: Annotated[
        SystolicBP,
        Field(description="Patient Systolic Blood Pressure information extracted from the medical note", default_factory=dict)
    ]
    diastolic: Annotated[
        DiastolicBP,
        Field(description="Patient Diastolic Blood Pressure information extracted from the medical note", default_factory=dict)
    ]
    # model_config = ConfigDict(extra='forbid')

class RespiratoryRate(BaseModel):
    value: Annotated[float, Field(descriptiom="Respiratory Rate (in breaths/min) of patient in medical note")]
    unit: Annotated[str, Field(description="Unit of Respiratory Rate - breaths/min", default="breaths/min")]

class HeartRate(BaseModel):
    value: Annotated[float, Field(descriptiom="Heart Rate (in beats per minute or bpm) of patient in medical note")]
    unit: Annotated[str, Field(description="Unit of Respiratory Rate - bpm", default="bpm")]

class VitalSigns(BaseModel):
    blood_pressure: Annotated[
        Optional[BloodPressure],
        Field(description="Patient blood pressure information extracted from the medical note")
    ] = None
    cholesterol_level: Annotated[
        Optional[CholesterolLevel],
        Field(description="Patient cholesterol level information extracted from the medical note")
    ] = None
    glucose_level: Annotated[
        Optional[GlucoseLevel],
        Field(description="Patient glucose level information extracted from the medical note")
    ] = None
    heart_rate: Annotated[
        Optional[HeartRate],
        Field(description="Patient heart rate information extracted from the medical note")
    ] = None
    oxygen_saturation: Annotated[
        Optional[OxygenSaturation],
        Field(description="Patient oxygen saturation (SpO2 level) information extracted from the medical note")
    ] = None
    respiratory_rate: Annotated[
        Optional[RespiratoryRate],
        Field(description="Patient respiratory rate information extracted from the medical note")
    ] = None
    temperature: Annotated[
        Optional[Temperature],
        Field(description="Patient body temperature information extracted from the medical note")
    ] = None
    # model_config = ConfigDict(extra='forbid')

class JsonOutput(BaseModel):
    patient_info: Annotated[
        PatientInfo,
        Field(description="Patient demographic information extracted from the medical note", default_factory=dict)
    ]
    visit_motivation: Annotated[
        Literal[*visit_motivation],
        Field(description="Patient visit motivation or main motive of patient to visit extracted from the medical note", default_factory=str)
    ]
    # symptoms: Annotated[List[SymptomCategory], Field(description="List of symptoms mentioned in the medical note")]
    symptoms: Annotated[
        List[Literal[*symptoms]],
        Field(description="List of symptoms mentioned in the medical note")
    ]
    vital_signs: Annotated[
        VitalSigns,
        Field(description="Patient vital signs information extracted from the medical note", default_factory=dict)
    ]

# Parser and format instructions used by the prompt.
parser = PydanticOutputParser(pydantic_object=JsonOutput)
format_instructions = parser.get_format_instructions()

# High-quality examples text block (kept identical to original).
EXAMPLES_TEXT = """
### Example 1
Medical Note:
**Clinical Notes:**

**Patient Information:**
- Age: 25
- Gender: Male

**Visit Motivation:**
- Primary complaint of sinusitis.

**Symptoms:**
- Cough
- Fatigue
- Chest pain
- Rash
- Weight loss
- Facial pain
- Heartburn
- Anxiety

**Vital Signs:**
- Cholesterol Level: 146.8 mg/dL (within normal range)
- Glucose Level: 72.5 mg/dL (within normal range)

**Assessment:**
The patient presents with a constellation of symptoms including cough, fatigue, facial pain, and chest pain, which may suggest a complex upper respiratory issue potentially related to sinusitis. The presence of a rash and weight loss raises concern for a possible systemic condition, such as an autoimmune disorder or an infectious etiology. Heartburn may indicate gastroesophageal reflux, which can be exacerbated by anxiety and stress. 

Given the vital signs are within normal limits, the immediate concern does not seem to be metabolic in nature. However, the combination of symptoms, particularly the weight loss and rash, warrants further investigation to rule out conditions such as sarcoidosis or other inflammatory processes.

**Plan:**
- Conduct a thorough physical examination focusing on respiratory and dermatological findings.
- Consider laboratory tests including a complete blood count, inflammatory markers (e.g., ESR, CRP), and possibly imaging studies (e.g., chest X-ray) to assess for any underlying pathology.
- Refer to a specialist if autoimmune or infectious disease is suspected based on further evaluation.
- Address anxiety with counseling and consider discussing lifestyle modifications that may alleviate symptoms.

Assistant: {"patient_info": {"age": 25, "gender": "Male"}, "visit_motivation": "Sinusitis", "symptoms": ["cough", "fatigue", "chest_pain", "rash", "weight_loss", "facial_pain", "heartburn", "anxiety"], "vital_signs": {"cholesterol_level": {"value": 146.8, "unit": "mg/dL"}, "glucose_level": {"value": 72.5, "unit": "mg/dL"}}}

### Example 2
Medical Note:
**Clinical Notes**

Patient: [Unknown]

**Chief Complaint:** Anemia

**History of Present Illness:** The patient presents with a 2-week history of progressive fatigue, cough, and difficulty breathing. He reports episodes of chest pain and pale skin, which worsened over the past week. The patient also experiences night sweats, weight loss, and an increased thirst.

**Physical Examination:**

* Vital Signs:
	+ Temperature: 37.2°C (out-of-range)
	+ Heart Rate: 107 bpm (elevated)
	+ Respiratory Rate: 20 breaths/min (elevated)
	+ Glucose Level: 102.4 mg/dL (slightly elevated)
* General Appearance: The patient appears pale and fatigued.
* Lungs: Bilateral wheezes and crackles heard on auscultation.

**Differential Diagnosis:** Based on the presentation, a diagnosis of **Pneumonia with possible Tuberculosis (TB) infection** is considered. The patient's symptoms, such as cough, difficulty breathing, chest pain, and fever, are consistent with pneumonia. However, the presence of night sweats and weight loss raises suspicion for TB infection.

**Additional Considerations:**

* Age and sex: A 21-year-old male with a history of anemia may be at increased risk for certain infections.
* Anemia history: The patient's anemia may be related to chronic disease or blood loss, which can increase the risk of infection.

**Plan:**

* Order chest X-ray and sputum smear for TB testing
* Perform complete blood count (CBC) with differential to assess anemia and infection markers
* Prescribe antibiotics for pneumonia, including TB prophylaxis if positive

**Follow-up:** Schedule follow-up appointment in 1 week to review test results and adjust treatment plan as needed.

Assistant: {"patient_info": {"age": 21, "gender": "Male"}, "visit_motivation": "Anemia", "symptoms": ["fever", "cough", "fatigue", "difficulty_breathing", "chest_pain", "runny_nose", "rash", "sneezing", "increased_thirst", "night_sweats", "weight_loss", "heartburn", "ear_pain", "pale_skin"], "vital_signs": {"temperature": {"value": 37.2, "unit": "\u00b0C"}, "heart_rate": {"value": 107, "unit": "bpm"}, "respiratory_rate": {"value": 20, "unit": "breaths/min"}, "glucose_level": {"value": 102.4, "unit": "mg/dL"}}}

### Example 3
Medical Note:
Clinical Note:

Patient: 82-year-old female
Chief Complaint: Chronic Obstructive Pulmonary Disease (COPD) exacerbation

History of Present Illness:
The patient, an 82-year-old female with a known history of COPD, presents today with worsening respiratory symptoms. She reports a persistent cough, increased fatigue, and difficulty breathing. The patient also complains of chest pain and dizziness, which are concerning given her age and underlying condition. Additionally, she mentions recent episodes of sneezing, increased thirst, and heartburn.

Past Medical History:
- Chronic Obstructive Pulmonary Disease (COPD)
- Hypertension (well-controlled)
- Gastroesophageal reflux disease (GERD)

Vital Signs:
- Blood Pressure: 95/62 mmHg (low)
- Oxygen Saturation: 96.5% on room air
- Glucose Level: 98.2 mg/dL

Physical Examination:
General: Elderly female, appears fatigued and mildly distressed
HEENT: Mild nasal congestion noted
Respiratory: Decreased breath sounds bilaterally, with occasional wheezes and crackles
Cardiovascular: Regular rate and rhythm, no murmurs appreciated
Abdomen: Soft, non-tender, no distension

Assessment and Plan:
1. COPD Exacerbation:
   - Likely triggered by recent respiratory infection (evidenced by sneezing)
   - Increase frequency of inhaled bronchodilators
   - Consider short course of oral corticosteroids
   - Educate on proper inhaler technique and importance of medication adherence

2. Hypotension:
   - Blood pressure is lower than usual for this patient
   - May be contributing to dizziness and fatigue
   - Review current medications, consider adjusting antihypertensive therapy
   - Encourage adequate hydration

3. Chest Pain:
   - Likely musculoskeletal or related to COPD exacerbation
   - Perform ECG to rule out cardiac involvement
   - Monitor closely and reassess if pain worsens or changes in character

4. GERD:
   - Heartburn symptoms noted
   - Recommend elevation of head of bed and avoidance of trigger foods
   - Consider increasing proton pump inhibitor dosage if symptoms persist

5. Increased Thirst:
   - Blood glucose within normal range, ruling out diabetic cause
   - May be related to medication side effects or dehydration
   - Encourage adequate fluid intake and monitor for changes

Follow-up:
- Schedule follow-up appointment in 1 week to reassess COPD symptoms and blood pressure
- Instruct patient to return sooner if symptoms worsen or new concerns arise
- Arrange for home health nurse to assess home environment and medication compliance

Assistant: {"patient_info": {"age": 82, "gender": "Female"}, "visit_motivation": "Chronic Obstructive Pulmonary Disease (COPD)", "symptoms": ["cough", "fatigue", "difficulty_breathing", "chest_pain", "dizziness", "sneezing", "increased_thirst", "heartburn"], "vital_signs": {"blood_pressure": {"systolic": {"value": 95, "unit": "mmHg"}, "diastolic": {"value": 62, "unit": "mmHg"}}, "oxygen_saturation": {"value": 96.5, "unit": "%"}, "glucose_level": {"value": 98.2, "unit": "mg/dL"}}}
"""

# Chat prompt template kept identical to original behavior.
prompt = ChatPromptTemplate([
    (
        "system",
        """
You are a highly precise and structured medical information extractor.
Your task is to extract **only** the requested patient information from the given medical note.

STRICT RULES:
- The output MUST appear immediately after the word "Assistant:" with no extra text before or after.
- Output ONLY a valid JSON object — no explanations, no text, no markdown.
- Do NOT repeat any values or keys.
- If a field is missing, set its value to null.
- Follow the JSON schema exactly as specified in the format instructions below.
- Never include any additional commentary or notes.
- Do NOT include quotation marks around the entire JSON object.
- Your final output must be exactly this structure:
    Assistant: {{ JSON dict here }}

{format_instructions}
"""
    ),
    (
        "human",
        """
Below are high-quality examples demonstrating the EXACT output format expected.

{EXAMPLES_TEXT}

Now extract the data from the following Medical Note. Return ONLY the JSON object and nothing else.

Medical Note:
{Note}

Assistant:
"""
    ),
])
