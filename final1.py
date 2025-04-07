import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore
import numpy as np
import matplotlib.pyplot as plt
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import random

# Initialize Firebase
if not firebase_admin._apps:
    cred = credentials.Certificate("/Users/spatika/Downloads//pcoscare-20ed8-firebase-adminsdk-fbsvc-1282a7a649.json")
    firebase_admin.initialize_app(cred)
db = firestore.client()

def train_and_encode(df):
    """
    Trains a RandomForestClassifier on the dataset and encodes categorical data.
    Assumes that the last column is the target and that the other columns match the user data keys.
    """
    # Determine categorical columns
    cat_columns = df.select_dtypes(include=['object']).columns.tolist()

    # Split data: all columns except the last one as features, and the last column as target.
    X = df.iloc[:, :-1]
    Y = df.iloc[:, -1]

    # Create a dictionary to store LabelEncoders for each categorical column
    label_encoders = {}

    # Encode categorical features
    for col in cat_columns:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            label_encoders[col] = le

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, Y)

    # Save the feature names as expected by the model
    model_feature_names = list(X.columns)

    return model, label_encoders, cat_columns, model_feature_names

def encode_new_data(user_data, label_encoders, cat_columns, model_feature_names):
    """
    Encodes the user input data using the LabelEncoders fitted on the training dataset.
    The keys in user_data should match those in model_feature_names.
    """
    encoded_data = []
    for feature in model_feature_names:
        value = user_data[feature]
        if feature in cat_columns:
            encoded_data.append(label_encoders[feature].transform([value])[0])
        else:
            encoded_data.append(value)
    return encoded_data

@st.cache_resource
def load_model():
    df = pd.read_csv('Cleaned-Data-new.csv', encoding='latin-1')
    model, label_encoders, cat_columns, model_feature_names = train_and_encode(df)
    return model, label_encoders, cat_columns, model_feature_names

# Unpack returned objects for later use
model, label_encoders, cat_columns, model_feature_names = load_model()

# MESA Components
class PatientAgent(Agent):
    def __init__(self, unique_id, model, treatment, risk_score, user_data):  # Pass user_data
        super().__init__(unique_id, model)
        self.treatment = treatment
        self.risk_score = risk_score


           # Initialize agent attributes based on user_data (or defaults if missing)
        self.weight = user_data.get("Weight_kg", 60) + (risk_score * 40) # use user_data if exists
        self.insulin_resistance = 1.0 + (risk_score * 4.0)  # Base value + risk adjustment
        #self.menstrual_regularity = user_data.get("Menstrual_Irregularity", "Irregular") if risk_score > 0.5 else "Regular"
        if user_data and "Menstrual_Irregularity" in user_data:
            self.menstrual_regularity = (
                "Irregular" if user_data["Menstrual_Irregularity"] in ["Yes", "Irregular"]
                else "Regular"
            )
        self.hirsutism = self.initialize_hirsutism(risk_score)
        self.pcos_medication = user_data.get("PCOS_Medication", "No")
        self.conception_difficulty = user_data.get("Conception_Difficulty", "No")
        self.mental_health = user_data.get("Mental_Health", "No")
        self.exercise = user_data.get("Exercise", "No")
        self.fast_food = user_data.get("Fast_Food", "No")

    def initialize_hirsutism(self, risk_score):
        if risk_score > 0.75:
            return "Severe"
        elif risk_score > 0.5:
            return "Moderate"
        elif risk_score > 0.25:
            return "Mild"
        else:
            return "None"

    def step(self):
        self.update_weight()
        self.update_insulin_resistance()
        self.update_menstrual_regularity()  
        self.update_hirsutism()  


    def update_weight(self):  # Example update, incorporate new aspects as needed
        if self.treatment == "Exercise" and self.exercise == "Yes":
            self.weight -= random.uniform(0.3, 0.7) * (1 - self.risk_score)
        elif self.treatment == "Diet" and self.fast_food == "No":
            self.weight -= random.uniform(0.4, 0.9) * (1 - self.risk_score)
        elif self.treatment == "Medication" and self.pcos_medication == "Yes": # Medication might indirectly affect weight
            self.weight -= random.uniform(0.1, 0.4) * (1 - self.risk_score) # Example (could be weight gain for some!)
        elif self.fast_food == "Yes":
            self.weight += random.uniform(0.2, 0.5) * self.risk_score
        # Incorporate other factors:
        if self.mental_health == "Yes": #Example: mental health could impact adherence or appetite
            self.weight += random.uniform(0.1, 0.3) # Possible weight change due to stress, etc. (needs research)

        # ... (Add logic for other treatments and interactions as needed)

        self.weight = max(40, self.weight) # Min weight
        self.weight = min(200, self.weight) # Max Weight



    def update_insulin_resistance(self):  # Example update, incorporate new aspects
        if self.treatment == "Medication" and self.pcos_medication == "Yes":
            self.insulin_resistance -= random.uniform(0.2, 0.5) * (1 - self.risk_score)
        elif self.treatment == "Exercise" and self.exercise == "Yes":
            self.insulin_resistance -= random.uniform(0.1, 0.35) * (1 - self.risk_score)
        elif self.treatment == "Diet" and self.fast_food == "No":
            self.insulin_resistance -= random.uniform(0.15, 0.4) * (1 - self.risk_score)
        elif self.fast_food == "Yes":  # Fast food can worsen insulin resistance
            self.insulin_resistance += random.uniform(0.1, 0.3) * self.risk_score
        # ... (Add logic for other treatments and interactions)

    def update_menstrual_regularity(self):
        if self.treatment == "Medication":
            # Medication is most effective
            improvement_prob = 0.7 * (1 - self.risk_score)
            if random.random() < improvement_prob:
                self.menstrual_regularity = "Regular"
        elif self.treatment == "Exercise":
            # Exercise has moderate effectiveness
            improvement_prob = 0.5 * (1 - self.risk_score)
            if random.random() < improvement_prob:
                self.menstrual_regularity = "Regular"
        elif self.treatment == "Diet":
            # Diet has mild effectiveness
            improvement_prob = 0.4 * (1 - self.risk_score)
            if random.random() < improvement_prob:
                self.menstrual_regularity = "Regular"  
        elif self.treatment == "None":
            # Without treatment, may worsen
            if random.random() < 0.3 * self.risk_score:
                self.menstrual_regularity = "Irregular"



    def update_hirsutism(self):

        if self.treatment == "Medication":
            # Example: Medication has a base effectiveness, but diminishing returns
            medication_effectiveness = 0.6 # Placeholder, needs research
            if self.hirsutism == "Severe":
                improvement_prob = medication_effectiveness * (1 - self.risk_score)

            elif self.hirsutism == "Moderate": #Smaller effect if already moderate
                 improvement_prob = medication_effectiveness * 0.7 * (1 - self.risk_score) #Example


            elif self.hirsutism == "Mild":  #Even smaller effect if mild.
                improvement_prob = medication_effectiveness * 0.3 * (1 - self.risk_score)

            else: # Already None, very small chance to worsen

                improvement_prob = 0.95


            if random.random() < improvement_prob:
                if self.hirsutism == "Severe":
                    self.hirsutism = "Moderate"
                elif self.hirsutism == "Moderate":
                    self.hirsutism = "Mild"
                elif self.hirsutism == "Mild":
                    self.hirsutism = "None"

        elif self.treatment == "Diet":

             # Similar logic as medication, but possibly different effectiveness
            diet_effectiveness = 0.4  # Placeholder
            if self.hirsutism == "Severe":
                 improvement_prob = diet_effectiveness * (1 - self.risk_score)
            elif self.hirsutism == "Moderate":
                 improvement_prob = diet_effectiveness * 0.7 * (1 - self.risk_score)
            elif self.hirsutism == "Mild":
                 improvement_prob = diet_effectiveness * 0.3 * (1 - self.risk_score)
            else:
                 improvement_prob = 0.95
            if random.random() < improvement_prob:  # Placeholder value
                 if self.hirsutism == "Severe":
                    self.hirsutism = "Moderate"
                 elif self.hirsutism == "Moderate":
                    self.hirsutism = "Mild"
                 elif self.hirsutism == "Mild":
                    self.hirsutism = "None"


class PCOSModel(Model):
    def __init__(self, num_patients, risk_score):
        super().__init__()
        self.schedule = RandomActivation(self)
        self.datacollector = DataCollector(
            agent_reporters={
                "Weight": "weight",
                "Insulin Resistance": "insulin_resistance",
                "Treatment": "treatment",
                "Menstrual Regularity": "menstrual_regularity", 
                "Hirsutism": "hirsutism" 
            }
        )
        # Create one patient for each treatment
        treatments = ["Medication", "Exercise", "Diet", "None"]
        for i, treatment in enumerate(treatments):
            patient = PatientAgent(i, self, treatment, risk_score, user_data)
            self.schedule.add(patient)

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()

# Streamlit UI with simplified questionnaire
st.title("PCOS Risk Assessment & Treatment Simulation")

with st.form("user_form"):
    st.header("Personal Information")
    name = st.text_input("Name")
    age = st.selectbox("Age", ["Less than 20", "20-25", "25-30", "30-35", "35-44", "45 and above"])
    weight = st.number_input("Weight (kg)", 30, 200)
    height = st.number_input("Height (cm)", 100, 250)

    st.header("Key Health Indicators")
    family_history_pcos = st.selectbox("Family History of PCOS?", ["No", "Yes"])
    menstrual_irregularity = st.selectbox("Menstrual Irregularity?", ["No", "Yes"])
    hormonal_imbalance = st.selectbox("Hormonal Imbalance?", ["No", "Yes"])
    hyperandrogenism = st.selectbox("Hyperandrogenism?", ["No", "Yes"])
    hirsutism = st.selectbox("Hirsutism?", ["No", "Yes"])
    insulin_resistance = st.selectbox("Insulin Resistance?", ["No", "Yes"])

    st.header("Lifestyle and Treatments")
    pcos_medication = st.selectbox("Are you currently taking PCOS medication?", ["No", "Yes"])
    conception_difficulty = st.selectbox("Have you experienced difficulty conceiving?", ["No", "Yes"])
    mental_health = st.selectbox("Have you experienced mental health challenges related to PCOS?", ["No", "Yes"])
    exercise = st.selectbox("Do you exercise regularly?", ["No", "Yes"])
    fast_food = st.selectbox("Do you consume fast food frequently?", ["No", "Yes"])

    submitted = st.form_submit_button("Submit & Simulate")

if submitted:
    # Prepare user data based on the simplified inputs.
    user_data = {
        "Age": age,
        "Weight_kg": weight,
        "Height_cm": height,
        "Family_History_PCOS": family_history_pcos,
        "Menstrual_Irregularity": menstrual_irregularity,
        "Hormonal_Imbalance": hormonal_imbalance,
        "Hyperandrogenism": hyperandrogenism,
        "Hirsutism": hirsutism,
        "Insulin_Resistance": insulin_resistance,
        "PCOS_Medication": pcos_medication,
        "Conception_Difficulty": conception_difficulty,
        "Mental_Health": mental_health,
        "Exercise": exercise,
        "Fast_Food": fast_food
    }
    # Store user response in Firebase
    db.collection("user_responses").document(name).set(user_data)

    # Encode user data to match model features
    encoded_data = encode_new_data(user_data, label_encoders, cat_columns, model_feature_names)

    # Predict risk using the trained model
    risk_score = model.predict_proba([encoded_data])[0][1]
    st.success(f"Predicted PCOS Risk: {risk_score*100:.1f}%")

    # Run simulation with all treatments
    num_patients = 1
    sim_model = PCOSModel(num_patients, risk_score)
    for _ in range(10):  # Simulation steps
        sim_model.step()

    results = sim_model.datacollector.get_agent_vars_dataframe()

    # Plotting for all treatments
    st.subheader("Simulation Results - Treatment Comparison")
    
    # Create tabs for different metrics
    tab1, tab2, tab3 , tab4= st.tabs(["Weight Trends", "Insulin Resistance Trends", "Hirsutism Trends", "Menstrual Regularity"])
    
    with tab1:
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        for treatment in ["Medication", "Exercise", "Diet", "None"]:
            agent_data = results[results["Treatment"] == treatment]
            ax1.plot(agent_data.index.get_level_values('Step'), 
                    agent_data["Weight"], 
                    label=treatment)
        ax1.set_title("Weight Trend by Treatment")
        ax1.set_xlabel("Time Step")
        ax1.set_ylabel("Weight (kg)")
        ax1.legend()
        st.pyplot(fig1)
    
    with tab2:
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        for treatment in ["Medication", "Exercise", "Diet", "None"]:
            agent_data = results[results["Treatment"] == treatment]
            ax2.plot(agent_data.index.get_level_values('Step'), 
                    agent_data["Insulin Resistance"], 
                    label=treatment)
        ax2.set_title("Insulin Resistance Trend by Treatment")
        ax2.set_xlabel("Time Step")
        ax2.set_ylabel("Insulin Resistance")
        ax2.legend()
        st.pyplot(fig2)
    
    with tab3:
        # Convert hirsutism to numerical values for plotting
        hirsutism_map = {"None": 0, "Mild": 1, "Moderate": 2, "Severe": 3}
        results["Hirsutism_Score"] = results["Hirsutism"].map(hirsutism_map)
        
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        for treatment in ["Medication", "Exercise", "Diet", "None"]:
            agent_data = results[results["Treatment"] == treatment]
            ax3.plot(agent_data.index.get_level_values('Step'), 
                    agent_data["Hirsutism_Score"], 
                    label=treatment)
        
        ax3.set_title("Hirsutism Severity Trend by Treatment")
        ax3.set_xlabel("Time Step")
        ax3.set_ylabel("Hirsutism Severity (0=None, 3=Severe)")
        ax3.set_yticks([0, 1, 2, 3])
        ax3.set_yticklabels(["None", "Mild", "Moderate", "Severe"])
        ax3.legend()
        st.pyplot(fig3)

    with tab4:
        # Convert to numerical values for plotting
        regularity_map = {"Regular": 1, "Irregular": 0}
        results["Regularity_Score"] = results["Menstrual Regularity"].map(regularity_map)
        
        fig4, ax4 = plt.subplots(figsize=(10, 5))
        for treatment in ["Medication", "Exercise", "Diet", "None"]:
            agent_data = results[results["Treatment"] == treatment]
            ax4.plot(agent_data.index.get_level_values('Step'),
                    agent_data["Regularity_Score"],
                    label=treatment)
        
        ax4.set_title("Menstrual Regularity Improvement")
        ax4.set_xlabel("Time Step")
        ax4.set_ylabel("Regularity (0=Irregular, 1=Regular)")
        ax4.set_yticks([0, 1])
        ax4.set_yticklabels(["Irregular", "Regular"])
        ax4.legend()
        st.pyplot(fig4)

    # Display final metrics table
    st.subheader("Final Health Metrics After Treatment")
    final_step = results.index.get_level_values('Step').max()
    final_results = results.xs(final_step, level='Step')
    
    # Format for readability
    st.dataframe(
        final_results[["Treatment", "Weight", "Insulin Resistance", "Hirsutism", "Menstrual Regularity"]]
        .sort_values("Treatment")
        .reset_index(drop=True)
        .style.format({
            "Weight": "{:.1f} kg",
            "Insulin Resistance": "{:.2f}"
        }),
        height=180
    )
    