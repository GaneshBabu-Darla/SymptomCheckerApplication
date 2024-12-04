import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

#Load Environment Variables:
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

#Initialize the Groq Model:
chatModel = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="gemma2-9b-it")

# Define the Prompt Template:
symptom_prompt_template = ChatPromptTemplate([
    ("system",
     """
     You are a medical assistant AI, trained to provide helpful insights about potential medical conditions based on user-reported symptoms. 
     Your task is to suggest possible conditions based on the symptoms described by the user. 
     Provide potential conditions with a confidence level (low, medium, high) and give actionable advice or next steps for each condition. 
     If the symptoms seem severe or urgent, recommend seeking immediate medical attention. 
     Always remind users that your suggestions are not a substitute for professional medical advice.
     """),
    ("human", "{symptoms}")
])


# Define the Parsing Logic:
parser = StrOutputParser()

def analyze_symptoms(symptoms: str):
    """Analyze symptoms and return potential conditions."""
    try:
        # Create the chain
        chain = symptom_prompt_template | chatModel | parser
        # Invoke the chain
        result = chain.invoke({"symptoms": symptoms})
        return result
    except Exception as e:
        return f"Error: {str(e)}"
    
# Set up the Streamlit app
st.title("AI-Powered Symptom Checker")
st.write("Describe your symptoms, and get potential medical insights.")

# Input text box for symptoms
user_symptoms = st.text_area("Enter your symptoms here:")

# Button to trigger the analysis
left, middle, right = st.columns(3)
if middle.button("Check Symptoms", type='primary', use_container_width=True):
    if user_symptoms.strip():
        st.write("Analyzing your symptoms...")
        # Call the backend function
        analysis_result = analyze_symptoms(user_symptoms)
        st.write("Here are the potential conditions:")
        st.write(analysis_result)
    else:
        st.warning("Please enter your symptoms before submitting.")
