import google.generativeai as genai
import streamlit as st

def query_gemini(question, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(question)
        return response.text
    except Exception as e:
        st.error(f"Gemini API error: {e}")
        return None
