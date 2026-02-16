import streamlit as st

st.title("BMI Calculator:")

# Correct syntax for using st.form as a context manager
def bmi_calc():
    st.write("BMI calculation logic goes here.")
    # Add your BMI calculation code using st components

def rate_yourself():
    with st.sidebar:
        st.title('Rate Yourself')
        languages = st.text_input('Enter programming languages you know with comma seperation', value='Python')
        # Ensure languages is always a list, even if the input is empty
        languages = [i.strip() for i in languages.split(',') if i.strip()]
        if not languages: # Handle case where input is empty after stripping
            languages = ['Python'] # Default if input is empty

    st.subheader('How would you rate your experience in the following programming languages and tools ')

    if languages: # Only show sliders if there are languages
        for language in languages:
            st.write(language)
            st.slider(f'{language}_rating', min_value=0.0, max_value=10.0, step=0.5, key=language) # Added a unique key

ch = st.sidebar.selectbox("Menu", ['BMI', 'Rate Yourself'])

if ch == 'BMI':
    bmi_calc()
elif ch == 'Rate Yourself':
    rate_yourself()