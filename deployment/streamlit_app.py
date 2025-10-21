"""
Streamlit demo application for Bengali Medical Chatbot.
"""

import sys
import streamlit as st
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.logger import setup_logger

logger = setup_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="BengaliMed Chatbot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main Streamlit application."""
    
    # Title and description
    st.title("🏥 BengaliMed: Bengali Medical Chatbot")
    st.markdown("""
    A culturally-aware medical chatbot for Bengali healthcare assistance.
    
    **⚠️ Disclaimer**: This is a research prototype. Always consult healthcare professionals for medical advice.
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        
        # Language selection
        language = st.selectbox(
            "Select Language / ভাষা নির্বাচন করুন",
            ["Bengali (বাংলা)", "English", "Mixed (মিশ্র)"]
        )
        
        # Specialization
        specialization = st.selectbox(
            "Medical Specialization",
            ["General Medicine", "Mental Health", "Pediatrics", "Women's Health"]
        )
        
        # Cultural sensitivity
        cultural_mode = st.checkbox("Cultural Sensitivity Mode", value=True)
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This chatbot is part of a research project to create 
        culturally-appropriate medical AI for Bengali speakers.
        """)
    
    # Main chat interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Chat Interface")
        
        # Chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("আপনার স্বাস্থ্য সংক্রান্ত প্রশ্ন লিখুন... / Type your health question..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response (placeholder)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # TODO: Replace with actual model inference
                    response = generate_response(prompt, language, specialization, cultural_mode)
                    st.markdown(response)
            
            # Add assistant message
            st.session_state.messages.append({"role": "assistant", "content": response})
    
    with col2:
        st.header("Quick Actions")
        
        # Common questions
        st.subheader("Common Questions")
        common_questions = [
            "জ্বর হলে কি করব?",
            "মাথাব্যথার কারণ কি?",
            "ডায়াবেটিসের লক্ষণ",
            "What are COVID symptoms?",
            "How to check blood pressure?"
        ]
        
        for question in common_questions:
            if st.button(question, key=f"q_{hash(question)}"):
                st.session_state.messages.append({"role": "user", "content": question})
                st.rerun()
        
        st.markdown("---")
        
        # Emergency contacts
        st.subheader("🚨 Emergency")
        st.error("""
        **Emergency Numbers:**
        - Bangladesh: 999
        - India: 102
        - Always call emergency services for urgent medical situations
        """)
        
        st.markdown("---")
        
        # Model info
        st.subheader("Model Information")
        st.info("""
        **Model**: BengaliMed v0.1.0
        **Languages**: Bengali, English
        **Specializations**: General, Mental Health, Pediatrics, Women's Health
        """)


def generate_response(prompt: str, language: str, specialization: str, cultural_mode: bool) -> str:
    """
    Generate chatbot response (placeholder implementation).
    
    Args:
        prompt: User input
        language: Selected language
        specialization: Medical specialization
        cultural_mode: Whether cultural sensitivity is enabled
        
    Returns:
        Generated response
    """
    # TODO: Replace with actual model inference
    
    # Simple rule-based responses for demo
    prompt_lower = prompt.lower()
    
    # Bengali responses
    if any(word in prompt_lower for word in ['জ্বর', 'fever']):
        if language.startswith("Bengali"):
            return """
জ্বরের জন্য:
- পর্যাপ্ত বিশ্রাম নিন
- প্রচুর পানি পান করুন  
- প্যারাসিটামল সেবন করতে পারেন
- জ্বর ১০২°F এর বেশি হলে ডাক্তারের পরামর্শ নিন

⚠️ এটি শুধুমাত্র সাধারণ পরামর্শ। গুরুতর অবস্থায় অবশ্যই ডাক্তারের কাছে যান।
            """
        else:
            return """
For fever:
- Take adequate rest
- Drink plenty of fluids
- You may take paracetamol
- Consult a doctor if fever exceeds 102°F

⚠️ This is general advice only. Please consult a healthcare professional for serious conditions.
            """
    
    elif any(word in prompt_lower for word in ['মাথাব্যথা', 'headache']):
        if language.startswith("Bengali"):
            return """
মাথাব্যথার জন্য:
- অন্ধকার ও শান্ত পরিবেশে বিশ্রাম নিন
- কপালে ঠান্ডা সেক দিন
- পর্যাপ্ত পানি পান করুন
- প্রয়োজনে ব্যথানাশক সেবন করুন

যদি মাথাব্যথা তীব্র হয় বা ঘন ঘন হয়, তাহলে ডাক্তারের পরামর্শ নিন।
            """
        else:
            return """
For headache:
- Rest in a dark, quiet environment
- Apply cold compress to forehead
- Stay hydrated
- Take pain relievers if needed

If headache is severe or frequent, please consult a doctor.
            """
    
    elif any(word in prompt_lower for word in ['ডায়াবেটিস', 'diabetes']):
        if language.startswith("Bengali"):
            return """
ডায়াবেটিসের লক্ষণ:
- অতিরিক্ত তৃষ্ণা
- ঘন ঘন প্রস্রাব
- অতিরিক্ত ক্ষুধা
- দুর্বলতা ও ক্লান্তি
- ওজন কমে যাওয়া
- ক্ষত শুকাতে দেরি

এই লক্ষণগুলো দেখা দিলে রক্তে সুগারের পরীক্ষা করান এবং ডাক্তারের পরামর্শ নিন।
            """
        else:
            return """
Diabetes symptoms:
- Excessive thirst
- Frequent urination
- Increased hunger
- Fatigue and weakness
- Unexplained weight loss
- Slow healing wounds

If you experience these symptoms, get your blood sugar tested and consult a doctor.
            """
    
    else:
        # Default response
        if language.startswith("Bengali"):
            return f"""
আপনার প্রশ্নটি আমি বুঝতে পেরেছি। তবে এই বিষয়ে সঠিক পরামর্শের জন্য একজন যোগ্য চিকিৎসকের সাথে পরামর্শ করা উত্তম।

বিশেষত্ব: {specialization}

⚠️ মনে রাখবেন, এটি একটি গবেষণা প্রকল্প। গুরুত্বপূর্ণ স্বাস্থ্য বিষয়ে সর্বদা পেশাদার চিকিৎসকের পরামর্শ নিন।
            """
        else:
            return f"""
I understand your question. However, for proper advice on this matter, it's best to consult with a qualified healthcare professional.

Specialization: {specialization}

⚠️ Remember, this is a research project. Always seek professional medical advice for important health matters.
            """


if __name__ == "__main__":
    main()
