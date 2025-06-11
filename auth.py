import streamlit as st
import hashlib
from settings import USERS

def login():
    if st.session_state.get('authenticated'):
        return True
    st.sidebar.title('🔐 Login')
    user = st.sidebar.text_input('Username')
    pwd = st.sidebar.text_input('Password', type='password')
    if st.sidebar.button('Login'):
        if user in USERS and USERS[user] == hashlib.sha256(pwd.encode()).hexdigest():
            st.session_state['authenticated'] = True
            st.sidebar.success('Logged in!')
            return True
        st.sidebar.error('Invalid credentials')
    return False
