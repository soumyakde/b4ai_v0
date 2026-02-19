import streamlit as st

def show_admin_dashboard(username):
    st.title("Admin Dashboard")
    st.write(f"Welcome {username}")