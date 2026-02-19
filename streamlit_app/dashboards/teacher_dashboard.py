import streamlit as st

def show_teacher_dashboard(username: str):
    st.title("Teacher Dashboard")
    st.write(f"Welcome, {username}!")
    st.success("Teacher routing is working correctly.")