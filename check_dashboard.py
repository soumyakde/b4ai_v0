with open('streamlit_app/dashboards/teacher_dashboard.py', 'r', encoding='utf-8') as f:
    content = f.read()

print('use_container_width count :', content.count('use_container_width'))
print('tolist count              :', content.count('tolist'))
print('width=stretch count       :', content.count("width='stretch'"))
