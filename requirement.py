import os
import sys

def generate_requirements():
    platform = "linux"
    
    if platform.startswith('win'):
        requirements_file = 'requirements_windows.txt'
    elif platform.startswith('linux'):
        requirements_file = 'requirements_linux.txt'
    else:
        requirements_file = 'requirements.txt'
    
    os.system(f"pip freeze > {requirements_file}")
    print(f"Generated requirements file for {platform}: {requirements_file}")

generate_requirements()
