from flask import Flask, request
import os
import subprocess

app = Flask(__name__)

@app.route('/update', methods=['POST'])
def update_repo():
    try:
        repo_path = "/mnt/c/Users/hanim/edf"  # Adjust for your local setup
        commands = [
            f"cd {repo_path}",
            "git fetch origin main",
            "git reset --hard origin/main"
        ]
        for command in commands:
            subprocess.run(command, shell=True, check=True)
        return "Repository updated successfully.", 200
    except Exception as e:
        return f"Error: {str(e)}", 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
