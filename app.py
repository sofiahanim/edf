from flask import Flask
from dotenv import load_dotenv
import os

# Explicitly load .env file
load_dotenv()

app = Flask(__name__)

# Example usage in the app
@app.route('/')
def index():
    # Using an environment variable
    host = os.getenv('REDSHIFT_HOST')
    return f'Redshift host is: {host}'

if __name__ == '__main__':
    app.run(debug=True)
