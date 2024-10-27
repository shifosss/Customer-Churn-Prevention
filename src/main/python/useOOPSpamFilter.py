# app.py
from flask import Flask, request, render_template, jsonify
import requests

app = Flask(__name__)

# Your API credentials
RAPIDAPI_KEY = "c58f72849amsh6cb62cf07171c3dp1e6746jsn77a4aa118614"
RAPIDAPI_HOST = "oopspam.p.rapidapi.com"


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/check_spam', methods=['POST'])
def check_spam():
    try:
        # Get content from the form
        content = request.form.get('content', '')

        # Prepare the payload for OOPSpam API
        payload = {
            # "senderIP": "127.0.0.1",  # You can modify this as needed
            "email": "test@example.com",  # You can make this dynamic if needed
            "content": content,
            "blockTempEmail": False,
            "logIt": False,
            "checkForLength": True,
            "urlFriendly": False,
            "allowedLanguages": ["en"],
            # "allowedCountries": ["it", "us"],
            # "blockedCountries": ["ru", "cn"]
        }

        # API request headers
        headers = {
            "x-rapidapi-key": RAPIDAPI_KEY,
            "x-rapidapi-host": RAPIDAPI_HOST,
            "Content-Type": "application/json"
        }

        # Make request to OOPSpam API
        response = requests.post(
            "https://oopspam.p.rapidapi.com/v1/spamdetection",
            json=payload,
            headers=headers
        )

        # Extract just the isContentSpam value
        result = response.json()
        is_spam = result.get('Details', {}).get('isContentSpam', 'unknown')

        return jsonify({'isSpam': is_spam})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)