from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
from openai import OpenAI

app = Flask(__name__)

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)


@app.route("/")
def hello_world():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    try:
        question = request.form.get("question")

        if not question:
            return jsonify({"error": "Please enter a question"}), 400

        response = client.chat.completions.create(
            model="gemini-3.6-flash",
            messages=[
                {
                    "role": "system",
                    "content": "Act like a helpful personal assistant"
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            temperature=0.7,
            max_tokens=512
        )

        answer = response.choices[0].message.content.strip()

        return jsonify({"response": answer}), 200

    except Exception as e:
        print("ASK ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


@app.route("/summarize", methods=["POST"])
def summarize():
    try:
        email_text = request.form.get("email")

        if not email_text:
            return jsonify({"error": "Please enter an email"}), 400

        prompt = f"""
Summarize the following email in 2-3 sentences.
Keep the summary clear and concise.

Email:
{email_text}
"""

        response = client.chat.completions.create(
            model="gemini-3.6-flash",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert email summarization assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=512
        )

        summary = response.choices[0].message.content.strip()

        return jsonify({"response": summary}), 200

    except Exception as e:
        print("SUMMARIZE ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
