<h1 align="center"><b>🧠 MindMate – Student Wellness Chatbot</b></h1>

A supportive AI-powered chatbot designed to promote student mental wellness by providing empathetic, stigma-free conversations. Built with **Streamlit**, **NVIDIA’s Llama-based API**, and **sentiment analysis (TextBlob)**, MindMate acts as a companion to help students navigate stress, anxiety, and academic challenges.

✨ Features

* 💬 **Conversational Support** – Provides calm, encouraging, and respectful responses.
* 🧾 **Sentiment Awareness** – Uses TextBlob to detect whether the student’s mood is *positive*, *neutral*, or *negative*.
* 🤝 **Adaptive Empathy** – Adjusts tone and supportiveness based on user sentiment.
* 🎨 **Clean Interface** – Built with Streamlit for a simple, interactive chat UI.
* ⚡ **Powered by NVIDIA AI** – Leverages cutting-edge large language models for natural dialogue.

🛠️ Tech Stack

* **Frontend / UI**: [Streamlit](https://streamlit.io/)
* **AI Model**: [NVIDIA API (Meta LLaMA-3.3-70B-Instruct)](https://build.nvidia.com/meta/llama-3_3-70b-instruct)
* **Sentiment Analysis**: [TextBlob](https://textblob.readthedocs.io/en/dev/)
* **Environment Management**: [python-dotenv](https://pypi.org/project/python-dotenv/)

🚀 Getting Started

1️⃣ Clone the Repository

git clone https://github.com/<your-username>/MindMate.git
cd MindMate

2️⃣ Install Dependencies

pip install -r requirements.txt

3️⃣ Add NVIDIA API Key

Create a `.env` file in the project root and add:
NVIDIA_API_KEY=your_api_key_here

4️⃣ Run the App
streamlit run app.py

📸 Demo Preview
<img width="1915" height="1079" alt="Screenshot 2025-08-26 002129" src="https://github.com/user-attachments/assets/5183b048-0dbd-4d0c-9754-ce160953992f" />

🤔 Why MindMate?

University life can be overwhelming — deadlines, exams, and personal struggles often pile up. MindMate is designed to:

* Provide a **safe, judgment-free space** to talk.
* Encourage **positive coping strategies**.
* Nudge students toward **professional support** if needed.

⚠️ **Disclaimer**: MindMate is *not a replacement for professional mental health care*. If you’re struggling, please reach out to a qualified counselor or trusted support system.

🌱 Future Enhancements

* 📊 **Mood Tracking Dashboard** – Visualize user sentiment over time.
* 🎙️ **Voice Input/Output** – Make conversations more natural.
* 🔔 **Wellness Reminders** – Send motivational quotes & self-care prompts.
* 🛡️ **Anonymous Cloud Deployment** – Allow students to access MindMate anywhere.

🤝 Contributing

Contributions are welcome! Feel free to fork this repo, open issues, or submit pull requests.

📄 License

This project is licensed under the **MIT License** – feel free to use and modify.

