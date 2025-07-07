# Auckland Air Discharge Consent Dashboard

A smart, AI-powered dashboard built with [Streamlit](https://streamlit.io) that turns static PDF resource consent reports from Auckland Council into a live, interactive web tool.

---

## 🌍 Purpose

Environmental consent data (particularly Air Discharge Resource Consents) is often buried in lengthy PDFs. This dashboard allows users to:

- Upload and process PDF decision reports
- Extract key information: applicant, address, issue/expiry dates, triggers, consent conditions
- Visualise expired/active/expiring consents
- Run AI-powered semantic searches and chatbot queries
- Geolocate sites and display them on an interactive map

Perfect for regulators, consultants, and planners who need fast insights into Auckland’s air discharge consent landscape.

---

## 🚀 Key Features

- ✅ **PDF Parsing** via PyMuPDF
- ✅ **Metadata Extraction** using Regex
- ✅ **Interactive Dashboard** with Streamlit & Plotly
- ✅ **Geocoding** using `geopy` and OpenStreetMap
- ✅ **Semantic Search** via Sentence Transformers
- ✅ **AI Chatbot** powered by Google Gemini or Langchain Groq
- ✅ **Weather Integration** (OpenWeatherMap)
- ✅ **Consent Status Monitoring** (Active, Expired, Expiring Soon)

---

## 🧠 Tech Stack

- Python 3.13
- Streamlit
- PyMuPDF
- geopy
- pandas
- plotly
- sentence-transformers
- Langchain + Groq / Google Gemini
- OpenWeatherMap API

---

## 🏗️ Deployment Instructions

You can deploy this project in minutes using [Streamlit Community Cloud](https://share.streamlit.io):

### 1. Clone this Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
```

### 2. Add Your Files

Ensure these two files exist:

- `dashboard.py` (main script)
- `requirements.txt` (with all dependencies listed)

### 3. Set Up Secrets

On [Streamlit Cloud](https://share.streamlit.io):

- Log in with GitHub
- Click **Create app**
- Select this GitHub repo and point to `dashboard.py`
- In **Advanced Settings**, set the following secrets:

```ini
GOOGLE_API_KEY="your_google_gemini_key"
GROQ_API_KEY="your_langchain_groq_key"
OPENWEATHER_API_KEY="your_openweather_key"
```

⚠️ **Never hardcode your API keys into your Python files.** Use Streamlit secrets to prevent accidental exposure.

### 4. Deploy the App

- Set Python version to **3.13**
- Click **Deploy**
- Your app will go live with a custom URL 🎉

---

## 📸 Screenshots (optional)

Include UI screenshots or GIFs here to showcase the dashboard in action.

---

## 📋 Sample Use Cases

- Find all consents expiring in 2025
- Check which sites trigger E14 or NES-AQ rules
- Map air discharge activity near a suburb
- Extract and analyse permit conditions for enforcement

---

## 👩‍💼 Target Users

- Local councils & planners
- Air quality specialists
- Environmental consultants
- Legal advisors & compliance teams

---

## 👏 Authors

Built by Earl Tavera, Alana Jacobson-Pepere & Louis Boamponsem| Auckland Air Discharge Intelligence © 2025

---

## 📬 Feedback

Feel free to submit issues, suggestions, or feature requests via GitHub Issues.

---

## 🛠️ Future Features (Ideas)

- 🔄 Auto-refresh with live data from council database
- 📎 PDF highlighting of extracted conditions
- 🧭 Navigation by suburb or zone
- 📊 Export to Excel or Power BI

---

Enjoy using the dashboard, and let us know if you'd like to expand it to other consent types!

