from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import joblib
import pickle
import os, traceback

# --------------------- CONFIG ------------------------
CSV_PATH = r"C:\Users\sunan\OneDrive\Documents\new project\cleaned_csv.csv"
MODEL_PATH = r"C:\Users\sunan\OneDrive\Documents\new project\new_model.pkl.txt"
# -----------------------------------------------------

app = Flask(__name__)

# NOTE: Removed all emojis + forced UTF-8 safe HTML
INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Renewable Energy Dashboard</title>
  <style>
    body { font-family: Arial, sans-serif; max-width:900px; margin:30px auto; }
    label { display:block; margin-top:10px; }
    input[type="number"]{ width:150px; }
    button{ margin-top:12px; padding:8px 14px; }
    .result{ margin-top:18px; font-weight:600; }
    nav a{ margin-right:12px }
  </style>
</head>
<body>
  <h1>Renewable Energy API (EDA + Predict)</h1>
  <nav>
    <a href="/">Home</a>
    <a href="/eda/summary" target="_blank">EDA Summary</a>
    <a href="/eda/correlation" target="_blank">EDA Correlation</a>
  </nav>

  <section>
    <h2>Predict Energy Output</h2>
    <form id="predictForm">
      <label>Solar (TWh): <input type="number" step="any" id="solar" required></label>
      <label>Wind (TWh): <input type="number" step="any" id="wind" required></label>
      <label>Hydro (TWh): <input type="number" step="any" id="hydro" required></label>
      <button type="submit">Predict</button>
    </form>
    <div class="result" id="result"></div>
  </section>

  <script>
    const form = document.getElementById('predictForm');
    form.addEventListener('submit', async (e)=> {
      e.preventDefault();
      const solar = parseFloat(document.getElementById('solar').value) || 0;
      const wind  = parseFloat(document.getElementById('wind').value) || 0;
      const hydro = parseFloat(document.getElementById('hydro').value) || 0;

      const url = `/predict?solar=${solar}&wind=${wind}&hydro=${hydro}`;
      const res = await fetch(url);
      const data = await res.json();

      if (res.ok)
        document.getElementById('result').innerText = 'Predicted Output: ' + data.predicted_output;
      else
        document.getElementById('result').innerText = 'Error: ' + (data.error || JSON.stringify(data));
    });
  </script>
</body>
</html>
"""

_loaded_model = None
def load_model():
    global _loaded_model
    if _loaded_model is not None:
        return _loaded_model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

    try:
        _loaded_model = joblib.load(MODEL_PATH)
        return _loaded_model
    except Exception:
        with open(MODEL_PATH, "rb") as f:
            _loaded_model = pickle.load(f)
        return _loaded_model

def load_df():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found at: {CSV_PATH}")
    return pd.read_csv(CSV_PATH)

@app.route("/")
def home():
    return render_template_string(INDEX_HTML)

@app.route("/eda/summary")
def eda_summary():
    try:
        df = load_df()
        return df.describe(include='all').to_html()
    except Exception:
        return f"<pre>{traceback.format_exc()}</pre>", 500

@app.route("/eda/correlation")
def eda_corr():
    try:
        df = load_df()
        corr = df.select_dtypes(include=['number']).corr()
        return corr.to_html()
    except Exception:
        return f"<pre>{traceback.format_exc()}</pre>", 500

@app.route("/predict")
def predict():
    try:
        solar = float(request.args.get("solar", 0))
        wind  = float(request.args.get("wind", 0))
        hydro = float(request.args.get("hydro", 0))

        model = load_model()
        pred = model.predict([[solar, wind, hydro]])
        return jsonify({"predicted_output": float(pred[0])})
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

if __name__ == "__main__":
    print("Starting Renewable Energy API…")
    print("CSV:", CSV_PATH)
    print("MODEL:", MODEL_PATH)
    app.run(debug=True)

