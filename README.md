# Heart Disease Prediction – End-to-End ML Deployment

**Name:** Abhi Pandey
**Registration Number:** 23BAI10909
**Assignment:** AI-ML Assignment – 10 (End-to-End Machine Learning Model Deployment using GitHub and Render)

A machine learning model that predicts whether a patient is at risk of heart
disease based on clinical parameters, served through a Flask REST API and
deployed live on Render.

**Render Deployment URL:** `<ADD_YOUR_RENDER_URL_HERE_AFTER_DEPLOYING>`

---

## 1. Problem Statement

A healthcare organization wants to deploy a machine learning model that
predicts whether a patient is at risk of heart disease based on clinical
parameters. This project builds that model, wraps it in a Flask REST API,
and deploys it as a live web service on Render.

## 2. Dataset

- **Source:** [Heart Disease Prediction Dataset – Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
- **Downloaded directly in code** using `kagglehub`:

```python
import kagglehub
path = kagglehub.dataset_download("johnsmith88/heart-disease-dataset")
print("Path to dataset files:", path)
```

- **Target variable:** `target` (1 = heart disease present, 0 = no heart disease)
- **Features (13):** `age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal`

## 3. Repository Structure

```
HeartDiseaseDeployment/
│
├── app.py                 # Flask REST API
├── model.pkl              # Trained model (Joblib bundle)
├── requirements.txt       # Runtime dependencies (Flask + Render)
├── requirements-train.txt # requirements.txt + kagglehub, for running train_model.py
├── Procfile                # Render start command
├── README.md                # Project documentation
├── train_model.py           # Data loading, preprocessing, training, evaluation
├── heart.csv                 # Dataset (downloaded via kagglehub, saved locally)
├── .gitignore
├── templates/                # (optional) HTML templates
└── static/                   # (optional) static assets
```

## 4. Task 1 – Data Understanding and Preprocessing

`train_model.py` performs:
1. Loads the dataset with Pandas after downloading it via `kagglehub`.
2. Displays the first five records (`df.head()`).
3. Identifies the 13 numerical/clinical features and the `target` variable.
4. Checks for missing values (`df.isnull().sum()`).
5. Splits the data 80/20 into train/test sets with stratification on the target.

## 5. Task 2 – Model Development

- **Algorithm used:** Random Forest Classifier (`n_estimators=200, max_depth=6`)
- **Evaluation metric:** Accuracy Score (also reports precision/recall/F1 and a confusion matrix)
- **Serialization:** The trained model, its feature order, and its accuracy are bundled and saved with `joblib` to `model.pkl`.

Run training locally:

```bash
pip install -r requirements-train.txt
python train_model.py
```

> Note: Downloading from Kaggle via `kagglehub` requires a Kaggle account and
> API credentials configured in your environment (`~/.kaggle/kaggle.json` or
> the `KAGGLE_USERNAME` / `KAGGLE_KEY` environment variables).

## 6. Task 3 – API Development

`app.py` exposes a Flask REST API:

| Route      | Method | Description                                   |
|------------|--------|------------------------------------------------|
| `/`        | GET    | Health/info message, lists required fields     |
| `/health`  | GET    | Simple health check for uptime monitoring       |
| `/predict` | POST   | Accepts patient JSON, returns prediction JSON   |

**Example request:**

```bash
curl -X POST https://<your-render-url>/predict \
  -H "Content-Type: application/json" \
  -d '{
        "age": 58, "sex": 1, "cp": 0, "trestbps": 128, "chol": 216,
        "fbs": 0, "restecg": 0, "thalach": 131, "exang": 1,
        "oldpeak": 2.2, "slope": 1, "ca": 3, "thal": 3
      }'
```

**Example response:**

```json
{
  "prediction": "Heart Disease Detected",
  "probability": 0.5828
}
```

The API also validates its input: missing fields return a `400` listing
which fields are missing, and non-numeric values return a `400` listing
which fields must be numeric.

Run locally:

```bash
pip install -r requirements.txt
python app.py
```

## 7. Task 4 – GitHub and Cloud Deployment

### GitHub
1. Create a **public** GitHub repository named `HeartDiseaseDeployment`.
2. Push all files listed in the repository structure above (source code, `model.pkl`, `app.py`, `requirements.txt`, `README.md`).

### Render
1. Sign in to [Render](https://render.com) and click **New → Web Service**.
2. Connect your GitHub repository.
3. Configure the service:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
   - **Environment:** Python 3
4. Deploy and wait for the service to go live.
5. Copy the generated Render URL (e.g. `https://heart-disease-api.onrender.com`) and paste it at the top of this README and in the Google Form submission.
6. Keep the service and the GitHub repository active/public until evaluation is complete.

> Render's free tier spins the service down after periods of inactivity, so
> the first request after idle time can take 30–60s to respond (cold start).
> Hit `/health` a minute or two before the evaluator is expected to test it,
> if you can.

## 8. Task 5 – Conclusion

The Random Forest model achieved a solid accuracy on the held-out test split,
showing that clinical features such as chest pain type, maximum heart rate,
exercise-induced angina, and the number of major vessels colored by
fluoroscopy carry strong predictive signal for heart disease risk. Precision
and recall were reasonably balanced across both classes, which matters for a
healthcare use case where both false negatives (missed at-risk patients) and
false positives (unnecessary alarm) carry real costs. The main challenges
during deployment involved keeping the feature order and data types
consistent between training and inference, handling missing or malformed
JSON fields gracefully in the API, and configuring Render's build/start
commands and free-tier cold starts correctly. This project highlighted why
MLOps practices matter in real-world ML systems: version-controlling code
and model artifacts together, packaging the model in a reproducible format,
serving it through a well-tested API, and continuously monitoring the live
endpoint are all essential to keeping a model reliable, reproducible, and
usable in production rather than just accurate on paper.

---

## Author

**Abhi Pandey**
Registration Number: 23BAI10909
