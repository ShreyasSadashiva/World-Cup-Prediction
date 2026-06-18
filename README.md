# ⚽ FIFA World Cup 2026 Prediction App

A Streamlit web application for match outcome predictions, team form analysis, head-to-head stats, and player data for the FIFA World Cup 2026.

## Features

- **Match Predictor** — Select any two WC 2026 teams and get ML-powered win/draw/loss probabilities
- **Head to Head** — Full H2H matrix across all 48 teams + detailed pair drill-down with match history
- **Team Form** — Last 15 pre-tournament international matches per team with form strip and goal charts
- **WC Fixtures** — All 104 fixtures grouped by stage, with live scores and a one-click sync button
- **Players** — WC top scorers leaderboard and squad viewer for every team

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Database | Firebase Firestore |
| ML Model | XGBoost (56% CV accuracy, 3-class classification) |
| Historical data | [martj42 international results dataset](https://github.com/martj42/international_results) |
| Live WC data | [football-data.org](https://www.football-data.org/) API |

## Project Structure

```
├── app.py                      # Dashboard (home page)
├── pages/
│   ├── 1_Match_Predictor.py
│   ├── 2_Head_to_Head.py
│   ├── 3_Team_Form.py
│   ├── 4_WC_Fixtures.py
│   └── 5_Players.py
├── src/
│   ├── db/
│   │   ├── client.py           # Firebase Firestore singleton
│   │   └── schema.sql          # Firestore collection reference
│   ├── data/
│   │   ├── football_api.py     # football-data.org API client
│   │   ├── historical.py       # Historical CSV loader
│   │   ├── seeder.py           # First-run DB population
│   │   └── team_mappings.py    # API name → canonical name normalization
│   └── ml/
│       ├── features.py         # Feature engineering (29 features)
│       └── model.py            # XGBoost train / load / predict
├── scripts/
│   ├── seed_db.py              # One-time database seed
│   ├── train_model.py          # Train and save the ML model
│   └── update_wc.py            # Sync latest WC results from API
├── models/                     # Saved model (gitignored)
├── requirements.txt
└── .env.example
```

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/ShreyasSadashiva/World-Cup-Prediction.git
cd World-Cup-Prediction
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```
FIREBASE_CREDENTIALS_PATH=/path/to/serviceAccountKey.json
FOOTBALL_DATA_API_KEY=your-key-from-football-data.org
```

- **Firebase**: Create a project at [console.firebase.google.com](https://console.firebase.google.com), enable Firestore, then go to **Project Settings → Service Accounts → Generate new private key**
- **football-data.org**: Register for a free API key at [football-data.org](https://www.football-data.org/client/register)

### 3. Seed the database (one-time)

Fetches WC 2026 teams, 15 pre-tournament matches per team, all fixtures, squad data, and current top scorers. Takes ~8–10 minutes due to API rate limits.

```bash
python -m scripts.seed_db
```

### 4. Train the ML model

Downloads the full historical international results dataset (~49k matches from 1872–present) and trains an XGBoost classifier with 5-fold cross-validation.

```bash
python -m scripts.train_model
```

### 5. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

## Updating WC Results

Click **"Update WC Results"** in the sidebar of the Fixtures page, or run:

```bash
python -m scripts.update_wc
```

This fetches the latest finished match scores and player stats from the API and syncs them to Firestore.

## ML Model

The model is trained on all international matches from 2000 onward using a rolling-window feature approach (no data leakage). For each match, features are computed from the previous 15 matches of each team at the time of prediction.

**Features (29 total):**
- Per-team form: win rate, draw rate, loss rate, avg goals scored/conceded, goal difference, weighted recency form, clean sheet rate, scoring rate (×2 for home/away)
- Differential features: form diff, goal diff diff, goals scored/conceded diff
- Head-to-head: historical win rate, draw rate, goal difference
- Context: neutral venue, current WC tournament form differential

**Results:** 56.1% cross-validated accuracy (vs 33% random baseline for 3-class prediction).
