# GenZ Datamind — Streamlit Predictor

A standalone, single-page version of the Predictor, built with Streamlit
so it can be hosted on **Streamlit Community Cloud** (which only runs
Streamlit apps — it can't host the main Flask + custom-HTML website).

This uses the exact same trained model as the main site
(`GenZ-Datamind-Backend/models/best_classifier.pkl` and
`best_regressor.pkl`, copied into this folder's `models/`), so predictions
match — just with Streamlit's own widget-based UI instead of the site's
custom Tailwind design.

## What's different from the main site

- **No shared History / SQLite log.** Streamlit Cloud's filesystem is
  ephemeral, so this app keeps a simple in-session prediction table
  (`st.session_state`) that resets when you refresh or close the tab —
  not a persisted database like `app.py`'s `/api/history`.
- **UI is Streamlit widgets**, not the custom dark neon design — sliders,
  dropdowns, and Streamlit's default styling. This is a Streamlit
  Cloud constraint, not something that can be worked around while staying
  on that host.

## Run it locally first

```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Opens at `http://localhost:8501`.

## Deploy to Streamlit Community Cloud

Streamlit Cloud deploys from a GitHub repo — there's no direct "upload a
folder" option, so:

1. **Push this folder to GitHub.** Create a new repo (public or private)
   and push `streamlit_app.py`, `requirements.txt`, and the `models/`
   folder (the three `.pkl` files — all small, a few MB total, fine to
   commit directly to git; no Git LFS needed).

   ```bash
   cd streamlit_app
   git init
   git add .
   git commit -m "GenZ Datamind Streamlit predictor"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<repo-name>.git
   git push -u origin main
   ```

2. **Go to** [share.streamlit.io](https://share.streamlit.io) **and sign
   in with GitHub.**

3. Click **"New app"**, pick the repo you just pushed, set:
   - Branch: `main`
   - Main file path: `streamlit_app.py`

4. Click **Deploy**. First build takes a couple of minutes (installing
   scikit-learn/xgboost). You'll get a public URL like
   `https://<something>.streamlit.app`.

5. Any time you `git push` a change to that repo, Streamlit Cloud
   auto-redeploys.

## If you'd rather deploy the real site instead

This Streamlit app is a simplified stand-in for one page (the Predictor).
If what you actually want online is the full custom-designed website with
History/Recommendations/Analyst reports and the real SQLite-backed log,
that needs a host that runs Flask — Streamlit Cloud can't serve that.
Render, Railway, and PythonAnywhere all have free tiers that work with the
existing `GenZ-Datamind-Backend/app.py` as-is; ask if you want deployment
steps for one of those instead.
