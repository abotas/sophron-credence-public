# Sophron Public Apps

Interactive dashboards for [Sophron Research](https://sophron.org) AI evaluation results.

## Credence: Measuring AI Belief

An automated pipeline to extract probabilistic beliefs — *expressed credences* — from frontier AI models. See results at [credence.streamlit.app](https://credence.streamlit.app/).

### Run locally

```bash
# Install uv (Python package manager) if you don't have it
brew install uv

# Clone and run
git clone https://github.com/sophron-research/public-apps.git
cd public-apps
uv sync
uv run streamlit run src/credence/viz/app.py
```

The app will open in your browser at `http://localhost:8501`.
