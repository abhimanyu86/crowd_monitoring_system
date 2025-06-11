# People Counter App

## Highlights
* **Dynamic model dropdown** – just drop `.pt` / `.onnx` weights into `models/`.
* Two modes: **Eagle‑Eye** (total) and **Lane Counter** (split frame).
* Capacity & restricted‑item email alerts (10 s cool‑down).
* Simple login (admin / password123).

### Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
