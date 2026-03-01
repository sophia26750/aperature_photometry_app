# 🌌 Stellar Photometry & Calibration Portal  
Precision • Automation • Scientific Integrity

A complete web platform for **automated stellar photometry**, **WCS calibration**, and **magnitude standardization** using Sloan **g/r** or Johnson **B/V** filters.  
Designed for observers, students, and researchers who want **professional‑grade results** with a clean, modern interface.

This system performs the entire workflow:

- Astrometry.net plate solving  
- APASS DR10 catalog querying  
- Star detection & background‑subtracted photometry  
- Color‑term and zero‑point calibration  
- Target‑object magnitude extraction  
- Star‑cluster photometry & CMD generation  
- Diagnostic plots for transparency and verification  

Whether you're analyzing a single star or an entire cluster, this platform delivers **reproducible, publication‑quality results**.

---

# 🚀 Quick Start

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

---

## 2. Create a Virtual Environment

### macOS / Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows (PowerShell)
```bash
python -m venv venv
venv\Scripts\Activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If you yourself added new libraries, make sure to run this in order to update your requirements.txt:

```bash
pip freeze > requirements.txt
```

---

## 4. Configure Your Astrometry.net API Key

This project requires an API key to perform WCS plate solving.

### Step 1 — Create a `.env` file

macOS / Linux:
```bash
touch .env
```

Windows:
```bash
New-Item -Path .env -ItemType File
```

### Step 2 — Get your API key

1. Go to https://nova.astrometry.net  
2. Sign in  
3. Click **Profile**  
4. Copy your API key (looks like: `asdfsdfgghjgho`)  

### Step 3 — Add it to `.env`

```
ASTRO_LOGIN=YOUR_ASTROMETRY_API_KEY
```

### Step 4 — Ensure Flask loads it

At the top of `app.py`:

```python
from dotenv import load_dotenv
load_dotenv()
```

### Step 5 — Verify

```bash
python app.py
```

Inside your code:

```python
print(os.environ.get("ASTRO_LOGIN"))
```

If it prints your key, you're good.

---

## 5. Run the Website

```bash
python app.py
```

or

```bash
flask run
```

Open:

```
http://127.0.0.1:5000
```

---

# 🌠 Using the Platform

## ⭐ Object Calibration (Single‑Star Photometry)

1. Enter RA/Dec (decimal or HMS/DMS)  
2. Choose photometric system (Sloan g/r or Johnson B/V)  
   - *Note: Johnson B/V is not yet implemented*  
3. Upload FITS files or provide file paths  
4. Click **Run Calibration**

The system automatically performs:

- WCS solving  
- APASS DR10 catalog query  
- Star detection  
- Aperture photometry with background subtraction  
- Color‑term calibration  
- Zero‑point calibration  
- Final standardized magnitudes  
- Diagnostic plots (WCS check, color term, offset plot)

Results appear instantly with:

- Standard g and r magnitudes  
- Uncertainties  
- Calibration coefficients  
- WCS verification images  
- Color‑term and zero‑point plots  

---

## ⭐ Star Cluster Calibration

1. Upload Sloan g and r FITS images  
2. Enter cluster center (RA/Dec)  
3. Choose inner/outer radii  
4. Run the calibration

The system:

- Splits APASS stars into cluster vs calibration sets  
- Performs photometry on both images  
- Computes calibrated magnitudes  
- Generates a **Color–Magnitude Diagram (CMD)**  
- Outputs calibration parameters and plots  

---

# 📁 Project Structure

```
project/
│
├── app.py                     # Main Flask application
├── templates/                 # HTML templates
│   ├── index.html
│   ├── object_calibration.html
│   ├── star_cluster_calibration.html
│   └── aavso_instructions.html
│
├── static/                    # Saved plots and assets
│
├── uploads/                   # User FITS uploads
│
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

# 🧪 Scientific Notes

### Photometry
- Aperture photometry with annulus background subtraction  
- Negative fluxes safely clipped  
- Instrumental magnitudes computed via  
  \(-2.5 \log_{10}(F)\)

### Calibration
- Color‑term calibration using APASS DR10  
- Linear least‑squares regression  
- Zero‑point offset correction  
- Physically meaningful constraints enforced  

### WCS
- Exact Astrometry.net WCS headers embedded into FITS  
- Verified via diagnostic overlay plots  

---

# 🟦 AAVSO Submission Support

The platform includes:

- Automatic RA/Dec conversion to **HH:MM:SS / DD:MM:SS**  
- Automatic Julian Date conversion  
- Copy‑to‑clipboard buttons  
- A full step‑by‑step WebObs submission guide  
- Notes on comparison/check star selection  
- Warnings about filter consistency (TG/TR)  

Everything you need to submit scientifically valid data.

---

# 🤝 Contributing

Pull requests are welcome!  
If you’d like to add features, improve UI/UX, or extend photometric support, feel free to open an issue.

---

# 📜 License

MIT License 
