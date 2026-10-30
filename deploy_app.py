"""Hosted entry point for the Streamlit application.

The application reads student-level files that are not stored in this repository. This wrapper asks for an access
password, downloads the three required files from a private repository with a read-only key held in the host's
secrets, and then runs the application unchanged.
"""

from __future__ import annotations

import hmac
import runpy
import shutil
import urllib.error
import urllib.request
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
APP = ROOT / "app" / "streamlit_app.py"
DATA_REPO = "SruthiGS-Gito/sar-deploy-data"
FILES = {
    "day30_feature_matrix.csv": ROOT / "data" / "processed" / "day30_feature_matrix.csv",
    "department_risk.csv": ROOT / "data" / "processed" / "department_risk.csv",
    "retention_funnel.csv": ROOT / "data" / "processed" / "retention_funnel.csv",
    "student_profiles.csv": ROOT / "data" / "processed" / "student_profiles.csv",
    "model.joblib": ROOT / "models" / "final" / "model.joblib",
    "calibrator.joblib": ROOT / "models" / "final" / "calibrator.joblib",
    "shap/2014J.parquet": ROOT / "models" / "final" / "shap" / "2014J.parquet",
    "shap/2014J.json": ROOT / "models" / "final" / "shap" / "2014J.json",
}

# The page configuration must be the first Streamlit command, and the application sets it as well. Set it here with the
# same values, then turn the application's own call into a no-op so that it does not raise an error.
if not hasattr(st, "_sar_set_page_config"):
    st._sar_set_page_config = st.set_page_config
st._sar_set_page_config(page_title="Student Academic Risk: Day 30 early warning", layout="wide")
st.set_page_config = lambda *args, **kwargs: None


def _secret(name: str) -> str:
    try:
        return str(st.secrets[name])
    except Exception:
        return ""


def _authorised() -> bool:
    expected = _secret("APP_PASSWORD")
    if not expected:
        st.error("Access is not configured for this deployment.")
        return False
    if st.session_state.get("access_granted"):
        return True
    st.title("Student Academic Risk")
    st.caption("Day 30 early warning. Access is restricted because the application shows individual student records.")
    with st.form("access"):
        entered = st.text_input("Access password", type="password")
        submitted = st.form_submit_button("Open")
    if submitted:
        if hmac.compare_digest(entered.encode("utf-8"), expected.encode("utf-8")):
            st.session_state["access_granted"] = True
            st.rerun()
        st.error("The password is not correct.")
    return False


def _download(name: str, destination: Path, token: str) -> None:
    request = urllib.request.Request(
        f"https://api.github.com/repos/{DATA_REPO}/contents/{name}?ref=main",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github.raw+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "sar-ai-hosted-app",
        },
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    with urllib.request.urlopen(request, timeout=180) as response, open(partial, "wb") as handle:
        shutil.copyfileobj(response, handle)
    partial.replace(destination)


def _ensure_files() -> bool:
    missing = {name: path for name, path in FILES.items() if not path.exists()}
    if not missing:
        return True
    token = _secret("DATA_TOKEN")
    if not token:
        st.error("The data files are not available and no download key is configured.")
        return False
    try:
        with st.spinner("Preparing the data for first use. This takes about a minute."):
            for name, path in missing.items():
                _download(name, path, token)
    except urllib.error.HTTPError as error:
        st.error(f"The data download failed (HTTP {error.code}).")
        return False
    except Exception:
        st.error("The data download failed.")
        return False
    return True


if _authorised() and _ensure_files():
    runpy.run_path(str(APP), run_name="__main__")