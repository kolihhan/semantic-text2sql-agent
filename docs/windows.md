# Windows

Use `run-demo.cmd` for the deterministic demo. It launches PowerShell with a process-local execution-policy bypass, creates project-local `.run/tmp`, and sets `TEMP`/`TMP` only for that process.

For the Streamlit UI:

```powershell
uv sync
uv run streamlit run app.py
```
