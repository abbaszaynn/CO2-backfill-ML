"""Build CO2_backfill_app.zip to send to collaborators. Run again after changing the app:  python make_package.py"""
import zipfile

ROOT = "CO2_backfill_app/"
APP_FILES = ["app.py", "backfill_model.py", "co2_backfill_train.csv", "requirements.txt", "README.md", ".streamlit/config.toml"]
TOP_FILES = {"START_HERE.txt": 0o644, "Start app (Windows).bat": 0o644, "Start app (Mac).command": 0o755}

def add(z, arcname, data, mode=0o644):
    info = zipfile.ZipInfo(ROOT + arcname, date_time=(2026, 11, 10, 12, 0, 0))
    info.external_attr = (0o100000 | mode) << 16       # keeps the Mac launcher double-clickable
    info.compress_type = zipfile.ZIP_DEFLATED
    z.writestr(info, data)

with zipfile.ZipFile("CO2_backfill_app.zip", "w") as z:
    for name, mode in TOP_FILES.items():
        data = open("package/" + name, "rb").read().replace(b"\r\n", b"\n")
        if not name.endswith(".command"): data = data.replace(b"\n", b"\r\n")   # Windows line endings
        add(z, name, data, mode)
    for name in APP_FILES:
        add(z, "app/" + name, open(name, "rb").read())
    print("\n".join(z.namelist()))
