from app.main import app, ensure_data_dirs

if __name__ == "__main__":
    ensure_data_dirs()
    app.launch()
