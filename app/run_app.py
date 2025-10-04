
from app.predict_app import run_app

def main():
    
    artifacts_dir = "models_artifact"
    app = run_app(artifacts_dir=artifacts_dir)

    return app

if __name__ == "__main__":
    main()