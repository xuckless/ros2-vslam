import os

from dotenv import load_dotenv
from roboflow import Roboflow


def main():
    load_dotenv()
    ROBOFLOW_WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE")
    ROBOFLOW_PROJECT_ID: str = os.getenv("ROBOFLOW_PROJECT_ID", "")

    rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))

    # Access the specified workspace
    workspace = rf.workspace(ROBOFLOW_WORKSPACE)

    # Deploy the model weights
    workspace.deploy_model(
        model_type="yolov8n",
        model_path=".",
        project_ids=[ROBOFLOW_PROJECT_ID],
        model_name="sign-detector-v4",
    )


if __name__ == "__main__":
    main()
