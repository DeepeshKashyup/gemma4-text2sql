"""
Deploy fine-tuned Gemma 4 to a Vertex AI endpoint using vLLM.

Usage:
    python serve/deploy_vertex.py
    python serve/deploy_vertex.py --model-gcs gs://bucket/output/gemma4-text2sql/merged
"""

import argparse
import os
import sys

import requests
from google import auth
from google.cloud import aiplatform

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import VertexConfig


VLLM_DOCKER_URI = (
    "us-docker.pkg.dev/vertex-ai/"
    "vertex-vision-model-garden-dockers/pytorch-vllm-serve:gemma4"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-gcs",
        type=str,
        default=None,
        help="GCS path to merged model weights (e.g., gs://bucket/output/merged)",
    )
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--region", type=str, default=None)
    parser.add_argument("--endpoint-name", type=str, default="gemma4-text2sql")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = VertexConfig()

    if args.project:
        cfg.project_id = args.project
    if args.region:
        cfg.region = args.region

    model_gcs = args.model_gcs or f"{cfg.staging_bucket}/output/gemma4-text2sql/merged"

    aiplatform.init(
        project=cfg.project_id,
        location=cfg.region,
        staging_bucket=cfg.staging_bucket,
    )

    # -----------------------------------------------------------------------
    # 1. Create a dedicated endpoint
    # -----------------------------------------------------------------------
    print(f"Creating endpoint: {args.endpoint_name}")
    endpoint = aiplatform.Endpoint.create(
        display_name=args.endpoint_name,
        dedicated_endpoint_enabled=True,
    )
    print(f"Endpoint created: {endpoint.resource_name}")

    # -----------------------------------------------------------------------
    # 2. Upload model with vLLM serving config
    # -----------------------------------------------------------------------
    vllm_args = [
        "python", "-m", "vllm.entrypoints.api_server",
        "--host=0.0.0.0",
        "--port=8080",
        f"--model={model_gcs}",
        "--tensor-parallel-size=1",
        "--max-model-len=4096",         # Sufficient for text-to-SQL
        "--gpu-memory-utilization=0.9",
        "--max-num-seqs=64",
        "--limit-mm-per-prompt.image=0",  # Text-only
        "--enable-auto-tool-choice",
        "--tool-call-parser=gemma4",
        "--reasoning-parser=gemma4",
    ]

    print("Uploading model to Vertex AI...")
    model = aiplatform.Model.upload(
        display_name=f"{args.endpoint_name}-model",
        serving_container_image_uri=VLLM_DOCKER_URI,
        serving_container_args=vllm_args,
        serving_container_ports=[8080],
        serving_container_predict_route="/generate",
        serving_container_health_route="/ping",
        serving_container_environment_variables={
            "MODEL_ID": model_gcs,
            "DEPLOY_SOURCE": "gemma4-text2sql-project",
        },
        serving_container_shared_memory_size_mb=(16 * 1024),
        serving_container_deployment_timeout=7200,
        model_garden_source_model_name="publishers/google/models/gemma4",
    )
    print(f"Model uploaded: {model.resource_name}")

    # -----------------------------------------------------------------------
    # 3. Deploy model to endpoint
    # -----------------------------------------------------------------------
    print("Deploying model to endpoint...")

    creds, _ = auth.default()
    auth_req = auth.transport.requests.Request()
    creds.refresh(auth_req)

    url = (
        f"https://{cfg.region}-aiplatform.googleapis.com/ui/"
        f"projects/{cfg.project_id}/locations/{cfg.region}/"
        f"endpoints/{endpoint.name}:deployModel"
    )
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {creds.token}",
    }
    data = {
        "deployedModel": {
            "model": model.resource_name,
            "displayName": f"{args.endpoint_name}-deployed",
            "dedicatedResources": {
                "machineSpec": {
                    "machineType": cfg.machine_type,
                    "acceleratorType": cfg.accelerator_type,
                    "acceleratorCount": cfg.accelerator_count,
                },
            },
        },
    }

    response = requests.post(url, headers=headers, json=data)
    print(f"Deploy response: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

    # -----------------------------------------------------------------------
    # 4. Test the endpoint
    # -----------------------------------------------------------------------
    print("\nTesting endpoint with a sample query...")

    test_prompt = """-- Database: concert_singer
CREATE TABLE stadium (
  Stadium_ID int PRIMARY KEY,
  Location text,
  Name text,
  Capacity int
);
CREATE TABLE singer (
  Singer_ID int PRIMARY KEY,
  Name text,
  Country text,
  Age int
);

-- Question: How many singers are from France?"""

    test_payload = {
        "prompt": test_prompt,
        "max_tokens": 256,
        "temperature": 0.0,
    }

    predict_url = f"https://{endpoint.resource_name}/predict"
    test_response = requests.post(
        predict_url,
        headers={"Authorization": f"Bearer {creds.token}", "Content-Type": "application/json"},
        json=test_payload,
    )

    if test_response.status_code == 200:
        print(f"Test response: {test_response.json()}")
    else:
        print(f"Test failed ({test_response.status_code}): {test_response.text}")
        print("The endpoint may still be warming up. Try again in a few minutes.")

    print(f"\nDeployment complete!")
    print(f"Endpoint:  {endpoint.resource_name}")
    print(f"Model GCS: {model_gcs}")


# ---------------------------------------------------------------------------
# Quick deploy using gcloud CLI (alternative)
# ---------------------------------------------------------------------------

def print_gcloud_command():
    """Print a one-liner gcloud deploy command."""
    print("\nAlternative: Deploy base Gemma 4 via gcloud CLI:")
    print("  gcloud ai model-garden models deploy \\")
    print("    --model=google/gemma4@gemma-4-31b")


if __name__ == "__main__":
    import json
    main()
