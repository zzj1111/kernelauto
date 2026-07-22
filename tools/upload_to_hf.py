#!/usr/bin/env python3
import argparse
import os
from huggingface_hub import HfApi, create_repo

def main():
    parser = argparse.ArgumentParser(description="Upload a local model folder to Hugging Face Hub")
    parser.add_argument("--local_dir", required=True, help="Path to local model folder")
    parser.add_argument("--hf_model_name", required=True, help="Target model name on Hugging Face (e.g. username/model-name)")
    parser.add_argument("--username", required=True, help="Target model name on Hugging Face (e.g. username/model-name)")
    parser.add_argument("--private", action="store_true", help="Make repo private (default: public)")
    parser.add_argument("--message", default="initial upload", help="Commit message")
    parser.add_argument("--hf_token", default=None,
                        help="HF token. If omitted, read from the HF_TOKEN environment variable.")
    args = parser.parse_args()

    # Prefer the CLI flag, else fall back to the HF_TOKEN environment variable.
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if not hf_token:
        raise SystemExit("No HF token provided. Pass --hf_token or set the HF_TOKEN environment variable.")

    api = HfApi(token=hf_token)
    
    # Create repository first
    try:
        create_repo(repo_id=args.hf_model_name, repo_type="model", private=args.private, exist_ok=True, token=hf_token)
        print(f"✅ Repository {args.hf_model_name} is ready")
    except Exception as e:
        print(f"❌ Failed to create repository: {e}")
        raise

    print(f"🚀 Uploading {args.local_dir} → {args.hf_model_name} ...")
    api.upload_folder(
        folder_path=args.local_dir,
        repo_id=args.username+'/'+args.hf_model_name,
        commit_message=args.message,
        token=hf_token,
    )
    print(f"✅ Done! View at: https://huggingface.co/{args.hf_model_name}")

if __name__ == "__main__":
    main()
