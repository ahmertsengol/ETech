#!/usr/bin/env python3
"""
🚀 Hugging Face Model Upload Script
================================

Uploads the AI Learning Psychology Analyzer models to Hugging Face Hub.
This script handles authentication, repository creation, and file uploads.

Usage:
    python upload_to_huggingface.py --username your-username --token your-token
    
Environment Variables:
    HUGGINGFACE_TOKEN: Your Hugging Face API token
    HUGGINGFACE_USERNAME: Your Hugging Face username
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from huggingface_hub import HfApi, Repository, create_repo, upload_file
    import huggingface_hub
    print(f"✅ Using huggingface_hub version: {huggingface_hub.__version__}")
except ImportError:
    print("❌ Error: huggingface_hub not installed. Install with: pip install huggingface-hub")
    sys.exit(1)


class HuggingFaceUploader:
    """Handles uploading AI Learning Psychology models to Hugging Face Hub."""
    
    def __init__(self, username: str, token: str, repo_name: str = "ai-learning-psychology-analyzer"):
        """
        Initialize the uploader.
        
        Args:
            username: Your Hugging Face username
            token: Your Hugging Face API token
            repo_name: Repository name (default: ai-learning-psychology-analyzer)
        """
        self.username = username
        self.token = token
        self.repo_name = repo_name
        self.repo_id = f"{username}/{repo_name}"
        
        # Initialize API client
        self.api = HfApi(token=token)
        
        # Model files to upload
        self.model_files = [
            "attention_tracker_model.pkl",
            "attention_tracker_scaler.pkl", 
            "attention_tracker_encoder.pkl",
            "cognitive_load_assessor_model.pkl",
            "cognitive_load_assessor_scaler.pkl",
            "learning_style_detector_model.pkl",
            "learning_style_detector_scaler.pkl",
            "learning_style_detector_encoder.pkl",
            "training_metrics.pkl",
            "training_report.txt"
        ]
        
        # Directories
        self.models_dir = Path("models")
        self.temp_dir = Path("temp_hf_upload")
        
        print(f"🎯 Target repository: {self.repo_id}")
        print(f"📁 Models directory: {self.models_dir}")
        
    def verify_files(self) -> bool:
        """Verify that all required model files exist."""
        print("\n📋 Verifying model files...")
        
        missing_files = []
        existing_files = []
        
        for file_name in self.model_files:
            file_path = self.models_dir / file_name
            if file_path.exists():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                existing_files.append(f"✅ {file_name} ({size_mb:.1f}MB)")
            else:
                missing_files.append(f"❌ {file_name}")
        
        print("\n📊 File Status:")
        for file_info in existing_files:
            print(f"  {file_info}")
        
        if missing_files:
            print("\n⚠️  Missing files:")
            for file_info in missing_files:
                print(f"  {file_info}")
            return False
        
        print(f"\n✅ All {len(existing_files)} model files verified!")
        return True
    
    def create_repository(self) -> bool:
        """Create the repository on Hugging Face Hub."""
        print(f"\n🔧 Creating repository: {self.repo_id}")
        
        try:
            # Check if repository already exists
            try:
                repo_info = self.api.repo_info(self.repo_id)
                print(f"ℹ️  Repository already exists: {repo_info.id}")
                return True
            except Exception:
                pass
            
            # Create new repository
            repo_url = create_repo(
                repo_id=self.repo_id,
                token=self.token,
                repo_type="model",
                private=False,
                exist_ok=True
            )
            
            print(f"✅ Repository created successfully!")
            print(f"🔗 Repository URL: {repo_url}")
            return True
            
        except Exception as e:
            print(f"❌ Error creating repository: {str(e)}")
            return False
    
    def prepare_upload_directory(self) -> bool:
        """Prepare temporary directory with all files for upload."""
        print(f"\n📦 Preparing upload directory: {self.temp_dir}")
        
        try:
            # Clean and create temp directory
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy model files
            for file_name in self.model_files:
                src_path = self.models_dir / file_name
                dst_path = self.temp_dir / file_name
                
                if src_path.exists():
                    shutil.copy2(src_path, dst_path)
                    print(f"  ✅ Copied {file_name}")
                else:
                    print(f"  ⚠️  Skipped {file_name} (not found)")
            
            # Copy model card (README.md)
            readme_src = Path("huggingface_model_card.md")
            readme_dst = self.temp_dir / "README.md"
            
            if readme_src.exists():
                shutil.copy2(readme_src, readme_dst)
                print(f"  ✅ Copied README.md")
            else:
                print(f"  ⚠️  Model card not found at {readme_src}")
            
            # Create requirements.txt for the model
            requirements_content = """# Hugging Face Model Requirements
huggingface-hub>=0.19.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
"""
            with open(self.temp_dir / "requirements.txt", "w") as f:
                f.write(requirements_content)
            print(f"  ✅ Created requirements.txt")
            
            # Create .gitignore
            gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# IDEs
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
"""
            with open(self.temp_dir / ".gitignore", "w") as f:
                f.write(gitignore_content)
            print(f"  ✅ Created .gitignore")
            
            print(f"\n✅ Upload directory prepared with {len(list(self.temp_dir.glob('*')))} files")
            return True
            
        except Exception as e:
            print(f"❌ Error preparing upload directory: {str(e)}")
            return False
    
    def upload_files(self) -> bool:
        """Upload all files to Hugging Face Hub."""
        print(f"\n🚀 Uploading files to {self.repo_id}...")
        
        try:
            uploaded_files = []
            failed_files = []
            
            # Upload each file
            for file_path in self.temp_dir.glob("*"):
                if file_path.is_file():
                    try:
                        print(f"  📤 Uploading {file_path.name}...")
                        
                        upload_file(
                            path_or_fileobj=str(file_path),
                            path_in_repo=file_path.name,
                            repo_id=self.repo_id,
                            token=self.token,
                            commit_message=f"Upload {file_path.name}"
                        )
                        
                        uploaded_files.append(file_path.name)
                        print(f"    ✅ Uploaded {file_path.name}")
                        
                    except Exception as e:
                        failed_files.append(f"{file_path.name}: {str(e)}")
                        print(f"    ❌ Failed to upload {file_path.name}: {str(e)}")
            
            # Summary
            print(f"\n📊 Upload Summary:")
            print(f"  ✅ Successful uploads: {len(uploaded_files)}")
            print(f"  ❌ Failed uploads: {len(failed_files)}")
            
            if uploaded_files:
                print(f"\n✅ Successfully uploaded files:")
                for file_name in uploaded_files:
                    print(f"  • {file_name}")
            
            if failed_files:
                print(f"\n❌ Failed to upload:")
                for error in failed_files:
                    print(f"  • {error}")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Error during upload: {str(e)}")
            return False
    
    def cleanup(self) -> None:
        """Clean up temporary files."""
        print(f"\n🧹 Cleaning up temporary files...")
        
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                print(f"  ✅ Cleaned up {self.temp_dir}")
        except Exception as e:
            print(f"  ⚠️  Error cleaning up: {str(e)}")
    
    def get_repository_info(self) -> Dict:
        """Get information about the uploaded repository."""
        try:
            repo_info = self.api.repo_info(self.repo_id)
            return {
                'id': repo_info.id,
                'url': f"https://huggingface.co/{self.repo_id}",
                'files': len(self.api.list_repo_files(self.repo_id)),
                'created': repo_info.created_at,
                'updated': repo_info.last_modified
            }
        except Exception as e:
            return {'error': str(e)}
    
    def run_upload(self) -> bool:
        """Run the complete upload process."""
        print("🚀 Starting Hugging Face Upload Process")
        print("=" * 50)
        
        try:
            # Step 1: Verify files
            if not self.verify_files():
                print("\n❌ File verification failed. Please ensure all models are trained and saved.")
                return False
            
            # Step 2: Create repository
            if not self.create_repository():
                print("\n❌ Repository creation failed.")
                return False
            
            # Step 3: Prepare upload directory
            if not self.prepare_upload_directory():
                print("\n❌ Upload directory preparation failed.")
                return False
            
            # Step 4: Upload files
            if not self.upload_files():
                print("\n❌ File upload failed.")
                return False
            
            # Step 5: Get repository info
            repo_info = self.get_repository_info()
            
            print("\n" + "=" * 50)
            print("🎉 UPLOAD COMPLETED SUCCESSFULLY!")
            print("=" * 50)
            
            if 'error' not in repo_info:
                print(f"📁 Repository: {repo_info['id']}")
                print(f"🔗 URL: {repo_info['url']}")
                print(f"📊 Files: {repo_info['files']}")
                print(f"🕒 Updated: {repo_info['updated']}")
            
            print("\n🎯 Next Steps:")
            print("  1. Visit your repository on Hugging Face")
            print("  2. Check that all files uploaded correctly")
            print("  3. Update any personal information in the model card")
            print("  4. Test the model downloads with the provided code examples")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Upload process failed: {str(e)}")
            return False
        
        finally:
            # Always cleanup
            self.cleanup()


def main():
    """Main function to handle command line arguments and run upload."""
    parser = argparse.ArgumentParser(
        description="Upload AI Learning Psychology models to Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python upload_to_huggingface.py --username johndoe --token your_token_here
    python upload_to_huggingface.py --username johndoe --repo custom-model-name
    
Environment Variables:
    HUGGINGFACE_TOKEN: Your Hugging Face API token
    HUGGINGFACE_USERNAME: Your Hugging Face username
        """
    )
    
    parser.add_argument(
        "--username",
        type=str,
        default=os.getenv("HUGGINGFACE_USERNAME"),
        help="Your Hugging Face username"
    )
    
    parser.add_argument(
        "--token",
        type=str,
        default=os.getenv("HUGGINGFACE_TOKEN"),
        help="Your Hugging Face API token"
    )
    
    parser.add_argument(
        "--repo",
        type=str,
        default="ai-learning-psychology-analyzer",
        help="Repository name (default: ai-learning-psychology-analyzer)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify files and setup without uploading"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.username:
        print("❌ Error: Username is required. Use --username or set HUGGINGFACE_USERNAME environment variable.")
        sys.exit(1)
    
    if not args.token:
        print("❌ Error: Token is required. Use --token or set HUGGINGFACE_TOKEN environment variable.")
        print("   Get your token from: https://huggingface.co/settings/tokens")
        sys.exit(1)
    
    # Create uploader
    uploader = HuggingFaceUploader(
        username=args.username,
        token=args.token,
        repo_name=args.repo
    )
    
    # Run upload process
    if args.dry_run:
        print("🔍 Running in dry-run mode (no actual upload)")
        success = uploader.verify_files() and uploader.create_repository()
        print(f"\n{'✅ Dry run completed successfully' if success else '❌ Dry run failed'}")
    else:
        success = uploader.run_upload()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 