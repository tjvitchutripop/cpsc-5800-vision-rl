import sys
import os
import zipfile
import re
from pathlib import Path


def extract_file_id(url_or_id: str) -> str:
    """Extract Google Drive file ID from URL or return the ID if already provided."""
    # If it's already just an ID (no slashes or http), return it
    if not ('/' in url_or_id or 'http' in url_or_id):
        return url_or_id
    
    # Try to extract ID from various Google Drive URL formats
    patterns = [
        r'/file/d/([a-zA-Z0-9_-]+)',
        r'id=([a-zA-Z0-9_-]+)',
        r'/d/([a-zA-Z0-9_-]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)
    
    raise ValueError(f"Could not extract file ID from: {url_or_id}")


def download_from_gdrive(file_id: str, output_path: str) -> None:
    """Download file from Google Drive using gdown."""
    try:
        import gdown
    except ImportError:
        print("Error: gdown is not installed.")
        print("Please install it with: pip install gdown")
        sys.exit(1)
    
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"Downloading from Google Drive (ID: {file_id})...")
    gdown.download(url, output_path, quiet=False)


def extract_zip(zip_path: str, extract_to: str) -> None:
    """Extract zip file to specified directory."""
    print(f"Extracting {zip_path} to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete!")


def main():
    url_or_id = "https://drive.google.com/file/d/1le0pnWjZoz0F0ocvD2Db8jugKI92-c5X/view?usp=sharing"
    
    # Extract file ID
    try:
        file_id = extract_file_id(url_or_id)
        print(f"Extracted file ID: {file_id}")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Setup paths
    script_dir = Path(__file__).parent
    models_dir = script_dir / "models"
    zip_path = script_dir / "models.zip"
    
    # Create models directory if it doesn't exist
    models_dir.mkdir(exist_ok=True)
    
    # Download the file
    try:
        download_from_gdrive(file_id, str(zip_path))
    except Exception as e:
        print(f"Error downloading file: {e}")
        sys.exit(1)
    
    # Verify the file was downloaded
    if not zip_path.exists():
        print("Error: Download failed - file not found")
        sys.exit(1)
    
    # Extract the zip file
    try:
        extract_zip(str(zip_path), str(script_dir))
    except Exception as e:
        print(f"Error extracting zip file: {e}")
        sys.exit(1)
    
    # Clean up zip file
    print("Cleaning up...")
    zip_path.unlink()
    
    print("\n✓ Models downloaded and extracted successfully!")
    print(f"Models location: {models_dir}")


if __name__ == "__main__":
    main()
