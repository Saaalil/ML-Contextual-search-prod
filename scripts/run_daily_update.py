import argparse
import subprocess
import sys
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parents[1]
META_PATH = ROOT_DIR / "data" / "index" / "metadata.jsonl"

def main():
    parser = argparse.ArgumentParser(description="Run the daily embedding update workflow.")
    parser.add_argument("--batch-size", type=int, default=1400, help="Number of new images to process today (default 1400 to stay under 1500 API limit).")
    parser.add_argument("--push", action="store_true", default=True, help="Automatically push to GitHub when done.")
    args = parser.parse_args()

    print("=== StyleGraph Daily Update Script ===")
    
    # 1. Figure out how many images are currently processed
    current_count = 0
    if META_PATH.exists():
        with open(META_PATH, "r", encoding="utf-8") as f:
            current_count = sum(1 for line in f if line.strip())
            
    print(f"📊 Current database size: {current_count} images.")
    
    new_target = current_count + args.batch_size
    print(f"🎯 Target size for today: {new_target} images.")
    
    # 2. Prepare the new subset of images
    print("\n[1/3] Extracting images from DeepFashion2...")
    subprocess.run([
        sys.executable, "-m", "scripts.prepare_demo_subset",
        "--limit", str(new_target)
    ], cwd=ROOT_DIR, check=True)
    
    # 3. Build the Gemini embeddings
    print(f"\n[2/3] Generating {args.batch_size} new embeddings using Gemini...")
    try:
        subprocess.run([
            sys.executable, "-m", "stylegraph.model.build_gemini_index",
            "--resume", "--limit", str(new_target)
        ], cwd=ROOT_DIR, check=True)
    except subprocess.CalledProcessError:
        print("\n❌ Indexing failed! (You likely hit your daily API quota or lost internet).")
        print("Wait 24 hours and run this script again.")
        sys.exit(1)

    # 4. Git Push
    if args.push:
        print("\n[3/3] Uploading database and images to GitHub...")
        try:
            subprocess.run(["git", "add", "data/"], cwd=ROOT_DIR, check=True)
            
            # Check if there's anything to commit
            status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT_DIR, capture_output=True, text=True)
            if "data/" in status.stdout or "data\\" in status.stdout or status.stdout.strip():
                subprocess.run(["git", "commit", "-m", f"Automated daily DB update: Added {args.batch_size} embeddings. Total: {new_target}"], cwd=ROOT_DIR, check=True)
                subprocess.run(["git", "push"], cwd=ROOT_DIR, check=True)
                print("✅ Successfully pushed to GitHub!")
            else:
                print("⚠️ Nothing to commit (database is already up to date).")
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to push to GitHub: {e}")
            sys.exit(1)

    print(f"\n🎉 Daily workflow complete! You now have {new_target} items in your database.")

if __name__ == "__main__":
    main()
