import os
import re
import time
from huggingface_hub import HfApi
from camoufox.sync_api import Camoufox

MAX_RUNTIME_SECONDS = 340 * 60  # Exit cleanly at 5h 40m before GitHub's 6h cutoff

def get_next_shard_num(api: HfApi, repo_id: str) -> int:
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        nums = [
            int(m.group(1))
            for f in files
            if (m := re.search(r"shard-(\d+)\.tar$", f))
        ]
        return max(nums, default=0) + 1
    except Exception as e:
        print(f"Could not list repo files ({e}); starting at shard 1.", flush=True)
        return 1

def main():
    hf_token = os.environ["HF_TOKEN"]
    repo_id = os.environ["HF_REPO_ID"]
    perchance_url = os.environ["PERCHANCE_URL"]

    api = HfApi(token=hf_token)
    next_shard = get_next_shard_num(api, repo_id)
    print(f"Target dataset: {repo_id} | Next shard index: {next_shard:03d}", flush=True)

    start_time = time.time()
    os.makedirs("/tmp/shards", exist_ok=True)

    with Camoufox(headless=False) as browser:
        page = browser.new_page()
        page.on("console", lambda msg: print(f"[PAGE] {msg.text}", flush=True))

        print(f"Opening {perchance_url}...", flush=True)
        page.goto(perchance_url, wait_until="networkidle", timeout=60000)

        # Locate #startBtn either directly on the page or inside Perchance's #outputIframeEl
        if page.locator("#startBtn").count() > 0:
            page.locator("#startBtn").click()
        else:
            frame = page.frame_locator("#outputIframeEl")
            frame.locator("#startBtn").wait_for(state="visible", timeout=30000)
            frame.locator("#startBtn").click()

        print("Clicked #startBtn. Waiting for 100-image .tar shard downloads...", flush=True)

        while (time.time() - start_time) < MAX_RUNTIME_SECONDS:
            remaining_ms = int((MAX_RUNTIME_SECONDS - (time.time() - start_time)) * 1000)
            if remaining_ms <= 0:
                break
            try:
                with page.expect_download(timeout=remaining_ms) as dl_info:
                    pass
                download = dl_info.value
                shard_name = f"shard-{next_shard:03d}.tar"
                local_path = f"/tmp/shards/{shard_name}"
                download.save_as(local_path)

                print(f"Uploading {shard_name} to {repo_id}...", flush=True)
                api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=shard_name,
                    repo_id=repo_id,
                    repo_type="dataset",
                    commit_message=f"Add {shard_name} (100 images)",
                )
                os.remove(local_path)
                print(f"Uploaded and cleaned up {shard_name}.", flush=True)
                next_shard += 1
            except Exception as e:
                if (time.time() - start_time) >= MAX_RUNTIME_SECONDS:
                    break
                print(f"Download wait interrupted: {e}", flush=True)
                time.sleep(5)

        print("Shift complete (5h 40m). Exiting cleanly for next scheduled run.", flush=True)

if __name__ == "__main__":
    main()
