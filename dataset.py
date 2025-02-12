# Data retrieved from https://github.com/sophos/SOREL-20M

import sqlite3
import subprocess
import os
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

class Downloader:
    def __init__(self, metadb_path, download_folder = "data/binaries", n_samples=10000, num_threads = 10):
        self.download_folder = download_folder
        self.total_downloads = 100000
        self.num_threads = num_threads
        os.makedirs(download_folder, exist_ok=True)

        conn = sqlite3.connect(metadb_path)
        cur = conn.cursor()

        self.families = ["adware", "flooder", "ransomware", "dropper", "spyware", "packed",
            "crypto_miner", "file_infector", "installer", "worm", "downloader"]
        
        self.file_paths = []

        for family in self.families:
            query = f"""
            SELECT sha256 FROM meta
            WHERE {family} = 1 AND is_malware = 1
            ORDER BY RANDOM()
            LIMIT ?
            """
            results = cur.execute(query, (n_samples,)).fetchall()
            self.file_paths.extend([(row[0], family) for row in results])

        conn.close()

    def download_file(self, sha256, family, download_counts, failed_downloads):
        """Downloads a single file from S3 and updates counters."""
        s3_url = f"s3://sorel-20m/09-DEC-2020/binaries/{sha256}"
        local_path = os.path.join(self.download_folder, f"{family}_{sha256}")
        
        result = subprocess.run(
            ["aws", "s3", "cp", s3_url, local_path, "--no-sign-request"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        
        if result.returncode == 0:  # Successful download
            download_counts[family] += 1
            return True
        else:  # Failed download
            failed_downloads += 1
            return False

    def download_files(self):
        print(f"Starting download of {self.total_downloads} files with {self.num_threads} threads...")

        download_counts = defaultdict(int)  # Count downloads per malware family
        failed_downloads = 0  # Track failed downloads

        # Run downloads in parallel
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [executor.submit(self.download_file, sha256, family, download_counts, failed_downloads)
                       for sha256, family in self.file_paths]

            for _ in tqdm(futures, desc="Downloading", unit="file"):
                _.result()  # Ensure all downloads complete

        # Print download summary
        print("\n Download Summary:")
        for family in self.families:
            print(f"   {family}: {download_counts[family]} files downloaded")
        print(f"   Total Downloads: {sum(download_counts.values())}\n")

        # Save failed downloads to a log file
        if failed_downloads:
            print(f" {len(failed_downloads)} downloads failed")


downloader = Downloader("data/meta.db", download_folder = "data/binaries")
downloader.download_files()
