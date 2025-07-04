#!/usr/bin/env python3
"""
🏗️ Real Dataset Downloader for AI Learning Psychology Analyzer
==============================================================

This script downloads high-quality real datasets for training our AI models.
Much better than synthetic data!

Usage:
    python download_datasets.py

Datasets to download:
1. Student Performance & Behavior Dataset (5k records) 
2. Student Learning Behavior Dataset (1.2k records)
3. UCI Student Performance Dataset (395 records)
4. xAPI Educational Data (480 records)
"""

import os
import sys
import pandas as pd
import requests
import zipfile
from pathlib import Path
import json

class DatasetDownloader:
    def __init__(self):
        self.data_dir = Path("data")
        self.raw_dir = self.data_dir / "raw"
        self.external_dir = self.data_dir / "external"
        
        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.external_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset configurations
        self.datasets = {
            "student_performance_behavior": {
                "kaggle_dataset": "mahmoudelhemaly/students-grading-dataset", 
                "description": "5,000 real student records with behavior data",
                "target_file": "Students Performance Dataset.csv",
                "priority": 1
            },
            "student_learning_behavior": {
                "kaggle_dataset": "ziya07/student-learning-behavior-dataset",
                "description": "1,200 records with physiological and emotional data", 
                "target_file": "simulated_student_learning_data.csv",
                "priority": 2
            },
            "uci_student_performance": {
                "kaggle_dataset": "dskagglemt/student-performance-data-set",
                "description": "UCI's classic student performance dataset",
                "target_file": "student-mat.csv", 
                "priority": 3
            },
            "xapi_educational": {
                "kaggle_dataset": "aljarah/xAPI-Edu-Data",
                "description": "Learning management system behavioral data",
                "target_file": "xAPI-Edu-Data.csv",
                "priority": 4
            }
        }

    def check_kaggle_setup(self):
        """Check if Kaggle API is properly configured."""
        print("🔍 Checking Kaggle API setup...")
        
        # Check if kaggle command exists
        try:
            result = os.system("kaggle --version > /dev/null 2>&1")
            if result != 0:
                print("❌ Kaggle command not found")
                return False
        except:
            print("❌ Kaggle command not available")
            return False
            
        # Check for API credentials  
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_json = kaggle_dir / "kaggle.json"
        
        if not kaggle_json.exists():
            print("⚠️  Kaggle API credentials not found!")
            print("📋 Setup instructions:")
            print("1. Go to https://www.kaggle.com/account")
            print("2. Click 'Create New API Token'")
            print("3. Save kaggle.json to ~/.kaggle/")
            print("4. Run: chmod 600 ~/.kaggle/kaggle.json")
            return False
            
        print("✅ Kaggle API is configured!")
        return True

    def download_kaggle_dataset(self, dataset_info, dataset_name):
        """Download a dataset from Kaggle."""
        print(f"\n📥 Downloading {dataset_name}...")
        print(f"📄 {dataset_info['description']}")
        
        try:
            # Download using Kaggle API
            cmd = f"kaggle datasets download -d {dataset_info['kaggle_dataset']} -p {self.external_dir} --unzip"
            result = os.system(cmd)
            
            if result == 0:
                print(f"✅ Downloaded {dataset_name}")
                return True
            else:
                print(f"❌ Failed to download {dataset_name}")
                return False
                
        except Exception as e:
            print(f"❌ Error downloading {dataset_name}: {str(e)}")
            return False

    def download_manual_fallback(self):
        """Provide manual download instructions if Kaggle API fails."""
        print("\n🔄 Kaggle API not available - Manual download instructions:")
        print("="*60)
        
        for name, info in self.datasets.items():
            print(f"\n📊 {name.upper()}:")
            print(f"   URL: https://www.kaggle.com/datasets/{info['kaggle_dataset']}")
            print(f"   File: {info['target_file']}")
            print(f"   Save to: {self.external_dir}")
            
        print(f"\n💾 After downloading, place all CSV files in: {self.external_dir}")

    def move_and_organize_data(self):
        """Move downloaded data to proper structure."""
        print("\n📁 Organizing downloaded data...")
        
        moved_files = []
        for name, info in self.datasets.items():
            # Look for the target file in external directory
            target_file = info['target_file']
            source_path = None
            
            # Search for the file (case insensitive)
            for file in self.external_dir.rglob("*.csv"):
                if file.name.lower() == target_file.lower():
                    source_path = file
                    break
                    
            if source_path and source_path.exists():
                # Move to raw directory with standardized name
                dest_path = self.raw_dir / f"{name}.csv"
                source_path.rename(dest_path)
                moved_files.append((name, dest_path))
                print(f"✅ Moved {name} → {dest_path}")
            else:
                print(f"❌ File not found: {target_file}")
                
        return moved_files

    def create_dataset_info(self, moved_files):
        """Create dataset information file."""
        info = {
            "download_date": pd.Timestamp.now().isoformat(),
            "datasets": {},
            "total_files": len(moved_files),
            "data_quality": "real_world"
        }
        
        for name, file_path in moved_files:
            try:
                df = pd.read_csv(file_path)
                info["datasets"][name] = {
                    "file": file_path.name,
                    "rows": len(df),
                    "columns": len(df.columns),
                    "size_mb": round(file_path.stat().st_size / 1024 / 1024, 2),
                    "description": self.datasets[name]["description"],
                    "priority": self.datasets[name]["priority"]
                }
            except Exception as e:
                print(f"⚠️  Could not analyze {name}: {str(e)}")
                
        # Save info
        info_file = self.data_dir / "dataset_info.json"
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=2)
            
        print(f"📄 Dataset info saved to {info_file}")
        return info

    def display_summary(self, info):
        """Display download summary."""
        print("\n" + "="*60)
        print("🎉 DATASET DOWNLOAD SUMMARY")
        print("="*60)
        
        total_rows = sum(ds.get('rows', 0) for ds in info['datasets'].values())
        total_size = sum(ds.get('size_mb', 0) for ds in info['datasets'].values())
        
        print(f"📊 Total datasets: {info['total_files']}")
        print(f"📈 Total records: {total_rows:,}")
        print(f"💾 Total size: {total_size:.1f} MB")
        print(f"🎯 Data quality: {info['data_quality']}")
        
        print("\n📋 Individual datasets:")
        for name, ds_info in info['datasets'].items():
            print(f"  • {name}: {ds_info['rows']:,} rows, {ds_info['columns']} cols ({ds_info['size_mb']} MB)")
            
        print(f"\n📁 Data location: {self.raw_dir}")
        print("🚀 Ready for preprocessing and model training!")

    def run(self):
        """Main download process."""
        print("🏗️  AI Learning Psychology Analyzer - Real Dataset Downloader")
        print("="*65)
        
        # Check Kaggle setup
        if self.check_kaggle_setup():
            # Download datasets
            successful_downloads = 0
            for name, info in self.datasets.items():
                if self.download_kaggle_dataset(info, name):
                    successful_downloads += 1
                    
            if successful_downloads == 0:
                self.download_manual_fallback()
                return
        else:
            self.download_manual_fallback()
            return
            
        # Organize data
        moved_files = self.move_and_organize_data()
        
        if moved_files:
            # Create info file and summary
            info = self.create_dataset_info(moved_files)
            self.display_summary(info)
        else:
            print("❌ No datasets were successfully downloaded and organized")

if __name__ == "__main__":
    print("🎓 Downloading Real Educational Datasets...")
    print("💪 Much better than synthetic data!")
    print()
    
    downloader = DatasetDownloader()
    downloader.run() 