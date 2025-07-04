#!/usr/bin/env python3
"""
🎯 Sample Real-Structure Dataset Creator
========================================

Creates realistic sample datasets that match the structure of real educational data.
This allows us to continue development while waiting for Kaggle API setup.

The data is more realistic than our previous synthetic data but still allows
immediate development and testing.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta

class SampleRealDataCreator:
    def __init__(self):
        self.data_dir = Path("data")
        self.raw_dir = self.data_dir / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        
        # Seed for reproducibility
        np.random.seed(42)

    def create_student_performance_behavior(self, n_students=1000):
        """Create sample data matching real student performance & behavior dataset structure."""
        print("📊 Creating Student Performance & Behavior sample...")
        
        # Realistic distributions based on real data
        students = []
        
        for i in range(n_students):
            # Demographics
            age = np.random.randint(18, 25)
            gender = np.random.choice(['Male', 'Female'])
            department = np.random.choice(['CS', 'Engineering', 'Business', 'Mathematics'])
            
            # Family background (affects performance)
            parent_education = np.random.choice(['None', 'High School', 'Bachelor', 'Master', 'PhD'], 
                                              p=[0.1, 0.3, 0.4, 0.15, 0.05])
            family_income = np.random.choice(['Low', 'Medium', 'High'], p=[0.3, 0.5, 0.2])
            internet_access = np.random.choice(['Yes', 'No'], p=[0.85, 0.15])
            
            # Behavioral factors
            study_hours = max(1, np.random.normal(15, 5))  # hours per week
            sleep_hours = max(4, np.random.normal(7, 1.5))  # hours per night
            stress_level = np.random.randint(1, 11)  # 1-10 scale
            extracurricular = np.random.choice(['Yes', 'No'], p=[0.6, 0.4])
            
            # Attendance (affects performance)
            attendance_base = 85 if internet_access == 'Yes' else 75
            attendance = max(0, min(100, np.random.normal(attendance_base, 15)))
            
            # Performance metrics (correlated with factors above)
            performance_base = 70
            
            # Adjustments based on factors
            if parent_education in ['Master', 'PhD']:
                performance_base += 10
            elif parent_education in ['None', 'High School']:
                performance_base -= 5
                
            if family_income == 'High':
                performance_base += 5
            elif family_income == 'Low':
                performance_base -= 5
                
            if study_hours > 20:
                performance_base += 8
            elif study_hours < 10:
                performance_base -= 8
                
            if sleep_hours < 6:
                performance_base -= 5
            elif sleep_hours > 8:
                performance_base += 3
                
            if stress_level > 7:
                performance_base -= 5
            elif stress_level < 4:
                performance_base += 3
                
            # Generate correlated scores with realistic noise
            midterm = max(0, min(100, np.random.normal(performance_base, 12)))
            final = max(0, min(100, np.random.normal(performance_base + 2, 10)))
            assignments = max(0, min(100, np.random.normal(performance_base + 5, 8)))
            quizzes = max(0, min(100, np.random.normal(performance_base - 3, 15)))
            participation = max(0, min(10, np.random.normal(performance_base/10, 1)))
            projects = max(0, min(100, np.random.normal(performance_base + 3, 12)))
            
            # Calculate total score (weighted)
            total_score = (0.15 * midterm + 0.25 * final + 0.15 * assignments + 
                          0.10 * quizzes + 0.05 * participation + 0.30 * projects)
            
            # Grade based on total score
            if total_score >= 90:
                grade = 'A'
            elif total_score >= 80:
                grade = 'B'
            elif total_score >= 70:
                grade = 'C'
            elif total_score >= 60:
                grade = 'D'
            else:
                grade = 'F'
            
            student = {
                'Student_ID': f'S{i+1000}',
                'First_Name': np.random.choice(['Ahmed', 'Maria', 'John', 'Sara', 'Omar', 'Emma', 'Ali', 'Liam']),
                'Last_Name': np.random.choice(['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Davis']),
                'Email': f'student{i}@university.com',
                'Gender': gender,
                'Age': age,
                'Department': department,
                'Attendance (%)': round(attendance, 2),
                'Midterm_Score': round(midterm, 2),
                'Final_Score': round(final, 2),
                'Assignments_Avg': round(assignments, 2),
                'Quizzes_Avg': round(quizzes, 2),
                'Participation_Score': round(participation, 2),
                'Projects_Score': round(projects, 2),
                'Total_Score': round(total_score, 2),
                'Grade': grade,
                'Study_Hours_per_Week': round(study_hours, 1),
                'Extracurricular_Activities': extracurricular,
                'Internet_Access_at_Home': internet_access,
                'Parent_Education_Level': parent_education,
                'Family_Income_Level': family_income,
                'Stress_Level (1-10)': stress_level,
                'Sleep_Hours_per_Night': round(sleep_hours, 1)
            }
            
            students.append(student)
        
        df = pd.DataFrame(students)
        return df

    def create_student_learning_behavior(self, n_sessions=800):
        """Create sample data matching learning behavior dataset structure."""
        print("🧠 Creating Student Learning Behavior sample...")
        
        sessions = []
        
        for i in range(n_sessions):
            # Session info
            session_id = i + 1
            student_id = np.random.randint(100, 300)
            
            # Physiological data (realistic ranges)
            hrv = np.random.normal(0, 1)  # Normalized
            skin_temp = np.random.normal(0, 1)  # Normalized
            
            # Emotional expressions (0-1 scale)
            base_mood = np.random.random()
            expression_joy = max(0, min(1, np.random.normal(base_mood, 0.2)))
            expression_confusion = max(0, min(1, np.random.normal(1-base_mood, 0.3)))
            
            # Activity level
            steps = np.random.randint(100, 1000)
            
            # Emotion based on expressions
            if expression_joy > 0.7:
                emotion = 'Happiness'
            elif expression_confusion > 0.6:
                emotion = 'Confusion'
            elif expression_joy > 0.4:
                emotion = 'Interest'
            else:
                emotion = 'Boredom'
                
            # Engagement level (1-5) correlated with emotion
            if emotion == 'Happiness':
                engagement = np.random.choice([4, 5], p=[0.3, 0.7])
            elif emotion == 'Interest':
                engagement = np.random.choice([3, 4, 5], p=[0.2, 0.5, 0.3])
            elif emotion == 'Confusion':
                engagement = np.random.choice([2, 3, 4], p=[0.4, 0.4, 0.2])
            else:  # Boredom
                engagement = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
            
            # Session details
            session_duration = np.random.randint(15, 61)  # 15-60 minutes
            learning_phase = np.random.choice(['Introduction', 'Practice', 'Conclusion'])
            
            # Learning outcome based on engagement
            if engagement >= 4:
                outcome = 'Successful'
            elif engagement >= 2:
                outcome = 'Partially Successful'
            else:
                outcome = 'Unsuccessful'
            
            # Frequency features (simulated from transforms)
            hrv_freq = np.random.normal(0, 0.5)
            temp_freq = np.random.normal(0, 0.3)
            
            # Labels for ML
            emotion_label = {'Interest': 0, 'Boredom': 1, 'Confusion': 2, 'Happiness': 3}[emotion]
            phase_label = {'Introduction': 0, 'Practice': 1, 'Conclusion': 2}[learning_phase]
            
            session = {
                'Session_ID': session_id,
                'Student_ID': student_id,
                'HRV': hrv,
                'Skin_Temperature': skin_temp,
                'Expression_Joy': expression_joy,
                'Expression_Confusion': expression_confusion,
                'Steps': steps,
                'Emotion': emotion,
                'Engagement_Level': engagement,
                'Session_Duration': session_duration,
                'Learning_Phase': learning_phase,
                'Start_Time': datetime.now() - timedelta(days=np.random.randint(0, 30)),
                'End_Time': datetime.now() - timedelta(days=np.random.randint(0, 30)),
                'Learning_Outcome': outcome,
                'HRV_Frequency_Feature': hrv_freq,
                'Skin_Temperature_Frequency_Feature': temp_freq,
                'Emotion_Label': emotion_label,
                'Learning_Phase_Label': phase_label
            }
            
            sessions.append(session)
        
        df = pd.DataFrame(sessions)
        return df

    def create_uci_student_performance(self, n_students=500):
        """Create sample data matching UCI student performance structure."""
        print("🏫 Creating UCI Student Performance sample...")
        
        students = []
        
        for i in range(n_students):
            # Demographics
            school = np.random.choice(['GP', 'MS'])
            sex = np.random.choice(['M', 'F'])
            age = np.random.randint(15, 23)
            address = np.random.choice(['U', 'R'], p=[0.7, 0.3])
            
            # Family
            famsize = np.random.choice(['LE3', 'GT3'], p=[0.3, 0.7])
            pstatus = np.random.choice(['T', 'A'], p=[0.85, 0.15])
            medu = np.random.choice([0, 1, 2, 3, 4], p=[0.05, 0.15, 0.25, 0.35, 0.2])
            fedu = np.random.choice([0, 1, 2, 3, 4], p=[0.05, 0.15, 0.25, 0.35, 0.2])
            
            # School related
            studytime = np.random.choice([1, 2, 3, 4], p=[0.2, 0.4, 0.3, 0.1])
            failures = np.random.choice([0, 1, 2, 3], p=[0.7, 0.2, 0.07, 0.03])
            
            # Social
            freetime = np.random.randint(1, 6)
            goout = np.random.randint(1, 6)
            health = np.random.randint(1, 6)
            absences = max(0, int(np.random.exponential(3)))
            
            # Performance (correlated with factors)
            performance_base = 12
            
            # Education level effect
            performance_base += (medu + fedu) * 0.5
            
            # Study time effect
            performance_base += studytime * 1.5
            
            # Failures effect
            performance_base -= failures * 2
            
            # Health effect
            performance_base += (health - 3) * 0.5
            
            # Absences effect
            performance_base -= absences * 0.1
            
            # Generate grades with noise
            g1 = max(0, min(20, np.random.normal(performance_base, 3)))
            g2 = max(0, min(20, np.random.normal(performance_base + 0.5, 2.5)))
            g3 = max(0, min(20, np.random.normal(performance_base + 1, 2)))
            
            student = {
                'school': school,
                'sex': sex,
                'age': age,
                'address': address,
                'famsize': famsize,
                'Pstatus': pstatus,
                'Medu': medu,
                'Fedu': fedu,
                'Mjob': np.random.choice(['teacher', 'health', 'services', 'at_home', 'other']),
                'Fjob': np.random.choice(['teacher', 'health', 'services', 'at_home', 'other']),
                'reason': np.random.choice(['home', 'reputation', 'course', 'other']),
                'guardian': np.random.choice(['mother', 'father', 'other']),
                'traveltime': np.random.choice([1, 2, 3, 4], p=[0.4, 0.35, 0.2, 0.05]),
                'studytime': studytime,
                'failures': failures,
                'schoolsup': np.random.choice(['yes', 'no'], p=[0.2, 0.8]),
                'famsup': np.random.choice(['yes', 'no'], p=[0.6, 0.4]),
                'paid': np.random.choice(['yes', 'no'], p=[0.3, 0.7]),
                'activities': np.random.choice(['yes', 'no'], p=[0.5, 0.5]),
                'nursery': np.random.choice(['yes', 'no'], p=[0.8, 0.2]),
                'higher': np.random.choice(['yes', 'no'], p=[0.9, 0.1]),
                'internet': np.random.choice(['yes', 'no'], p=[0.8, 0.2]),
                'romantic': np.random.choice(['yes', 'no'], p=[0.3, 0.7]),
                'famrel': np.random.randint(1, 6),
                'freetime': freetime,
                'goout': goout,
                'Dalc': np.random.randint(1, 6),
                'Walc': np.random.randint(1, 6),
                'health': health,
                'absences': absences,
                'G1': round(g1, 0),
                'G2': round(g2, 0),
                'G3': round(g3, 0)
            }
            
            students.append(student)
        
        df = pd.DataFrame(students)
        return df

    def save_datasets(self):
        """Create and save all sample datasets."""
        print("🏗️  Creating Sample Real-Structure Datasets")
        print("="*50)
        
        datasets_info = {
            "creation_date": datetime.now().isoformat(),
            "data_quality": "sample_real_structure",
            "description": "Sample datasets matching real educational data structures",
            "datasets": {}
        }
        
        # Create datasets
        datasets = {
            "student_performance_behavior": self.create_student_performance_behavior(1000),
            "student_learning_behavior": self.create_student_learning_behavior(800),
            "uci_student_performance": self.create_uci_student_performance(500)
        }
        
        # Save datasets
        for name, df in datasets.items():
            file_path = self.raw_dir / f"{name}.csv"
            df.to_csv(file_path, index=False)
            
            datasets_info["datasets"][name] = {
                "file": file_path.name,
                "rows": len(df),
                "columns": len(df.columns),
                "size_mb": round(file_path.stat().st_size / 1024 / 1024, 2),
                "description": f"Sample {name} data with realistic structure"
            }
            
            print(f"✅ Created {name}: {len(df)} rows, {len(df.columns)} columns")
        
        # Save info file
        info_file = self.data_dir / "dataset_info.json"
        with open(info_file, 'w') as f:
            json.dump(datasets_info, f, indent=2)
        
        # Display summary
        total_rows = sum(info["rows"] for info in datasets_info["datasets"].values())
        total_size = sum(info["size_mb"] for info in datasets_info["datasets"].values())
        
        print("\n" + "="*50)
        print("🎉 SAMPLE DATASETS CREATED")
        print("="*50)
        print(f"📊 Total datasets: {len(datasets)}")
        print(f"📈 Total records: {total_rows:,}")
        print(f"💾 Total size: {total_size:.1f} MB")
        print(f"📁 Location: {self.raw_dir}")
        print("\n🚀 Ready for real data preprocessing pipeline!")
        print("💡 These match real dataset structures for immediate development")

if __name__ == "__main__":
    creator = SampleRealDataCreator()
    creator.save_datasets() 