"""
ML-Based Crop Recommendation Service
Uses trained ML models and crop dataset CSV to generate detailed crop recommendations
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple


class MLCropRecommendationService:
    """Generate crop recommendations using ML models and CSV dataset"""
    
    def __init__(self, csv_path='crop_recommendation_dataset.csv'):
        self.csv_path = csv_path
        self.crop_data = None
        self.crop_stats = None
        self._load_dataset()
    
    def _load_dataset(self):
        """Load and analyze the crop recommendation dataset"""
        try:
            if not os.path.exists(self.csv_path):
                print(f"[!] Dataset not found: {self.csv_path}")
                return False
            
            self.crop_data = pd.read_csv(self.csv_path)
            print(f"[+] Loaded {len(self.crop_data)} records from crop dataset")
            print(f"[+] Crops available: {self.crop_data['crop'].unique().tolist()}")
            
            # Calculate statistics per crop
            self.crop_stats = {}
            for crop in self.crop_data['crop'].unique():
                crop_records = self.crop_data[self.crop_data['crop'] == crop]
                self.crop_stats[crop] = {
                    'count': len(crop_records),
                    'N': {'mean': crop_records['N'].mean(), 'min': crop_records['N'].min(), 'max': crop_records['N'].max()},
                    'P': {'mean': crop_records['P'].mean(), 'min': crop_records['P'].min(), 'max': crop_records['P'].max()},
                    'K': {'mean': crop_records['K'].mean(), 'min': crop_records['K'].min(), 'max': crop_records['K'].max()},
                    'temperature': {'mean': crop_records['temperature'].mean(), 'min': crop_records['temperature'].min(), 'max': crop_records['temperature'].max()},
                    'humidity': {'mean': crop_records['humidity'].mean(), 'min': crop_records['humidity'].min(), 'max': crop_records['humidity'].max()},
                    'pH': {'mean': crop_records['pH'].mean(), 'min': crop_records['pH'].min(), 'max': crop_records['pH'].max()},
                    'rainfall': {'mean': crop_records['rainfall'].mean(), 'min': crop_records['rainfall'].min(), 'max': crop_records['rainfall'].max()}
                }
            
            return True
        
        except Exception as e:
            print(f"[!] Error loading dataset: {e}")
            return False
    
    def get_crop_details(self, crop_name: str) -> Dict:
        """Get detailed information about a crop from the dataset"""
        if not self.crop_stats or crop_name not in self.crop_stats:
            return None
        
        stats = self.crop_stats[crop_name]
        
        return {
            'name': crop_name,
            'dataset_samples': stats['count'],
            'nitrogen_requirement': {
                'mean': round(stats['N']['mean'], 2),
                'range': f"{round(stats['N']['min'], 2)}-{round(stats['N']['max'], 2)} kg/ha"
            },
            'phosphorus_requirement': {
                'mean': round(stats['P']['mean'], 2),
                'range': f"{round(stats['P']['min'], 2)}-{round(stats['P']['max'], 2)} kg/ha"
            },
            'potassium_requirement': {
                'mean': round(stats['K']['mean'], 2),
                'range': f"{round(stats['K']['min'], 2)}-{round(stats['K']['max'], 2)} kg/ha"
            },
            'temperature': {
                'optimal': round(stats['temperature']['mean'], 2),
                'range': f"{round(stats['temperature']['min'], 2)}-{round(stats['temperature']['max'], 2)}°C"
            },
            'humidity': {
                'optimal': round(stats['humidity']['mean'], 2),
                'range': f"{round(stats['humidity']['min'], 2)}-{round(stats['humidity']['max'], 2)}%"
            },
            'pH': {
                'optimal': round(stats['pH']['mean'], 2),
                'range': f"{round(stats['pH']['min'], 2)}-{round(stats['pH']['max'], 2)}"
            },
            'rainfall': {
                'optimal': round(stats['rainfall']['mean'], 2),
                'range': f"{round(stats['rainfall']['min'], 2)}-{round(stats['rainfall']['max'], 2)} mm"
            }
        }
    
    def calculate_suitability_explanation(self, crop_name: str, field_data: Dict) -> str:
        """Generate explanation of why a crop is suitable for the field"""
        if crop_name not in self.crop_stats:
            return "Suitable for your field conditions."
        
        stats = self.crop_stats[crop_name]
        
        # Compare field conditions with crop requirements
        temp_optimal = stats['temperature']['mean']
        humidity_optimal = stats['humidity']['mean']
        rainfall_optimal = stats['rainfall']['mean']
        n_optimal = stats['N']['mean']
        
        field_temp = float(field_data.get('temperature', 25))
        field_humidity = float(field_data.get('humidity', 65))
        field_rainfall = float(field_data.get('rainfall', 500))
        field_n = float(field_data.get('nitrogen', 75))
        
        explanation = f"{crop_name} is well-suited for your field because:\n\n"
        
        # Temperature analysis
        temp_diff = abs(field_temp - temp_optimal)
        if temp_diff < 2:
            explanation += f"🌡️ Temperature ({field_temp}°C) matches optimal requirements ({temp_optimal:.1f}°C)\n"
        elif temp_diff < 5:
            explanation += f"🌡️ Temperature ({field_temp}°C) is quite suitable (optimal: {temp_optimal:.1f}°C)\n"
        else:
            explanation += f"🌡️ Temperature ({field_temp}°C) is acceptable (optimal: {temp_optimal:.1f}°C)\n"
        
        # Humidity analysis
        humidity_diff = abs(field_humidity - humidity_optimal)
        if humidity_diff < 5:
            explanation += f"💧 Humidity ({field_humidity}%) matches optimal levels ({humidity_optimal:.1f}%)\n"
        elif humidity_diff < 15:
            explanation += f"💧 Humidity ({field_humidity}%) is suitable (optimal: {humidity_optimal:.1f}%)\n"
        else:
            explanation += f"💧 Humidity ({field_humidity}%) requires management (optimal: {humidity_optimal:.1f}%)\n"
        
        # Rainfall analysis
        rainfall_diff = abs(field_rainfall - rainfall_optimal)
        if rainfall_diff < 100:
            explanation += f"🌧️ Rainfall ({field_rainfall:.0f}mm) is well-suited (optimal: {rainfall_optimal:.0f}mm)\n"
        else:
            explanation += f"🌧️ Rainfall ({field_rainfall:.0f}mm) is manageable (optimal: {rainfall_optimal:.0f}mm)\n"
        
        # Soil nutrients analysis
        if field_n >= n_optimal * 0.8:
            explanation += f"🌱 Nitrogen availability ({field_n:.0f} kg/ha) meets crop needs ({n_optimal:.0f} kg/ha)"
        else:
            explanation += f"🌱 Nitrogen levels ({field_n:.0f} kg/ha) are below optimal ({n_optimal:.0f} kg/ha) but manageable"
        
        return explanation
    
    def generate_ml_recommendations(self, top_crops: List[Tuple[str, float]], field_data: Dict) -> Dict:
        """Generate ML-based recommendations for top 3 crops"""
        
        recommendations = {
            'recommended_crops': []
        }
        
        # Add top 3 crop recommendations only
        rank_symbols = ['🥇', '🥈', '🥉']
        for idx, (crop_name, confidence) in enumerate(top_crops[:3]):
            crop_details = self.get_crop_details(crop_name)
            
            if crop_details:
                recommendation = {
                    'name': crop_name,
                    'rank': idx + 1,
                    'confidence': round(confidence, 2),
                    'variety': f"Select varieties suited to {field_data.get('location', 'your region')}",
                    'why_suitable': self.calculate_suitability_explanation(crop_name, field_data),
                    'market_potential': f"Strong market demand for {crop_name} with consistent pricing; Potential yield: {crop_details['rainfall']['optimal']/5:.0f} quintals/hectare",
                    'growing_tips': f"Use {crop_details['nitrogen_requirement']['mean']:.0f} kg/ha N, {crop_details['phosphorus_requirement']['mean']:.0f} kg/ha P, {crop_details['potassium_requirement']['mean']:.0f} kg/ha K; maintain soil pH {crop_details['pH']['optimal']:.1f}"
                }
                recommendations['recommended_crops'].append(recommendation)
        
        return recommendations
