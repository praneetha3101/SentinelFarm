"""
API endpoint for soil property prediction from satellite data
Add this to your Flask backend (app.py)
"""

from flask import Blueprint, request, jsonify
from soil_prediction_service import predict_soil_from_satellite, soil_model
import json

soil_bp = Blueprint('soil', __name__, url_prefix='/api/soil')

@soil_bp.route('/predict', methods=['POST'])
def predict_soil():
    """
    Predict soil properties from satellite data over drawn field area
    
    Request body:
    {
        "coordinates": [[lat, lng], [lat, lng], ...],
        "start_date": "2024-01-01",
        "end_date": "2024-12-31"
    }
    
    Returns soil pH, type, moisture, and recommendations
    """
    try:
        data = request.json
        coordinates = data.get('coordinates')
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        
        if not coordinates:
            return jsonify({'error': 'Coordinates required'}), 400
        
        # Call prediction service
        result = predict_soil_from_satellite(coordinates, start_date, end_date)
        
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@soil_bp.route('/indices', methods=['POST'])
def calculate_indices():
    """
    Calculate satellite indices from spectral band values
    
    Request body:
    {
        "red": 0.1,
        "nir": 0.3,
        "swir1": 0.15,
        "blue": 0.05
    }
    """
    try:
        data = request.json
        from soil_prediction_service import calculate_satellite_indices
        
        indices = calculate_satellite_indices(
            data.get('red'),
            data.get('nir'),
            data.get('swir1'),
            data.get('blue')
        )
        
        # Predict soil from indices
        soil_properties = soil_model.predict_soil_properties(
            indices['ndvi'],
            indices['ndbi'],
            indices['ndmi'],
            indices['savi']
        )
        
        return jsonify({
            'status': 'success',
            'indices': indices,
            'soil_properties': soil_properties
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@soil_bp.route('/ndvi-to-properties', methods=['POST'])
def ndvi_to_properties():
    """
    Quick endpoint to predict soil properties from just NDVI value
    (for quick testing without full satellite data)
    
    Request body:
    {
        "ndvi": 0.65
    }
    """
    try:
        data = request.json
        ndvi = data.get('ndvi')
        
        # Estimate other indices from NDVI
        ndbi = -0.15 if ndvi > 0.5 else 0.0  # Rough estimation
        ndmi = 0.35 if ndvi > 0.5 else 0.2   # Rough estimation
        savi = ndvi * 0.7                     # Rough estimation
        
        soil_properties = soil_model.predict_soil_properties(ndvi, ndbi, ndmi, savi)
        
        return jsonify({
            'status': 'success',
            'input_ndvi': ndvi,
            'soil_properties': soil_properties
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
