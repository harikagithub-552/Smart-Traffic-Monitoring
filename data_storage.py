import json
import os
from datetime import datetime
from typing import List, Dict, Any

class PredictionStorage:
    def __init__(self, storage_file: str = "predictions.json"):
        self.storage_file = storage_file
        self.data = self._load_data()
    
    def _load_data(self) -> Dict[str, Any]:
        """Load existing data from file or create new structure"""
        if os.path.exists(self.storage_file):
            try:
                with open(self.storage_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        
        return {
            "predictions": [],
            "statistics": {
                "total_vehicles": 0,
                "total_images": 0,
                "lane_totals": [0, 0, 0, 0]
            }
        }
    
    def _save_data(self):
        """Save data to file"""
        with open(self.storage_file, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def add_prediction(self, lanes: List[int], image_filename: str, signal_output: str) -> Dict[str, Any]:
        """Add a new prediction result"""
        timestamp = datetime.now().isoformat()
        total_vehicles = sum(lanes)
        winning_lane = lanes.index(max(lanes)) if any(lanes) else None
        
        prediction = {
            "id": len(self.data["predictions"]) + 1,
            "timestamp": timestamp,
            "image_filename": image_filename,
            "lanes": lanes,
            "total_vehicles": total_vehicles,
            "winning_lane": winning_lane,
            "signal_output": signal_output
        }
        
        self.data["predictions"].append(prediction)
        
        # Update statistics
        self.data["statistics"]["total_vehicles"] += total_vehicles
        self.data["statistics"]["total_images"] += 1
        for i, count in enumerate(lanes):
            self.data["statistics"]["lane_totals"][i] += count
        
        self._save_data()
        return prediction
    
    def get_recent_predictions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent predictions"""
        return sorted(self.data["predictions"], 
                     key=lambda x: x["timestamp"], 
                     reverse=True)[:limit]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get overall statistics"""
        return self.data["statistics"]
    
    def get_lane_distribution(self) -> List[int]:
        """Get total vehicle count per lane"""
        return self.data["statistics"]["lane_totals"]
    
    def get_average_vehicles_per_image(self) -> float:
        """Calculate average vehicles per image"""
        if self.data["statistics"]["total_images"] == 0:
            return 0
        return self.data["statistics"]["total_vehicles"] / self.data["statistics"]["total_images"]
