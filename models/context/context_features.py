import torch
import torch.nn as nn
from typing import Dict
import math

class ContextFeatureExtractor(nn.Module):
    """Extracts and processes driving context features for gating network"""
    def __init__(self, 
                 context_dim: int = 64,
                 include_weather: bool = True,
                 include_time: bool = True,
                 include_road: bool = True):
        super().__init__()
        self.context_dim = context_dim
        self.include_weather = include_weather
        self.include_time = include_time
        self.include_road = include_road
        
        self.input_dim = 4  # speed, steering, throttle, brake
        
        if self.include_weather:
            self.input_dim += 4  # rain, fog, wetness, sun_angle
        if self.include_time:
            self.input_dim += 2  # hour (sin/cos), minute (sin/cos)
        if self.include_road:
            self.input_dim += 3  # road_type, lane_count, road_curvature
        
        self.context_encoder = nn.Sequential(
            nn.Linear(self.input_dim, context_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(context_dim, context_dim),
            nn.LayerNorm(context_dim)
        )
        
    def _encode_time(self, hour: torch.Tensor, minute: torch.Tensor) -> torch.Tensor:
        """Encode time using cyclical encoding"""
        hour_rad = 2 * math.pi * hour / 24.0
        minute_rad = 2 * math.pi * minute / 60.0
        
        hour_sin = torch.sin(hour_rad)
        hour_cos = torch.cos(hour_rad)
        minute_sin = torch.sin(minute_rad)
        minute_cos = torch.cos(minute_rad)
        
        return torch.stack([hour_sin, hour_cos, minute_sin, minute_cos], dim=-1)
    
    def _encode_weather(self, weather_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode weather features"""
        features = []
        
        # Rain intensity (0-1)
        rain = weather_data.get('rain', torch.zeros(weather_data['speed'].size(0), 1, device=weather_data['speed'].device))
        features.append(rain)
        
        # Fog density (0-1)
        fog = weather_data.get('fog', torch.zeros(weather_data['speed'].size(0), 1, device=weather_data['speed'].device))
        features.append(fog)
        
        # Road wetness (0-1)
        wetness = weather_data.get('wetness', torch.zeros(weather_data['speed'].size(0), 1, device=weather_data['speed'].device))
        features.append(wetness)
        
        # Sun angle (0-1)
        sun_angle = weather_data.get('sun_angle', torch.zeros(weather_data['speed'].size(0), 1, device=weather_data['speed'].device))
        features.append(sun_angle)
        
        return torch.cat(features, dim=-1)
    
    def _encode_road(self, road_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode road features"""
        features = []
        
        # Road type
        road_type = road_data.get('road_type', torch.zeros(road_data['speed'].size(0), 1, device=road_data['speed'].device))
        features.append(road_type)
        
        # Number of lanes
        lane_count = road_data.get('lane_count', torch.ones(road_data['speed'].size(0), 1, device=road_data['speed'].device))
        features.append(lane_count)
        
        # Road curvature
        curvature = road_data.get('curvature', torch.zeros(road_data['speed'].size(0), 1, device=road_data['speed'].device))
        features.append(curvature)
        
        return torch.cat(features, dim=-1)
    
    def forward(self, context_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            context_data: Dict containing:
                - speed: [B, 1] - vehicle speed
                - steering: [B, 1] - steering angle
                - throttle: [B, 1] - throttle input
                - brake: [B, 1] - brake input
                - hour: [B, 1] - hour of day (0-23)
                - minute: [B, 1] - minute of hour (0-59)
                - weather: Dict with weather features
                - road: Dict with road features
        Returns:
            context_features: [B, context_dim] - encoded context features
        """
        batch_size = context_data['speed'].size(0)
        device = context_data['speed'].device
        
        features = []
        
        vehicle_features = torch.cat([
            context_data['speed'],
            context_data['steering'],
            context_data['throttle'],
            context_data['brake']
        ], dim=-1)
        features.append(vehicle_features)
        
        if self.include_weather:
            weather_features = self._encode_weather(context_data.get('weather', {}))
            features.append(weather_features)
        
        if self.include_time:
            hour = context_data.get('hour', torch.zeros(batch_size, 1, device=device))
            minute = context_data.get('minute', torch.zeros(batch_size, 1, device=device))
            time_features = self._encode_time(hour, minute)
            features.append(time_features)
        
        if self.include_road:
            road_features = self._encode_road(context_data.get('road', {}))
            features.append(road_features)
        
        combined_features = torch.cat(features, dim=-1)
        
        context_features = self.context_encoder(combined_features)
        
        return context_features


class SimpleContextExtractor(nn.Module):
    """Simplified context extractor for basic vehicle state"""
    def __init__(self, context_dim: int = 64):
        super().__init__()
        self.context_dim = context_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(4, 32),  # speed, steering, throttle, brake
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, context_dim),
            nn.LayerNorm(context_dim)
        )
        
    def forward(self, speed: torch.Tensor, 
                steering: torch.Tensor, 
                throttle: torch.Tensor, 
                brake: torch.Tensor) -> torch.Tensor:
        """
        Args:
            speed: [B, 1] - vehicle speed
            steering: [B, 1] - steering angle
            throttle: [B, 1] - throttle input
            brake: [B, 1] - brake input
        Returns:
            context_features: [B, context_dim] - encoded context
        """
        vehicle_state = torch.cat([speed, steering, throttle, brake], dim=-1)
        return self.encoder(vehicle_state)


def create_context_extractor(config: Dict) -> nn.Module:
    """
    Create context extractor based on configuration
    
    Args:
        config: Dict with extractor configuration
    Returns:
        Context extractor module
    """
    extractor_type = config.get('type', 'simple')
    
    if extractor_type == 'simple':
        return SimpleContextExtractor(
            context_dim=config.get('context_dim', 64)
        )
    elif extractor_type == 'full':
        return ContextFeatureExtractor(
            context_dim=config.get('context_dim', 64),
            include_weather=config.get('include_weather', True),
            include_time=config.get('include_time', True),
            include_road=config.get('include_road', True)
        )
    else:
        raise ValueError(f"Unknown context extractor type: {extractor_type}")

