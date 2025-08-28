import unittest
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.gating.gating_network import GatingNetwork, ContextEncoder, ExpertOutputProcessor
from models.experts.expert_extractors import (
    DetectionExpertExtractor, 
    SegmentationExpertExtractor, 
    DrivableExpertExtractor,
    NuScenesExpertExtractor,
    create_expert_extractors
)
from models.context.context_features import SimpleContextExtractor, create_context_extractor
from models.automoe import create_automoe_model

class TestGatingNetwork(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 4
        self.num_experts = 4
        
    def test_context_encoder(self):
        """Test context encoder"""
        context_dim = 64
        hidden_dim = 128
        
        encoder = ContextEncoder(context_dim, hidden_dim)
        context = torch.randn(self.batch_size, context_dim)
        
        output = encoder(context)
        
        self.assertEqual(output.shape, (self.batch_size, hidden_dim))
        self.assertFalse(torch.isnan(output).any())
        
    def test_expert_output_processor(self):
        """Test expert output processor"""
        expert_output_dim = 512
        processed_dim = 256
        
        processor = ExpertOutputProcessor(expert_output_dim, processed_dim)
        expert_output = torch.randn(self.batch_size, expert_output_dim)
        
        output = processor(expert_output)
        
        self.assertEqual(output.shape, (self.batch_size, processed_dim))
        self.assertFalse(torch.isnan(output).any())
        
    def test_gating_network(self):
        """Test gating network"""
        context_dim = 64
        expert_output_dims = [256, 256, 256, 256]
        processed_dim = 256
        
        gating_network = GatingNetwork(
            num_experts=self.num_experts,
            context_dim=context_dim,
            expert_output_dims=expert_output_dims,
            processed_dim=processed_dim
        )
        
        # Create dummy inputs
        expert_outputs = [torch.randn(self.batch_size, dim) for dim in expert_output_dims]
        context = torch.randn(self.batch_size, context_dim)
        
        # Forward pass
        output = gating_network(expert_outputs, context)
        
        # Check outputs
        self.assertEqual(output['combined_output'].shape, (self.batch_size, processed_dim))
        self.assertEqual(output['expert_weights'].shape, (self.batch_size, self.num_experts))
        
        # Check that weights sum to 1
        weight_sums = output['expert_weights'].sum(dim=1)
        self.assertTrue(torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-6))
        
        # Check that weights are non-negative
        self.assertTrue((output['expert_weights'] >= 0).all())
        
    def test_expert_extractors(self):
        """Test expert output extractors"""
        # Test detection extractor
        detection_extractor = DetectionExpertExtractor(output_dim=256, num_classes=10)
        detection_output = {
            'class_logits': torch.randn(self.batch_size, 10, 32, 32),
            'bbox_deltas': torch.randn(self.batch_size, 4, 32, 32)
        }
        detection_features = detection_extractor(detection_output)
        self.assertEqual(detection_features.shape, (self.batch_size, 256))
        
        # Test segmentation extractor
        segmentation_extractor = SegmentationExpertExtractor(output_dim=256, num_classes=19)
        segmentation_output = torch.randn(self.batch_size, 19, 224, 224)
        segmentation_features = segmentation_extractor(segmentation_output)
        self.assertEqual(segmentation_features.shape, (self.batch_size, 256))
        
        # Test drivable extractor
        drivable_extractor = DrivableExpertExtractor(output_dim=256, num_classes=3)
        drivable_output = torch.randn(self.batch_size, 3, 224, 224)
        drivable_features = drivable_extractor(drivable_output)
        self.assertEqual(drivable_features.shape, (self.batch_size, 256))
        
        # Test nuScenes extractor
        nuscenes_extractor = NuScenesExpertExtractor(output_dim=256, num_queries=100, num_classes=10)
        nuscenes_output = {
            'class_logits': torch.randn(self.batch_size, 100, 10),
            'bbox_preds': torch.randn(self.batch_size, 100, 7)
        }
        nuscenes_features = nuscenes_extractor(nuscenes_output)
        self.assertEqual(nuscenes_features.shape, (self.batch_size, 256))
        
    def test_expert_extractor_manager(self):
        """Test expert extractor manager"""
        expert_configs = [
            {'type': 'detection', 'output_dim': 256, 'num_classes': 10},
            {'type': 'segmentation', 'output_dim': 256, 'num_classes': 19},
            {'type': 'drivable', 'output_dim': 256, 'num_classes': 3},
            {'type': 'nuscenes', 'output_dim': 256, 'num_queries': 100, 'num_classes': 10}
        ]
        
        extractor_manager = create_expert_extractors(expert_configs)
        
        # Create dummy expert outputs
        expert_outputs = [
            {'class_logits': torch.randn(self.batch_size, 10, 32, 32), 'bbox_deltas': torch.randn(self.batch_size, 4, 32, 32)},
            torch.randn(self.batch_size, 19, 224, 224),
            torch.randn(self.batch_size, 3, 224, 224),
            {'class_logits': torch.randn(self.batch_size, 100, 10), 'bbox_preds': torch.randn(self.batch_size, 100, 7)}
        ]
        
        features = extractor_manager.extract_features(expert_outputs)
        
        self.assertEqual(len(features), self.num_experts)
        for feature in features:
            self.assertEqual(feature.shape, (self.batch_size, 256))
            
    def test_context_extractor(self):
        """Test context extractor"""
        context_dim = 64
        
        # Test simple context extractor
        simple_extractor = SimpleContextExtractor(context_dim)
        speed = torch.randn(self.batch_size, 1)
        steering = torch.randn(self.batch_size, 1)
        throttle = torch.randn(self.batch_size, 1)
        brake = torch.randn(self.batch_size, 1)
        
        context_features = simple_extractor(speed, steering, throttle, brake)
        self.assertEqual(context_features.shape, (self.batch_size, context_dim))
        
        # Test context extractor factory
        config = {'type': 'simple', 'context_dim': context_dim}
        extractor = create_context_extractor(config)
        self.assertIsInstance(extractor, SimpleContextExtractor)
        
    def test_automoe_model(self):
        """Test complete AutoMoE model"""
        config = {
            'experts': [
                {'type': 'detection', 'num_classes': 10, 'output_dim': 256, 'pretrained_backbone': False},
                {'type': 'segmentation', 'num_classes': 19, 'output_dim': 256, 'pretrained_backbone': False},
                {'type': 'drivable', 'num_classes': 3, 'output_dim': 256, 'pretrained_backbone': False},
                {'type': 'nuscenes', 'num_queries': 100, 'num_classes': 10, 'output_dim': 256, 'fusion': 'concat'}
            ],
            'gating': {
                'processed_dim': 256,
                'hidden_dim': 128,
                'temperature': 1.0,
                'use_softmax': True
            },
            'context': {
                'type': 'simple',
                'context_dim': 64
            },
            'policy': {
                'hidden_dim': 256,
                'num_waypoints': 10,
                'waypoint_dim': 2
            }
        }
        
        model = create_automoe_model(config, self.device)
        
        # Create dummy batch
        batch = {
            'image': torch.randn(self.batch_size, 3, 224, 224),
            'lidar': torch.randn(self.batch_size, 1000, 3),
            'speed': torch.randn(self.batch_size, 1),
            'steering': torch.randn(self.batch_size, 1),
            'throttle': torch.randn(self.batch_size, 1),
            'brake': torch.randn(self.batch_size, 1),
            'waypoints': torch.randn(self.batch_size, 10, 2),
            'speed_target': torch.randn(self.batch_size, 1)
        }
        
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Forward pass
        output = model(batch)
        
        # Check outputs
        self.assertEqual(output['waypoints'].shape, (self.batch_size, 10, 2))
        self.assertEqual(output['speed'].shape, (self.batch_size, 1))
        self.assertEqual(output['expert_weights'].shape, (self.batch_size, self.num_experts))
        self.assertEqual(output['context_features'].shape, (self.batch_size, 64))
        self.assertEqual(output['combined_features'].shape, (self.batch_size, 256))
        
        # Check that expert weights sum to 1
        weight_sums = output['expert_weights'].sum(dim=1)
        self.assertTrue(torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-6))
        
    def test_model_freeze_unfreeze(self):
        """Test expert freezing and unfreezing"""
        config = {
            'experts': [
                {'type': 'detection', 'num_classes': 10, 'output_dim': 256, 'pretrained_backbone': False},
                {'type': 'segmentation', 'num_classes': 19, 'output_dim': 256, 'pretrained_backbone': False}
            ],
            'gating': {'processed_dim': 256, 'hidden_dim': 128, 'temperature': 1.0, 'use_softmax': True},
            'context': {'type': 'simple', 'context_dim': 64},
            'policy': {'hidden_dim': 256, 'num_waypoints': 10, 'waypoint_dim': 2}
        }
        
        model = create_automoe_model(config, self.device)
        
        # Check initial state (should be trainable)
        for expert in model.experts:
            for param in expert.parameters():
                self.assertTrue(param.requires_grad)
        
        # Freeze experts
        model.freeze_experts()
        for expert in model.experts:
            for param in expert.parameters():
                self.assertFalse(param.requires_grad)
        
        # Unfreeze experts
        model.unfreeze_experts()
        for expert in model.experts:
            for param in expert.parameters():
                self.assertTrue(param.requires_grad)


if __name__ == '__main__':
    unittest.main()

