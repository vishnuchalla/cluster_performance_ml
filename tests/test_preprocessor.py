"""
Unit tests for the data preprocessor module.
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessor import ClusterDataPreprocessor

class TestClusterDataPreprocessor(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = ClusterDataPreprocessor()
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'clusterType': ['self-managed', 'managed', 'self-managed'],
            'controlPlaneArch': ['amd64', 'amd64', 'arm64'],
            'masterNodesCount': [3, 3, 5],
            'workerNodesCount': [6, 4, 8],
            'jobConfig.qps': [20, 25, 30],
            'cpu-kubelet': [0.1, 0.15, 0.12],
            'memory-kubelet': [100, 120, 110],
            'avg-ro-apicalls-latency': [0.01, 0.02, 0.015],
            'max-cpu-masters': [0.5, 0.6, 0.55],
        })
    
    def test_identify_columns(self):
        """Test column identification."""
        feature_cols, target_cols = self.preprocessor.identify_columns(self.sample_data)
        
        # Check that we identified some features and targets
        self.assertGreater(len(feature_cols), 0)
        self.assertGreater(len(target_cols), 0)
        
        # Check specific columns
        self.assertIn('clusterType', feature_cols)
        self.assertIn('cpu-kubelet', target_cols)
    
    def test_clean_data(self):
        """Test data cleaning."""
        # Add some missing values
        data_with_missing = self.sample_data.copy()
        data_with_missing.loc[0, 'masterNodesCount'] = np.nan
        data_with_missing.loc[1, 'clusterType'] = np.nan
        
        cleaned_data = self.preprocessor.clean_data(data_with_missing)
        
        # Check that no missing values remain
        self.assertEqual(cleaned_data.isnull().sum().sum(), 0)
    
    def test_encode_categorical_features(self):
        """Test categorical encoding."""
        self.preprocessor.identify_columns(self.sample_data)
        encoded_data = self.preprocessor.encode_categorical_features(self.sample_data, fit=True)
        
        # Check that categorical columns are now numeric
        categorical_features = [col for col in self.preprocessor.feature_columns 
                              if col in self.sample_data.select_dtypes(include=['object']).columns]
        
        for col in categorical_features:
            self.assertTrue(pd.api.types.is_numeric_dtype(encoded_data[col]))
    
    def test_preprocess_data(self):
        """Test complete preprocessing pipeline."""
        X, y = self.preprocessor.preprocess_data(self.sample_data, fit=True)
        
        # Check output shapes
        self.assertEqual(len(X), len(self.sample_data))
        self.assertEqual(len(y), len(self.sample_data))
        
        # Check that we have features and targets
        self.assertGreater(X.shape[1], 0)
        self.assertGreater(y.shape[1], 0)

if __name__ == '__main__':
    unittest.main()
