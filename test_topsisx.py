"""
Comprehensive test suite for TOPSISX library
Run this to verify all functionality works correctly
"""

import pandas as pd
import numpy as np
from topsisx.pipeline import DecisionPipeline
from topsisx.topsis import topsis
from topsisx.ahp import ahp
from topsisx.vikor import vikor
from topsisx.entropy import entropy_weights

def test_topsis():
    """Test TOPSIS method"""
    print("\n" + "="*60)
    print("Testing TOPSIS Method")
    print("="*60)
    
    # Sample data
    data = pd.DataFrame({
        'Cost': [250, 200, 300, 275, 225],
        'Quality': [16, 16, 32, 32, 16],
        'Time': [12, 8, 16, 8, 16],
        'Efficiency': [5, 3, 4, 4, 2]
    })
    
    weights = [0.25, 0.25, 0.25, 0.25]
    impacts = ['-', '+', '-', '+']  # Cost and Time are to be minimized
    
    print("\nInput Data:")
    print(data)
    print(f"\nWeights: {weights}")
    print(f"Impacts: {impacts}")
    
    result = topsis(data, weights, impacts)
    print("\nTOPSIS Results:")
    print(result)
    
    return result

def test_ahp():
    """Test AHP method"""
    print("\n" + "="*60)
    print("Testing AHP Method")
    print("="*60)
    
    # Pairwise comparison matrix
    pairwise = pd.DataFrame([
        [1, 3, 5],
        ['1/3', 1, 4],
        ['1/5', '1/4', 1]
    ])
    
    print("\nPairwise Comparison Matrix:")
    print(pairwise)
    
    weights = ahp(pairwise)
    print("\nAHP Weights:")
    for i, w in enumerate(weights):
        print(f"Criterion {i+1}: {w:.4f}")
    
    return weights

def test_vikor():
    """Test VIKOR method"""
    print("\n" + "="*60)
    print("Testing VIKOR Method")
    print("="*60)
    
    data = pd.DataFrame({
        'C1': [7, 8, 6, 9],
        'C2': [9, 7, 8, 6],
        'C3': [9, 6, 8, 7]
    })
    
    weights = [0.33, 0.33, 0.34]
    impacts = ['+', '+', '+']
    
    print("\nInput Data:")
    print(data)
    print(f"\nWeights: {weights}")
    print(f"Impacts: {impacts}")
    
    result = vikor(data, weights, impacts)
    print("\nVIKOR Results:")
    print(result)
    
    return result

def test_entropy():
    """Test Entropy weighting"""
    print("\n" + "="*60)
    print("Testing Entropy Weighting")
    print("="*60)
    
    data = np.array([
        [250, 16, 12, 5],
        [200, 16, 8, 3],
        [300, 32, 16, 4],
        [275, 32, 8, 4],
        [225, 16, 16, 2]
    ])
    
    print("\nInput Matrix:")
    print(data)
    
    weights = entropy_weights(data)
    print("\nEntropy Weights:")
    for i, w in enumerate(weights):
        print(f"Criterion {i+1}: {w:.4f}")
    
    return weights

def test_pipeline_topsis_entropy():
    """Test pipeline with TOPSIS and Entropy weights"""
    print("\n" + "="*60)
    print("Testing Pipeline: TOPSIS + Entropy Weights")
    print("="*60)
    
    data = pd.DataFrame({
        'ID': ['A', 'B', 'C', 'D', 'E'],
        'Cost': [250, 200, 300, 275, 225],
        'Quality': [16, 16, 32, 32, 16],
        'Time': [12, 8, 16, 8, 16]
    })
    
    print("\nInput Data:")
    print(data)
    
    pipeline = DecisionPipeline(weights='entropy', method='topsis')
    impacts = ['-', '+', '-']
    
    result = pipeline.run(data.iloc[:, 1:], impacts=impacts)
    result.insert(0, 'ID', data['ID'].values)
    
    print("\nPipeline Results:")
    print(result)
    
    return result

def test_pipeline_vikor_equal():
    """Test pipeline with VIKOR and equal weights"""
    print("\n" + "="*60)
    print("Testing Pipeline: VIKOR + Equal Weights")
    print("="*60)
    
    data = pd.DataFrame({
        'Alternative': ['A1', 'A2', 'A3', 'A4'],
        'C1': [7, 8, 6, 9],
        'C2': [9, 7, 8, 6],
        'C3': [9, 6, 8, 7]
    })
    
    print("\nInput Data:")
    print(data)
    
    pipeline = DecisionPipeline(weights='equal', method='vikor')
    impacts = ['+', '+', '+']
    
    result = pipeline.run(data.iloc[:, 1:], impacts=impacts)
    result.insert(0, 'Alternative', data['Alternative'].values)
    
    print("\nPipeline Results:")
    print(result)
    
    return result

def test_csv_workflow():
    """Test workflow with CSV files"""
    print("\n" + "="*60)
    print("Testing CSV Workflow")
    print("="*60)
    
    # Test with data.csv
    try:
        data = pd.read_csv('data.csv')
        print("\nLoaded data.csv:")
        print(data)
        
        pipeline = DecisionPipeline(weights='entropy', method='topsis')
        impacts = ['-', '+', '-']  # Cost-, Quality+, Time-
        
        result = pipeline.run(data.iloc[:, 1:], impacts=impacts)
        result.insert(0, 'ID', data['ID'].values)
        
        print("\nResults:")
        print(result)
        
    except FileNotFoundError:
        print("\n⚠️  data.csv not found in current directory")
    
    # Test with sample.csv
    try:
        data = pd.read_csv('sample.csv')
        print("\nLoaded sample.csv:")
        print(data)
        
        pipeline = DecisionPipeline(weights='equal', method='topsis')
        impacts = ['+', '+', '+']
        
        result = pipeline.run(data.iloc[:, 1:], impacts=impacts)
        result.insert(0, 'Alternative', data['Alternative'].values)
        
        print("\nResults:")
        print(result)
        
    except FileNotFoundError:
        print("\n⚠️  sample.csv not found in current directory")

def run_all_tests():
    """Run all tests"""
    print("\n" + "#"*60)
    print("#" + " "*18 + "TOPSISX TEST SUITE" + " "*20 + "#")
    print("#"*60)
    
    try:
        test_topsis()
        test_ahp()
        test_vikor()
        test_entropy()
        test_pipeline_topsis_entropy()
        test_pipeline_vikor_equal()
        test_csv_workflow()
        
        print("\n" + "="*60)
        print("✅ All tests completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_all_tests()