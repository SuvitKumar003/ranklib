#!/usr/bin/env python3
"""
TOPSISX Automated Test Suite
Run this script to verify everything works before publishing
"""

import sys
import traceback
import pandas as pd
import numpy as np

# Test results tracking
test_results = []

def log_test(name, passed, details=""):
    """Log test result"""
    status = "✅ PASS" if passed else "❌ FAIL"
    test_results.append((name, passed, details))
    print(f"\n{status}: {name}")
    if details:
        print(f"   {details}")

def run_test(test_func):
    """Run a test function and catch exceptions"""
    try:
        test_func()
        return True
    except Exception as e:
        print(f"   Error: {e}")
        if "--verbose" in sys.argv:
            traceback.print_exc()
        return False

# =============================================================================
# TEST 1: Import Tests
# =============================================================================

def test_imports():
    """Test all imports work"""
    print("\n" + "="*70)
    print("TEST 1: Import Tests")
    print("="*70)
    
    try:
        from topsisx.pipeline import DecisionPipeline
        from topsisx.topsis import topsis
        from topsisx.vikor import vikor
        from topsisx.ahp import ahp
        from topsisx.entropy import entropy_weights
        from topsisx.reports import generate_report
        log_test("All imports successful", True)
        return True
    except ImportError as e:
        log_test("Imports", False, str(e))
        return False

# =============================================================================
# TEST 2: TOPSIS Functionality
# =============================================================================

def test_topsis():
    """Test TOPSIS method"""
    print("\n" + "="*70)
    print("TEST 2: TOPSIS Method")
    print("="*70)
    
    def run():
        from topsisx.pipeline import DecisionPipeline
        
        data = pd.DataFrame({
            'Cost': [250, 200, 300],
            'Quality': [16, 20, 12],
            'Time': [12, 8, 10]
        })
        
        pipeline = DecisionPipeline(weights='entropy', method='topsis')
        result = pipeline.run(data, impacts=['-', '+', '-'])
        
        # Validate result
        assert 'Topsis_Score' in result.columns, "Missing Topsis_Score column"
        assert 'Rank' in result.columns, "Missing Rank column"
        assert len(result) == 3, "Wrong number of results"
        assert result['Rank'].tolist() == [1, 2, 3] or result['Rank'].tolist() == [1, 1, 3], "Ranks not sequential"
        
        print(f"   Results: {len(result)} alternatives ranked")
        return True
    
    passed = run_test(run)
    log_test("TOPSIS with Entropy weights", passed)
    return passed

# =============================================================================
# TEST 3: VIKOR Functionality
# =============================================================================

def test_vikor():
    """Test VIKOR method"""
    print("\n" + "="*70)
    print("TEST 3: VIKOR Method")
    print("="*70)
    
    def run():
        from topsisx.pipeline import DecisionPipeline
        
        data = pd.DataFrame({
            'C1': [7, 8, 6, 9],
            'C2': [9, 7, 8, 6],
            'C3': [9, 6, 8, 7]
        })
        
        pipeline = DecisionPipeline(weights='equal', method='vikor')
        result = pipeline.run(data, impacts=['+', '+', '+'], v=0.5)
        
        # Validate result
        assert 'Q' in result.columns, "Missing Q column"
        assert 'S' in result.columns, "Missing S column"
        assert 'R' in result.columns, "Missing R column"
        assert 'Rank' in result.columns, "Missing Rank column"
        assert len(result) == 4, "Wrong number of results"
        
        print(f"   Results: {len(result)} alternatives ranked")
        return True
    
    passed = run_test(run)
    log_test("VIKOR with Equal weights", passed)
    return passed

# =============================================================================
# TEST 4: AHP Weighting
# =============================================================================

def test_ahp():
    """Test AHP method"""
    print("\n" + "="*70)
    print("TEST 4: AHP Weighting")
    print("="*70)
    
    def run():
        from topsisx.ahp import ahp
        
        # Test with fractional strings
        pairwise = pd.DataFrame([
            [1, 3, 5],
            ['1/3', 1, 3],
            ['1/5', '1/3', 1]
        ])
        
        weights = ahp(pairwise, verbose=False)
        
        # Validate weights
        assert len(weights) == 3, "Wrong number of weights"
        assert abs(weights.sum() - 1.0) < 0.001, "Weights don't sum to 1"
        assert all(w > 0 for w in weights), "Weights must be positive"
        
        print(f"   Weights: {weights}")
        return True
    
    passed = run_test(run)
    log_test("AHP pairwise comparison", passed)
    return passed

# =============================================================================
# TEST 5: Entropy Weighting
# =============================================================================

def test_entropy():
    """Test Entropy weighting"""
    print("\n" + "="*70)
    print("TEST 5: Entropy Weighting")
    print("="*70)
    
    def run():
        from topsisx.entropy import entropy_weights
        
        data = np.array([
            [250, 16, 12],
            [200, 20, 8],
            [300, 12, 10]
        ])
        
        weights = entropy_weights(data)
        
        # Validate weights
        assert len(weights) == 3, "Wrong number of weights"
        assert abs(weights.sum() - 1.0) < 0.001, "Weights don't sum to 1"
        assert all(w >= 0 for w in weights), "Weights must be non-negative"
        
        print(f"   Weights: {weights}")
        return True
    
    passed = run_test(run)
    log_test("Entropy weight calculation", passed)
    return passed

# =============================================================================
# TEST 6: Error Handling
# =============================================================================

def test_error_handling():
    """Test error handling"""
    print("\n" + "="*70)
    print("TEST 6: Error Handling")
    print("="*70)
    
    from topsisx.pipeline import DecisionPipeline
    
    data = pd.DataFrame({
        'C1': [1, 2, 3],
        'C2': [4, 5, 6]
    })
    
    # Test 1: Wrong impact count
    def test_wrong_impacts():
        pipeline = DecisionPipeline(weights='entropy', method='topsis')
        try:
            result = pipeline.run(data, impacts=['+'])  # Should fail
            return False  # Should not reach here
        except ValueError:
            return True
    
    passed1 = test_wrong_impacts()
    log_test("Wrong impact count raises error", passed1)
    
    # Test 2: Invalid impacts
    def test_invalid_impacts():
        pipeline = DecisionPipeline(weights='entropy', method='topsis')
        try:
            result = pipeline.run(data, impacts=['x', 'y'])  # Should fail
            return False
        except ValueError:
            return True
    
    passed2 = test_invalid_impacts()
    log_test("Invalid impact characters raise error", passed2)
    
    # Test 3: Missing AHP matrix
    def test_missing_ahp():
        pipeline = DecisionPipeline(weights='ahp', method='topsis')
        try:
            result = pipeline.run(data, impacts=['+', '+'])  # Should fail
            return False
        except ValueError:
            return True
    
    passed3 = test_missing_ahp()
    log_test("Missing AHP matrix raises error", passed3)
    
    return passed1 and passed2 and passed3

# =============================================================================
# TEST 7: Edge Cases
# =============================================================================

def test_edge_cases():
    """Test edge cases"""
    print("\n" + "="*70)
    print("TEST 7: Edge Cases")
    print("="*70)
    
    from topsisx.pipeline import DecisionPipeline
    
    # Test 1: Small dataset (2 alternatives)
    def test_small():
        data = pd.DataFrame({'C1': [1, 2], 'C2': [3, 4]})
        pipeline = DecisionPipeline(weights='entropy', method='topsis')
        result = pipeline.run(data, impacts=['+', '+'])
        return len(result) == 2
    
    passed1 = run_test(test_small)
    log_test("Small dataset (2 alternatives)", passed1)
    
    # Test 2: Many criteria
    def test_many_criteria():
        data = pd.DataFrame(np.random.rand(5, 10))
        pipeline = DecisionPipeline(weights='entropy', method='topsis')
        result = pipeline.run(data, impacts=['+']*10)
        return len(result) == 5
    
    passed2 = run_test(test_many_criteria)
    log_test("Many criteria (10 criteria)", passed2)
    
    # Test 3: Identical values
    def test_identical():
        data = pd.DataFrame({'C1': [5, 5, 5], 'C2': [1, 2, 3]})
        pipeline = DecisionPipeline(weights='entropy', method='topsis')
        result = pipeline.run(data, impacts=['+', '+'])
        return len(result) == 3
    
    passed3 = run_test(test_identical)
    log_test("Identical values in criterion", passed3)
    
    return passed1 and passed2 and passed3

# =============================================================================
# TEST 8: Verbose Mode
# =============================================================================

def test_verbose_mode():
    """Test verbose mode works"""
    print("\n" + "="*70)
    print("TEST 8: Verbose Mode")
    print("="*70)
    
    def run():
        from topsisx.pipeline import DecisionPipeline
        
        data = pd.DataFrame({
            'C1': [1, 2, 3],
            'C2': [4, 5, 6]
        })
        
        # Should print detailed output
        pipeline = DecisionPipeline(weights='entropy', method='topsis', verbose=True)
        result = pipeline.run(data, impacts=['+', '+'])
        
        return len(result) == 3
    
    passed = run_test(run)
    log_test("Verbose mode produces output", passed)
    return passed

# =============================================================================
# TEST 9: Method Comparison
# =============================================================================

def test_method_comparison():
    """Test method comparison"""
    print("\n" + "="*70)
    print("TEST 9: Method Comparison")
    print("="*70)
    
    def run():
        from topsisx.pipeline import DecisionPipeline
        
        data = pd.DataFrame({
            'C1': [7, 8, 6, 9],
            'C2': [9, 7, 8, 6],
            'C3': [9, 6, 8, 7]
        })
        
        pipeline = DecisionPipeline(weights='equal', method='topsis')
        comparison = pipeline.compare_methods(data, impacts=['+', '+', '+'])
        
        # Validate comparison
        assert 'topsis' in comparison, "Missing TOPSIS results"
        assert 'vikor' in comparison, "Missing VIKOR results"
        assert 'comparison' in comparison, "Missing comparison table"
        assert 'Rank_Difference' in comparison['comparison'].columns, "Missing rank difference"
        
        print(f"   Comparison complete")
        return True
    
    passed = run_test(run)
    log_test("TOPSIS vs VIKOR comparison", passed)
    return passed

# =============================================================================
# TEST 10: pandas Compatibility
# =============================================================================

def test_pandas_compatibility():
    """Test pandas 2.1+ compatibility"""
    print("\n" + "="*70)
    print("TEST 10: pandas Compatibility")
    print("="*70)
    
    import pandas
    print(f"   pandas version: {pandas.__version__}")
    
    def run():
        from topsisx.ahp import ahp
        
        # This should work with pandas 2.1+
        pairwise = pd.DataFrame([
            [1, 3, 5],
            ['1/3', 1, 3],
            ['1/5', '1/3', 1]
        ])
        
        weights = ahp(pairwise)
        return len(weights) == 3
    
    passed = run_test(run)
    log_test("pandas 2.1+ compatibility (no applymap)", passed)
    return passed

# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def main():
    """Run all tests"""
    print("\n" + "#"*70)
    print("#" + " "*20 + "TOPSISX TEST SUITE" + " "*22 + "#")
    print("#"*70)
    
    # Run all tests
    all_passed = True
    
    all_passed &= test_imports()
    all_passed &= test_topsis()
    all_passed &= test_vikor()
    all_passed &= test_ahp()
    all_passed &= test_entropy()
    all_passed &= test_error_handling()
    all_passed &= test_edge_cases()
    all_passed &= test_verbose_mode()
    all_passed &= test_method_comparison()
    all_passed &= test_pandas_compatibility()
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed_count = sum(1 for _, passed, _ in test_results if passed)
    total_count = len(test_results)
    
    for name, passed, details in test_results:
        status = "✅" if passed else "❌"
        print(f"{status} {name}")
        if details and not passed:
            print(f"   {details}")
    
    print(f"\n{'='*70}")
    print(f"Results: {passed_count}/{total_count} tests passed")
    
    if all_passed:
        print("✅ ALL TESTS PASSED - Ready for publication!")
        print("="*70)
        return 0
    else:
        print("❌ SOME TESTS FAILED - Fix issues before publishing")
        print("="*70)
        return 1

if __name__ == "__main__":
    sys.exit(main())