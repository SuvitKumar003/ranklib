"""
TOPSISX Quick Demo
Run this script to see all features in action
"""

import pandas as pd
from topsisx.pipeline import DecisionPipeline
from topsisx.topsis import topsis
from topsisx.vikor import vikor
from topsisx.ahp import ahp

def print_header(text):
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def demo_1_simple_topsis():
    """Demo 1: Simple TOPSIS Analysis"""
    print_header("DEMO 1: Simple TOPSIS Analysis")
    
    # Create sample data
    data = pd.DataFrame({
        'Laptop': ['Model A', 'Model B', 'Model C', 'Model D'],
        'Price_USD': [800, 1200, 1000, 900],
        'RAM_GB': [8, 16, 16, 8],
        'Battery_Hours': [6, 4, 8, 7],
        'Weight_KG': [2.0, 2.5, 1.8, 2.2]
    })
    
    print("📊 Laptop Comparison Data:")
    print(data.to_string(index=False))
    
    # Run TOPSIS
    print("\n🔄 Running TOPSIS analysis...")
    print("   Weights: Equal (auto-calculated)")
    print("   Impacts: Price(-), RAM(+), Battery(+), Weight(-)")
    
    pipeline = DecisionPipeline(weights='equal', method='topsis', verbose=True)
    result = pipeline.run(
        data,
        impacts=['-', '+', '+', '-']
    )
    
    print("\n🏆 Results:")
    print(result.to_string(index=False))
    
    winner = result.iloc[0]['Laptop']
    score = result.iloc[0]['Topsis_Score']
    print(f"\n✨ Winner: {winner} (Score: {score:.4f})")
    
    return result

def demo_2_entropy_weighting():
    """Demo 2: TOPSIS with Entropy Weights"""
    print_header("DEMO 2: TOPSIS with Entropy Weighting")
    
    data = pd.DataFrame({
        'Supplier': ['S1', 'S2', 'S3', 'S4', 'S5'],
        'Cost': [250, 200, 300, 275, 225],
        'Quality_Score': [16, 16, 32, 32, 16],
        'Delivery_Days': [12, 8, 16, 8, 16],
        'Service_Rating': [5, 3, 4, 4, 2]
    })
    
    print("📊 Supplier Data:")
    print(data.to_string(index=False))
    
    print("\n🔄 Using Entropy weighting (objective, data-driven)...")
    
    pipeline = DecisionPipeline(weights='entropy', method='topsis', verbose=True)
    result = pipeline.run(
        data,
        impacts=['-', '+', '-', '+']
    )
    
    print("\n🏆 Supplier Rankings:")
    print(result[['Supplier', 'Topsis_Score', 'Rank']].to_string(index=False))
    
    return result

def demo_3_ahp_weighting():
    """Demo 3: TOPSIS with AHP Weights"""
    print_header("DEMO 3: AHP Expert Weighting")
    
    # Criteria: Cost, Quality, Delivery
    print("📊 Decision: Best Supplier")
    print("   Criteria: Cost, Quality, Delivery Time")
    print("\n🎯 Expert Judgment (Pairwise Comparisons):")
    print("   - Quality is 3x more important than Cost")
    print("   - Quality is 5x more important than Delivery")
    print("   - Cost is 2x more important than Delivery")
    
    # AHP pairwise matrix
    ahp_matrix = pd.DataFrame([
        [1,   '1/3', '1/2'],  # Cost
        [3,   1,     5    ],  # Quality (most important)
        [2,   '1/5', 1    ]   # Delivery
    ])
    
    print("\n📐 Pairwise Comparison Matrix:")
    print(ahp_matrix.to_string(index=False, header=False))
    
    # Calculate weights
    print("\n⚖️  Calculating AHP weights...")
    weights = ahp(ahp_matrix, verbose=True)
    
    # Apply to data
    data = pd.DataFrame({
        'Supplier': ['A', 'B', 'C'],
        'Cost': [250, 200, 300],
        'Quality': [16, 16, 32],
        'Delivery': [12, 8, 16]
    })
    
    pipeline = DecisionPipeline(weights='ahp', method='topsis')
    result = pipeline.run(
        data,
        impacts=['-', '+', '-'],
        pairwise_matrix=ahp_matrix
    )
    
    print("\n🏆 Final Rankings:")
    print(result.to_string(index=False))
    
    return result

def demo_4_vikor_method():
    """Demo 4: VIKOR Compromise Solution"""
    print_header("DEMO 4: VIKOR Compromise Ranking")
    
    data = pd.DataFrame({
        'Investment': ['Stocks', 'Bonds', 'Real Estate', 'Gold'],
        'Expected_Return': [12, 5, 8, 6],
        'Risk_Level': [8, 2, 5, 3],
        'Liquidity': [9, 7, 4, 6],
        'Min_Investment': [1000, 100, 50000, 500]
    })
    
    print("📊 Investment Options:")
    print(data.to_string(index=False))
    
    print("\n🔄 Running VIKOR (Compromise solution)...")
    print("   VIKOR finds balanced solutions considering:")
    print("   - Group utility (majority preference)")
    print("   - Individual regret (worst-case scenario)")
    
    pipeline = DecisionPipeline(weights='entropy', method='vikor', verbose=True)
    result = pipeline.run(
        data,
        impacts=['+', '-', '+', '-'],
        v=0.5
    )
    
    print("\n🏆 VIKOR Results:")
    print(result[['Investment', 'Q', 'Rank']].to_string(index=False))
    
    return result

def demo_5_compare_methods():
    """Demo 5: Compare TOPSIS vs VIKOR"""
    print_header("DEMO 5: Comparing TOPSIS vs VIKOR")
    
    data = pd.DataFrame({
        'Option': ['A', 'B', 'C', 'D'],
        'Criterion_1': [7, 8, 6, 9],
        'Criterion_2': [9, 7, 8, 6],
        'Criterion_3': [9, 6, 8, 7]
    })
    
    print("📊 Decision Data:")
    print(data.to_string(index=False))
    
    print("\n🔄 Running both TOPSIS and VIKOR...")
    
    pipeline = DecisionPipeline(weights='equal', method='topsis')
    comparison = pipeline.compare_methods(
        data,
        impacts=['+', '+', '+']
    )
    
    print("\n📊 Rank Comparison:")
    print(comparison['comparison'].to_string(index=False))
    
    avg_diff = comparison['comparison']['Rank_Difference'].mean()
    print(f"\n📈 Average rank difference: {avg_diff:.2f}")
    
    if avg_diff < 1:
        print("✅ Methods show high agreement!")
    else:
        print("⚠️  Methods show different preferences - consider both results")
    
    return comparison

def demo_6_complete_workflow():
    """Demo 6: Complete Real-World Workflow"""
    print_header("DEMO 6: Complete Decision-Making Workflow")
    
    print("🎯 Scenario: Selecting Best Cloud Service Provider")
    print("   Evaluating 5 providers on 4 criteria\n")
    
    # Real-world data
    data = pd.DataFrame({
        'Provider': ['AWS', 'Azure', 'Google Cloud', 'IBM Cloud', 'Oracle Cloud'],
        'Monthly_Cost': [500, 480, 450, 520, 490],
        'Performance_Score': [95, 92, 94, 88, 85],
        'Support_Rating': [4.5, 4.7, 4.3, 4.0, 3.8],
        'Feature_Count': [120, 115, 125, 100, 95]
    })
    
    print("📊 Provider Data:")
    print(data.to_string(index=False))
    
    # Step 1: Calculate objective weights
    print("\n" + "-"*70)
    print("STEP 1: Calculate Objective Weights (Entropy Method)")
    print("-"*70)
    
    pipeline = DecisionPipeline(weights='entropy', method='topsis', verbose=True)
    
    # Step 2: Run analysis
    print("\n" + "-"*70)
    print("STEP 2: Perform TOPSIS Ranking")
    print("-"*70)
    
    result = pipeline.run(
        data,
        impacts=['-', '+', '+', '+']  # Cost is negative, others positive
    )
    
    print("\n📊 Complete Results:")
    print(result.to_string(index=False))
    
    # Step 3: Insights
    print("\n" + "-"*70)
    print("STEP 3: Key Insights")
    print("-"*70)
    
    winner = result.iloc[0]
    runner_up = result.iloc[1]
    
    print(f"\n🥇 Winner: {winner['Provider']}")
    print(f"   Score: {winner['Topsis_Score']:.4f}")
    print(f"   Strengths: Best overall balance of all criteria")
    
    print(f"\n🥈 Runner-up: {runner_up['Provider']}")
    print(f"   Score: {runner_up['Topsis_Score']:.4f}")
    
    score_gap = winner['Topsis_Score'] - runner_up['Topsis_Score']
    print(f"\n📊 Decision Confidence: {score_gap:.4f}")
    
    if score_gap > 0.1:
        print("   ✅ Clear winner - high confidence")
    elif score_gap > 0.05:
        print("   ⚠️  Moderate difference - consider both options")
    else:
        print("   ⚠️  Very close - may need additional analysis")
    
    # Save results
    result.to_csv('cloud_provider_results.csv', index=False)
    print("\n💾 Results saved to: cloud_provider_results.csv")
    
    return result

def main():
    """Run all demos"""
    print("\n" + "="*70)
    print(" "*15 + "🎯 TOPSISX COMPLETE DEMO")
    print("="*70)
    print("\nThis demo showcases all features of TOPSISX library")
    print("Each demo demonstrates different capabilities and use cases")
    
    try:
        # Run all demos
        demo_1_simple_topsis()
        input("\n⏎ Press Enter to continue to Demo 2...")
        
        demo_2_entropy_weighting()
        input("\n⏎ Press Enter to continue to Demo 3...")
        
        demo_3_ahp_weighting()
        input("\n⏎ Press Enter to continue to Demo 4...")
        
        demo_4_vikor_method()
        input("\n⏎ Press Enter to continue to Demo 5...")
        
        demo_5_compare_methods()
        input("\n⏎ Press Enter to continue to Demo 6...")
        
        demo_6_complete_workflow()
        
        # Summary
        print("\n" + "="*70)
        print(" "*20 + "🎉 DEMO COMPLETE!")
        print("="*70)
        print("\n✨ You've seen:")
        print("   ✅ TOPSIS ranking")
        print("   ✅ Entropy weighting")
        print("   ✅ AHP expert weighting")
        print("   ✅ VIKOR compromise solutions")
        print("   ✅ Method comparisons")
        print("   ✅ Real-world workflow")
        
        print("\n🚀 Next Steps:")
        print("   1. Try with your own data")
        print("   2. Launch web interface: topsisx --web")
        print("   3. Read documentation: README.md")
        
        print("\n💡 Quick Commands:")
        print("   topsisx --web              # Launch web interface")
        print("   topsisx data.csv --impacts '+,-,+'  # CLI analysis")
        print("   python demo.py             # Run this demo again")
        
        print("\n" + "="*70 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted. Run again anytime!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease ensure all required packages are installed:")
        print("   pip install -r requirements.txt")

if __name__ == "__main__":
    main()