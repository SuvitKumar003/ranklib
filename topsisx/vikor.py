import numpy as np
import pandas as pd

def vikor(data, weights, impacts, v=0.5, verbose=False):
    """
    Complete VIKOR ranking method with C1 and C2 acceptance conditions.
    
    Parameters:
    - data: DataFrame or numpy array with criteria values
    - weights: Array of criteria weights
    - impacts: List of '+' (benefit) or '-' (cost) for each criterion
    - v: Strategy weight (0-1), default 0.5
         v=0 focuses on maximum group utility (consensus)
         v=1 focuses on minimum individual regret
    - verbose: If True, print detailed C1/C2 condition checks
    
    Returns:
    - DataFrame with S, R, Q scores, ranks, and compromise solution info
    
    Notes:
    - Lower Q values indicate better alternatives
    - Checks C1 (acceptable advantage) and C2 (acceptable stability)
    - Identifies compromise solution set when conditions are not met
    """
    # Convert to DataFrame if needed
    if isinstance(data, (list, np.ndarray)):
        data = pd.DataFrame(data)
    elif not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a DataFrame, list, or NumPy array.")
    
    # Make a copy to avoid modifying original
    df = data.copy()
    
    # Validate inputs
    matrix = df.values.astype(float)
    weights = np.array(weights, dtype=float)
    m = matrix.shape[0]  # number of alternatives
    n = matrix.shape[1]  # number of criteria
    
    if len(weights) != n:
        raise ValueError(f"Number of weights ({len(weights)}) must match number of criteria ({n}).")
    
    if len(impacts) != n:
        raise ValueError(f"Number of impacts ({len(impacts)}) must match number of criteria ({n}).")
    
    if not all(i in ['+', '-'] for i in impacts):
        raise ValueError("Impacts must be '+' or '-' only.")
    
    # Normalize weights if needed
    if abs(weights.sum() - 1.0) > 1e-6:
        weights = weights / weights.sum()
    
    # Calculate ideal and anti-ideal for each criterion
    ideal = np.zeros(n)
    anti_ideal = np.zeros(n)
    
    for j in range(n):
        if impacts[j] == '+':
            # Benefit criterion: higher is better
            ideal[j] = np.max(matrix[:, j])
            anti_ideal[j] = np.min(matrix[:, j])
        else:
            # Cost criterion: lower is better
            ideal[j] = np.min(matrix[:, j])
            anti_ideal[j] = np.max(matrix[:, j])
    
    # Calculate S (group utility) and R (individual regret)
    S = np.zeros(m)
    R = np.zeros(m)
    
    for i in range(m):
        # Calculate weighted normalized distances for each alternative
        weighted_distances = weights * (ideal - matrix[i, :]) / (ideal - anti_ideal + 1e-10)
        S[i] = np.sum(weighted_distances)  # Sum for group utility
        R[i] = np.max(weighted_distances)  # Max for individual regret
    
    # Calculate Q values using the strategy weight v
    S_star = np.min(S)   # Best group utility
    S_minus = np.max(S)  # Worst group utility
    R_star = np.min(R)   # Best individual regret
    R_minus = np.max(R)  # Worst individual regret
    
    # VIKOR Q formula: compromise between S and R
    Q = v * (S - S_star) / (S_minus - S_star + 1e-10) + \
        (1 - v) * (R - R_star) / (R_minus - R_star + 1e-10)
    
    # Create result dataframe with original data
    result = df.copy()
    result['S'] = S
    result['R'] = R
    result['Q'] = Q
    
    # Rank by Q, S, and R (lower is better for all)
    Q_series = pd.Series(Q)
    S_series = pd.Series(S)
    R_series = pd.Series(R)
    
    result['Q_Rank'] = Q_series.rank(ascending=True, method='min').astype(int)
    result['S_Rank'] = S_series.rank(ascending=True, method='min').astype(int)
    result['R_Rank'] = R_series.rank(ascending=True, method='min').astype(int)
    
    # Main ranking by Q
    result['Rank'] = result['Q_Rank']
    
    # Sort by Q rank to check conditions
    sorted_result = result.sort_values('Q_Rank')
    
    # Check C1: Acceptable Advantage
    DQ = 1.0 / (m - 1) if m > 1 else 1.0
    Q_sorted = sorted_result['Q'].values
    
    if m > 1:
        Q_diff = Q_sorted[1] - Q_sorted[0]  # Difference between 1st and 2nd
        C1_satisfied = Q_diff >= DQ
    else:
        C1_satisfied = True
    
    # Check C2: Acceptable Stability in Decision Making
    # The best alternative by Q should also be best by S or R
    best_Q_idx = sorted_result.index[0]
    best_S_idx = result[result['S_Rank'] == 1].index[0]
    best_R_idx = result[result['R_Rank'] == 1].index[0]
    
    C2_satisfied = (best_Q_idx == best_S_idx) or (best_Q_idx == best_R_idx)
    
    # Determine compromise solution
    result['Is_Compromise'] = False
    compromise_set = []
    
    if C1_satisfied and C2_satisfied:
        # Best alternative is the compromise solution
        result.loc[best_Q_idx, 'Is_Compromise'] = True
        compromise_set = [best_Q_idx]
        compromise_status = "Single compromise solution (C1 and C2 satisfied)"
    elif not C1_satisfied:
        # Multiple alternatives in compromise set
        # All alternatives a(M) where Q(a(M)) - Q(a(1)) < DQ
        for idx in sorted_result.index:
            if sorted_result.loc[idx, 'Q'] - Q_sorted[0] < DQ:
                result.loc[idx, 'Is_Compromise'] = True
                compromise_set.append(idx)
        compromise_status = f"Multiple compromise solutions: {len(compromise_set)} alternatives (C1 not satisfied)"
    else:  # C2 not satisfied
        # Compromise set includes best by Q and best by S and R
        for idx in [best_Q_idx, best_S_idx, best_R_idx]:
            result.loc[idx, 'Is_Compromise'] = True
            if idx not in compromise_set:
                compromise_set.append(idx)
        compromise_status = f"Compromise set: {len(compromise_set)} alternatives (C2 not satisfied)"
    
    # Add metadata to result
    result.attrs['C1_satisfied'] = C1_satisfied
    result.attrs['C2_satisfied'] = C2_satisfied
    result.attrs['DQ'] = DQ
    result.attrs['compromise_status'] = compromise_status
    result.attrs['compromise_set'] = compromise_set
    
    # Print verification info if verbose mode enabled
    if verbose:
        print("\n" + "="*70)
        print("VIKOR ACCEPTANCE CONDITIONS CHECK")
        print("="*70)
        print(f"\nC1 (Acceptable Advantage): {'✅ SATISFIED' if C1_satisfied else '❌ NOT SATISFIED'}")
        print(f"   Required difference (DQ): {DQ:.4f}")
        if m > 1:
            print(f"   Actual difference Q(2) - Q(1): {Q_diff:.4f}")
        
        print(f"\nC2 (Acceptable Stability): {'✅ SATISFIED' if C2_satisfied else '❌ NOT SATISFIED'}")
        print(f"   Best by Q: Alternative {result.loc[best_Q_idx].name + 1}")
        print(f"   Best by S: Alternative {result.loc[best_S_idx].name + 1}")
        print(f"   Best by R: Alternative {result.loc[best_R_idx].name + 1}")
        
        print(f"\n📊 COMPROMISE SOLUTION:")
        print(f"   {compromise_status}")
        print(f"   Alternative(s): {[i+1 for i in compromise_set]}")
        print("="*70 + "\n")
    
    # Return in ORIGINAL order (no sorting)
    return result