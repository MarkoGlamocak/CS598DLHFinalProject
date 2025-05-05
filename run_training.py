from bp_dann import train_and_evaluate
import os

if __name__ == "__main__":
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Train models with different amounts of training data
    results = []
    
    for minutes in [3, 4, 5]:
        print(f"\n{'='*50}")
        print(f"TRAINING WITH {minutes} MINUTES OF DATA")
        print(f"{'='*50}\n")
        
        result = train_and_evaluate(data_dir='data', minutes=minutes, epochs=100)
        if result:
            results.append(result)
    
    # Print summary of results
    if results:
        print("\n\n" + "="*80)
        print("SUMMARY OF RESULTS")
        print("="*80)
        print(f"{'Minutes':<10} {'DBP RMSE':<12} {'SBP RMSE':<12} {'DBP r':<10} {'SBP r':<10} {'DBP %':<10} {'SBP %':<10}")
        print("-"*80)
        
        for r in results:
            print(f"{r['minutes']:<10} {r['dbp_rmse']:<12.2f} {r['sbp_rmse']:<12.2f} "
                 f"{r['dbp_r']:<10.2f} {r['sbp_r']:<10.2f} {r['dbp_within_10']:<10.1f} {r['sbp_within_10']:<10.1f}")