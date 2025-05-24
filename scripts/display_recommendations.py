#!/usr/bin/env python3
"""
Display detailed position recommendations from the Unity trading pipeline
"""
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

def display_recommendations():
    # Find the latest recommendations file
    outputs_dir = Path("outputs")
    rec_files = list(outputs_dir.glob("recommendations_*.json"))
    
    if not rec_files:
        print("❌ No recommendation files found in outputs/")
        return
    
    # Get the most recent file
    latest_file = max(rec_files, key=lambda x: x.stat().st_mtime)
    timestamp = latest_file.stem.split('_', 1)[1]
    
    print("=" * 80)
    print(f"🎯 UNITY (U) TRADING RECOMMENDATIONS")
    print(f"📅 Generated: {timestamp}")
    print(f"📁 File: {latest_file.name}")
    print("=" * 80)
    
    # Load recommendations
    with open(latest_file) as f:
        recommendations = json.load(f)
    
    print(f"📊 Total Recommendations: {len(recommendations)}")
    print()
    
    # Separate options recommendations, stock recommendations, and portfolio summary
    options_recs = {}
    stock_recs = {}
    portfolio_summary = None
    
    for pos_id, rec in recommendations.items():
        if pos_id == "_portfolio_summary":
            portfolio_summary = rec
        elif pos_id.startswith("STOCK_"):
            stock_recs[pos_id] = rec
        else:
            options_recs[pos_id] = rec
    
    print(f"📊 Total Recommendations: {len(options_recs) + len(stock_recs)}")
    print(f"   • Options recommendations: {len(options_recs)}")
    print(f"   • Stock recommendations: {len(stock_recs)}")
    print()
    
    # Convert options to DataFrame for analysis
    options_data = []
    for pos_id, rec in options_recs.items():
        options_data.append({
            'Position': pos_id,
            'Action': rec['action'],
            'Size': rec['size'],
            'Confidence': rec['certainty'],
            'Predicted_Move': rec['predicted_move'],
            'Delta': rec['delta'],
            'Strike': rec['strike'],
            'Sharpe_Score': rec.get('sharpe_score', 0),
            'Portfolio_Delta_Impact': rec.get('portfolio_delta_impact', 0),
            'Estimated_Cost': rec.get('estimated_cost', 0)
        })
    
    # Convert stock recommendations for display
    stock_data = []
    for pos_id, rec in stock_recs.items():
        stock_data.append({
            'Position': pos_id,
            'Action': rec['action'],
            'Shares': rec['shares'],
            'Type': rec['type'],
            'Current_Price': rec['current_price'],
            'Delta_Impact': rec['delta_impact'],
            'Estimated_Cost': rec['estimated_cost'],
            'Sharpe_Contribution': rec.get('sharpe_contribution', 0)
        })
    
    df_options = pd.DataFrame(options_data)
    df_stocks = pd.DataFrame(stock_data)
    
        
    # Display portfolio summary first
    if portfolio_summary:
        print("🏦 PORTFOLIO SUMMARY:")
        print(f"   • Current delta: {portfolio_summary.get('current_delta', 'N/A')}")
        print(f"   • Target delta range: {portfolio_summary.get('target_delta_range', 'N/A')}")
        print(f"   • Available cash: ${portfolio_summary.get('available_cash', 0):,.2f}")
        print(f"   • Current positions: {portfolio_summary.get('current_positions', 0)}")
        print(f"   • Portfolio value: ${portfolio_summary.get('portfolio_value', 0):,.2f}")
        print()
    
    # Display options recommendations
    if len(df_options) > 0:
        print("📈 OPTIONS RECOMMENDATIONS:")
        print("-" * 80)
        
        # Summary statistics
        print(f"   • BUY positions: {len(df_options[df_options['Action'] == 'BUY'])}")
        print(f"   • SELL positions: {len(df_options[df_options['Action'] == 'SELL'])}")
        print(f"   • Average confidence: {df_options['Confidence'].mean():.1%}")
        print(f"   • Total position size: {df_options['Size'].sum()} contracts")
        print(f"   • Strike range: ${df_options['Strike'].min():.1f} - ${df_options['Strike'].max():.1f}")
        print()
        
        # Top recommendations by Sharpe score
        print("🌟 TOP OPTIONS RECOMMENDATIONS (by Sharpe score):")
        print("-" * 80)
        top_options = df_options.nlargest(min(10, len(df_options)), 'Sharpe_Score')
        for idx, row in top_options.iterrows():
            cost_str = f"${row['Estimated_Cost']:,.0f}" if row['Estimated_Cost'] > 0 else "N/A"
            delta_impact_str = f"{row['Portfolio_Delta_Impact']:+.1f}" if row['Portfolio_Delta_Impact'] != 0 else "N/A"
            print(f"   {row['Position']:12} | {row['Action']:4} | Size: {row['Size']:2} | "
                  f"Conf: {row['Confidence']:5.1%} | Sharpe: {row['Sharpe_Score']:5.1f} | "
                  f"Strike: ${row['Strike']:6.1f} | Δ Impact: {delta_impact_str:>8} | Cost: {cost_str:>10}")
        print()
        
        # Action breakdown for options
        print("📋 OPTIONS BREAKDOWN BY ACTION:")
        print("-" * 80)
        
        for action in ['BUY', 'SELL']:
            action_recs = df_options[df_options['Action'] == action].sort_values('Sharpe_Score', ascending=False)
            if len(action_recs) > 0:
                print(f"\n🔹 {action} POSITIONS ({len(action_recs)} total):")
                for idx, row in action_recs.iterrows():
                    cost_str = f"${row['Estimated_Cost']:,.0f}" if row['Estimated_Cost'] > 0 else "N/A"
                    delta_impact_str = f"{row['Portfolio_Delta_Impact']:+.1f}" if row['Portfolio_Delta_Impact'] != 0 else "N/A"
                    print(f"   {row['Position']:12} | Size: {row['Size']:2} | "
                          f"Conf: {row['Confidence']:5.1%} | Move: {row['Predicted_Move']:6.3f} | "
                          f"Delta: {row['Delta']:6.3f} | Δ Impact: {delta_impact_str:>8} | Cost: {cost_str:>10}")
    
    # Display stock recommendations
    if len(df_stocks) > 0:
        print("\n📊 STOCK RECOMMENDATIONS:")
        print("-" * 80)
        
        for idx, row in df_stocks.iterrows():
            ticker = row['Position'].replace('STOCK_', '')
            cost_str = f"${row['Estimated_Cost']:,.2f}" if row['Estimated_Cost'] > 0 else "N/A"
            delta_impact_str = f"{row['Delta_Impact']:+.0f}" if row['Delta_Impact'] != 0 else "N/A"
            sharpe_str = f"{row['Sharpe_Contribution']:5.2f}" if row['Sharpe_Contribution'] != 0 else "N/A"
            
            print(f"   {ticker:6} | {row['Action']:4} | Shares: {row['Shares']:3} | "
                  f"Price: ${row['Current_Price']:6.2f} | Δ Impact: {delta_impact_str:>8} | "
                  f"Cost: {cost_str:>10} | Sharpe: {sharpe_str:>8}")
        print()
    
    print("\n" + "=" * 80)
    print("📝 TRADING STRATEGY NOTES:")
    print("   • OPTIONS:")
    print("     - BUY: Open long positions in these options")
    print("     - SELL: Open short positions in these options")
    print("     - Sharpe score = risk-adjusted return potential")
    print("     - Δ Impact = expected change to portfolio delta")
    print("   • STOCKS:")
    print("     - BUY/SELL: Adjust stock position to balance portfolio delta")
    print("     - Δ Impact = direct delta exposure change")
    print("   • Higher confidence = stronger recommendation")
    print("   • Predicted move = expected price movement")
    print("=" * 80)
    
    # Save summary to text file
    summary_file = outputs_dir / f"recommendations_summary_{timestamp}.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Unity Trading Recommendations Summary\n")
        f.write(f"Generated: {timestamp}\n")
        f.write(f"Total Recommendations: {len(options_recs) + len(stock_recs)}\n\n")
        
        if len(df_options) > 0:
            f.write("OPTIONS RECOMMENDATIONS:\n")
            f.write(df_options.to_string(index=False))
            f.write("\n\n")
        
        if len(df_stocks) > 0:
            f.write("STOCK RECOMMENDATIONS:\n")
            f.write(df_stocks.to_string(index=False))
            f.write("\n\n")
        
        if portfolio_summary:
            f.write("PORTFOLIO SUMMARY:\n")
            for key, value in portfolio_summary.items():
                f.write(f"{key}: {value}\n")
    
    print(f"💾 Summary saved to: {summary_file.name}")

if __name__ == "__main__":
    display_recommendations()
