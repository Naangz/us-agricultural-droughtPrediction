import os
import re
import matplotlib.pyplot as plt

# Try to import seaborn for a cleaner aesthetic, fallback if not available
try:
    import seaborn as sns
    sns.set_theme(style="whitegrid")
except ImportError:
    pass

# Hardcoded fallback values in case files are missing or moved
FALLBACK_SCORES = {
    'kansas': {
        15: 0.4460,  # Scenario 2
        20: 0.4376,  # Scenario 2A
        25: 0.8014,  # Scenario 2B
        30: 0.8014   # Scenario 2C
    },
    'nebraska': {
        15: 0.5262,  # Scenario 2
        20: 0.4942,  # Scenario 2A
        25: 0.7517,  # Scenario 2B
        30: 0.7603   # Scenario 2C
    }
}

def extract_macro_f1(file_path):
    """
    Extracts the tuned Macro F1 score from the results_summary.txt file.
    Looks for the line starting with "Macro F1:" (avoiding "Macro F1 (raw):").
    """
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Regex to match "Macro F1: 0.xxxx" but not "Macro F1 (raw): 0.xxxx"
        match = re.search(r'(?<!\(raw\)\s)Macro\s+F1:\s*([\d\.]+)', content)
        if match:
            return float(match.group(1))
        
        # Fallback to parse classification report macro average f1-score
        report_match = re.search(r'macro avg\s+[\d\.]+\s+[\d\.]+\s+([\d\.]+)', content)
        if report_match:
            return float(report_match.group(1))
    except Exception as e:
        print(f"Warning: Error reading {file_path}: {e}")
    return None

def get_data():
    """
    Get the F1 scores for Kansas and Nebraska from files or fallback defaults.
    """
    features = [15, 20, 25, 30]
    scenarios = {
        15: 'scenario2',
        20: 'scenario2A',
        25: 'scenario2B',
        30: 'scenario2C'
    }
    
    data = {'kansas': [], 'nebraska': []}
    
    for region in ['kansas', 'nebraska']:
        for num_feat in features:
            scen = scenarios[num_feat]
            # Construct path to results_summary.txt
            file_path = os.path.join(region, f"output_weekly_{region}_{scen}", "results_summary.txt")
            
            val = extract_macro_f1(file_path)
            if val is not None:
                data[region].append(val)
                print(f"Extracted {region.upper()} {scen} ({num_feat} features): Macro F1 = {val:.4f}")
            else:
                fallback_val = FALLBACK_SCORES[region][num_feat]
                data[region].append(fallback_val)
                print(f"Using fallback for {region.upper()} {scen} ({num_feat} features): Macro F1 = {fallback_val:.4f} (file not found/parsed)")
                
    return features, data['kansas'], data['nebraska']

def main():
    features, kansas_f1, nebraska_f1 = get_data()
    
    # ------------------ STYLING & SETUP ------------------
    # Custom matplotlib parameters for premium presentation
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica', 'Segoe UI'],
        'axes.edgecolor': '#cccccc',
        'axes.linewidth': 0.8,
        'xtick.color': '#444444',
        'ytick.color': '#444444',
        'text.color': '#222222',
        'axes.labelcolor': '#222222',
        'axes.titlesize': 14,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
    })
    
    # Create side-by-side subplots (1 row, 2 columns)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.patch.set_facecolor('#F8F9FA') # Soft light gray background
    
    regions = [
        {
            'name': 'Kansas',
            'ax': ax1,
            'f1': kansas_f1,
            'color': '#0288D1', # Deep Sky Blue
            'marker': 'o',
            'jump_text': 'Kansas jump: +36.38% F1\n(0.4376 \u2192 0.8014)',
            'key_features': 'Key features added in Scenario 2B:\n- D2_lag1, D4_lag1\n- severe_carryover_lag1\n- week_sin, PREC_lag8'
        },
        {
            'name': 'Nebraska',
            'ax': ax2,
            'f1': nebraska_f1,
            'color': '#F57C00', # Vibrant Orange
            'marker': 's', # Square marker
            'jump_text': 'Nebraska jump: +25.75% F1\n(0.4942 \u2192 0.7517)',
            'key_features': 'Key features added in Scenario 2B:\n- D2_lag1, D4_lag2\n- severe_carryover_lag1\n- D0_lag2, PREC_lag4'
        }
    ]
    
    # Plot configuration for both subplots
    for r in regions:
        ax = r['ax']
        ax.set_facecolor('#FFFFFF') # White plot area
        ax.grid(True, linestyle='--', alpha=0.5, color='#CCCCCC')
        
        # Plot the main bars
        ax.bar(features, r['f1'], width=3.2, color=r['color'], alpha=0.85,
               edgecolor='white', linewidth=1.5, label=r['name'], zorder=4)
        
        # Shaded Recovery Zone (X=20 to X=25)
        ax.axvspan(20, 25, color='#E8F5E9', alpha=0.6, label='Performance Recovery Zone', zorder=2)
        
        # Annotate each bar with its value
        for x, y in zip(features, r['f1']):
            ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points", 
                        xytext=(0, 10), ha='center', fontweight='bold', fontsize=9.5,
                        bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.3, ec="none"))
            
        # Draw arrow to show the dramatic jump
        ax.annotate(r['jump_text'], 
                    xy=(22.5, (r['f1'][1] + r['f1'][2])/2), 
                    xytext=(16, 0.65),
                    arrowprops=dict(facecolor='#2E7D32', shrink=0.08, width=2, headwidth=8, headlength=8),
                    fontsize=10.5, color='#1B5E20', fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", fc="#E8F5E9", ec="#2E7D32", lw=1))
        
        # Display explanatory text block about added features
        ax.text(15.2, 0.76, r['key_features'], fontsize=9, color='#333333',
                bbox=dict(boxstyle="round,pad=0.4", fc="#F1F8E9", ec="#81C784", lw=0.8))
        
        # Titles and limits
        ax.set_title(f"Wilayah {r['name']}", fontweight='bold', pad=12)
        ax.set_xlabel("Jumlah Fitur (Number of Features)", labelpad=8)
        ax.set_xticks(features)
        ax.set_xlim(13, 32)
        
    # Set y-axis labels and limits
    ax1.set_ylabel("Macro F1-Score", labelpad=10, fontweight='bold')
    ax1.set_ylim(0.35, 0.88)
    
    # Main super title
    plt.suptitle("Analisis Sensitivitas Fitur: Fenomena 'Performance Recovery'\nMatriks F1-Score vs Jumlah Fitur (Scenario 2, 2A, 2B, 2C)", 
                 fontsize=16, fontweight='bold', y=0.96, color='#1A237E')
    
    # Legend (Combining from both ax1 and ax2)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles = handles1 + handles2
    labels = labels1 + labels2
    
    # Keep unique legends while preserving order
    by_label = {}
    for lbl, hnd in zip(labels, handles):
        if lbl not in by_label:
            by_label[lbl] = hnd
            
    fig.legend(by_label.values(), by_label.keys(), loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.01))
    
    plt.tight_layout(rect=[0, 0.07, 1, 0.90])
    
    # Save the output image
    output_filename = "performance_recovery_comparison.png"
    plt.savefig(output_filename, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    print(f"\nPlot saved successfully as '{output_filename}' in the root directory.")
    
if __name__ == "__main__":
    main()
