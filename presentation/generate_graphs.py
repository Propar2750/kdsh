"""
KDSH Hackathon - Claim Verification Pipeline Improvement Visualization
Generates beautiful graphs showing our iterative improvement journey.

Run: python presentation/generate_graphs.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.ticker import MaxNLocator

# ============================================================================
# Style Configuration - Black background with red (#e10600) accent
# ============================================================================

# Color palette
BACKGROUND_COLOR = '#0a0a0a'  # Near black
TEXT_COLOR = '#ffffff'  # White
PRIMARY_COLOR = '#e10600'  # Ferrari red
SECONDARY_COLOR = '#ff4444'  # Lighter red
ACCENT_COLOR = '#ff8888'  # Even lighter red for highlights
GRID_COLOR = '#333333'  # Dark gray for grid
SUCCESS_COLOR = '#00e106'  # Green for success markers

# Apply dark style
plt.style.use('dark_background')
plt.rcParams.update({
    'figure.facecolor': BACKGROUND_COLOR,
    'axes.facecolor': BACKGROUND_COLOR,
    'axes.edgecolor': TEXT_COLOR,
    'axes.labelcolor': TEXT_COLOR,
    'text.color': TEXT_COLOR,
    'xtick.color': TEXT_COLOR,
    'ytick.color': TEXT_COLOR,
    'grid.color': GRID_COLOR,
    'grid.alpha': 0.3,
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'legend.facecolor': BACKGROUND_COLOR,
    'legend.edgecolor': PRIMARY_COLOR,
})

# ============================================================================
# Data from our experiment runs
# ============================================================================

# Key test runs with their results
test_data = {
    'runs': ['test6', 'test7', 'test10', 'test15', 'test17', 'test20', 
             'test24', 'test27', 'test32', 'test37', 'test40', 'test44', 'test45'],
    'overall': [52.5, 65.0, 43.3, 56.7, 66.7, 70.0, 60.0, 72.5, 75.0, 70.0, 67.5, 63.8, 58.8],
    'consistent': [45.1, 100.0, 11.1, 50.0, 72.2, 77.8, 96.0, 72.0, 76.0, 72.0, 68.0, 68.6, 66.7],
    'contradict': [65.5, 0.0, 91.7, 66.7, 58.3, 58.3, 0.0, 73.3, 73.3, 66.7, 66.7, 55.2, 44.8],
    'milestones': [False, False, False, True, False, True, False, True, True, True, True, True, False]
}

# Phase labels
phases = {
    'test6': 'Initial\nStruggle',
    'test7': 'Too\nConservative',
    'test10': 'Too\nAggressive',
    'test15': 'First\nBalance',
    'test17': 'Query\nExpansion',
    'test20': 'Prompt\nImproved',
    'test24': 'Crisis\n(0% contra)',
    'test27': 'Breakthrough\n(72%/73%)',
    'test32': 'Best 40\n(76%/73%)',
    'test37': 'Stable\nBalance',
    'test40': 'FP\nFiltering',
    'test44': 'Best 80\n(69%/55%)',
    'test45': 'Failed\n(67%/45%)'
}

# ============================================================================
# Graph 1: Overall Progress Line Chart
# ============================================================================

def create_progress_chart():
    """Create the main progress chart showing all three metrics over time."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(test_data['runs']))
    
    # Plot lines with markers
    ax.plot(x, test_data['overall'], 'o-', color=TEXT_COLOR, linewidth=2.5, 
            markersize=10, label='Overall Accuracy', zorder=5)
    ax.plot(x, test_data['consistent'], 's--', color=PRIMARY_COLOR, linewidth=2.5, 
            markersize=10, label='Consistent Accuracy', zorder=5)
    ax.plot(x, test_data['contradict'], '^:', color=SECONDARY_COLOR, linewidth=2.5, 
            markersize=10, label='Contradict Accuracy', zorder=5)
    
    # Add 60% target line
    ax.axhline(y=60, color=SUCCESS_COLOR, linestyle='--', linewidth=2, 
               alpha=0.7, label='60% Target')
    
    # Highlight milestone runs
    for i, is_milestone in enumerate(test_data['milestones']):
        if is_milestone:
            ax.axvspan(i-0.3, i+0.3, alpha=0.15, color=PRIMARY_COLOR)
    
    # Styling
    ax.set_xlabel('Experiment Runs', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Claim Verification Pipeline: Improvement Journey', 
                 fontsize=18, fontweight='bold', pad=20)
    
    ax.set_xticks(x)
    ax.set_xticklabels([phases[r] for r in test_data['runs']], fontsize=9, rotation=45, ha='right')
    ax.set_ylim(0, 105)
    ax.set_xlim(-0.5, len(x)-0.5)
    
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Add annotations for key moments
    ax.annotate('Crisis!', xy=(6, 0), xytext=(6, 15),
                fontsize=10, color=PRIMARY_COLOR, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=PRIMARY_COLOR),
                ha='center')
    
    ax.annotate('Breakthrough!', xy=(7, 73.3), xytext=(7, 88),
                fontsize=10, color=SUCCESS_COLOR, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=SUCCESS_COLOR),
                ha='center')
    
    plt.tight_layout()
    plt.savefig('presentation/graphs/01_progress_chart.png', dpi=150, 
                facecolor=BACKGROUND_COLOR, edgecolor='none', bbox_inches='tight')
    plt.close()
    print("✅ Created: 01_progress_chart.png")

# ============================================================================
# Graph 2: Consistent vs Contradict Balance
# ============================================================================

def create_balance_chart():
    """Create a scatter plot showing the balance between consistent and contradict accuracy."""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create scatter plot
    scatter = ax.scatter(test_data['consistent'], test_data['contradict'], 
                         c=range(len(test_data['runs'])), cmap='Reds', 
                         s=200, edgecolors=TEXT_COLOR, linewidths=2, zorder=5)
    
    # Add run labels
    for i, run in enumerate(test_data['runs']):
        ax.annotate(run.replace('test', 't'), 
                    (test_data['consistent'][i], test_data['contradict'][i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=9,
                    color=TEXT_COLOR)
    
    # Add target zone (60%+ on both)
    ax.axvline(x=60, color=SUCCESS_COLOR, linestyle='--', linewidth=2, alpha=0.7)
    ax.axhline(y=60, color=SUCCESS_COLOR, linestyle='--', linewidth=2, alpha=0.7)
    ax.fill_between([60, 100], 60, 100, alpha=0.1, color=SUCCESS_COLOR)
    ax.text(80, 80, 'TARGET\nZONE', fontsize=14, fontweight='bold', 
            color=SUCCESS_COLOR, ha='center', va='center')
    
    # Draw path showing evolution
    for i in range(len(test_data['runs'])-1):
        ax.annotate('', xy=(test_data['consistent'][i+1], test_data['contradict'][i+1]),
                    xytext=(test_data['consistent'][i], test_data['contradict'][i]),
                    arrowprops=dict(arrowstyle='->', color=PRIMARY_COLOR, alpha=0.4, lw=1.5))
    
    # Styling
    ax.set_xlabel('Consistent Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Contradict Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Finding the Balance: Consistent vs Contradict', 
                 fontsize=18, fontweight='bold', pad=20)
    
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, label='Experiment Progress')
    cbar.set_ticks([0, len(test_data['runs'])-1])
    cbar.set_ticklabels(['Early', 'Final'])
    
    plt.tight_layout()
    plt.savefig('presentation/graphs/02_balance_chart.png', dpi=150,
                facecolor=BACKGROUND_COLOR, edgecolor='none', bbox_inches='tight')
    plt.close()
    print("✅ Created: 02_balance_chart.png")

# ============================================================================
# Graph 3: Before/After Comparison
# ============================================================================

def create_comparison_chart():
    """Create a bar chart comparing initial vs final results."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    categories = ['Overall\nAccuracy', 'Consistent\nAccuracy', 'Contradict\nAccuracy']
    initial = [52.5, 45.1, 65.5]  # test6
    final = [63.8, 68.6, 55.2]    # test44
    best_40 = [75.0, 76.0, 73.3]  # test32
    
    x = np.arange(len(categories))
    width = 0.25
    
    bars1 = ax.bar(x - width, initial, width, label='Initial (test6)', 
                   color='#333333', edgecolor=TEXT_COLOR, linewidth=2)
    bars2 = ax.bar(x, final, width, label='Final 80-sample (test44)', 
                   color=PRIMARY_COLOR, edgecolor=TEXT_COLOR, linewidth=2)
    bars3 = ax.bar(x + width, best_40, width, label='Best 40-sample (test32)', 
                   color=SECONDARY_COLOR, edgecolor=TEXT_COLOR, linewidth=2)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add 60% target line
    ax.axhline(y=60, color=SUCCESS_COLOR, linestyle='--', linewidth=2, 
               alpha=0.7, label='60% Target')
    
    # Styling
    ax.set_xlabel('Metric', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Performance Comparison: Initial vs Final', 
                 fontsize=18, fontweight='bold', pad=20)
    
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 90)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('presentation/graphs/03_comparison_chart.png', dpi=150,
                facecolor=BACKGROUND_COLOR, edgecolor='none', bbox_inches='tight')
    plt.close()
    print("✅ Created: 03_comparison_chart.png")

# ============================================================================
# Graph 4: Key Improvements Timeline
# ============================================================================

def create_timeline_chart():
    """Create a timeline showing key improvements and their impact."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    improvements = [
        ('test6', 'Baseline', 52.5),
        ('test15', 'Structured\nPrompt', 56.7),
        ('test17', 'Query\nExpansion', 66.7),
        ('test27', 'Balanced\nPrompt', 72.5),
        ('test32', 'FP\nFiltering', 75.0),
        ('test44', 'Verification\nPriority', 63.8),
    ]
    
    x_pos = np.arange(len(improvements))
    accuracies = [imp[2] for imp in improvements]
    labels = [imp[1] for imp in improvements]
    
    # Create bar chart with gradient effect
    bars = ax.bar(x_pos, accuracies, color=PRIMARY_COLOR, edgecolor=TEXT_COLOR, 
                  linewidth=2, width=0.6)
    
    # Add gradient effect to bars
    for i, bar in enumerate(bars):
        bar.set_alpha(0.5 + 0.5 * (i / len(bars)))
    
    # Add improvement arrows
    for i in range(len(accuracies)-1):
        diff = accuracies[i+1] - accuracies[i]
        color = SUCCESS_COLOR if diff > 0 else PRIMARY_COLOR
        symbol = '↑' if diff > 0 else '↓'
        ax.annotate(f'{symbol}{abs(diff):.1f}%', 
                    xy=(i + 0.5, max(accuracies[i], accuracies[i+1]) + 3),
                    fontsize=10, color=color, fontweight='bold', ha='center')
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        ax.annotate(f'{acc:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold',
                    color=TEXT_COLOR)
    
    # Add 60% target line
    ax.axhline(y=60, color=SUCCESS_COLOR, linestyle='--', linewidth=2, alpha=0.7)
    ax.text(len(improvements)-0.5, 61, '60% Target', fontsize=10, color=SUCCESS_COLOR)
    
    # Styling
    ax.set_xlabel('Key Improvements', fontsize=14, fontweight='bold')
    ax.set_ylabel('Overall Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Key Improvements and Their Impact', 
                 fontsize=18, fontweight='bold', pad=20)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 85)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('presentation/graphs/04_timeline_chart.png', dpi=150,
                facecolor=BACKGROUND_COLOR, edgecolor='none', bbox_inches='tight')
    plt.close()
    print("✅ Created: 04_timeline_chart.png")

# ============================================================================
# Graph 5: Final Results Donut Chart
# ============================================================================

def create_final_results_chart():
    """Create a donut chart showing final classification results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # Consistent results
    consistent_correct = 35
    consistent_wrong = 51 - 35
    
    ax1 = axes[0]
    sizes1 = [consistent_correct, consistent_wrong]
    colors1 = [SUCCESS_COLOR, PRIMARY_COLOR]
    explode1 = (0.05, 0)
    
    wedges1, texts1, autotexts1 = ax1.pie(sizes1, explode=explode1, colors=colors1,
                                           autopct='%1.1f%%', startangle=90,
                                           wedgeprops=dict(width=0.5, edgecolor=TEXT_COLOR),
                                           textprops=dict(color=TEXT_COLOR, fontsize=14, fontweight='bold'))
    ax1.set_title('Consistent Samples\n(51 total)', fontsize=16, fontweight='bold', pad=20)
    ax1.legend(['Correct (35)', 'Wrong (16)'], loc='lower center', fontsize=11)
    
    # Add center text
    ax1.text(0, 0, '68.6%', ha='center', va='center', fontsize=24, 
             fontweight='bold', color=SUCCESS_COLOR)
    
    # Contradict results
    contradict_correct = 16
    contradict_wrong = 29 - 16
    
    ax2 = axes[1]
    sizes2 = [contradict_correct, contradict_wrong]
    colors2 = [SUCCESS_COLOR, PRIMARY_COLOR]
    explode2 = (0.05, 0)
    
    wedges2, texts2, autotexts2 = ax2.pie(sizes2, explode=explode2, colors=colors2,
                                           autopct='%1.1f%%', startangle=90,
                                           wedgeprops=dict(width=0.5, edgecolor=TEXT_COLOR),
                                           textprops=dict(color=TEXT_COLOR, fontsize=14, fontweight='bold'))
    ax2.set_title('Contradict Samples\n(29 total)', fontsize=16, fontweight='bold', pad=20)
    ax2.legend(['Correct (16)', 'Wrong (13)'], loc='lower center', fontsize=11)
    
    # Add center text
    ax2.text(0, 0, '55.2%', ha='center', va='center', fontsize=24, 
             fontweight='bold', color=SECONDARY_COLOR)
    
    plt.suptitle('Final Results Breakdown (test44 - 80 samples)', 
                 fontsize=20, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig('presentation/graphs/05_final_results_chart.png', dpi=150,
                facecolor=BACKGROUND_COLOR, edgecolor='none', bbox_inches='tight')
    plt.close()
    print("✅ Created: 05_final_results_chart.png")

# ============================================================================
# Graph 6: Architecture Diagram
# ============================================================================

def create_architecture_diagram():
    """Create a visual representation of the pipeline architecture."""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Define box positions and sizes
    boxes = [
        {'name': 'Input\nBackstory', 'pos': (1, 7), 'size': (2, 1.5)},
        {'name': 'Claim\nExtractor', 'pos': (4.5, 7), 'size': (2, 1.5)},
        {'name': 'Hybrid\nRetriever', 'pos': (8, 7), 'size': (2, 1.5)},
        {'name': 'LLM\nVerifier', 'pos': (11.5, 7), 'size': (2, 1.5)},
        {'name': 'Aggregator', 'pos': (11.5, 4), 'size': (2, 1.5)},
        {'name': 'Final\nPrediction', 'pos': (11.5, 1), 'size': (2, 1.5)},
    ]
    
    # Draw boxes
    for box in boxes:
        rect = mpatches.FancyBboxPatch(
            box['pos'], box['size'][0], box['size'][1],
            boxstyle="round,pad=0.05,rounding_size=0.2",
            facecolor=BACKGROUND_COLOR, edgecolor=PRIMARY_COLOR, linewidth=3
        )
        ax.add_patch(rect)
        ax.text(box['pos'][0] + box['size'][0]/2, box['pos'][1] + box['size'][1]/2,
                box['name'], ha='center', va='center', fontsize=12, 
                fontweight='bold', color=TEXT_COLOR)
    
    # Draw arrows
    arrows = [
        ((3, 7.75), (4.5, 7.75)),      # Input → Extractor
        ((6.5, 7.75), (8, 7.75)),       # Extractor → Retriever
        ((10, 7.75), (11.5, 7.75)),     # Retriever → Verifier
        ((12.5, 7), (12.5, 5.5)),       # Verifier → Aggregator
        ((12.5, 4), (12.5, 2.5)),       # Aggregator → Output
    ]
    
    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle='->', color=PRIMARY_COLOR, lw=3))
    
    # Add retriever sub-components
    ax.text(9, 5.5, 'BM25', ha='center', va='center', fontsize=10, 
            color=SECONDARY_COLOR, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=BACKGROUND_COLOR, edgecolor=SECONDARY_COLOR))
    ax.text(9, 4.5, 'Vector', ha='center', va='center', fontsize=10, 
            color=SECONDARY_COLOR, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=BACKGROUND_COLOR, edgecolor=SECONDARY_COLOR))
    ax.text(9, 3.5, 'RRF Fusion', ha='center', va='center', fontsize=10, 
            color=ACCENT_COLOR, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=BACKGROUND_COLOR, edgecolor=ACCENT_COLOR))
    
    # Connect retriever sub-components
    ax.annotate('', xy=(9, 6.5), xytext=(9, 8.5),
                arrowprops=dict(arrowstyle='-', color=GRID_COLOR, lw=1, ls='--'))
    
    # Add title
    ax.text(8, 9.5, 'Pipeline Architecture', ha='center', va='center',
            fontsize=22, fontweight='bold', color=TEXT_COLOR)
    
    # Add stats box
    stats_text = "Stats:\n- 5 claims per backstory\n- 8 evidence chunks\n- Groq llama-3.1-8b\n- ~3 sec/sample"
    ax.text(2, 3, stats_text, ha='left', va='top', fontsize=11, color=TEXT_COLOR,
            bbox=dict(boxstyle='round', facecolor=BACKGROUND_COLOR, edgecolor=PRIMARY_COLOR, linewidth=2))
    
    plt.tight_layout()
    plt.savefig('presentation/graphs/06_architecture_diagram.png', dpi=150,
                facecolor=BACKGROUND_COLOR, edgecolor='none', bbox_inches='tight')
    plt.close()
    print("✅ Created: 06_architecture_diagram.png")

# ============================================================================
# Main execution
# ============================================================================

if __name__ == '__main__':
    import os
    
    # Create output directory
    os.makedirs('presentation/graphs', exist_ok=True)
    
    print("🎨 Generating presentation graphs...")
    print("=" * 50)
    
    create_progress_chart()
    create_balance_chart()
    create_comparison_chart()
    create_timeline_chart()
    create_final_results_chart()
    create_architecture_diagram()
    
    print("=" * 50)
    print("✅ All graphs generated successfully!")
    print("📁 Output folder: presentation/graphs/")
