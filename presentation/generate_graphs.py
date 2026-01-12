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
             'test24', 'test27', 'test32', 'test37', 'test40', 'test44', 'test45', 'test46'],
    'overall': [52.5, 65.0, 43.3, 56.7, 66.7, 70.0, 60.0, 72.5, 75.0, 70.0, 67.5, 63.8, 58.8, 66.2],
    'consistent': [45.1, 100.0, 11.1, 50.0, 72.2, 77.8, 96.0, 72.0, 76.0, 72.0, 68.0, 68.6, 66.7, 68.6],
    'contradict': [65.5, 0.0, 91.7, 66.7, 58.3, 58.3, 0.0, 73.3, 73.3, 66.7, 66.7, 55.2, 44.8, 62.1],
    'milestones': [False, False, False, True, False, True, False, True, True, True, True, True, False, True]
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
    'test44': 'Stable 80\n(69%/55%)',
    'test45': 'Failed\n(67%/45%)',
    'test46': '🎉 GOAL!\n(69%/62%)'
}

# ============================================================================
# Graph 1: Overall Progress Line Chart
# ============================================================================

def create_progress_chart():
    """Create the main progress chart showing all three metrics over time."""
    fig, ax = plt.subplots(figsize=(16, 9))
    
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
    ax.set_title('Claim Verification Pipeline: The Journey to 60%+ on Both Classes', 
                 fontsize=20, fontweight='bold', pad=20)
    
    ax.set_xticks(x)
    ax.set_xticklabels([phases[r] for r in test_data['runs']], fontsize=9, rotation=45, ha='right')
    ax.set_ylim(0, 105)
    ax.set_xlim(-0.5, len(x)-0.5)
    
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Add key moment annotations with better styling
    # Crisis annotation
    ax.annotate('CRISIS!\n0% Contradict', xy=(6, 0), xytext=(6, 20),
                fontsize=11, color=PRIMARY_COLOR, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=PRIMARY_COLOR, lw=2),
                ha='center', bbox=dict(boxstyle='round,pad=0.3', facecolor=BACKGROUND_COLOR, edgecolor=PRIMARY_COLOR))
    
    # Breakthrough annotation
    ax.annotate('First\nBreakthrough!', xy=(7, 73.3), xytext=(7.5, 88),
                fontsize=11, color=SUCCESS_COLOR, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=SUCCESS_COLOR, lw=2),
                ha='center', bbox=dict(boxstyle='round,pad=0.3', facecolor=BACKGROUND_COLOR, edgecolor=SUCCESS_COLOR))
    
    # Failed experiment annotation
    ax.annotate('Failed\nExperiment', xy=(12, 44.8), xytext=(11.5, 30),
                fontsize=10, color=PRIMARY_COLOR, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=PRIMARY_COLOR, lw=2),
                ha='center', bbox=dict(boxstyle='round,pad=0.3', facecolor=BACKGROUND_COLOR, edgecolor=PRIMARY_COLOR))
    
    # GOAL ACHIEVED annotation
    ax.annotate('GOAL\nACHIEVED!', xy=(13, 62.1), xytext=(13, 80),
                fontsize=12, color='#00ff00', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#00ff00', lw=3),
                ha='center', bbox=dict(boxstyle='round,pad=0.5', facecolor=BACKGROUND_COLOR, edgecolor='#00ff00', linewidth=2))
    
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
    fig, ax = plt.subplots(figsize=(14, 8))
    
    categories = ['Overall\nAccuracy', 'Consistent\nAccuracy', 'Contradict\nAccuracy']
    initial = [52.5, 45.1, 65.5]  # test6
    final = [66.2, 68.6, 62.1]    # test46 - GOAL ACHIEVED!
    best_40 = [75.0, 76.0, 73.3]  # test32
    
    x = np.arange(len(categories))
    width = 0.22
    
    bars1 = ax.bar(x - width, initial, width, label='Initial (test6)', 
                   color='#333333', edgecolor=TEXT_COLOR, linewidth=2)
    bars2 = ax.bar(x, final, width, label='Final (test46) - GOAL!', 
                   color=SUCCESS_COLOR, edgecolor=TEXT_COLOR, linewidth=2)
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
    ax.set_title('Performance Comparison: Initial vs Final (GOAL ACHIEVED!)', 
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
# Graph 5: Final Results Donut Chart (Improved)
# ============================================================================

def create_final_results_chart():
    """Create professional donut charts showing final classification results."""
    fig = plt.figure(figsize=(16, 8))
    
    # Data
    total_correct = 53
    total_wrong = 27
    consistent_correct = 35
    consistent_wrong = 16
    contradict_correct = 18
    contradict_wrong = 11
    
    # Create 3 subplots
    gs = fig.add_gridspec(1, 3, wspace=0.3)
    
    def draw_donut(ax, correct, wrong, title, subtitle, highlight_color):
        """Draw a single donut chart with center text."""
        total = correct + wrong
        accuracy = correct / total * 100
        
        # Donut data
        sizes = [correct, wrong]
        colors = [SUCCESS_COLOR, '#444444']  # Green for correct, dark gray for wrong
        
        # Draw outer ring
        wedges, texts, autotexts = ax.pie(
            sizes, 
            colors=colors,
            autopct='',
            startangle=90,
            wedgeprops=dict(width=0.35, edgecolor=BACKGROUND_COLOR, linewidth=2),
        )
        
        # Draw inner decorative ring
        inner_circle = mpatches.Circle((0, 0), 0.5, facecolor=BACKGROUND_COLOR, 
                                        edgecolor=highlight_color, linewidth=3)
        ax.add_patch(inner_circle)
        
        # Center percentage
        ax.text(0, 0.08, f'{accuracy:.1f}%', ha='center', va='center',
               fontsize=36, fontweight='bold', color=highlight_color)
        
        # Center label
        ax.text(0, -0.25, f'{correct}/{total}', ha='center', va='center',
               fontsize=14, color=ACCENT_COLOR)
        
        # Title above
        ax.set_title(title, fontsize=16, fontweight='bold', color=TEXT_COLOR, pad=15)
        
        # Subtitle below
        ax.text(0, -1.4, subtitle, ha='center', va='center',
               fontsize=11, color=TEXT_COLOR)
        
        # Add legend-style labels
        ax.text(0, -1.65, f'✓ Correct: {correct}  ✗ Wrong: {wrong}', ha='center', va='center',
               fontsize=10, color=ACCENT_COLOR)
        
        ax.set_aspect('equal')
    
    # Overall Accuracy (center, slightly larger)
    ax1 = fig.add_subplot(gs[0, 1])
    draw_donut(ax1, total_correct, total_wrong, 
               'OVERALL', '80 samples total', TEXT_COLOR)
    
    # Consistent Accuracy (left)
    ax2 = fig.add_subplot(gs[0, 0])
    draw_donut(ax2, consistent_correct, consistent_wrong,
               'CONSISTENT', '51 samples', PRIMARY_COLOR)
    
    # Contradict Accuracy (right)  
    ax3 = fig.add_subplot(gs[0, 2])
    draw_donut(ax3, contradict_correct, contradict_wrong,
               'CONTRADICT', '29 samples', SECONDARY_COLOR)
    
    # Main title
    fig.suptitle('Final Results', 
                 fontsize=22, fontweight='bold', color=TEXT_COLOR, y=0.98)
    
    # Add target indicator
    fig.text(0.5, 0.02, '60% Target Achieved on All Metrics', ha='center',
            fontsize=14, color=SUCCESS_COLOR, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.93])
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
# Graph 7: Confusion Matrix Heatmap (Improved)
# ============================================================================

def create_confusion_matrix():
    """Create a clean, professional confusion matrix."""
    import json
    
    # Load results from eval_results_fast.json
    with open('eval_results_fast.json', 'r') as f:
        data = json.load(f)
    
    results = data['results']
    
    # Calculate confusion matrix values
    tp = sum(1 for r in results if r['true'] == 'contradict' and r['pred'] == 'contradict')  # True Positive
    tn = sum(1 for r in results if r['true'] == 'consistent' and r['pred'] == 'consistent')  # True Negative
    fp = sum(1 for r in results if r['true'] == 'consistent' and r['pred'] == 'contradict')  # False Positive
    fn = sum(1 for r in results if r['true'] == 'contradict' and r['pred'] == 'consistent')  # False Negative
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    fig, (ax_matrix, ax_metrics) = plt.subplots(1, 2, figsize=(14, 7), 
                                                 gridspec_kw={'width_ratios': [1.2, 1]})
    
    # Left side: Confusion Matrix with custom colors
    # Use green for correct (diagonal), red for errors (off-diagonal)
    cell_colors = [
        [SUCCESS_COLOR, PRIMARY_COLOR],  # Row 0: TN (correct), FP (error)
        [PRIMARY_COLOR, SUCCESS_COLOR]   # Row 1: FN (error), TP (correct)
    ]
    
    # Draw cells manually for better control
    cell_values = [[tn, fp], [fn, tp]]
    cell_labels = [['True Negative', 'False Positive'], ['False Negative', 'True Positive']]
    
    for i in range(2):
        for j in range(2):
            # Draw rectangle
            rect = mpatches.Rectangle((j-0.5, i-0.5), 1, 1, 
                                  facecolor=cell_colors[i][j], alpha=0.7,
                                  edgecolor=TEXT_COLOR, linewidth=3)
            ax_matrix.add_patch(rect)
            
            # Add count (large)
            ax_matrix.text(j, i, f'{cell_values[i][j]}', ha='center', va='center',
                          fontsize=48, fontweight='bold', color=TEXT_COLOR)
            
            # Add label (small, below count)
            ax_matrix.text(j, i+0.35, cell_labels[i][j], ha='center', va='center',
                          fontsize=10, color=TEXT_COLOR, alpha=0.8)
    
    ax_matrix.set_xlim(-0.5, 1.5)
    ax_matrix.set_ylim(-0.5, 1.5)
    ax_matrix.set_xticks([0, 1])
    ax_matrix.set_yticks([0, 1])
    ax_matrix.set_xticklabels(['Consistent', 'Contradict'], fontsize=14, fontweight='bold')
    ax_matrix.set_yticklabels(['Consistent', 'Contradict'], fontsize=14, fontweight='bold')
    ax_matrix.set_xlabel('PREDICTED', fontsize=16, fontweight='bold', labelpad=15)
    ax_matrix.set_ylabel('ACTUAL', fontsize=16, fontweight='bold', labelpad=15)
    ax_matrix.invert_yaxis()
    
    # Right side: Metrics display
    ax_metrics.axis('off')
    
    # Title for metrics
    ax_metrics.text(0.5, 0.95, 'Performance Metrics', ha='center', va='top',
                   fontsize=18, fontweight='bold', color=TEXT_COLOR, transform=ax_metrics.transAxes)
    
    # Metrics with visual bars
    metrics = [
        ('Accuracy', accuracy, TEXT_COLOR),
        ('Precision', precision, PRIMARY_COLOR),
        ('Recall', recall, SECONDARY_COLOR),
        ('F1 Score', f1, SUCCESS_COLOR),
    ]
    
    y_start = 0.78
    bar_height = 0.08
    spacing = 0.18
    
    for i, (name, value, color) in enumerate(metrics):
        y_pos = y_start - i * spacing
        
        # Metric name and value
        ax_metrics.text(0.05, y_pos + 0.04, name, ha='left', va='center',
                       fontsize=14, fontweight='bold', color=TEXT_COLOR, transform=ax_metrics.transAxes)
        ax_metrics.text(0.95, y_pos + 0.04, f'{value:.1%}', ha='right', va='center',
                       fontsize=14, fontweight='bold', color=color, transform=ax_metrics.transAxes)
        
        # Progress bar background
        bar_bg = mpatches.Rectangle((0.05, y_pos - 0.02), 0.9, bar_height,
                               facecolor=GRID_COLOR, transform=ax_metrics.transAxes)
        ax_metrics.add_patch(bar_bg)
        
        # Progress bar fill
        bar_fill = mpatches.Rectangle((0.05, y_pos - 0.02), 0.9 * value, bar_height,
                                 facecolor=color, alpha=0.8, transform=ax_metrics.transAxes)
        ax_metrics.add_patch(bar_fill)
    
    # Summary box at bottom
    summary_y = 0.12
    ax_metrics.text(0.5, summary_y, f'Total Samples: 80', ha='center', va='center',
                   fontsize=12, color=TEXT_COLOR, transform=ax_metrics.transAxes)
    ax_metrics.text(0.5, summary_y - 0.08, f'Correct: {tn + tp} | Wrong: {fp + fn}', ha='center', va='center',
                   fontsize=12, color=ACCENT_COLOR, transform=ax_metrics.transAxes)
    
    plt.suptitle('Confusion Matrix Analysis (test46)', fontsize=22, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('presentation/graphs/07_confusion_matrix.png', dpi=150,
                facecolor=BACKGROUND_COLOR, edgecolor='none', bbox_inches='tight')
    plt.close()
    print("✅ Created: 07_confusion_matrix.png")

# ============================================================================
# Graph 8: Error Analysis Bar Chart
# ============================================================================

def create_error_analysis():
    """Create a bar chart showing types of errors."""
    import json
    
    with open('eval_results_fast.json', 'r') as f:
        data = json.load(f)
    
    results = data['results']
    
    # Count error types
    false_positives = sum(1 for r in results if r['true'] == 'consistent' and r['pred'] == 'contradict')
    false_negatives = sum(1 for r in results if r['true'] == 'contradict' and r['pred'] == 'consistent')
    correct_consistent = sum(1 for r in results if r['true'] == 'consistent' and r['pred'] == 'consistent')
    correct_contradict = sum(1 for r in results if r['true'] == 'contradict' and r['pred'] == 'contradict')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Error breakdown
    categories = ['False Positives\n(Said contradict,\nwas consistent)', 
                  'False Negatives\n(Said consistent,\nwas contradict)']
    errors = [false_positives, false_negatives]
    colors = [SECONDARY_COLOR, PRIMARY_COLOR]
    
    bars = ax1.barh(categories, errors, color=colors, edgecolor=TEXT_COLOR, linewidth=2, height=0.6)
    
    for bar, val in zip(bars, errors):
        ax1.text(val + 0.5, bar.get_y() + bar.get_height()/2, f'{val}', 
                va='center', fontsize=16, fontweight='bold', color=TEXT_COLOR)
    
    ax1.set_xlabel('Number of Errors', fontsize=14, fontweight='bold')
    ax1.set_title('Error Type Breakdown', fontsize=16, fontweight='bold')
    ax1.set_xlim(0, max(errors) + 5)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Right: Correct vs Wrong pie chart
    sizes = [correct_consistent + correct_contradict, false_positives + false_negatives]
    labels = [f'Correct\n({sizes[0]})', f'Wrong\n({sizes[1]})']
    colors_pie = [SUCCESS_COLOR, PRIMARY_COLOR]
    explode = (0.05, 0.05)
    
    wedges, texts, autotexts = ax2.pie(sizes, explode=explode, labels=labels, colors=colors_pie,
                                        autopct='%1.1f%%', startangle=90,
                                        wedgeprops=dict(edgecolor=TEXT_COLOR, linewidth=2),
                                        textprops=dict(color=TEXT_COLOR, fontsize=12, fontweight='bold'))
    ax2.set_title('Overall Predictions', fontsize=16, fontweight='bold')
    
    plt.suptitle('Error Analysis (test46 - 80 samples)', fontsize=20, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('presentation/graphs/08_error_analysis.png', dpi=150,
                facecolor=BACKGROUND_COLOR, edgecolor='none', bbox_inches='tight')
    plt.close()
    print("✅ Created: 08_error_analysis.png")

# ============================================================================
# Graph 9: Per-Book Performance
# ============================================================================

def create_per_book_performance():
    """Create a chart comparing performance across different books."""
    import json
    import pandas as pd
    
    # Load results and train data
    with open('eval_results_fast.json', 'r') as f:
        data = json.load(f)
    
    train_df = pd.read_csv('Dataset/train.csv')
    
    results = data['results']
    result_ids = {r['id']: r for r in results}
    
    # Calculate per-book performance
    books = {}
    for _, row in train_df.iterrows():
        if row['id'] in result_ids:
            book = row['book_name']
            if book not in books:
                books[book] = {'correct': 0, 'total': 0, 'consistent_correct': 0, 'consistent_total': 0,
                              'contradict_correct': 0, 'contradict_total': 0}
            
            r = result_ids[row['id']]
            books[book]['total'] += 1
            if r['correct']:
                books[book]['correct'] += 1
            
            if r['true'] == 'consistent':
                books[book]['consistent_total'] += 1
                if r['correct']:
                    books[book]['consistent_correct'] += 1
            else:
                books[book]['contradict_total'] += 1
                if r['correct']:
                    books[book]['contradict_correct'] += 1
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    book_names = list(books.keys())
    x = np.arange(len(book_names))
    width = 0.25
    
    overall_acc = [books[b]['correct']/books[b]['total']*100 for b in book_names]
    consistent_acc = [books[b]['consistent_correct']/books[b]['consistent_total']*100 if books[b]['consistent_total'] > 0 else 0 for b in book_names]
    contradict_acc = [books[b]['contradict_correct']/books[b]['contradict_total']*100 if books[b]['contradict_total'] > 0 else 0 for b in book_names]
    
    bars1 = ax.bar(x - width, overall_acc, width, label='Overall', color=TEXT_COLOR, edgecolor='white', linewidth=2)
    bars2 = ax.bar(x, consistent_acc, width, label='Consistent', color=PRIMARY_COLOR, edgecolor='white', linewidth=2)
    bars3 = ax.bar(x + width, contradict_acc, width, label='Contradict', color=SECONDARY_COLOR, edgecolor='white', linewidth=2)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add sample counts
    for i, book in enumerate(book_names):
        ax.text(i, -8, f'n={books[book]["total"]}', ha='center', fontsize=10, color=ACCENT_COLOR)
    
    ax.axhline(y=60, color=SUCCESS_COLOR, linestyle='--', linewidth=2, alpha=0.7, label='60% Target')
    
    ax.set_xlabel('Book', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Performance by Source Book', fontsize=18, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([b.replace(' ', '\n') for b in book_names], fontsize=11)
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('presentation/graphs/09_per_book_performance.png', dpi=150,
                facecolor=BACKGROUND_COLOR, edgecolor='none', bbox_inches='tight')
    plt.close()
    print("✅ Created: 09_per_book_performance.png")

# ============================================================================
# Graph 10: Class Distribution & Performance
# ============================================================================

def create_class_distribution():
    """Show the class imbalance and how we handled it."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Class distribution in dataset
    consistent_count = 51
    contradict_count = 29
    
    colors = [PRIMARY_COLOR, SECONDARY_COLOR]
    sizes = [consistent_count, contradict_count]
    labels = [f'Consistent\n({consistent_count})', f'Contradict\n({contradict_count})']
    explode = (0.02, 0.02)
    
    wedges, texts, autotexts = ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
                                        autopct='%1.1f%%', startangle=90,
                                        wedgeprops=dict(edgecolor=TEXT_COLOR, linewidth=2),
                                        textprops=dict(color=TEXT_COLOR, fontsize=12, fontweight='bold'))
    ax1.set_title('Dataset Class Distribution\n(Imbalanced)', fontsize=16, fontweight='bold')
    
    # Right: Our balanced performance
    categories = ['Consistent\n(51 samples)', 'Contradict\n(29 samples)']
    accuracies = [68.6, 62.1]
    colors_bar = [PRIMARY_COLOR, SECONDARY_COLOR]
    
    bars = ax2.bar(categories, accuracies, color=colors_bar, edgecolor=TEXT_COLOR, linewidth=2, width=0.6)
    
    for bar, acc in zip(bars, accuracies):
        ax2.annotate(f'{acc:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=18, fontweight='bold')
    
    ax2.axhline(y=60, color=SUCCESS_COLOR, linestyle='--', linewidth=2, alpha=0.7)
    ax2.text(1.5, 61, '60% Target', fontsize=10, color=SUCCESS_COLOR)
    
    ax2.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax2.set_title('Our Balanced Performance\n(Both Above 60%!)', fontsize=16, fontweight='bold')
    ax2.set_ylim(0, 85)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Handling Class Imbalance', fontsize=20, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('presentation/graphs/10_class_distribution.png', dpi=150,
                facecolor=BACKGROUND_COLOR, edgecolor='none', bbox_inches='tight')
    plt.close()
    print("✅ Created: 10_class_distribution.png")

# ============================================================================
# Graph 11: LLM Stats (Improved)
# ============================================================================

def create_llm_stats():
    """Create a clean, professional LLM statistics dashboard."""
    import json
    
    with open('eval_results_fast.json', 'r') as f:
        data = json.load(f)
    
    stats = data['llm_stats']
    
    # Calculate derived stats
    total_samples = 80
    calls_per_sample = stats['call_count'] / total_samples
    minutes = stats['total_time'] / 60
    per_sample = stats['total_time'] / total_samples
    
    fig = plt.figure(figsize=(14, 8))
    
    # Create grid layout
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Top row: 3 main stats with circular progress indicators
    stat_data = [
        {'value': stats['call_count'], 'label': 'API Calls', 'sublabel': f'{calls_per_sample:.1f} per sample', 'color': PRIMARY_COLOR},
        {'value': f"{stats['avg_time']:.2f}s", 'label': 'Avg Response', 'sublabel': 'per LLM call', 'color': SECONDARY_COLOR},
        {'value': f"{per_sample:.1f}s", 'label': 'Per Backstory', 'sublabel': f'{minutes:.1f} min total', 'color': SUCCESS_COLOR},
    ]
    
    for i, stat in enumerate(stat_data):
        ax = fig.add_subplot(gs[0, i])
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Draw outer ring
        circle_outer = mpatches.Circle((0, 0), 1.2, fill=False, 
                                   edgecolor=stat['color'], linewidth=6, alpha=0.3)
        ax.add_patch(circle_outer)
        
        # Draw progress arc (decorative)
        theta = np.linspace(0, 2*np.pi*0.75, 100)
        x = 1.2 * np.cos(theta - np.pi/2)
        y = 1.2 * np.sin(theta - np.pi/2)
        ax.plot(x, y, color=stat['color'], linewidth=6, solid_capstyle='round')
        
        # Inner circle background
        circle_inner = mpatches.Circle((0, 0), 0.95, facecolor=BACKGROUND_COLOR, 
                                   edgecolor=stat['color'], linewidth=2)
        ax.add_patch(circle_inner)
        
        # Value text
        ax.text(0, 0.1, str(stat['value']), ha='center', va='center',
               fontsize=32, fontweight='bold', color=TEXT_COLOR)
        
        # Label text
        ax.text(0, -0.35, stat['label'], ha='center', va='center',
               fontsize=14, fontweight='bold', color=stat['color'])
        
        # Sublabel
        ax.text(0, -1.45, stat['sublabel'], ha='center', va='center',
               fontsize=11, color=ACCENT_COLOR)
    
    # Bottom row: Summary bar
    ax_summary = fig.add_subplot(gs[1, :])
    ax_summary.set_xlim(0, 10)
    ax_summary.set_ylim(0, 2)
    ax_summary.axis('off')
    
    # Summary background box
    summary_rect = mpatches.FancyBboxPatch(
        (0.2, 0.3), 9.6, 1.4,
        boxstyle="round,pad=0.05,rounding_size=0.2",
        facecolor=BACKGROUND_COLOR, edgecolor=PRIMARY_COLOR, linewidth=2
    )
    ax_summary.add_patch(summary_rect)
    
    # Summary title
    ax_summary.text(5, 1.45, 'Pipeline Efficiency Summary', ha='center', va='center',
                   fontsize=16, fontweight='bold', color=TEXT_COLOR)
    
    # Summary stats in a row
    summary_items = [
        (f"Model: llama-3.1-8b", 1.5),
        (f"Errors: {stats['errors']}", 3.5),
        (f"Samples: {total_samples}", 5.5),
        (f"Total Time: {minutes:.1f} min", 7.5),
    ]
    
    for text, x_pos in summary_items:
        ax_summary.text(x_pos, 0.85, text, ha='center', va='center',
                       fontsize=13, color=TEXT_COLOR)
    
    # Efficiency indicator
    efficiency = 100 - (stats['errors'] / stats['call_count'] * 100) if stats['call_count'] > 0 else 100
    ax_summary.text(5, 0.45, f"Success Rate: {efficiency:.1f}%", ha='center', va='center',
                   fontsize=14, fontweight='bold', color=SUCCESS_COLOR)
    
    plt.suptitle('LLM Performance Dashboard', fontsize=22, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('presentation/graphs/11_llm_stats.png', dpi=150,
                facecolor=BACKGROUND_COLOR, edgecolor='none', bbox_inches='tight')
    plt.close()
    print("✅ Created: 11_llm_stats.png")

# ============================================================================
# Graph 12: Journey Summary Infographic
# ============================================================================

def create_journey_summary():
    """Create a summary infographic of the entire journey."""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(8, 9.5, 'The Journey to 60%+ on Both Classes', ha='center', va='center',
            fontsize=28, fontweight='bold', color=TEXT_COLOR)
    ax.text(8, 8.8, '46 experiments • 14 major changes • 1 goal achieved', ha='center', va='center',
            fontsize=14, color=ACCENT_COLOR)
    
    # Key milestones boxes
    milestones = [
        {'pos': (1, 6), 'title': 'START', 'value': '52.5%', 'subtitle': 'test6', 'color': '#ff4444'},
        {'pos': (4.5, 6), 'title': 'CRISIS', 'value': '0%', 'subtitle': '0% contradict!', 'color': PRIMARY_COLOR},
        {'pos': (8, 6), 'title': 'BREAKTHROUGH', 'value': '72.5%', 'subtitle': 'Both >60%', 'color': '#ffaa00'},
        {'pos': (11.5, 6), 'title': 'FAILED', 'value': '44.8%', 'subtitle': 'Aggressive backfired', 'color': PRIMARY_COLOR},
        {'pos': (15, 6), 'title': 'GOAL!', 'value': '62.1%', 'subtitle': 'Both above 60%!', 'color': SUCCESS_COLOR},
    ]
    
    for m in milestones:
        rect = mpatches.FancyBboxPatch(
            (m['pos'][0]-1.2, m['pos'][1]-1.2), 2.4, 2.4,
            boxstyle="round,pad=0.05,rounding_size=0.3",
            facecolor=BACKGROUND_COLOR, edgecolor=m['color'], linewidth=3
        )
        ax.add_patch(rect)
        ax.text(m['pos'][0], m['pos'][1]+0.7, m['title'], ha='center', va='center',
                fontsize=12, fontweight='bold', color=m['color'])
        ax.text(m['pos'][0], m['pos'][1], m['value'], ha='center', va='center',
                fontsize=24, fontweight='bold', color=TEXT_COLOR)
        ax.text(m['pos'][0], m['pos'][1]-0.7, m['subtitle'], ha='center', va='center',
                fontsize=10, color=ACCENT_COLOR)
    
    # Arrows between milestones
    for i in range(len(milestones)-1):
        ax.annotate('', xy=(milestones[i+1]['pos'][0]-1.4, milestones[i+1]['pos'][1]),
                   xytext=(milestones[i]['pos'][0]+1.4, milestones[i]['pos'][1]),
                   arrowprops=dict(arrowstyle='->', color=GRID_COLOR, lw=2))
    
    # Key learnings box
    learnings = [
        "✓ Balance is key - too aggressive or conservative both fail",
        "✓ Detective mindset: 'Precise but not assumptional'",
        "✓ Failed experiments teach valuable lessons",
        "✓ Iterative improvement beats one-shot solutions",
    ]
    
    y_start = 3
    ax.text(8, y_start + 0.8, 'Key Learnings', ha='center', fontsize=16, fontweight='bold', color=PRIMARY_COLOR)
    for i, learning in enumerate(learnings):
        ax.text(8, y_start - i*0.6, learning, ha='center', fontsize=12, color=TEXT_COLOR)
    
    # Final stats box
    ax.add_patch(mpatches.FancyBboxPatch((5.5, 0.3), 5, 1.2, boxstyle="round,pad=0.1",
                                          facecolor=BACKGROUND_COLOR, edgecolor=SUCCESS_COLOR, linewidth=3))
    ax.text(8, 1.1, 'FINAL: 68.6% Consistent | 62.1% Contradict | 66.2% Overall', 
            ha='center', va='center', fontsize=14, fontweight='bold', color=SUCCESS_COLOR)
    ax.text(8, 0.6, 'Both classes above 60% target!', ha='center', va='center',
            fontsize=12, color=TEXT_COLOR)
    
    plt.tight_layout()
    plt.savefig('presentation/graphs/12_journey_summary.png', dpi=150,
                facecolor=BACKGROUND_COLOR, edgecolor='none', bbox_inches='tight')
    plt.close()
    print("✅ Created: 12_journey_summary.png")

# ============================================================================
# Main execution
# ============================================================================

if __name__ == '__main__':
    import os
    
    # Create output directory
    os.makedirs('presentation/graphs', exist_ok=True)
    
    print("🎨 Generating presentation graphs...")
    print("=" * 50)
    
    # Generate selected graphs
    create_progress_chart()
    create_final_results_chart()
    create_confusion_matrix()
    create_llm_stats()
    print("✅ Created: 11_llm_stats.png")
    
    print("=" * 50)
    print("✅ Selected graphs (1, 5, 7, 11) generated successfully!")
    print("📁 Output folder: presentation/graphs/")
