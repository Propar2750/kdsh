import json

with open('eval_results_fast.json') as f:
    data = json.load(f)

# Count breakdown
consistent_correct = sum(1 for r in data['results'] if r['true'] == 'consistent' and r['correct'])
consistent_wrong = sum(1 for r in data['results'] if r['true'] == 'consistent' and not r['correct'])
contradict_correct = sum(1 for r in data['results'] if r['true'] == 'contradict' and r['correct'])
contradict_wrong = sum(1 for r in data['results'] if r['true'] == 'contradict' and not r['correct'])

total_consistent = consistent_correct + consistent_wrong
total_contradict = contradict_correct + contradict_wrong

print(f'=== ACCURACY BREAKDOWN ===')
print(f'Consistent: {consistent_correct}/{total_consistent} = {consistent_correct/total_consistent*100:.1f}%')
print(f'Contradict: {contradict_correct}/{total_contradict} = {contradict_correct/total_contradict*100:.1f}%')
print(f'Overall: {data["correct"]}/{data["total"]} = {data["accuracy"]*100:.1f}%')
print()
print(f'=== FALSE POSITIVES (consistent wrongly marked as contradict) - {consistent_wrong} cases ===')
fp_ids = []
for r in data['results']:
    if r['true'] == 'consistent' and r['pred'] == 'contradict':
        fp_ids.append(r['id'])
        print(f"  ID {r['id']}")
print()
print(f'=== FALSE NEGATIVES (contradict wrongly marked as consistent) - {contradict_wrong} cases ===')
fn_ids = []
for r in data['results']:
    if r['true'] == 'contradict' and r['pred'] == 'consistent':
        fn_ids.append(r['id'])
        print(f"  ID {r['id']}")
