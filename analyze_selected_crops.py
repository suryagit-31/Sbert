import json

data = json.load(open('PlantVillage_train_prompts - Copy.json', encoding='utf-8'))

selected_crops = ['tomato', 'apple', 'corn', 'maize', 'potato', 'grape', 'soybean', 'orange']
selected = {k: len(v) for k, v in data.items() if any(crop in k.lower() for crop in selected_crops)}

print(f'Selected Crops: {len(selected)} categories\n')
print('=' * 60)

total = 0
for crop in ['Tomato', 'Apple', 'Corn', 'Potato', 'Grape', 'Soybean', 'Orange']:
    crop_items = {k: v for k, v in selected.items() if crop.lower() in k.lower()}
    if crop_items:
        print(f'\n{crop} ({len(crop_items)} categories):')
        for i, (k, v) in enumerate(crop_items.items()):
            print(f'  {i+1}. {k} - {v} descriptions')
            total += v

print(f'\n' + '=' * 60)
print(f'TOTAL: {len(selected)} categories with {total} descriptions')
print(f'Reduction: 38 total → {len(selected)} selected ({len(selected)/38*100:.1f}%)')
