import json, sys
sys.stdout.reconfigure(encoding='utf-8')
with open('docs/image_audit_raw.json', encoding='utf-8') as f:
    d = json.load(f)

print('=== SPLIT SUMMARY KEYS ===')
first = next(iter(d['splits_summary'].values()))
print(list(first.keys()))

print('\n=== SPLITS ===')
for k, v in d['splits_summary'].items():
    dup_g = v.get('exact_dup_groups', v.get('exact_dup_groups_within','?'))
    dup_f = v.get('exact_dup_files', v.get('exact_dup_files_within','?'))
    print(f"{k}: n={v['total']} corrupt={v['corrupt_count']} "
          f"w={v['width_min']}-{v['width_max']} mean_w={v['width_mean']} "
          f"h={v['height_min']}-{v['height_max']} mean_h={v['height_mean']} "
          f"modes={v['mode_dist']} within_dup_groups={dup_g} within_dup_files={dup_f}")

print('\n=== CROSS MD5 ===')
cs = d['cross_split_md5_dups']
print(f"total cross-dup groups: {cs['count']}")
for pair, cnt in cs['by_pair'].items():
    print(f"  {pair}: {cnt}")

print('\n=== PHASH ===')
ph = d['perceptual_hashing']
for k,v in ph.items():
    if 'sample' not in k:
        print(f"  {k}: {v}")

print('\n=== BBOX AGGREGATE ===')
ba = d['bbox_visualisation']['aggregate_all_train_labels']
for k,v in ba.items():
    print(f"  {k}: {v}")

print('\n=== BBOX SAMPLE (20 images) ===')
for item in d['bbox_visualisation']['sample_details']:
    print(f"  {item['image']}: {item['num_boxes']} boxes avg_area={item['avg_box_area_frac']}")
