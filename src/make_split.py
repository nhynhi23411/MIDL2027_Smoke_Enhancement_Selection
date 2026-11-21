import csv, json
rows = list(csv.DictReader(open('outputs/image_mask_audit.csv')))
def role(c):
    c = int(c)
    return 'selector_dev_train' if c <= 170 else 'selector_dev_val' if c <= 227 else 'final_test'
note = ('Procedure ID unverified; clip index used as proxy. Official AutoLaparo split is contiguous by clip '
        '(001-170 / 171-227 / 228-300), which likely keeps source videos disjoint across blocks - assumption, not verified.')
with open('outputs/source_exposure.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['image','clip','paper1_exposure','evidence','independent_external'])
    for r in rows:
        w.writerow([r['name'], r['clip'], 'external_test_in_JBHI; earlier_rejected_finetune_membership_unresolved',
                    'NoiSoi/GynSurg_Clinical_Followup/manifests/master_manifest_v1.csv', 'NO'])
with open('outputs/procedure_split.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['clip','group_proxy','role','n_images','note'])
    for c in sorted({r['clip'] for r in rows}):
        w.writerow([c, f'clip_{c}', role(c), sum(r['clip']==c for r in rows), note])
label = {'0':'background'} | {str(v): n for v, n in zip(range(20,170,20), ['tool1m','tool1s','tool2m','tool2s','tool3m','tool3s','tool4m','tool4s'])} | {'180':'uterus'}
json.dump({'uterus':[180], 'instrument_union':list(range(20,161,20)), 'raw_values':label,
 'note':'Missing annotation category is not evidence of target absence; uterus absent in 743/1800 reference masks.'},
 open('outputs/label_mapping.json','w'), indent=1)
import collections
print(collections.Counter(role(r['clip']) for r in rows))
print({k: sum(1 for r in rows if role(r['clip'])==k and int(r['uterus_px'])>0) for k in ['selector_dev_train','selector_dev_val','final_test']})
