import os
import sys
import yaml
import json

old_config = sys.argv[1]
new_args = sys.argv[2]
stage = sys.argv[3]

tmp = "TEMP"
os.makedirs(tmp, exist_ok=True)

with open(old_config) as f:
    old_data = f.read()
old_data = yaml.load(old_data, Loader=yaml.FullLoader) if old_config[-4:] == "yaml" else json.loads(old_data)
with open(new_args) as f:
    new_data = f.read()
new_data = yaml.load(new_data, Loader=yaml.FullLoader) if new_args[-4:] == "yaml" else json.loads(new_data)

def deep_merge_dict(d1, d2):
    for k, v in d2.items():
        if k in d1 and isinstance(d1[k], dict) and isinstance(v, dict):
            deep_merge_dict(d1[k], v)  # 递归合并子字典
        else:
            d1[k] = v  # 如果没有这个键或不是字典，就直接赋值覆盖
    return d1
tmp_data = deep_merge_dict(old_data, new_data)

tmp_config_path1=f"{tmp}/tmp_s{stage}.yaml"
tmp_config_path2=f"{tmp}/tmp_s{stage}.json"
if old_config[-4:] == "yaml":
    with open(tmp_config_path1, "w") as f:
        f.write(yaml.dump(tmp_data, default_flow_style=False))
else:
    with open(tmp_config_path2, "w") as f:
        f.write(json.dumps(tmp_data, indent=2))
