import json
import os

path = "../../test.json"
with open(path, 'w') as f:
    json.dump({}, f)
if os.path.exists(path):
    print("..")