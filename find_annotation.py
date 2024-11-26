import json

# MIMIC json annotation
file_path = "mimic_annotation_all.json"

# The id of the image
search_id = "23ca57bb-19b06ff8-ca4007f3-d63391f9-fef651a7"

with open(file_path, 'r') as file:
    data = json.load(file)

# Find id, then return report
def find_report(data, search_id):
    for split in ['train', 'val', 'test']:
        if split in data:
            for entry in data[split]:
                if entry.get('id') == search_id:
                    return entry.get('report')
    return None

report = find_report(data, search_id)

if report:
    print(f"Report for ID {search_id}:\n{report}")
else:
    print(f"ID {search_id} not found.")
