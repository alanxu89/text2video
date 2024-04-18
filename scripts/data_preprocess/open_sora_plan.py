import json
import os

cap_data_path = "/mnt/sda/open_sora_plan_data/llava_path_cap_64x512x512.json"
dataset_name = "mixkit"

cap_data_dir = os.path.dirname(cap_data_path)
with open(cap_data_path) as f:
    data = json.load(f)
# print(type(data))
# print(len(data))
# print(data[0])
# print(data[0].keys())

csv_save_path = os.path.join(cap_data_dir, dataset_name + ".csv")
csvfile = open(csv_save_path, "w")
csvfile.write("videoid,contentUrl,duration,page_dir,name\n")

for item in data:
    video_path = item['path']
    captions = item['cap']
    if len(captions) == 0:
        continue
    cap = captions[0]
    cap = cap.replace("\"", "")
    cap = cap.replace("\'", "")
    splits = video_path.split('/')
    dataset, category, videoname = splits[-3:]
    if dataset == dataset_name:
        videoid = videoname.removesuffix(".mp4")
        csvfile.write(f"{videoid},null,null,{dataset}/{category},\"{cap}\"\n")
    else:
        continue

csvfile.close()
