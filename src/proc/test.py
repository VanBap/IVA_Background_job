import cv2
from ai_match_scene_images import SceneMatcher
# === test detect camera function ===
# try:
#     img1 = cv2.imread('/home/vbd-vanhk-l1-ubuntu/background.jpg')
#     img2 = cv2.imread('/home/vbd-vanhk-l1-ubuntu/work/camera_1_snapshot.jpg')
#     matcher = SceneMatcher(visualize=False)
#     result = matcher.match_scenes(img1, img2)
#     if result == True:
#         print("Matched scenes")
#     else:
#         print("No matches")
# except AttributeError as e:
#     print("SceneMatcher.match_scenes khong chay duoc: ", e)

# Detect Camera change

try:
    img_1 = cv2.imread('/home/vbd-vanhk-l1-ubuntu/work/white_background.jpg')
    img_2 = cv2.imread('/home/vbd-vanhk-l1-ubuntu/work/camera_7_snapshot.jpg')
    matcher = SceneMatcher(visualize=False)
    is_matched = matcher.match_scenes(img_1, img_2)
    if is_matched == True:
        print('Matched scenes')
    else:
        print('Not Matched scenes')
except AttributeError as e:
    print("SceneMatcher.match_scenes khong chay duoc: ", e)
