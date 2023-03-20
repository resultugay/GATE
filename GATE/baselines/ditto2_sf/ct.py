for line in open("/data/qiu/ditto2_sf/test2.txt", 'r', encoding='utf-8'):
    print(line)

import os
absolute_path = os.path.abspath(__file__)
print("Full path: " + absolute_path)
print("Directory Path: " + os.path.dirname(absolute_path))