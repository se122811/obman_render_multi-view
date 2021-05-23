import json
import sys
import pathlib

root = '.'
sys.path.insert(0, root)
new_path = str(pathlib.Path(__file__).parent.absolute())
sys.path.append(new_path)

from blender_hands import ex as hand_ex

recover_json_string = ' '.join(sys.argv[sys.argv.index('--') + 1:])
#recover_json_string = '{"frame_nb": 10, "frame_start": 0, "results_root": "/home/donguk/make_db/results", "background_datasets": ["white"]}'

json_config = json.loads(recover_json_string)

# Generate hands only by sampling from random hand poses
r = hand_ex.run(config_updates=json_config)
