from argparse import ArgumentParser
import json
from pathlib import Path

def change_label(label):
    if label == 'Mais' or label == 'mais' or label == 'None' or label == 'corn':
        return 'corn'
    elif label == 'Half':
        return 'half_corn'
    elif label == 'roggen':
        return 'rye'
    elif label == 'half_roggen':
        return 'half_rye'
    elif label == 'triticale' or label == 'trictale':
        return 'triticale'
    elif label == 'half_tritctale' or label == 'half_trictale':
        return 'half_triticale'
    elif label == 'weizen' or label == 'wheat':
        return 'wheat'
    elif label == 'half_weizen':
        return 'half_wheat'


parser = ArgumentParser()
parser.add_argument('labelfolder')
args = parser.parse_args()

labelfolder_path = Path(args.labelfolder)

for labelpath in labelfolder_path.glob('*.json'):
    labels = {}
    with labelpath.open(mode='r') as jsonfile:
        labels = json.load(jsonfile)
        shapes = labels['shapes']
        for shape in shapes:
            newlabel = change_label(shape['label'])
            print(newlabel)
            shape['label'] = newlabel
    with labelpath.open(mode='w') as jsonfile:
        print('saved')
        json.dump(labels, jsonfile, indent="\t")