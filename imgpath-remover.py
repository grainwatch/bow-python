from argparse import ArgumentParser
import json
from pathlib import Path

parser = ArgumentParser()
parser.add_argument('labelfolder')
parser.add_argument('imgfolder')
args = parser.parse_args()

labelfolder_path = Path(args.labelfolder)
imgfolder_path = Path(args.imgfolder)

for labelpath in labelfolder_path.glob('*.json'):
    labels = {}
    with labelpath.open(mode='r') as jsonfile:
        labels = json.load(jsonfile)
        oldimgpath = Path(labels['imagePath'])
        newimgpath = imgfolder_path.joinpath(oldimgpath.name)
        labels['imagePath'] = str(newimgpath)
    with labelpath.open(mode='w') as jsonfile:
        json.dump(labels, jsonfile, indent="\t")

