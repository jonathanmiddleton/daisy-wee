import json
from dataclasses import asdict

from models import model_from_spec
from tools.model_report import build_report
from training.hparams import load_hparams_from_yaml

hparams = load_hparams_from_yaml("/Users/jonathanmiddleton/projects/daisy-wee/config/pretrain_350m.yml")
model = model_from_spec(asdict(hparams), device="cpu")

report = build_report(model,hparams)
report = json.dumps(report, indent=2)
print(report)