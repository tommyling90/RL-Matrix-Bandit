from runResults import *
from runFigures import *

run_results()
aggregate_metrics_from_pkl(f"{folder}/pkl")

with open("../graph_config.yaml", "r") as f:
    config = yaml.safe_load(f)

for key, val in config.items():
    if len(val['algos']) > 1 and len(val['noise']) > 1 :
        raise "Too many pairs to compare. Either compare different algo combos on one noise level or different noise levels on one algo combo."
    cumul_y = val['cumul_y']
    generate_fig(cumul_y, val['algos'], val['noise'], val['name'])