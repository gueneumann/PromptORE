python experiments/run_experiments.py --spec experiments/fewrel_mlm_spec.json --output experiments/fewrel_mlm_results.json
python experiments/render_results_table.py --results experiments/fewrel_mlm_results.json --output experiments/fewrel_mlm.png

python experiments/run_experiments.py --spec experiments/tacred_mlm_spec.json --output experiments/tacred_mlm_results.json
python experiments/render_results_table.py --results experiments/tacred_mlm_results.json --output experiments/tacred_mlm.png

python experiments/run_experiments.py --spec experiments/meddistant19_mlm_spec.json --output experiments/meddistant19_mlm_results.json
python experiments/render_results_table.py --results experiments/meddistant19_mlm_results.json --output experiments/meddistant19_mlm.png