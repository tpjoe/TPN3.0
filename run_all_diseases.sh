#!/bin/bash

for disease in dated_nec dated_bpd dated_death dated_max_chole_TPNEHR; do
    echo "Running for disease: $disease"
    sed -i.bak "s/target_disease: .*/target_disease: $disease/" config/dataset/synthetic_neonatal.yaml
    python run_synthetic_neonatal.py --repeats 1 --max-epochs 1
done