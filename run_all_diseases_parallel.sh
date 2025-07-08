#!/bin/bash

# Create temporary config files for each disease to avoid conflicts
echo "Creating temporary config files..."
for disease in dated_nec dated_bpd dated_death dated_max_chole_TPNEHR; do
    cp config/dataset/synthetic_neonatal.yaml config/dataset/synthetic_neonatal_${disease}.yaml
    sed -i.bak "s/target_disease: .*/target_disease: $disease/" config/dataset/synthetic_neonatal_${disease}.yaml
done

# Run all diseases in parallel
echo "Starting parallel execution..."
for disease in dated_nec dated_bpd dated_death dated_max_chole_TPNEHR; do
    echo "Starting $disease..."
    python run_synthetic_neonatal.py --repeats 2 --max-epochs 100 > outputs/log_${disease}.txt 2>&1 &
done

# Wait for all background jobs to complete
echo "Waiting for all jobs to complete..."
wait

echo "All diseases completed!"

# Clean up temporary config files
echo "Cleaning up temporary files..."
rm -f config/dataset/synthetic_neonatal_dated_*.yaml
rm -f config/dataset/synthetic_neonatal_dated_*.yaml.bak

echo "Done!"