#!/usr/bin/env python3
import pysubgroup as ps
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score

# Load predictions
df_val = pd.read_csv('outputs/predictions_val.csv')

# Load original data to get features
df_orig = pd.read_csv('data/processed/autoregressive.csv', low_memory=False)

# Get patient features (take first row per patient since static features are constant)
patient_features = df_orig.groupby('person_id').first().reset_index()

# Select key features to analyze
features_to_use = ['gest_age', 'bw', 'gender_concept_id_8507', 'race_concept_id_8515']

# Get features for our patients - merge on person_id
df_val_with_features = pd.merge(df_val, patient_features[['person_id'] + features_to_use], on='person_id', how='left')


aa = df_val_with_features.loc[df_val_with_features.bw<1.5, :]
average_precision_score(aa.y_true, aa.y_prob)
average_precision_score(df_val_with_features.y_true, df_val_with_features.y_prob)


# Create analysis dataframe
df_analysis = df_val_with_features[features_to_use].copy()
df_analysis['y_true'] = df_val_with_features['y_true']
df_analysis['y_prob'] = df_val_with_features['y_prob']

# Print basic stats
print(f"Total samples: {len(df_analysis)}")
print(f"Positive rate: {df_analysis['y_true'].mean():.3f}")
print(f"Overall AP: {average_precision_score(df_analysis['y_true'], df_analysis['y_prob']):.3f}")

# Minimum subgroup size constraint
min_size = 5  # or 0.01 * len(df_analysis) for 1% of data

# Custom quality function class for AP lift
class APLiftQF(ps.AbstractInterestingnessMeasure):
    def __init__(self, min_size=5):
        self.min_size = min_size
        self.dataset = None
        self.overall_ap = None
    
    def calculate_constant_statistics(self, data, target):
        self.dataset = data
        self.overall_ap = average_precision_score(data['y_true'], data['y_prob'])
    
    def calculate_statistics(self, subgroup, target, data, statistics=None):
        # Return the data covered by the subgroup - required by pysubgroup
        return subgroup
    
    def evaluate(self, subgroup, target, data, statistics=None):
        sg_data = data[subgroup.covers(data)]
        
        if len(sg_data) < self.min_size or sg_data['y_true'].sum() == 0:
            return 0
        
        sg_ap = average_precision_score(sg_data['y_true'], sg_data['y_prob'])
        size_factor = len(sg_data) / len(data)
        
        # Return worse performance as positive (higher = worse)
        return (self.overall_ap - sg_ap) * size_factor
    
    @property
    def optimistic_estimate(self):
        # Required property - return 1 as we want to explore all branches
        return 1
    
    def optimistic_estimate_from_statistics(self, statistics, target, data):
        # Required method - return 1 to explore all branches
        return 1


# Create search space only on selected features
searchspace = ps.create_selectors(df_analysis, ignore=['y_true', 'y_prob'])

# Run discovery with custom quality function
task = ps.SubgroupDiscoveryTask(
    df_analysis, 
    ps.BinaryTarget('y_true', True),  # Just used for framework structure
    searchspace,
    result_set_size=10,
    depth=2,
    qf=APLiftQF(min_size)
)

result = ps.BeamSearch().execute(task)

# Analyze results
print("\nSubgroups with poor Average Precision performance:")
print("="*80)

overall_ap = average_precision_score(df_analysis['y_true'], df_analysis['y_prob'])

# Access results using the results attribute
for result_item in result.results:
    quality = result_item[0]
    subgroup = result_item[1]
    
    mask = subgroup.covers(df_analysis)
    sg_data = df_analysis[mask]
    
    # Calculate metrics
    sg_ap = average_precision_score(sg_data['y_true'], sg_data['y_prob']) if sg_data['y_true'].sum() > 0 else 0
    
    print(f"\nRule: {subgroup}")
    print(f"Quality score: {quality:.4f}")
    print(f"Size: {len(sg_data)} ({100*len(sg_data)/len(df_analysis):.1f}%)")
    print(f"Positive cases: {sg_data['y_true'].sum()}")
    print(f"Positive rate: {sg_data['y_true'].mean():.3f}")
    print(f"Subgroup AP: {sg_ap:.3f}")
    print(f"Overall AP: {overall_ap:.3f}")
    print(f"AP Lift: {sg_ap - overall_ap:.3f}")


























import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
from pysubgroup import SubgroupDiscoveryTask, Apriori, create_nominal_selectors, create_numeric_selectors
from pysubgroup import QualityFunction

## --- 1. Generate Fake Data ---
# Let's create a synthetic dataset suitable for subgroup discovery.
# It includes features and a binary target along with pre-computed scores
# (e.g., from a global classifier) which are necessary for AUPRC calculation.

np.random.seed(42) # for reproducibility
n_samples = 200

data = pd.DataFrame({
    'gender': np.random.choice(['Male', 'Female'], n_samples),
    'age': np.random.randint(20, 60, n_samples),
    'city_tier': np.random.choice(['Tier_1', 'Tier_2', 'Tier_3'], n_samples),
    'income': np.random.normal(50000, 15000, n_samples),
    'target': np.random.choice([0, 1], n_samples, p=[0.75, 0.25]) # Imbalanced target (0=No, 1=Yes)
})

# Create fake 'scores' (probabilities of target=1) for AUPRC calculation.
# Let's assume some subgroups (e.g., 'gender' == 'Female' or 'age' > 45)
# might have slightly higher scores or more positive cases.
data['scores'] = data['target'] * np.random.uniform(0.65, 0.95, n_samples) + \
                 (1 - data['target']) * np.random.uniform(0.05, 0.35, n_samples)
# Add some noise and clip to valid probability range
data['scores'] += np.random.normal(0, 0.03, n_samples)
data['scores'] = np.clip(data['scores'], 0.01, 0.99)

# Calculate the baseline AUPRC for the entire dataset
precision_base, recall_base, _ = precision_recall_curve(data['target'], data['scores'])
baseline_auprc = auc(recall_base, precision_base)
print(f"📊 Baseline AUPRC for the entire dataset: {baseline_auprc:.4f}\n")

## --- 2. Define Custom AUPRC Gain Quality Function ---

class AUPRCGainQF(QualityFunction):
    """
    Custom Quality Function to calculate AUPRC Gain for subgroups.
    Maximizes: AUPRC(subgroup) - AUPRC(overall dataset)
    Enforces: Minimum number of instances in the subgroup.
    """
    def __init__(self, target_variable: str, score_variable: str, baseline_auprc: float, min_instances: int = 4):
        self.target_variable = target_variable
        self.score_variable = score_variable
        self.baseline_auprc = baseline_auprc
        self.min_instances = min_instances
        self.is_constant = False # Indicates that the quality measure varies across subgroups

    def __call__(self, subgroup, data_frame: pd.DataFrame) -> float:
        """Alias for the evaluate method."""
        return self.evaluate(subgroup, data_frame)

    def evaluate(self, subgroup, data_frame: pd.DataFrame) -> float:
        """
        Calculates the AUPRC gain for the given subgroup.
        """
        # Get a boolean mask for instances covered by the subgroup
        subgroup_mask = subgroup.covers(data_frame)
        
        # Enforce the minimum number of instances constraint
        if subgroup_mask.sum() < self.min_instances:
            # Return a very low value to penalize subgroups that are too small
            return -np.inf # or 0, depending on desired behavior for too small groups

        # Extract target and scores for the instances within the subgroup
        y_true_subgroup = data_frame.loc[subgroup_mask, self.target_variable]
        y_scores_subgroup = data_frame.loc[subgroup_mask, self.score_variable]

        # AUPRC cannot be calculated if there's only one unique class in the subgroup
        if len(np.unique(y_true_subgroup)) < 2:
            return -np.inf # Or 0, as no meaningful AUPRC gain can be computed

        # Calculate AUPRC for the subgroup
        precision_subgroup, recall_subgroup, _ = precision_recall_curve(y_true_subgroup, y_scores_subgroup)
        auprc_subgroup = auc(recall_subgroup, precision_subgroup)

        # Calculate and return the AUPRC gain
        auprc_gain = auprc_subgroup - self.baseline_auprc
        return auprc_gain

    def optimistic_estimate(self, subgroup, data_frame: pd.DataFrame) -> float:
        """
        Provides an optimistic estimate of the quality for any subgroup
        that can be formed by extending the current partial subgroup description.
        For AUPRC gain, the maximum possible AUPRC is 1.0, so the most optimistic
        gain is 1.0 - baseline_auprc. This is a loose but valid upper bound.
        """
        return 1.0 - self.baseline_auprc

    def get_name(self) -> str:
        """Returns the name of the quality function."""
        return f"AUPRC Gain (min_instances={self.min_instances})"

## --- 3. Set Up and Run Subgroup Discovery Task ---

# Define the search space using selectors from your features.
# pysubgroup will automatically discretize numerical features for ranges.
selectors = create_nominal_selectors(data, ['gender', 'city_tier'])
selectors.extend(create_numeric_selectors(data, ['age', 'income']))

# Initialize your custom AUPRCGainQF
auprc_gain_qf = AUPRCGainQF(
    target_variable='target',
    score_variable='scores',
    baseline_auprc=baseline_auprc,
    min_instances=4 # Enforces the "at least 4 cases" constraint
)

# Create the SubgroupDiscoveryTask.
# Note: The 'target' parameter in SubgroupDiscoveryTask is set to None
# because our custom quality function handles the target variable internally.
task = SubgroupDiscoveryTask(
    data=data,
    target=None,
    search_space=selectors,
    quality_function=auprc_gain_qf,
    result_set_size=5, # Find the top 5 subgroups
    depth=2 # Maximum number of conjuncts in a subgroup description (e.g., gender='Female' AND age > 40)
)

# Run the Apriori algorithm to discover subgroups.
# Apriori is a common algorithm for this. Other algorithms are available in pysubgroup.
search_algorithm = Apriori()
results = search_algorithm.discover_subgroups(task)

## --- 4. Print Results ---

print("--- 🚀 Top Subgroups by AUPRC Gain ---")
if not results:
    print("No subgroups found that meet the criteria.")
else:
    for i, (subgroup, quality_value) in enumerate(results):
        # Retrieve the actual data points covered by the subgroup
        covered_indices = subgroup.covers(data)
        covered_data = data[covered_indices]
        
        print(f"\n{i+1}. Subgroup Description: {subgroup.representation}")
        print(f"   AUPRC Gain: {quality_value:.4f}")
        print(f"   Number of Instances: {len(covered_data)}")
        
        # For verification, calculate actual AUPRC for the subgroup if possible
        if len(covered_data) >= 2 and len(np.unique(covered_data['target'])) >= 2:
            prec_sub, rec_sub, _ = precision_recall_curve(covered_data['target'], covered_data['scores'])
            actual_subgroup_auprc = auc(rec_sub, prec_sub)
            print(f"   Actual Subgroup AUPRC: {actual_subgroup_auprc:.4f}")
        else:
            print("   Actual Subgroup AUPRC: N/A (too few instances or single class)")