from ml_denoising.noise_modeling import get_quera_noise_model_cudaq
from ml_denoising.data_generation import generate_proper_circuits, generate_proper_observables, prepare_mitigator_dataset
from ml_denoising.circuit import circuit_to_graph_data, qasm_hash
from ml_denoising.model import build_model, visualize_embeddings
from ml_denoising.train import train_mitigator, evaluate_mitigator, save_scaling_results, create_scaling_summary_plots

from qiskit_aer.primitives import Estimator as AerEstimator
import cudaq
from cudaq import spin

import os
import time
import json
import torch

# Define parameters for data generation
circuit_types = ['random']
min_qubits = 2
max_qubits = 8
min_depth = 100
max_depth = 500
output_dir = "ml_mitigation_output"
seed = 31415

variants_to_test = ['simple']
noise_factors = [1.0]

data_scaling_configs = [600]
model_depth_configs = [2]


def main():
    """Main function with comprehensive scaling experiments."""
    
    # Create main output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a comprehensive results tracker
    all_scaling_results = {
        'experiment_start_time': time.strftime("%Y-%m-%d %H:%M:%S"),
        'configurations': {
            'data_scaling_configs': data_scaling_configs,
            'model_depth_configs': model_depth_configs,
            'variants_tested': variants_to_test
        },
        'results': {}
    }
    
    # Generate a base set of circuits for feature size calculation
    print("\n--- Generating Base Circuits for Feature Size Calculation ---")
    qiskit_circuits, cudaq_ciruits = generate_proper_circuits(
        num_circuits=100,  # Small number just for feature size calculation
        min_qubits=min_qubits,
        max_qubits=max_qubits,
        min_depth=min_depth,
        max_depth=max_depth,
        seed=seed,
        circuit_types=circuit_types
    )
    
    # --- DYNAMIC FEATURE SIZE CALCULATION ---
    print("\n--- Calculating Feature Sizes from Sample Circuit ---")
    if not qiskit_circuits:
        raise ValueError("No valid circuits were generated. Cannot determine feature sizes.")
    
    # Create a sample data object from the first circuit
    sample_data = circuit_to_graph_data(qiskit_circuits[0])

    # Extract the feature sizes from the tensor shapes
    node_feature_size = sample_data.x.shape[1]

    print(f"Determined Feature Sizes:")
    print(f"  Node Features:   {node_feature_size}")
    
    # Check for GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    for variant in variants_to_test:
        print(f"\n{'='*50}")
        print(f"STARTING SCALING STUDY FOR VARIANT: {variant}")
        print(f"{'='*50}")
        
        all_scaling_results['results'][variant] = {}
        
        for num_circuits_config in data_scaling_configs:
            print(f"\n{'-'*30}")
            print(f"DATA SCALING: {num_circuits_config} circuits")
            print(f"{'-'*30}")
            
            all_scaling_results['results'][variant][f'circuits_{num_circuits_config}'] = {}
            
            for model_depth_config in model_depth_configs:
                print(f"\n>>> Model Depth: {model_depth_config}")
                
                # Create unique output directory for this configuration
                config_output_dir = os.path.join(
                    output_dir, 
                    f"scaling_study_{variant}_circuits_{num_circuits_config}_depth_{model_depth_config}"
                )
                os.makedirs(config_output_dir, exist_ok=True)
                
                experiment_config = {
                    'variant': variant,
                    'num_circuits': num_circuits_config,
                    'model_depth': model_depth_config,
                    'node_feature_size': node_feature_size,
                    'noise_factors': noise_factors
                }
                
                try:
                    # Generate circuits for this configuration
                    print(f"Generating {num_circuits_config} circuits...")
                    circuits_qiskit, circuits_cudaq = generate_proper_circuits(
                        num_circuits=num_circuits_config,
                        min_qubits=min_qubits,
                        max_qubits=max_qubits,
                        min_depth=min_depth,
                        max_depth=max_depth,
                        seed=seed,
                        circuit_types=circuit_types
                    )
                    
                    qiskit_observables, cudaq_observables = generate_proper_observables(circuits_qiskit, seed=seed)
                    data_pairs = list(zip(circuits_cudaq, cudaq_observables))
                    n_qubits = max(c.num_qubits for c in circuits_qiskit)

                    # Run simulations
                    all_circuits_by_nf = []
                    all_observables_by_nf = []
                    all_noisy_values_by_nf = []
                    all_noiseless_values_by_nf = []
                    
                    for noise_factor in noise_factors:
                        print(f"Running simulations with noise factor: {noise_factor}")
                        
                        noise_model = get_quera_noise_model_cudaq(n_qubits, noise_factor)
                        noiseless_vals, noisy_vals = [], []

                        for kernel, H in data_pairs:
                            # ---------- noiseless ------------------------------------------------
                            zero_noise_model = cudaq.NoiseModel()          # ideal, no noise
                            res0 = cudaq.observe(kernel, H, noise_model=zero_noise_model)
                            noiseless_vals.append(float(res0.expectation()))

                            # ---------- noisy ----------------------------------------------------
                            res1 = cudaq.observe(kernel, H, noise_model=noise_model)
                            noisy_vals.append(float(res1.expectation()))
                            

                        
                        all_circuits_by_nf.append(circuits_qiskit)
                        all_observables_by_nf.append(cudaq_observables)
                        all_noisy_values_by_nf.append(noisy_vals)
                        all_noiseless_values_by_nf.append(noiseless_vals)
                    
                    # Prepare dataset
                    print("  Preparing dataset...")
                    dataset = prepare_mitigator_dataset(
                        all_circuits_by_nf,
                        all_observables_by_nf,
                        all_noisy_values_by_nf,
                        all_noiseless_values_by_nf,
                        noise_factors
                    )
                    
                    # Split dataset to prevent data leakage
                    circuit_to_entries = {}
                    for entry in dataset:
                        circuit_hash = qasm_hash(entry['circuit'])
                        if circuit_hash not in circuit_to_entries:
                            circuit_to_entries[circuit_hash] = []
                        circuit_to_entries[circuit_hash].append(entry)

                    unique_hashes = list(circuit_to_entries.keys())
                    
                    from sklearn.model_selection import train_test_split
                    train_hashes, test_hashes = train_test_split(unique_hashes, test_size=0.2, random_state=42)

                    train_dataset = [entry for h in train_hashes for entry in circuit_to_entries[h]]
                    test_dataset = [entry for h in test_hashes for entry in circuit_to_entries[h]]
                    
                    print(f"  Dataset: {len(train_dataset)} train, {len(test_dataset)} test samples")
                    
                    # Build and train model
                    print("  Building model...")
                    model = build_model(
                        variant,
                        node_fs=node_feature_size,
                        device=device,
                        depth=model_depth_config
                    )
                    
                    
                    print("  Training model...")
                    training_start_time = time.time()
                    trained_model, history = train_mitigator(
                        train_dataset,
                        model,
                        epochs=100,
                        batch_size=16,
                        lr=0.001,
                        device=device
                    )
                    training_time = time.time() - training_start_time
                    
                    # Evaluate model
                    print("  Evaluating model...")
                    eval_start_time = time.time()
                    eval_results = evaluate_mitigator(
                        trained_model, 
                        test_dataset, 
                        device=device, 
                        output_prefix=config_output_dir
                    )
                    eval_time = time.time() - eval_start_time
                    
                    # Visualize embeddings
                    print("  Visualizing embeddings...")
                    visualize_embeddings(
                        trained_model,
                        test_dataset,
                        device=device,
                        output_prefix=config_output_dir
                    )
                    
                    # Save model
                    torch.save({
                        'model_state_dict': trained_model.state_dict(),
                        'architecture': f'QErrorMitigationModel_{variant}_depth_{model_depth_config}',
                        'creation_date': time.strftime("%Y-%m-%d %H:%M:%S"),
                        'experiment_config': experiment_config,
                        'training_time': training_time,
                        'evaluation_time': eval_time
                    }, os.path.join(config_output_dir, "quantum_error_mitigator.pt"))
                    
                    print(f"  Saved model to {config_output_dir}")
                    
                    # Store results in comprehensive tracker
                    experiment_results = {
                        'experiment_config': experiment_config,
                        'training_time': training_time,
                        'evaluation_time': eval_time,
                        'dataset_sizes': {
                            'train': len(train_dataset),
                            'test': len(test_dataset),
                            'unique_circuits_train': len(train_hashes),
                            'unique_circuits_test': len(test_hashes)
                        },
                        'performance_metrics': eval_results['metrics'] if eval_results else {},
                        'confidence_intervals': eval_results.get('confidence_intervals', {}) if eval_results else {}
                    }
                    
                    all_scaling_results['results'][variant][f'circuits_{num_circuits_config}'][f'depth_{model_depth_config}'] = experiment_results
                    
                    # Save individual experiment results
                    save_scaling_results(experiment_config, eval_results, config_output_dir)
                    
                    print(f"  ✓ Completed: {variant}, {num_circuits_config} circuits, depth {model_depth_config}")
                    
                except Exception as e:
                    print(f"  ✗ ERROR in configuration {variant}, {num_circuits_config} circuits, depth {model_depth_config}: {str(e)}")
                    error_info = {
                        'error': str(e),
                        'experiment_config': experiment_config,
                        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    all_scaling_results['results'][variant][f'circuits_{num_circuits_config}'][f'depth_{model_depth_config}'] = error_info
                    continue
    
    # Save comprehensive results
    all_scaling_results['experiment_end_time'] = time.strftime("%Y-%m-%d %H:%M:%S")
    comprehensive_results_path = os.path.join(output_dir, "comprehensive_scaling_results.json")
    with open(comprehensive_results_path, 'w') as f:
        json.dump(all_scaling_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("SCALING STUDY COMPLETED!")
    print(f"Comprehensive results saved to: {comprehensive_results_path}")
    print(f"Individual experiment folders created in: {output_dir}")
    print(f"{'='*60}")
    
    # Create summary plots
    create_scaling_summary_plots(all_scaling_results, output_dir)

if __name__ == "__main__":
    main()

