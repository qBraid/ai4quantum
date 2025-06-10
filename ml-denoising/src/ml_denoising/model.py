from torch_geometric.nn import GATv2Conv, global_mean_pool
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt 
from torch_geometric.data import Batch


class QErrorMitigationModel(nn.Module):
    """A GAT-based quantum error mitigation model.
    
    This model uses Graph Attention Networks (GAT) to process quantum circuit graphs
    and predict error corrections. It's designed as a lightweight baseline model
    that doesn't use edge features or complex circuit representations.
    
    Args:
        node_feature_size (int): Dimension of input node features from circuit graphs.
        depth (int, optional): Number of GAT layers. Defaults to 2.
    """
    def __init__(self, node_feature_size, depth=2):
        super().__init__()
        
        self.conv_layers = nn.ModuleList()
        in_channels = node_feature_size

        for i in range(depth):
            out_channels = 128 if i < depth-1 else 32
            # Edge features are not used in the simple model
            self.conv_layers.append(GATv2Conv(in_channels=in_channels, out_channels=out_channels))
            in_channels = out_channels
        
        self.obs_encoder = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 8)
        )
        
        self.noise_encoder = nn.Linear(1, 4)
        
        # Fusion network for the simple model
        self.fusion = nn.Sequential(
            nn.Linear(32 + 8 + 4 + 1, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )
        
    def forward(self, circuit_data, return_embeddings=False):
        """Forward pass through the simple error mitigation model.
        
        Args:
            circuit_data: PyTorch Geometric batch containing circuit graph data with
                         attributes x, edge_index, batch, observable_features, 
                         noise_factor, and noisy_exp.
            return_embeddings (bool, optional): Whether to return intermediate embeddings
                                               for visualization. Defaults to False.
        
        Returns:
            torch.Tensor: Predicted corrected expectation values.
            dict (optional): Dictionary containing circuit embeddings if return_embeddings=True.
        """
        x, edge_index, batch = circuit_data.x, circuit_data.edge_index, circuit_data.batch
        
        # The simple model only expects node features
        for conv in self.conv_layers:
            x = F.elu(conv(x, edge_index))
        
        circuit_embedding = global_mean_pool(x, batch)
        
        obs_embedding = self.obs_encoder(circuit_data.observable_features.squeeze(1))
        noise_embedding = self.noise_encoder(circuit_data.noise_factor)
        
        combined = torch.cat([
            circuit_embedding, 
            obs_embedding, 
            noise_embedding, 
            circuit_data.noisy_exp
        ], dim=1)
        
        # Predict the correction
        correction = self.fusion(combined)
        
        # Return the corrected expectation value
        prediction = circuit_data.noisy_exp + correction
        
        if return_embeddings:
            embeddings = {
                'circuit': circuit_embedding.detach().cpu().numpy()
            }
            return prediction, embeddings
            
        return prediction
    

def visualize_embeddings(model, dataset, device='cpu', output_prefix="ml_mitigation_output"):
    """Visualizes learned circuit embeddings using t-SNE dimensionality reduction.
    
    Generates 2D visualizations of high-dimensional circuit embeddings learned by
    the error mitigation model, colored by various circuit properties.
    
    Args:
        model: Trained error mitigation model (QErrorMitigationModel).
        dataset: Dataset containing circuit samples for embedding generation.
        device (str, optional): Device to run inference on ('cpu' or 'cuda'). 
                               Defaults to 'cpu'.
        output_prefix (str, optional): Directory to save visualization plots. 
                                     Defaults to "ml_mitigation_output".
    
    Returns:
        None: Saves visualization plots to disk.
    """
    from sklearn.manifold import TSNE
    import pandas as pd
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    if isinstance(model, tuple): # Handle case where (model, history) is passed
        model = model[0]
        
    model.eval()
    model.to(device)

    # Use a collate function to handle batching and metadata extraction
    def collate_fn(batch):
        data_list = [item['circuit_graph'] for item in batch]
        for i, data in enumerate(data_list):
            item = batch[i]
            data.observable_features = item['observable_features']
            data.noise_factor = item['noise_factor']
            data.noisy_exp = item['noisy_exp']
            data.true_exp = item['true_exp']
            data.correction = item['correction']
            data.num_qubits = torch.tensor([item['num_qubits']], dtype=torch.float)
            if hasattr(item['circuit_graph'], 'global_features') and item['circuit_graph'].global_features.shape[1] > 1:
                data.depth = torch.tensor([item['circuit_graph'].global_features[0, 1]], dtype=torch.float)
            else:
                 data.depth = torch.tensor([0], dtype=torch.float) # Fallback

        return Batch.from_data_list(data_list)

    loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    all_embeddings = []
    all_metadata = []

    print("\n--- Generating Embeddings for Visualization ---")
    with torch.no_grad():
        for batch in tqdm(loader, desc="Generating Embeddings", leave=False):
            circuit_data = batch.to(device)
            
            # The model's forward method now returns embeddings
            _, embeddings = model(circuit_data, return_embeddings=True)
            
            all_embeddings.append(embeddings['circuit'])
            
            # Collect metadata for coloring the plots
            for i in range(circuit_data.num_graphs):
                metadata = {
                    'num_qubits': circuit_data.num_qubits[i].item(),
                    'depth': circuit_data.depth[i].item(),
                    'noise_factor': circuit_data.noise_factor[i].item(),
                    'error_magnitude': abs(circuit_data.correction[i].item())
                }
                all_metadata.append(metadata)

    if not all_embeddings:
        print("No embeddings were generated. Skipping visualization.")
        return

    embeddings_np = np.concatenate(all_embeddings, axis=0)
    metadata_df = pd.DataFrame(all_metadata)

    # --- Perform t-SNE reduction ---
    print("Running t-SNE... (this may take a moment)")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings_np) - 1), max_iter=1000, init='pca')
    embeddings_2d = tsne.fit_transform(embeddings_np)
    
    # --- Create Plots ---
    print("Creating visualization plots...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(f't-SNE Visualization of Learned Circuit Embeddings ({os.path.basename(output_prefix)} model)', fontsize=20)
    
    # Plot 1: Color by Number of Qubits
    scatter1 = axes[0, 0].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=metadata_df['num_qubits'], cmap='viridis', alpha=0.7)
    axes[0, 0].set_title('Colored by Number of Qubits')
    axes[0, 0].grid(True, linestyle='--', alpha=0.6)
    fig.colorbar(scatter1, ax=axes[0, 0], label='Number of Qubits')

    # Plot 2: Color by Circuit Depth
    scatter2 = axes[0, 1].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=metadata_df['depth'], cmap='plasma', alpha=0.7)
    axes[0, 1].set_title('Colored by Circuit Depth')
    axes[0, 1].grid(True, linestyle='--', alpha=0.6)
    fig.colorbar(scatter2, ax=axes[0, 1], label='Circuit Depth')

    # Plot 3: Color by Noise Factor
    if metadata_df['noise_factor'].nunique() > 1:
        scatter3 = axes[1, 0].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=metadata_df['noise_factor'], cmap='magma', alpha=0.7)
        axes[1, 0].set_title('Colored by Noise Factor')
        axes[1, 0].grid(True, linestyle='--', alpha=0.6)
        fig.colorbar(scatter3, ax=axes[1, 0], label='Noise Factor')
    else:
        axes[1, 0].text(0.5, 0.5, 'Single Noise Factor in Dataset', ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Colored by Noise Factor')

    # Plot 4: Color by Error Magnitude
    scatter4 = axes[1, 1].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=metadata_df['error_magnitude'], cmap='cividis', alpha=0.7, vmax=np.percentile(metadata_df['error_magnitude'], 95))
    axes[1, 1].set_title('Colored by Error Magnitude (|True - Noisy|)')
    axes[1, 1].grid(True, linestyle='--', alpha=0.6)
    fig.colorbar(scatter4, ax=axes[1, 1], label='Correction Magnitude')

    for ax in axes.flat:
        ax.set_xlabel('t-SNE Component 1')
        ax.set_ylabel('t-SNE Component 2')
        
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(output_prefix, "embedding_visualization.png")
    plt.savefig(save_path)
    print(f"Saved embedding visualization to {save_path}")
    plt.close(fig)

def build_model(variant: str, node_fs: int, device='cpu', depth=None):
    """Factory function to build an error mitigation model based on the specified variant.
    
    Args:
        variant (str): Model variant to build ('simple').
        node_fs (int): Node feature size for the model.
        device (str, optional): Device to place the model on. Defaults to 'cpu'.
        depth (int, optional): Number of layers in the model. Uses variant-specific 
                              defaults if None.
    
    Returns:
        nn.Module: Configured error mitigation model moved to specified device.
    
    Raises:
        ValueError: If an unknown model variant is specified.
    """
    print(f"Building model variant: '{variant}' with depth: {depth}")
    if variant == 'simple':
        # The simple model doesn't use edge or global features.
        model_depth = depth if depth is not None else 2
        model = QErrorMitigationModel(node_feature_size=node_fs, depth=model_depth).to(device)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model has {num_params:,} trainable parameters")
        return model
    else:
        raise ValueError(f"Unknown model variant: {variant}")