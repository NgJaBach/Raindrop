"""
Raindrop Encoder - A Self-Contained Implementation

This module provides a consolidated, drop-in encoder implementation of the Raindrop model
from the paper "Graph-Guided Network for Irregularly Sampled Multivariate Time Series"

The Raindrop model is designed for irregularly sampled multivariate time series classification.
It uses graph neural networks to capture inter-sensor dependencies and handles missing values
through observation propagation mechanisms.

=== INPUT FORMAT SPECIFICATION ===

The model expects inputs in the following format:

1. src (P): Tensor of shape [max_len, batch_size, 2 * n_features]
   - First half ([:, :, :n_features]): Normalized time series values
   - Second half ([:, :, n_features:]): Binary mask (1 = observed, 0 = missing)
   - max_len: Maximum sequence length (e.g., 215 for P12, 60 for P19, 600 for PAM)
   - batch_size: Number of samples in the batch
   - n_features: Number of sensor/variable channels (e.g., 36 for P12, 34 for P19, 17 for PAM)
   
2. static (Pstatic): Tensor of shape [batch_size, d_static] or None
   - Static patient/sample features (e.g., demographics)
   - d_static: Dimension of static features (e.g., 9 for P12, 6 for P19)
   - Set to None if no static features are available
   
3. times (Ptime): Tensor of shape [max_len, batch_size]
   - Timestamps for each observation (typically in hours)
   - Zero-padded for shorter sequences
   
4. lengths: Tensor of shape [batch_size]
   - Actual sequence length for each sample (number of non-zero timestamps)

=== OUTPUT FORMAT ===

Returns a tuple of (output, distance, None):
- output: Tensor of shape [batch_size, n_classes] - class logits
- distance: Scalar tensor - regularization term (inter-sample attention distance)
- None: Placeholder for compatibility

=== USAGE EXAMPLE ===

```python
# Initialize the model
model = Raindrop_v2(
    d_inp=36,           # Number of input features/sensors
    d_model=36*4,       # Model dimension (d_inp * d_ob)
    nhead=2,            # Number of attention heads
    nhid=2*36*4,        # Hidden dimension in transformer
    nlayers=2,          # Number of transformer layers
    dropout=0.2,        # Dropout rate
    max_len=215,        # Maximum sequence length
    d_static=9,         # Static feature dimension (set to 0 if no static features)
    MAX=100,            # Positional encoding parameter
    perc=0.5,           # Not used in v2
    aggreg='mean',      # Aggregation method ('mean')
    n_classes=2,        # Number of output classes
    global_structure=torch.ones(36, 36),  # Initial graph structure (can be learned)
    sensor_wise_mask=False,  # Whether to use sensor-wise masking
    static=True         # Whether static features are used
)

# Forward pass
output, distance, _ = model(src, static, times, lengths)
```

=== DEPENDENCIES ===

- torch
- torch_geometric (for MessagePassing, softmax, glorot initialization)
- torch_sparse (for SparseTensor)
- torch_scatter (for scatter operations)

Author: Consolidated from original Raindrop implementation
"""

import math
from typing import Union, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear
from torch.nn.parameter import Parameter
from torch.nn import init

# PyTorch Geometric imports
from torch_geometric.typing import PairTensor, Adj, OptTensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import softmax
from torch_sparse import SparseTensor
from torch_scatter import scatter


# =============================================================================
# POSITIONAL ENCODING
# =============================================================================

class PositionalEncodingTF(nn.Module):
    """
    Temporal positional encoding for irregularly sampled time series.
    
    Uses sinusoidal encoding based on actual timestamps rather than positions.
    
    Args:
        d_model: Output dimension of positional encoding
        max_len: Maximum expected sequence length (used for timescale computation)
        MAX: Scaling factor for positional encoding (default: 10000)
    
    Input:
        P_time: Tensor of shape [max_len, batch_size] containing timestamps
        
    Output:
        Positional encoding of shape [max_len, batch_size, d_model]
    """
    
    def __init__(self, d_model: int, max_len: int = 500, MAX: int = 10000):
        super(PositionalEncodingTF, self).__init__()
        self.max_len = max_len
        self.d_model = d_model
        self.MAX = MAX
        self._num_timescales = d_model // 2

    def getPE(self, P_time: Tensor) -> Tensor:
        """Compute positional encoding for given timestamps."""
        timescales = self.max_len ** torch.linspace(0, 1, self._num_timescales)
        
        times = P_time.unsqueeze(2)  # [max_len, batch_size, 1]
        scaled_time = times / timescales[None, None, :]
        pe = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=-1)
        return pe.float()

    def forward(self, P_time: Tensor) -> Tensor:
        """
        Args:
            P_time: Timestamps tensor of shape [max_len, batch_size]
            
        Returns:
            Positional encoding of shape [max_len, batch_size, d_model]
        """
        device = P_time.device
        P_time_cpu = P_time.cpu()
        pe = self.getPE(P_time_cpu)
        return pe.to(device)


# =============================================================================
# TRANSFORMER CONVOLUTION (Graph Attention)
# =============================================================================

class TransformerConv(MessagePassing):
    """
    Graph Transformer Convolution layer.
    
    Implements attention-based message passing on graph structures.
    Used in the first version of Raindrop for inter-sensor communication.
    
    Based on "Masked Label Prediction: Unified Message Passing Model for 
    Semi-Supervised Classification" (https://arxiv.org/abs/2009.03509)
    
    Args:
        in_channels: Size of input features
        out_channels: Size of output features
        heads: Number of attention heads
        concat: Whether to concatenate or average multi-head outputs
        beta: Whether to use skip connection weighting
        dropout: Dropout probability on attention weights
        edge_dim: Edge feature dimension (if any)
        bias: Whether to use bias in linear layers
        root_weight: Whether to add transformed root node features
    """
    _alpha: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        beta: bool = False,
        dropout: float = 0.,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        root_weight: bool = True,
        **kwargs
    ):
        kwargs.setdefault('aggr', 'add')
        super(TransformerConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_key = Linear(in_channels[0], heads * out_channels)
        self.lin_query = Linear(in_channels[1], heads * out_channels)
        self.lin_value = Linear(in_channels[0], heads * out_channels)
        
        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.lin_edge = self.register_parameter('lin_edge', None)

        if concat:
            self.lin_skip = Linear(in_channels[1], heads * out_channels, bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * heads * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)
        else:
            self.lin_skip = Linear(in_channels[1], out_channels, bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if self.edge_dim:
            self.lin_edge.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()

    def forward(
        self,
        x: Union[Tensor, PairTensor],
        edge_index: Adj,
        edge_weights: OptTensor = None,
        edge_attr: OptTensor = None,
        return_attention_weights: Optional[bool] = None
    ):
        """
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Graph connectivity [2, num_edges]
            edge_weights: Optional edge weights
            edge_attr: Optional edge features
            return_attention_weights: Whether to return attention weights
            
        Returns:
            Updated node features, optionally with attention weights
        """
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        out = self.propagate(edge_index, x=x, edge_weights=edge_weights, 
                            edge_attr=edge_attr, size=None)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.root_weight:
            x_r = self.lin_skip(x[1])
            if self.lin_beta is not None:
                beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1))
                beta = beta.sigmoid()
                out = beta * x_r + (1 - beta) * out
            else:
                out += x_r

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(
        self,
        x_i: Tensor,
        x_j: Tensor,
        edge_weights: Tensor,
        edge_attr: OptTensor,
        index: Tensor,
        ptr: OptTensor,
        size_i: Optional[int]
    ) -> Tensor:
        query = self.lin_query(x_i).view(-1, self.heads, self.out_channels)
        key = self.lin_key(x_j).view(-1, self.heads, self.out_channels)

        if self.lin_edge is not None:
            assert edge_attr is not None
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
            key += edge_attr

        alpha = (query * key).sum(dim=-1) / math.sqrt(self.out_channels)
        if edge_weights is not None:
            alpha = edge_weights.unsqueeze(-1)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = self.lin_value(x_j).view(-1, self.heads, self.out_channels)
        out *= alpha.view(-1, self.heads, 1)
        return out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.heads
        )


# =============================================================================
# OBSERVATION PROPAGATION
# =============================================================================

class ObservationPropagation(MessagePassing):
    """
    Observation Propagation layer for inter-sensor message passing.
    
    This is the core component of Raindrop v2 that handles the propagation
    of observations between sensors based on learned graph structure.
    
    Args:
        in_channels: Input feature dimension
        out_channels: Output feature dimension
        n_nodes: Number of sensor nodes
        ob_dim: Observation dimension per sensor
        heads: Number of attention heads
        concat: Whether to concatenate multi-head outputs
        beta: Whether to use adaptive skip connections
        dropout: Dropout rate
        edge_dim: Edge feature dimension
        bias: Whether to use bias
        root_weight: Whether to use root node weights
    """
    _alpha: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        n_nodes: int,
        ob_dim: int,
        heads: int = 1,
        concat: bool = True,
        beta: bool = False,
        dropout: float = 0.,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        root_weight: bool = True,
        **kwargs
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_key = Linear(in_channels[0], heads * out_channels)
        self.lin_query = Linear(in_channels[1], heads * out_channels)
        self.lin_value = Linear(in_channels[0], heads * out_channels)
        
        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.lin_edge = self.register_parameter('lin_edge', None)

        if concat:
            self.lin_skip = Linear(in_channels[1], heads * out_channels, bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * heads * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)
        else:
            self.lin_skip = Linear(in_channels[1], out_channels, bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)

        self.weight = Parameter(torch.Tensor(in_channels[1], heads * out_channels))
        self.bias_param = Parameter(torch.Tensor(heads * out_channels))

        self.n_nodes = n_nodes
        self.nodewise_weights = Parameter(torch.Tensor(self.n_nodes, heads * out_channels))

        self.increase_dim = Linear(in_channels[1], heads * out_channels * 8)
        self.map_weights = Parameter(torch.Tensor(self.n_nodes, heads * 16))

        self.ob_dim = ob_dim
        self.index = None

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if self.edge_dim:
            self.lin_edge.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()
        glorot(self.weight)
        if self.bias_param is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias_param, -bound, bound)
        glorot(self.nodewise_weights)
        glorot(self.map_weights)
        self.increase_dim.reset_parameters()

    def forward(
        self,
        x: Union[Tensor, PairTensor],
        p_t: Tensor,
        edge_index: Adj,
        edge_weights: OptTensor = None,
        use_beta: bool = False,
        edge_attr: OptTensor = None,
        return_attention_weights: Optional[bool] = None
    ):
        """
        Args:
            x: Node features [n_nodes, in_channels]
            p_t: Positional/temporal encoding
            edge_index: Graph connectivity
            edge_weights: Edge weights from global structure
            use_beta: Whether to use beta weighting (for edge pruning)
            edge_attr: Edge attributes
            return_attention_weights: Whether to return attention
            
        Returns:
            Updated node features, optionally with attention weights
        """
        self.edge_index = edge_index
        self.p_t = p_t
        self.use_beta = use_beta

        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        out = self.propagate(edge_index, x=x, edge_weights=edge_weights, 
                            edge_attr=edge_attr, size=None)

        alpha = self._alpha
        self._alpha = None
        edge_index = self.edge_index

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(
        self,
        x_i: Tensor,
        x_j: Tensor,
        edge_weights: Tensor,
        edge_attr: OptTensor,
        index: Tensor,
        ptr: OptTensor,
        size_i: Optional[int]
    ) -> Tensor:
        use_beta = self.use_beta
        
        if use_beta:
            n_step = self.p_t.shape[0]
            n_edges = x_i.shape[0]

            h_W = self.increase_dim(x_i).view(-1, n_step, 32)
            w_v = self.map_weights[self.edge_index[1]].unsqueeze(1)

            p_emb = self.p_t.unsqueeze(0)

            aa = torch.cat([w_v.repeat(1, n_step, 1), p_emb.repeat(n_edges, 1, 1)], dim=-1)
            beta = torch.mean(h_W * aa, dim=-1)

        if edge_weights is not None:
            if use_beta:
                gamma = beta * (edge_weights.unsqueeze(-1))
                gamma = torch.repeat_interleave(gamma, self.ob_dim, dim=-1)

                # Edge pruning: keep top 50% of edges
                all_edge_weights = torch.mean(gamma, dim=1)
                K = int(gamma.shape[0] * 0.5)
                index_top_edges = torch.argsort(all_edge_weights, descending=True)[:K]
                gamma = gamma[index_top_edges]
                self.edge_index = self.edge_index[:, index_top_edges]
                index = self.edge_index[0]
                x_i = x_i[index_top_edges]
            else:
                gamma = edge_weights.unsqueeze(-1)

        self.index = index
        if use_beta:
            self._alpha = torch.mean(gamma, dim=-1)
        else:
            self._alpha = gamma

        gamma = softmax(gamma, index, ptr, size_i)
        gamma = F.dropout(gamma, p=self.dropout, training=self.training)

        out = F.relu(self.lin_value(x_i)).view(-1, self.heads, self.out_channels)
        
        if use_beta:
            out = out * gamma.view(-1, self.heads, out.shape[-1])
        else:
            out = out * gamma.view(-1, self.heads, 1)
        return out

    def aggregate(
        self,
        inputs: Tensor,
        index: Tensor,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None
    ) -> Tensor:
        """Custom aggregation using stored index."""
        index = self.index
        return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.heads
        )


# =============================================================================
# RAINDROP MODEL (Version 2 - Main Implementation)
# =============================================================================

class Raindrop(nn.Module):
    """
    Raindrop: Graph-Guided Network for Irregularly Sampled Multivariate Time Series.
    
    This is the main encoder class that implements the Raindrop architecture.
    It processes irregularly sampled multivariate time series by:
    1. Learning inter-sensor relationships via graph neural networks
    2. Using observation propagation to share information between sensors
    3. Applying transformer encoding for temporal modeling
    4. Aggregating representations for final classification
    
    Args:
        d_inp: Number of input features/sensors (e.g., 36 for P12 dataset)
        d_model: Model dimension, typically d_inp * d_ob
        nhead: Number of attention heads in transformer
        nhid: Hidden dimension in transformer feedforward layers
        nlayers: Number of transformer encoder layers
        dropout: Dropout probability
        max_len: Maximum sequence length
        d_static: Dimension of static features (0 if not used)
        MAX: Positional encoding scaling parameter
        perc: Unused parameter (kept for compatibility)
        aggreg: Aggregation method ('mean' recommended)
        n_classes: Number of output classes
        global_structure: Initial graph structure [d_inp, d_inp], can be all ones
        sensor_wise_mask: Whether to use sensor-wise masking for aggregation
        static: Whether to use static features
        
    Example Config for Different Datasets:
        P12:  d_inp=36, max_len=215, d_static=9, n_classes=2
        P19:  d_inp=34, max_len=60,  d_static=6, n_classes=2
        PAM:  d_inp=17, max_len=600, d_static=0, n_classes=8, static=False
    """

    def __init__(
        self,
        d_inp: int = 36,
        d_model: int = 64,
        nhead: int = 4,
        nhid: int = 128,
        nlayers: int = 2,
        dropout: float = 0.3,
        max_len: int = 215,
        d_static: int = 9,
        MAX: int = 100,
        perc: float = 0.5,
        aggreg: str = 'mean',
        n_classes: int = 2,
        global_structure: Optional[Tensor] = None,
        sensor_wise_mask: bool = False,
        static: bool = True
    ):
        super().__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        
        self.model_type = 'Transformer'
        self.global_structure = global_structure
        self.sensor_wise_mask = sensor_wise_mask

        # Dimensions
        d_pe = 16  # Positional encoding dimension
        self.d_inp = d_inp
        self.d_model = d_model
        self.static = static
        self.d_ob = int(d_model / d_inp)  # Observation embedding dimension per sensor
        
        # Static feature embedding
        if self.static:
            self.emb = nn.Linear(d_static, d_inp)

        # Input encoder
        self.encoder = nn.Linear(d_inp * self.d_ob, d_inp * self.d_ob)

        # Positional encoding
        self.pos_encoder = PositionalEncodingTF(d_pe, max_len, MAX)

        # Transformer encoder
        if self.sensor_wise_mask:
            encoder_layers = TransformerEncoderLayer(
                d_inp * (self.d_ob + 16), nhead, nhid, dropout
            )
        else:
            encoder_layers = TransformerEncoderLayer(
                d_model + 16, nhead, nhid, dropout
            )
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        # Graph structure (adjacency)
        self.adj = None  # Will be set based on device

        # Learnable sensor representation
        self.R_u = Parameter(torch.Tensor(1, d_inp * self.d_ob))

        # Observation propagation layers (2 layers for depth)
        self.ob_propagation = ObservationPropagation(
            in_channels=max_len * self.d_ob,
            out_channels=max_len * self.d_ob,
            heads=1,
            n_nodes=d_inp,
            ob_dim=self.d_ob
        )
        self.ob_propagation_layer2 = ObservationPropagation(
            in_channels=max_len * self.d_ob,
            out_channels=max_len * self.d_ob,
            heads=1,
            n_nodes=d_inp,
            ob_dim=self.d_ob
        )

        # Output MLP
        if static:
            d_final = d_model + d_pe + d_inp
        else:
            d_final = d_model + d_pe

        self.mlp_static = nn.Sequential(
            nn.Linear(d_final, d_final),
            nn.ReLU(),
            nn.Linear(d_final, n_classes),
        )

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_classes),
        )

        self.aggreg = aggreg
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        self.init_weights()

    def init_weights(self):
        """Initialize model weights."""
        initrange = 1e-10
        self.encoder.weight.data.uniform_(-initrange, initrange)
        if self.static:
            self.emb.weight.data.uniform_(-initrange, initrange)
        glorot(self.R_u)

    def forward(
        self,
        src: Tensor,
        static: Optional[Tensor],
        times: Tensor,
        lengths: Tensor
    ) -> Tuple[Tensor, Tensor, None]:
        """
        Forward pass of the Raindrop model.
        
        Args:
            src: Input tensor of shape [max_len, batch_size, 2*d_inp]
                 First half contains values, second half contains observation mask
            static: Static features [batch_size, d_static] or None
            times: Timestamps [max_len, batch_size]
            lengths: Sequence lengths [batch_size]
            
        Returns:
            Tuple of:
                - output: Class logits [batch_size, n_classes]
                - distance: Regularization term (mean pairwise distance of attention)
                - None: Placeholder for compatibility
        """
        device = src.device
        maxlen, batch_size = src.shape[0], src.shape[1]
        
        # Extract mask and values
        missing_mask = src[:, :, self.d_inp:int(2 * self.d_inp)]
        src = src[:, :, :int(src.shape[2] / 2)]
        n_sensor = self.d_inp

        # Expand input with learned sensor representation
        src = torch.repeat_interleave(src, self.d_ob, dim=-1)
        h = F.relu(src * self.R_u)
        
        # Get positional encoding
        pe = self.pos_encoder(times)
        
        # Static feature embedding
        if static is not None:
            emb = self.emb(static)

        h = self.dropout(h)

        # Create padding mask
        mask = torch.arange(maxlen, device='cpu')[None, :] >= (lengths.cpu()[:, None])
        mask = mask.squeeze(1).to(device)

        # === OBSERVATION PROPAGATION ===
        # Build graph from global structure
        adj = self.global_structure.to(device)
        adj[torch.eye(self.d_inp, device=device).bool()] = 1

        edge_index = torch.nonzero(adj).T
        edge_weights = adj[edge_index[0], edge_index[1]]

        n_step = src.shape[0]
        output = torch.zeros([n_step, batch_size, self.d_inp * self.d_ob], device=device)

        use_beta = False
        if use_beta:
            alpha_all = torch.zeros([int(edge_index.shape[1] / 2), batch_size], device=device)
        else:
            alpha_all = torch.zeros([edge_index.shape[1], batch_size], device=device)

        # Process each sample in batch
        for unit in range(batch_size):
            stepdata = h[:, unit, :]
            p_t = pe[:, unit, :]

            # Reshape: [max_len, d_inp*d_ob] -> [d_inp, max_len*d_ob]
            stepdata = stepdata.reshape([n_step, self.d_inp, self.d_ob]).permute(1, 0, 2)
            stepdata = stepdata.reshape(self.d_inp, n_step * self.d_ob)

            # First observation propagation layer
            stepdata, attentionweights = self.ob_propagation(
                stepdata, p_t=p_t, edge_index=edge_index, edge_weights=edge_weights,
                use_beta=use_beta, edge_attr=None, return_attention_weights=True
            )

            edge_index_layer2 = attentionweights[0]
            edge_weights_layer2 = attentionweights[1].squeeze(-1)

            # Second observation propagation layer
            stepdata, attentionweights = self.ob_propagation_layer2(
                stepdata, p_t=p_t, edge_index=edge_index_layer2, 
                edge_weights=edge_weights_layer2,
                use_beta=False, edge_attr=None, return_attention_weights=True
            )

            # Reshape back: [d_inp, max_len*d_ob] -> [max_len, d_inp*d_ob]
            stepdata = stepdata.view([self.d_inp, n_step, self.d_ob])
            stepdata = stepdata.permute([1, 0, 2])
            stepdata = stepdata.reshape([-1, self.d_inp * self.d_ob])

            output[:, unit, :] = stepdata
            alpha_all[:, unit] = attentionweights[1].squeeze(-1)

        # Compute regularization term
        distance = torch.cdist(alpha_all.T, alpha_all.T, p=2)
        distance = torch.mean(distance)

        # === COMBINE WITH POSITIONAL ENCODING ===
        if self.sensor_wise_mask:
            extend_output = output.view(-1, batch_size, self.d_inp, self.d_ob)
            extended_pe = pe.unsqueeze(2).repeat([1, 1, self.d_inp, 1])
            output = torch.cat([extend_output, extended_pe], dim=-1)
            output = output.view(-1, batch_size, self.d_inp * (self.d_ob + 16))
        else:
            output = torch.cat([output, pe], dim=2)

        # === TRANSFORMER ENCODING ===
        r_out = self.transformer_encoder(output, src_key_padding_mask=mask)

        # === AGGREGATION ===
        lengths2 = lengths.unsqueeze(1)
        mask2 = mask.permute(1, 0).unsqueeze(2).long()
        
        if self.sensor_wise_mask:
            output = torch.zeros([batch_size, self.d_inp, self.d_ob + 16], device=device)
            extended_missing_mask = missing_mask.view(-1, batch_size, self.d_inp)
            for se in range(self.d_inp):
                r_out_view = r_out.view(-1, batch_size, self.d_inp, (self.d_ob + 16))
                out = r_out_view[:, :, se, :]
                len_se = torch.sum(extended_missing_mask[:, :, se], dim=0).unsqueeze(1)
                out_sensor = torch.sum(
                    out * (1 - extended_missing_mask[:, :, se].unsqueeze(-1)), dim=0
                ) / (len_se + 1)
                output[:, se, :] = out_sensor
            output = output.view([-1, self.d_inp * (self.d_ob + 16)])
        elif self.aggreg == 'mean':
            output = torch.sum(r_out * (1 - mask2), dim=0) / (lengths2 + 1)

        # === OUTPUT ===
        if static is not None:
            output = torch.cat([output, emb], dim=1)
        output = self.mlp_static(output)

        return output, distance, None
    
    def get_representation(
        self,
        src: Tensor,
        static: Optional[Tensor],
        times: Tensor,
        lengths: Tensor
    ) -> Tensor:
        """
        Get the learned representation before the final classification layer.
        
        Useful for using Raindrop as a feature extractor / encoder for downstream tasks.
        
        Args:
            Same as forward()
            
        Returns:
            Representation tensor of shape [batch_size, d_final]
            where d_final = d_model + d_pe (+ d_inp if static features used)
        """
        device = src.device
        maxlen, batch_size = src.shape[0], src.shape[1]
        
        missing_mask = src[:, :, self.d_inp:int(2 * self.d_inp)]
        src = src[:, :, :int(src.shape[2] / 2)]

        src = torch.repeat_interleave(src, self.d_ob, dim=-1)
        h = F.relu(src * self.R_u)
        pe = self.pos_encoder(times)
        
        if static is not None:
            emb = self.emb(static)

        h = self.dropout(h)

        mask = torch.arange(maxlen, device='cpu')[None, :] >= (lengths.cpu()[:, None])
        mask = mask.squeeze(1).to(device)

        adj = self.global_structure.to(device)
        adj[torch.eye(self.d_inp, device=device).bool()] = 1
        edge_index = torch.nonzero(adj).T
        edge_weights = adj[edge_index[0], edge_index[1]]

        n_step = src.shape[0]
        output = torch.zeros([n_step, batch_size, self.d_inp * self.d_ob], device=device)

        for unit in range(batch_size):
            stepdata = h[:, unit, :]
            p_t = pe[:, unit, :]

            stepdata = stepdata.reshape([n_step, self.d_inp, self.d_ob]).permute(1, 0, 2)
            stepdata = stepdata.reshape(self.d_inp, n_step * self.d_ob)

            stepdata, attentionweights = self.ob_propagation(
                stepdata, p_t=p_t, edge_index=edge_index, edge_weights=edge_weights,
                use_beta=False, edge_attr=None, return_attention_weights=True
            )

            edge_index_layer2 = attentionweights[0]
            edge_weights_layer2 = attentionweights[1].squeeze(-1)

            stepdata, _ = self.ob_propagation_layer2(
                stepdata, p_t=p_t, edge_index=edge_index_layer2,
                edge_weights=edge_weights_layer2,
                use_beta=False, edge_attr=None, return_attention_weights=True
            )

            stepdata = stepdata.view([self.d_inp, n_step, self.d_ob])
            stepdata = stepdata.permute([1, 0, 2])
            stepdata = stepdata.reshape([-1, self.d_inp * self.d_ob])

            output[:, unit, :] = stepdata

        if self.sensor_wise_mask:
            extend_output = output.view(-1, batch_size, self.d_inp, self.d_ob)
            extended_pe = pe.unsqueeze(2).repeat([1, 1, self.d_inp, 1])
            output = torch.cat([extend_output, extended_pe], dim=-1)
            output = output.view(-1, batch_size, self.d_inp * (self.d_ob + 16))
        else:
            output = torch.cat([output, pe], dim=2)

        r_out = self.transformer_encoder(output, src_key_padding_mask=mask)

        lengths2 = lengths.unsqueeze(1)
        mask2 = mask.permute(1, 0).unsqueeze(2).long()

        if self.sensor_wise_mask:
            representation = torch.zeros([batch_size, self.d_inp, self.d_ob + 16], device=device)
            extended_missing_mask = missing_mask.view(-1, batch_size, self.d_inp)
            for se in range(self.d_inp):
                r_out_view = r_out.view(-1, batch_size, self.d_inp, (self.d_ob + 16))
                out = r_out_view[:, :, se, :]
                len_se = torch.sum(extended_missing_mask[:, :, se], dim=0).unsqueeze(1)
                out_sensor = torch.sum(
                    out * (1 - extended_missing_mask[:, :, se].unsqueeze(-1)), dim=0
                ) / (len_se + 1)
                representation[:, se, :] = out_sensor
            representation = representation.view([-1, self.d_inp * (self.d_ob + 16)])
        elif self.aggreg == 'mean':
            representation = torch.sum(r_out * (1 - mask2), dim=0) / (lengths2 + 1)

        if static is not None:
            representation = torch.cat([representation, emb], dim=1)

        return representation


# =============================================================================
# UTILITY FUNCTIONS FOR INFERENCE
# =============================================================================

def evaluate(
    model: nn.Module,
    P_tensor: Tensor,
    P_time_tensor: Tensor,
    P_static_tensor: Optional[Tensor],
    batch_size: int = 100,
    n_classes: int = 2,
    static: bool = True
) -> Tensor:
    """
    Evaluate model on a dataset in batches.
    
    Args:
        model: The Raindrop model
        P_tensor: Input data [max_len, N, 2*d_inp]
        P_time_tensor: Timestamps [max_len, N]
        P_static_tensor: Static features [N, d_static] or None
        batch_size: Batch size for evaluation
        n_classes: Number of classes
        static: Whether static features are used
        
    Returns:
        Output logits of shape [N, n_classes]
    """
    model.eval()
    device = next(model.parameters()).device
    
    P_tensor = P_tensor.to(device)
    P_time_tensor = P_time_tensor.to(device)
    
    if static and P_static_tensor is not None:
        P_static_tensor = P_static_tensor.to(device)
        N = P_static_tensor.shape[0]
    else:
        P_static_tensor = None
        N = P_tensor.shape[1]

    T, _, Ff = P_tensor.shape
    n_batches, rem = N // batch_size, N % batch_size
    out = torch.zeros(N, n_classes)
    
    start = 0
    for i in range(n_batches):
        P = P_tensor[:, start:start + batch_size, :]
        Ptime = P_time_tensor[:, start:start + batch_size]
        Pstatic = P_static_tensor[start:start + batch_size] if P_static_tensor is not None else None
        lengths = torch.sum(Ptime > 0, dim=0)
        
        with torch.no_grad():
            middleoutput, _, _ = model.forward(P, Pstatic, Ptime, lengths)
        out[start:start + batch_size] = middleoutput.detach().cpu()
        start += batch_size
        
    if rem > 0:
        P = P_tensor[:, start:start + rem, :]
        Ptime = P_time_tensor[:, start:start + rem]
        Pstatic = P_static_tensor[start:start + rem] if P_static_tensor is not None else None
        lengths = torch.sum(Ptime > 0, dim=0)
        
        with torch.no_grad():
            output, _, _ = model.forward(P, Pstatic, Ptime, lengths)
        out[start:start + rem] = output.detach().cpu()
        
    return out


def evaluate_standard(
    model: nn.Module,
    P_tensor: Tensor,
    P_time_tensor: Tensor,
    P_static_tensor: Optional[Tensor],
    batch_size: int = 100,
    n_classes: int = 2,
    static: bool = True
) -> Tensor:
    """
    Simple evaluation without batching (for smaller datasets).
    
    Args:
        Same as evaluate()
        
    Returns:
        Output logits of shape [N, n_classes]
    """
    device = next(model.parameters()).device
    
    P_tensor = P_tensor.to(device)
    P_time_tensor = P_time_tensor.to(device)
    
    if static and P_static_tensor is not None:
        P_static_tensor = P_static_tensor.to(device)
    else:
        P_static_tensor = None

    lengths = torch.sum(P_time_tensor > 0, dim=0)
    
    with torch.no_grad():
        out, _, _ = model.forward(P_tensor, P_static_tensor, P_time_tensor, lengths)
    return out


# =============================================================================
# CONVENIENCE FUNCTION FOR CREATING MODEL
# =============================================================================

def create_raindrop_encoder(
    n_features: int,
    max_len: int,
    n_classes: int = 2,
    d_static: int = 0,
    d_ob: int = 4,
    nhead: int = 2,
    nlayers: int = 2,
    dropout: float = 0.2,
    use_static: bool = True,
    sensor_wise_mask: bool = False,
    device: str = 'cuda'
) -> Raindrop:
    """
    Convenience function to create a Raindrop encoder with sensible defaults.
    
    Args:
        n_features: Number of input features/sensors
        max_len: Maximum sequence length
        n_classes: Number of output classes
        d_static: Dimension of static features (0 if not used)
        d_ob: Observation dimension multiplier (default 4)
        nhead: Number of attention heads
        nlayers: Number of transformer layers
        dropout: Dropout rate
        use_static: Whether to use static features
        sensor_wise_mask: Whether to use sensor-wise masking
        device: Device to place model on
        
    Returns:
        Configured Raindrop model
    """
    d_model = n_features * d_ob
    nhid = 2 * d_model
    
    global_structure = torch.ones(n_features, n_features)
    
    model = Raindrop(
        d_inp=n_features,
        d_model=d_model,
        nhead=nhead,
        nhid=nhid,
        nlayers=nlayers,
        dropout=dropout,
        max_len=max_len,
        d_static=d_static if use_static else 0,
        MAX=100,
        perc=0.5,
        aggreg='mean',
        n_classes=n_classes,
        global_structure=global_structure,
        sensor_wise_mask=sensor_wise_mask,
        static=use_static
    )
    
    return model.to(device)


# =============================================================================
# MAIN - Example Usage
# =============================================================================

if __name__ == "__main__":
    # Example: Create model for P12-like dataset
    print("Creating Raindrop encoder for P12-like dataset...")
    
    # Model configuration
    n_features = 36      # Number of sensors
    max_len = 215        # Maximum sequence length
    n_classes = 2        # Binary classification
    d_static = 9         # Static feature dimension
    batch_size = 8       # Small batch for demo
    
    # Create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = create_raindrop_encoder(
        n_features=n_features,
        max_len=max_len,
        n_classes=n_classes,
        d_static=d_static,
        device=device
    )
    
    print(f"Model created on device: {device}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dummy input
    # src: [max_len, batch_size, 2*n_features] - values + mask
    src = torch.randn(max_len, batch_size, 2 * n_features).to(device)
    src[:, :, n_features:] = (torch.rand(max_len, batch_size, n_features) > 0.3).float().to(device)
    
    # static: [batch_size, d_static]
    static = torch.randn(batch_size, d_static).to(device)
    
    # times: [max_len, batch_size]
    times = torch.linspace(0, 48, max_len).unsqueeze(1).repeat(1, batch_size).to(device)
    
    # lengths: [batch_size]
    lengths = torch.full((batch_size,), max_len).to(device)
    
    # Forward pass
    print("\nRunning forward pass...")
    model.eval()
    with torch.no_grad():
        output, distance, _ = model(src, static, times, lengths)
    
    print(f"Output shape: {output.shape}")  # [batch_size, n_classes]
    print(f"Distance (regularization): {distance.item():.4f}")
    
    # Get representations
    print("\nGetting representations...")
    with torch.no_grad():
        representations = model.get_representation(src, static, times, lengths)
    print(f"Representation shape: {representations.shape}")
    
    print("\n✓ Raindrop encoder is working correctly!")
