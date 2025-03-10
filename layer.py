import torch
from torch import nn
from opt import *
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, ChebConv
from torch_geometric.nn.pool.select.topk import topk
from torch_geometric.nn.pool.connect.filter_edges import filter_adj
from util import dataloader
from edge import EdgeFeatureParser

opt = OptInit().initialize()

class NodeEmbeddingModule(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels):
        super(NodeEmbeddingModule, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):
        x = x.to(torch.float32)
        edge_attr = edge_attr.to(torch.float32)
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        return x

class AttentionPoolingModule(nn.Module):

    def __init__(self, in_channels, ratio=0.9):
        super(AttentionPoolingModule, self).__init__()
        self.attention_layer = GATConv(in_channels, 1, heads=1, concat=False)
        self.ratio = ratio

    def forward(self, x, edge_index, edge_attr, batch=None):
        score = self.attention_layer(x, edge_index, edge_attr).squeeze()

        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        perm = topk(score, self.ratio, batch)
        filtered_x = x[perm]
        filtered_edge_index, filtered_edge_attr = filter_adj(edge_index, edge_attr, perm)
        return filtered_x, filtered_edge_index, filtered_edge_attr, perm

class MutualInformationEstimator(nn.Module):

    def __init__(self, in_channels):
        super(MutualInformationEstimator, self).__init__()
        self.gat = GATConv(in_channels, in_channels, heads=1, concat=False)
        self.fc = nn.Linear(in_channels * 2, 1)

    def forward(self, x, edge_index):
        neg_samples = x[torch.randperm(x.size(0))]
        embeddings = self.gat(x, edge_index)

        joint = torch.cat([embeddings, x], dim=-1)
        margin = torch.cat([embeddings, neg_samples], dim=-1)

        joint_score = F.normalize(self.fc(joint), dim=1)
        margin_score = F.normalize(self.fc(margin), dim=1)

        mi_estimate = torch.mean(joint_score) - torch.log(torch.mean(torch.exp(margin_score)))
        return mi_estimate

class ResidualGraphBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ResidualGraphBlock, self).__init__()
        self.conv1 = ChebConv(in_channels, out_channels, K=3)
        self.conv2 = ChebConv(out_channels, out_channels, K=3)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.ln = nn.LayerNorm(out_channels)

        self.skip = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()

    def forward(self, x, edge_index):
        residual = self.skip(x)
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = self.bn2(self.conv2(x, edge_index)) + residual
        return F.relu(self.ln(x))

class HierarchicalGraphModule(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(HierarchicalGraphModule, self).__init__()
        self.res_blocks = nn.ModuleList([
            ResidualGraphBlock(input_dim, hidden_dim),
            ResidualGraphBlock(hidden_dim, hidden_dim),
            ResidualGraphBlock(hidden_dim, hidden_dim)
        ])
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, edge_weight):
        for block in self.res_blocks:
            x = block(x, edge_index)
        return self.fc_out(x)

class GraphClassifier(nn.Module):

    def __init__(self, node_in_dim, local_hidden_dim, hierarchical_in_dim, hierarchical_out_dim, nonimg_data,
                 phonetic_data):
        super(GraphClassifier, self).__init__()

        # Modules
        self.node_embedding_module = NodeEmbeddingModule(node_in_dim, local_hidden_dim, 20)
        self.attention_pooling = AttentionPoolingModule(20)
        self.mi_estimator = MutualInformationEstimator(20)
        self.hierarchical_module = HierarchicalGraphModule(hierarchical_in_dim, 20, hierarchical_out_dim)
        self.edge_parser = EdgeFeatureParser(2, dropout=0.2).to(opt.device)

        # External Data
        self.nonimg_data = nonimg_data
        self.phonetic_data = phonetic_data

    def forward(self, graphs):
        # dataloader_instance = dataloader().to(opt.device)
        dataloader_instance = dataloader()
        all_embeddings = []
        total_mi_loss = 0

        for graph in graphs:
            # Move graph data to device
            graph.x = graph.x.to(opt.device)
            graph.edge_index = graph.edge_index.to(opt.device)
            graph.edge_attr = graph.edge_attr.to(opt.device)

            # Check for batch attribute
            if hasattr(graph, 'batch') and graph.batch is not None:
                graph.batch = graph.batch.to(opt.device)

            # Node Embedding
            node_embeddings = self.node_embedding_module(graph.x, graph.edge_index, graph.edge_attr)

            # Pooling with Self-Attention
            pooled_x, pooled_edge_index, pooled_edge_attr, perm = self.attention_pooling(
                node_embeddings,
                graph.edge_index,
                graph.edge_attr,
                graph.batch
            )

            # Estimate MI loss
            mi_loss = self.mi_estimator(node_embeddings, graph.edge_index)
            total_mi_loss += mi_loss

            all_embeddings.append(pooled_x.view(1, -1))

        all_embeddings = torch.cat(all_embeddings)
        avg_mi_loss = total_mi_loss / len(graphs)
        edge_index, edge_input = dataloader_instance.get_inputs(
            self.nonimg_data,
            all_embeddings.detach().cpu().numpy(),  # 转换为numpy
            self.phonetic_data
        )

        edge_index = torch.tensor(edge_index, dtype=torch.long).to(opt.device)
        edge_input = torch.tensor(edge_input, dtype=torch.float32).to(opt.device)
        edge_weight = torch.squeeze(self.edge_parser(edge_input))

        # Hierarchical model prediction
        predictions = self.hierarchical_module(all_embeddings, edge_index, edge_weight)

        return predictions, avg_mi_loss
