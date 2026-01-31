import torch
import torch.nn as nn

def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result

def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)

class E_GCL(nn.Module):
    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, nodes_att_dim=0, act_fn=nn.ReLU(), recurrent=True, coords_weight=1.0, attention=False, clamp=False, norm_diff=False, tanh=False):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.coords_weight = coords_weight
        self.recurrent = recurrent
        self.attention = attention
        self.norm_diff = norm_diff
        self.tanh = tanh
        edge_coords_nf = 1
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        self.clamp = clamp
        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
            self.coords_range = nn.Parameter(torch.ones(1)) * 3
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        trans = torch.clamp(trans, min=-100, max=100)
        agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        coord += agg * self.coords_weight
        # 这里将coord的维度从3调整为1
        coord = coord.mean(dim=1, keepdim=True)
        return coord


    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)

        if self.norm_diff:
            norm = torch.sqrt(radial) + 1
            coord_diff = coord_diff / norm

        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)
        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        return h, coord

class AE(nn.Module):
    def __init__(self, hidden_nf, K=8, device='cuda', act_fn=nn.SiLU(), n_layers=3, reg=1e-3, clamp=False):
        super(AE, self).__init__()
        self.hidden_nf = hidden_nf
        self.K = K
        self.device = device
        self.n_layers = n_layers
        self.reg = reg
        self.clamp = clamp
        self.act_fn = act_fn

        self.encoder_layers = nn.ModuleList()
        self.encoder_layers.append(E_GCL(8, 4, self.hidden_nf, edges_in_d=0, act_fn=act_fn, recurrent=False, clamp=clamp))
        self.encoder_layers.append(act_fn)
        self.encoder_layers.append(E_GCL(4, 2, self.hidden_nf, edges_in_d=0, act_fn=act_fn, recurrent=True, clamp=clamp))
        self.encoder_layers.append(act_fn)
        self.encoder_layers.append(E_GCL(2, 1, self.hidden_nf, edges_in_d=0, act_fn=act_fn, recurrent=False, clamp=clamp))

        self.decoder_layers = nn.ModuleList()
        self.decoder_layers.append(E_GCL(1, 2, self.hidden_nf, edges_in_d=0, act_fn=act_fn, recurrent=False, clamp=clamp))
        self.decoder_layers.append(act_fn)
        self.decoder_layers.append(E_GCL(2, 4, self.hidden_nf, edges_in_d=0, act_fn=act_fn, recurrent=True, clamp=clamp))
        self.decoder_layers.append(act_fn)
        self.decoder_layers.append(E_GCL(4, 8, self.hidden_nf, edges_in_d=0, act_fn=act_fn, recurrent=False, clamp=clamp))

        self.w = nn.Parameter(-0.1 * torch.ones(1)).to(self.device)
        self.b = nn.Parameter(torch.ones(1)).to(self.device)
        
        self.coord_decoder = nn.Sequential(
            nn.Linear(1, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 3)
        )
        self.to(self.device)


    def encode(self, h, edge_index_pos, edge_index_cov, edge_index_hb, coord, edge_attr):
        coords = coord  # 使用传入的coord
        # print(coords)
        h, coords = self.encoder_layers[0](h, edge_index_pos, coords, edge_attr=edge_attr)
        h = self.encoder_layers[1](h)
        coords = self.encoder_layers[1](coords)
        h, coords = self.encoder_layers[2](h, edge_index_cov, coords, edge_attr=edge_attr)
        h = self.encoder_layers[3](h)
        coords = self.encoder_layers[3](coords)
        h, coords = self.encoder_layers[4](h, edge_index_hb, coords, edge_attr=edge_attr)
        # print(coords)
        return h, coords


    def decode(self, h, edge_index_pos, edge_index_cov, edge_index_hb, coord, edge_attr):
        h, coords = self.decoder_layers[0](h, edge_index_hb, coord, edge_attr=edge_attr)
        h = self.decoder_layers[1](h)
        coords = self.decoder_layers[1](coords)
        h, coords = self.decoder_layers[2](h, edge_index_cov, coord, edge_attr=edge_attr)
        h = self.decoder_layers[3](h)
        coords = self.decoder_layers[3](coords)
        h, coords = self.decoder_layers[4](h, edge_index_pos, coord, edge_attr=edge_attr)
        coords = self.coord_decoder(coords)
        return h, coords

    def forward(self, nodes, edge_index_pos, edge_index_cov, edge_index_hb, coord, edge_attr=None):
        h, x = self.encode(nodes, edge_index_pos, edge_index_cov, edge_index_hb, coord, edge_attr)
        h_pred, x_pred = self.decode(h, edge_index_pos, edge_index_cov, edge_index_hb, x, edge_attr)
        return h_pred, x_pred


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_nf = 8
    hidden_nf = 32
    latent_nf = 1
    autoencoder = AE(hidden_nf=hidden_nf, K=1, device=device, act_fn=nn.SiLU(), n_layers=3, reg=1e-3, clamp=False)


    h = torch.randn(25, input_nf).to(device)   
    edge_index_pos = torch.tensor([[0, 1, 2], [1, 2, 0]]).to(device)
    edge_index_cov = torch.tensor([[0, 1, 2], [1, 2, 0]]).to(device)
    edge_index_hb = torch.tensor([[0, 1, 2], [1, 2, 0]]).to(device)
    coord = torch.randn(25, 3).to(device)


    h_pred, coords_pred = autoencoder(h, edge_index_pos, edge_index_cov, edge_index_hb, coord)
    print('h_pred:', h_pred.size())
    print('coords_pred:', coords_pred.size())
    print(coords_pred)
