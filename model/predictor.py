import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import glob
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("/home/cpns/Chen/paper/DIPP_copy/network_visualization")

class DrivingData(Dataset):
    def __init__(self, data_dir):
        self.data_list = glob.glob(data_dir)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = np.load(self.data_list[idx])
        ego = data['ego']
        neighbors = data['neighbors']
        ref_line = data['ref_line'] 
        map_lanes = data['map_lanes']
        map_crosswalks = data['map_crosswalks']
        gt_future_states = data['gt_future_states']

        return ego, neighbors, map_lanes, map_crosswalks, ref_line, gt_future_states


# ego future plan encoder
class EgoPlanEncoder(nn.Module):
    def __init__(self):
        super(EgoPlanEncoder, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(5, 128),nn.ReLU(),nn.Linear(128, 256))

    def forward(self,inputs):
        plancode = self.mlp(inputs[:, :, :5])
        return plancode
    

# Agent history encoder
class AgentEncoder(nn.Module):
    def __init__(self):
        super(AgentEncoder, self).__init__()
        self.motion = nn.LSTM(8, 256, 2, batch_first=True)

    def forward(self, inputs):
        traj, _ = self.motion(inputs[:, :, :8])
        output = traj[:, -1]

        return output

# Local context encoders
class LaneEncoder(nn.Module):
    def __init__(self):
        super(LaneEncoder, self).__init__()
        # encdoer layer
        self.self_line = nn.Linear(3, 128)
        self.left_line = nn.Linear(3, 128)
        self.right_line = nn.Linear(3, 128)
        self.speed_limit = nn.Linear(1, 64)
        self.self_type = nn.Embedding(4, 64, padding_idx=0)
        self.left_type = nn.Embedding(11, 64, padding_idx=0)
        self.right_type = nn.Embedding(11, 64, padding_idx=0)
        self.traffic_light_type = nn.Embedding(9, 64, padding_idx=0)
        self.interpolating = nn.Embedding(2, 64)
        self.stop_sign = nn.Embedding(2, 64)
        self.stop_point = nn.Embedding(2, 64)

        # hidden layers
        self.pointnet = nn.Sequential(nn.Linear(512, 384), nn.ReLU(), nn.Linear(384, 256), nn.ReLU())

    def forward(self, inputs):
        # embedding
        self_line = self.self_line(inputs[..., :3])
        left_line = self.left_line(inputs[..., 3:6])
        right_line = self.right_line(inputs[...,  6:9])
        speed_limit = self.speed_limit(inputs[..., 9].unsqueeze(-1))
        self_type = self.self_type(inputs[..., 10].int())
        left_type = self.left_type(inputs[..., 11].int())
        right_type = self.right_type(inputs[..., 12].int()) 
        traffic_light = self.traffic_light_type(inputs[..., 13].int())
        stop_point = self.stop_point(inputs[..., 14].int())
        interpolating = self.interpolating(inputs[..., 15].int()) 
        stop_sign = self.stop_sign(inputs[..., 16].int())

        lane_attr = self_type + left_type + right_type + traffic_light + stop_point + interpolating + stop_sign
        
        lane_embedding = torch.cat([self_line, left_line, right_line, speed_limit, lane_attr], dim=-1)
    
        # process
        output = self.pointnet(lane_embedding)

        return output

class CrosswalkEncoder(nn.Module):
    def __init__(self):
        super(CrosswalkEncoder, self).__init__()
        self.point_net = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 256), nn.ReLU())
    
    def forward(self, inputs):
        output = self.point_net(inputs)

        return output

# Transformer modules
class CrossTransformer(nn.Module):
    def __init__(self,embed_dim=256):
        super(CrossTransformer, self).__init__()
        self.cross_attention = nn.MultiheadAttention(embed_dim, 8, 0.1, batch_first=True)
        self.transformer = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1024), nn.ReLU(), nn.Dropout(0.1), nn.Linear(1024, embed_dim), nn.LayerNorm(embed_dim))
        # self.cross_attention = nn.MultiheadAttention(512, 8, 0.1, batch_first=True)
        # self.transformer = nn.Sequential(nn.LayerNorm(512), nn.Linear(256, 1024), nn.ReLU(), nn.Dropout(0.1), nn.Linear(1024, 256), nn.LayerNorm(256))

    def forward(self, query, key, value, mask=None):
        attention_output, _ = self.cross_attention(query, key, value, key_padding_mask=mask)
        output = self.transformer(attention_output)

        return output



class MultiModalTransformer(nn.Module):
    def __init__(self, modes=3, output_dim=256):
        super(MultiModalTransformer, self).__init__()
        self.modes = modes
        self.attention = nn.ModuleList([nn.MultiheadAttention(256, 4, 0.1, batch_first=True) for _ in range(modes)])
        self.ffn = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 1024), nn.ReLU(), nn.Dropout(0.1), nn.Linear(1024, output_dim), nn.LayerNorm(output_dim))

    def forward(self, query, key, value, mask=None):
        attention_output = []
        for i in range(self.modes):
            attention_output.append(self.attention[i](query, key, value, key_padding_mask=mask)[0])
        attention_output = torch.stack(attention_output, dim=1)
        output = self.ffn(attention_output)

        return output

# Transformer-based encoders
class Agent2Agent(nn.Module):
    def __init__(self):
        super(Agent2Agent, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=1024, activation='relu', batch_first=True)
        self.interaction_net = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, inputs, mask=None):
        output = self.interaction_net(inputs, src_key_padding_mask=mask)

        return output

class Agent2Map(nn.Module):
    def __init__(self):
        super(Agent2Map, self).__init__()
        self.lane_attention = CrossTransformer()
        self.crosswalk_attention = CrossTransformer()
        self.map_attention = MultiModalTransformer() 

    def forward(self, actor, lanes, crosswalks, mask):
        query = actor.unsqueeze(1)
        # k : lanes v: lanes
        lanes_actor = [self.lane_attention(query, lanes[:, i], lanes[:, i]) for i in range(lanes.shape[1])]
        # k: crosswalks v : crosswalks
        crosswalks_actor = [self.crosswalk_attention(query, crosswalks[:, i], crosswalks[:, i]) for i in range(crosswalks.shape[1])]
        map_actor = torch.cat(lanes_actor+crosswalks_actor, dim=1)

        #自注意力
        output = self.map_attention(query, map_actor, map_actor, mask).squeeze(2)

        return map_actor, output 
    
class Ego2Ego(nn.Module):
    def __init__(self):
        super(Ego2Ego, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=1024, activation='relu', batch_first=True)
        self.ego_plan_selfattention=nn.TransformerEncoder(encoder_layer,num_layers=2)

    def forward(self, inputs, mask=None):
        output = self.ego_plan_selfattention(inputs, src_key_padding_mask=mask)
        return output

class Ego2Agent(nn.Module):
    def __init__(self):
        super(Ego2Agent, self).__init__()
        self.ego_attention = CrossTransformer()

    def forward(self,actor, ego ,mask=None):
        output = self.ego_attention(actor,ego,ego)
        return output

# Decoders
class AgentDecoder(nn.Module):
    def __init__(self, future_steps):
        super(AgentDecoder, self).__init__()
        self._future_steps = future_steps 
        self.decode = nn.Sequential(nn.Dropout(0.1), nn.Linear(512, 256), nn.ELU(), nn.Linear(256, future_steps*4))

    def transform(self, prediction, current_state):
        x = current_state[:, 0] 
        y = current_state[:, 1]
        theta = current_state[:, 2]
        delta_x = prediction[:, :, 0]
        delta_y = prediction[:, :, 1]
        delta_theta = prediction[:, :, 2]
        new_x = x.unsqueeze(1) + delta_x 
        new_y = y.unsqueeze(1) + delta_y 
        new_theta = theta.unsqueeze(1) + delta_theta
        traj = torch.stack([new_x, new_y, new_theta], dim=-1)

        return traj
       
    def forward(self, agent_map, agent_agent, current_state):
        # feature = torch.cat([agent_map, agent_agent.unsqueeze(1).repeat(1, 3, 1, 1)], dim=-1)
        # decoded = self.decode(feature).view(-1, 3, 10, self._future_steps, 3)
        # trajs = torch.stack([self.transform(decoded[:, i, j], current_state[:, j]) for i in range(3) for j in range(10)], dim=1)
        # trajs = torch.reshape(trajs, (-1, 3, 10, self._future_steps, 3))
        # print(f'fagent_map:{agent_map.shape} , fagent_agent:{agent_agent.shape}')
        feature = torch.cat([agent_map, agent_agent.unsqueeze(1).repeat(1, 5, 1, 1)], dim=-1)
        # print(f'feature:{feature.shape}')
        decoded = self.decode(feature).view(-1, 5, 10, self._future_steps, 4)
        # print(f'decoded:{decoded.shape}')
        trajs = torch.stack([self.transform(decoded[:, i, j], current_state[:, j]) for i in range(4) for j in range(10)], dim=1)
        trajs = torch.reshape(trajs, (-1, 5, 10, self._future_steps, 4))
        return trajs


# Decoders
class AgentDecoder_1(nn.Module):
    def __init__(self, future_steps):
        super(AgentDecoder_1, self).__init__()
        self._future_steps = future_steps 
        self.decode = nn.Sequential(nn.Dropout(0.1), nn.Linear(768, 256), nn.ELU(), nn.Linear(256, future_steps*4))

    def transform(self, prediction, current_state):
        x = current_state[:, 0] 
        y = current_state[:, 1]
        theta = current_state[:, 2]
        delta_x = prediction[:, :, 0]
        delta_y = prediction[:, :, 1]
        delta_theta = prediction[:, :, 2]
        new_x = x.unsqueeze(1) + delta_x 
        new_y = y.unsqueeze(1) + delta_y 
        new_theta = theta.unsqueeze(1) + delta_theta
        traj = torch.stack([new_x, new_y, new_theta], dim=-1)

        return traj
       
    def forward(self, agent_map, agent_agent, agent_ego ,current_state):
        feature = torch.cat([agent_map, agent_agent.unsqueeze(1).repeat(1, 3, 1, 1),agent_ego.unsqueeze(1).repeat(1, 3, 1, 1)], dim=-1)
        # print(f'feature:{feature.shape}')
        decoded = self.decode(feature).view(-1, 3, 10, self._future_steps, 4)
        # print(f'decoded:{decoded.shape}')
        trajs = torch.stack([self.transform(decoded[:, i, j], current_state[:, j]) for i in range(3) for j in range(10)], dim=1)
        trajs = torch.reshape(trajs, (-1, 3, 10, self._future_steps, 3))
        return trajs


class AVDecoder(nn.Module):
    def __init__(self, future_steps=50, feature_len=9):
        super(AVDecoder, self).__init__()
        self._future_steps = future_steps
        self.control = nn.Sequential(nn.Dropout(0.1), nn.Linear(512, 256), nn.ELU(), nn.Linear(256, future_steps*2))
        self.cost = nn.Sequential(nn.Linear(1, 128), nn.ReLU(), nn.Linear(128, feature_len), nn.Softmax(dim=-1))
        self.register_buffer('scale', torch.tensor([1, 1, 1, 1, 1, 10, 100]))
        self.register_buffer('constraint', torch.tensor([[10, 10]]))

    def forward(self, agent_map, agent_agent):
        feature = torch.cat([agent_map, agent_agent.unsqueeze(1).repeat(1, 3, 1)], dim=-1)
        actions = self.control(feature).view(-1, 3, self._future_steps, 2)
        dummy = torch.ones(1, 1).to(self.cost[0].weight.device)
        cost_function_weights = torch.cat([self.cost(dummy)[:, :7] * self.scale, self.constraint], dim=-1)

        return actions, cost_function_weights

class Score(nn.Module):
    def __init__(self):
        super(Score, self).__init__()
        self.reduce = nn.Sequential(nn.Dropout(0.1), nn.Linear(512, 256), nn.ELU())
        self.decode = nn.Sequential(nn.Dropout(0.1), nn.Linear(512, 128), nn.ELU(), nn.Linear(128, 1))

    def forward(self, map_feature, agent_agent, agent_map):
        # pooling
        map_feature = map_feature.view(map_feature.shape[0], -1, map_feature.shape[-1])
        map_feature = torch.max(map_feature, dim=1)[0]
        agent_agent = torch.max(agent_agent, dim=1)[0]
        agent_map = torch.max(agent_map, dim=2)[0]

        feature = torch.cat([map_feature, agent_agent], dim=-1)
        feature = self.reduce(feature.detach())
        feature = torch.cat([feature.unsqueeze(1).repeat(1, 3, 1), agent_map.detach()], dim=-1)
        scores = self.decode(feature).squeeze(-1)

        return scores

# Build predictor
class Predictor(nn.Module):
    def __init__(self, future_steps):
        super(Predictor, self).__init__()
        self._future_steps = future_steps

        self.plan_net = EgoPlanEncoder()
        # agent layer
        self.vehicle_net = AgentEncoder()
        self.pedestrian_net = AgentEncoder()
        self.cyclist_net = AgentEncoder()
        

        # map layer
        self.lane_net = LaneEncoder()
        self.crosswalk_net = CrosswalkEncoder()
        
        # attention layers
        self.agent_map = Agent2Map()
        self.agent_agent = Agent2Agent()
        self.ego_ego = Ego2Ego()
        self.agent_ego =  Ego2Agent()

        # decode layers
        self.plan = AVDecoder(self._future_steps)
        self.predict = AgentDecoder(self._future_steps)
        self.predict_1 = AgentDecoder_1(self._future_steps)
        self.score = Score()

    def forward(self, ego, neighbors, map_lanes, map_crosswalks):
        # actors 利用lstm做编码 
        # ego : [batch_size,20,8] neighbors:[batch_size,10,20,9]

        #ego_actor [batch_size,256]
        ego_actor = self.vehicle_net(ego)
        # print(f'ego_actor:{ego_actor.shape}')

        ego_plan = self.plan_net(ego)
        # print(f'ego_plan:{ego_plan.shape}')
        #[batch_size,10,256]
        vehicles = torch.stack([self.vehicle_net(neighbors[:, i]) for i in range(10)], dim=1) 
        pedestrians = torch.stack([self.pedestrian_net(neighbors[:, i]) for i in range(10)], dim=1) 
        cyclists = torch.stack([self.cyclist_net(neighbors[:, i]) for i in range(10)], dim=1)
        #遍历所有neighbors第三维最后一个值，判断neighbor_actors是哪一种类型的neighbor 

        '''
        to_do_list:  试一试只留周围的vehicle

        '''
        neighbor_actors = torch.where(neighbors[:, :, -1, -1].unsqueeze(2)==2, pedestrians, vehicles)
        neighbor_actors = torch.where(neighbors[:, :, -1, -1].unsqueeze(2)==3, cyclists, neighbor_actors)
        #把所有actor包括ego的编码拼在一起
        #actor [batch_size,11,256]
        actors = torch.cat([ego_actor.unsqueeze(1), neighbor_actors], dim=1)
        # print(f'ego_actor: {ego_actor.shape}')
        # print(f'actors : {actors.shape}')
        #用于生成掩码，判断actor有效无效
        actor_mask = torch.eq(torch.cat([ego.unsqueeze(1), neighbors[..., :-1]], dim=1), 0)[:, :, -1, -1]

        # maps
        lane_feature = self.lane_net(map_lanes)
        # print(f'lane_feature:{lane_feature.shape}')
        crosswalk_feature = self.crosswalk_net(map_crosswalks)
        # print(f'crosswalk_feature:{crosswalk_feature.shape}')
        lane_mask = torch.eq(map_lanes, 0)[:, :, :, 0, 0]
        crosswalk_mask = torch.eq(map_crosswalks, 0)[:, :, :, 0, 0]
        map_mask = torch.cat([lane_mask, crosswalk_mask], dim=2)
        map_mask[:, :, 0] = False # prevent nan
        
        # actor to actor
        #actor 的 多头自注意力层

        #  ego_actor           neighbor_actor           actors                    agent_agent
        # [batch_size,256]+[batch_size,10,256]--->[batch_size,11.256] ----> [[batch_size,11.256]]

        agent_agent = self.agent_agent(actors, actor_mask)
        # print(f'agent_agent:{agent_agent.shape}')

        #ego的自注意力编码
        ego_ego = self.ego_ego(ego_plan)
        # print(f'ego_ego :{ego_ego.shape}')

        #ego做k，v agent 做q
        agent_ego= self.agent_ego(agent_agent , ego_ego , ego_ego )
        # print(f'agent_ego :{agent_ego.shape}')

        #agent_agentego = torch.cat([agent_ego,agent_agent],dim=-1)
        agent_agentego = agent_ego + agent_agent
        # print(f'agent_agentego :{agent_agentego.shape}')

        # map to actor
        map_feature = []
        agent_map = []
        #遍历场景下的所有actor
        for i in range(actors.shape[1]):           
            output = self.agent_map(agent_agent[:, i], lane_feature[:, i], crosswalk_feature[:, i], map_mask[:, i])
            map_feature.append(output[0])
            agent_map.append(output[1])
        # for i in range(actors.shape[1]):
        #     output = self.agent_map(agent_agentego[:, i], lane_feature[:, i], crosswalk_feature[:, i], map_mask[:, i])
        #     map_feature.append(output[0])
        #     agent_map.append(output[1])
        map_feature = torch.stack(map_feature, dim=1)
        # print(f'map_feature:{map_feature.shape}')
        agent_map = torch.stack(agent_map, dim=2)
        # print(f'agent_map:{agent_map.shape}')


        #做预测时，去除了代理自交互中的ego，
        predictions = self.predict_1(agent_map[:, :, 1:], agent_agent[:, 1:], agent_ego[:,1:],neighbors[:, :, -1])


        plans=0
        cost_function_weights=0
        scores=0

        # later_combine=torch.cat([torch.cat([agent_agentego.unsqueeze(1),agent_agent.unsqueeze(1)],dim=1),agent_map],dim=1)
        # print(f'later_combine : {later_combine.shape}') 
        # predictions = self.predict(later_combine[:, :, 1:], agent_agent[:, 1:], neighbors[:, :, -1])


        # plan + prediction 
        #plans, cost_function_weights = self.plan(agent_map[:, :, 0], agent_agent[:, 0])
        #predictions = self.predict(agent_map[:, :, 1:], agent_agent[:, 1:], neighbors[:, :, -1])
        #在预测时，去掉了自车的特征
        # predictions = self.predict(later_combine[:, :, 1:], agent_agent[:, 1:], neighbors[:, :, -1])
        #scores = self.score(map_feature, agent_agent, agent_map)
        
        return plans, predictions, scores, cost_function_weights


if __name__ == "__main__":
    # set up model
    model = Predictor(50).to('cuda')
    # train_set = DrivingData('/home/cpns/Chen/paper/DIPP_copy/mini_dataset/ef9d83924b897f33_110.npz')
    #train_set = DrivingData('/home/cpns/Chen/paper/DIPP_copy/mini_dataset/*')
    train_set = DrivingData('/mnt/HDD/dataset/waymo_valid/*')

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, pin_memory=True,num_workers=8)
    #model.train()
    model.eval()
    for batch in tqdm(train_loader):
        ego = batch[0].to('cuda')
        neighbors = batch[1].to('cuda')
        map_lanes = batch[2].to('cuda')
        map_crosswalks = batch[3].to('cuda')
        plans, predictions, scores, cost_function_weights = model(ego, neighbors, map_lanes, map_crosswalks)
        
    writer.close()
    print(model)
    print('Model Params:', sum(p.numel() for p in model.parameters()))



