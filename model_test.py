from utils.train_utils import *
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm



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
        #print(f'inputs:{inputs.shape} ')
        #print(f'self_type:{self_type.shape} lane_attr:{lane_attr.shape}')
        #print(f'lane_embedding:{lane_embedding.shape} self_line :{self_line.shape}')
    
        # process
        output = self.pointnet(lane_embedding)

        return output


def main():
    train_set = DrivingData('/home/cpns/Chen/paper/DIPP_copy/mini_dataset/*')
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, pin_memory=True,num_workers=1)

    vehicle_net=AgentEncoder().to('cuda')
    pedestrian_net=AgentEncoder().to('cuda')
    cyclist_net=AgentEncoder().to('cuda')

    lane_net = LaneEncoder().to('cuda')

    # a=[[[1.1,3],
    #     [1.2,7],
    #     [1.3,11]]]
    # b=[[[[1,3,6],[1,3,6],[1,3,6]],
    #     [[1,3,6],[1,3,6],[1,3,6]],
    #     [[1,3,6],[1,3,6],[1,3,6]],
    #     [[1,3,6],[1,1,6],[1,1,6]]]]
    # b=torch.tensor(b)
    # a=torch.tensor(a)
    # c=torch.cat([a.unsqueeze(1), b[..., :-1]], dim=1)
    # print(f'a:{a.shape}b:{b.shape}c:{c.shape}')
    # ab_mask = torch.eq(c, 0)
    # print(ab_mask,ab_mask[:,:,-1,-1])
    
    for batch in train_loader:
        ego = batch[0].to('cuda')
        neighbors=batch[1].to('cuda')
        map_lanes = batch[2].to('cuda')
        map_crosswalks = batch[3].to('cuda')
        ref_line_info = batch[4].to('cuda')
        ground_truth = batch[5].to('cuda')


        '''
        agent
        '''
        ego_actor = vehicle_net(ego)
        vehicles = torch.stack([vehicle_net(neighbors[:, i]) for i in range(10)], dim=1) 
        pedestrians = torch.stack([pedestrian_net(neighbors[:, i]) for i in range(10)], dim=1) 
        cyclists = torch.stack([cyclist_net(neighbors[:, i]) for i in range(10)], dim=1)
        neighbor_actors = torch.where(neighbors[:, :, -1, -1].unsqueeze(2)==2, pedestrians, vehicles)
        neighbor_actors = torch.where(neighbors[:, :, -1, -1].unsqueeze(2)==3, cyclists, neighbor_actors)
        actors = torch.cat([ego_actor.unsqueeze(1), neighbor_actors], dim=1)
        actor_mask = torch.eq(torch.cat([ego.unsqueeze(1), neighbors[..., :-1]], dim=1), 0)[:, :, -1, -1]       
        #print(f'ego : {ego.shape} neighbors : {neighbors.shape} \nego_actor: {ego_actor.shape}vehicles:{vehicles.shape}')
        #print(f'actors : {actors.shape} actor_mask : {actor_mask.shape}')

        '''
        map
        '''
        lane_feature = lane_net(map_lanes)




if __name__ == "__main__" :
    main()
