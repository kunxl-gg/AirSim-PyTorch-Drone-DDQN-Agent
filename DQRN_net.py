import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary

class AttentionModule(nn.Module):
    def __init__(self, r, m, a):
        super(AttentionModule, self).__init__()
        self.w = nn.Parameter(torch.randn(a, 1))
        self.Wa = nn.Linear(r, a, bias=False)
        self.Ua = nn.Linear(m, a, bias=False)
        self.ba = nn.Parameter(torch.zeros(1,a))

    def forward(self, hidden_state, feature_vectors):
        L = feature_vectors.size(1) # time steps, eg. 3
        r = hidden_state.size(-1) # hidden state size, eg. 512
        m = feature_vectors.size(-1) # feature vector size, eg. 3136
        a = self.w.size(0) # attention size, eg. 128

        attention_weights = []
        for i in range(L):
            v_t = feature_vectors[:,i,:]  # (batch_size, m)
            #print(v_t.shape)
            e_t = torch.matmul(torch.tanh(self.Wa(hidden_state) + self.Ua(v_t) + self.ba),self.w)  # (batch_size, 1)
            attention_weights.append(e_t)

        attention_weights = torch.stack(attention_weights, dim=1)  # (batch_size, L, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)  # (batch_size, L, 1)

        context_vector = torch.zeros(hidden_state.size(0), m).to(feature_vectors.device)  # (batch_size, m)
        for i in range(L):
            context_vector += attention_weights[:, i] * feature_vectors[:,i,:]

        return context_vector

class QNetwork(nn.Module):
    def __init__(self, input_shape=(84,84,1), num_actions=4, mode="duel", recurrent=True, a_t=True, bidir=False, device="cuda:0"):
        super(QNetwork, self).__init__()
        self.mode = mode
        self.recurrent = recurrent
        self.a_t = a_t
        self.bidir = bidir
        self.num_actions = num_actions

        if mode == "linear":
            self.flatten = nn.Flatten()
            self.output = nn.Linear(self._get_flattened_size(input_shape), num_actions)
        else:
            if not recurrent:
                self.conv1 = nn.Conv2d(input_shape[2], 32, kernel_size=8, stride=4) #84x84x3 -> 20x20x32
                self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2) #20x20x32 -> 9x9x64
                self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1) #9x9x64 -> 7x7x64
                self.flatten = nn.Flatten() #7x7x64 -> 3136
            else:
                # print('>>>> Defining Recurrent Modules...')
                self.conv1 = nn.Conv2d(input_shape[2], 32, kernel_size=8, stride=4)
                self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
                self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
                self.flatten = nn.Flatten()


                if not a_t:
                    self.lstm = nn.LSTM(3136, 512, batch_first=True)
                else:
                    if bidir:
                        self.lstm = nn.LSTM(3136, 512, batch_first=True, bidirectional=True)
                    else:
                        self.lstm = nn.LSTM(3136, 512, batch_first=True)

                    # Attention mechanism
                    self.attention = AttentionModule(512, 3136, 128).to(device)  # Assuming r=512, m=512, a=128

            if mode == "dqn":
                self.fc = nn.Linear(512, 512)
                self.output = nn.Linear(512, num_actions)
            elif mode == "duel":
                self.value_fc = nn.Linear(3136, 512)
                self.action_fc = nn.Linear(3136, 512)
                self.value = nn.Linear(512, 1)
                self.action = nn.Linear(512, num_actions)

    def forward(self, x):
        #print(x.shape)
        x = F.interpolate(x, size=(x.size(2),84, 84))
        if self.mode == "linear":
            x = self.flatten(x)
            return self.output(x)
        else:
            if not self.recurrent:
                x = F.relu(self.conv1(x))
                x = F.relu(self.conv2(x))
                x = F.relu(self.conv3(x))
                x = self.flatten(x)
            else:
                # print('>>>> Defining Recurrent Modules...')
                batch_size, timesteps, C, H, W = x.size()

                x = x.view(batch_size * timesteps, C, H, W)
                x = F.relu(self.conv1(x))
                x = F.relu(self.conv2(x))
                x = F.relu(self.conv3(x))
                x = self.flatten(x)

                if not self.a_t:
                    r_out, _ = self.lstm(x)
                else:
                    x = x.view(batch_size, timesteps, -1) # vt-(L-1) to vt
                    #print(x.shape)
                    r_out, _ = self.lstm(x)
                    #print(r_out[:,-2,:].shape) # ht-1

                    # Apply attention mechanism
                    context_vec = self.attention(r_out[:,-2,:], x)  # Pass x as a list with a single element

            if self.mode == "dqn":
                x = F.relu(self.fc(context_vec))
                return self.output(x)
            elif self.mode == "duel":
                value = F.relu(self.value_fc(context_vec))
                action = F.relu(self.action_fc(context_vec))
                value = self.value(value)
                action = self.action(action)
                action_mean = torch.mean(action, dim=1, keepdim=True)
                return action + value - action_mean


if __name__ == "__main__":
    input_shape = (84, 84, 3)
    num_actions = 4
    q_network = QNetwork(input_shape, num_actions, "duel", recurrent=True, a_t=True)
    x = torch.randn(5, 3, 3, 84, 84)
    #x = x.permute(0, 3, 1, 2)
    print(x.shape)
    # summary = torchsummary.summary(q_network, x)
    # print(summary)
    q_values = q_network(x)
    print(q_values)
    print(q_values.shape)
