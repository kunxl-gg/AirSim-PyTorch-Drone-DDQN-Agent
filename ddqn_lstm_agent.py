import math
import random
from collections import deque
import airsim
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from setuptools import glob
from env import DroneEnv
from torch.utils.tensorboard import SummaryWriter
import time
from prioritized_memory import Memory
from DQRN_net import QNetwork, AttentionModule
import wandb

writer = SummaryWriter()
wandb.init(project="my-project", name="run-name")

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class DDQN_Agent_LSTM:
    def __init__(self,num_frames=7, useDepth=False):
        self.useDepth = useDepth
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 30000
        self.gamma = 0.8
        self.learning_rate = 0.001
        self.batch_size = 256
        self.memory = Memory(10000)
        self.max_episodes = 10000
        self.save_interval = 2
        self.test_interval = 10
        self.network_update_interval = 10
        self.episode = -1
        self.steps_done = 0
        self.max_steps = 34
        self.num_frames = num_frames

        self.policy = QNetwork()
        self.target = QNetwork()
        self.test_network = QNetwork()
        self.target.eval()
        self.test_network.eval()
        self.updateNetworks()

        self.env = DroneEnv(useDepth)
        self.optimizer = optim.Adam(self.policy.parameters(), self.learning_rate)

        if torch.cuda.is_available():
            print('Using device:', device)
            print(torch.cuda.get_device_name(0))
        else:
            print("Using CPU")

        # LOGGING
        cwd = os.getcwd()
        self.save_dir = os.path.join(cwd, "saved models")
        print("Save directory: ", self.save_dir)
        if not os.path.exists(self.save_dir):
            os.mkdir("saved models")
        if not os.path.exists(os.path.join(cwd, "videos")):
            os.mkdir("videos")

        if torch.cuda.is_available():
            self.policy = self.policy.to(device)  # to use GPU
            self.target = self.target.to(device)  # to use GPU
            self.test_network = self.test_network.to(device)  # to use GPU

        # model backup
        files = glob.glob("saved models" + '/*.pt')
        print(len(files))
        if len(files) > 0:
            files.sort(key=os.path.getmtime)
            file = files[-1]
            checkpoint = torch.load(file)
            self.policy.load_state_dict(checkpoint['state_dict'])
            self.episode = checkpoint['episode']
            self.steps_done = checkpoint['steps_done']
            self.updateNetworks()
            print("Saved parameters loaded"
                  "\nModel: ", file,
                  "\nSteps done: ", self.steps_done,
                  "\nEpisode: ", self.episode)


        else:
            if os.path.exists("log.txt"):
                open('log.txt', 'w').close()
            if os.path.exists("last_episode.txt"):
                open('last_episode.txt', 'w').close()
            if os.path.exists("last_episode.txt"):
                open('saved_model_params.txt', 'w').close()

        self.optimizer = optim.Adam(self.policy.parameters(), self.learning_rate)
        obs, _ = self.env.reset_lstm(num_images=self.num_frames)
        tensor = self.transformSeqToTensor(obs)
        writer.add_graph(self.policy, tensor)

    def updateNetworks(self):
        self.target.load_state_dict(self.policy.state_dict())

    def transformToTensor(self, img):
        tensor = torch.FloatTensor(img).to(device)
        tensor = tensor.unsqueeze(0)
        tensor = tensor.unsqueeze(0)
        tensor = tensor.float()
        return tensor
    
    def transformSeqToTensor(self, img_sequence):
        """
        Convert a sequence of images to a PyTorch tensor.
        
        Args:
            img_sequence (list): A list of images.
            
        Returns:
            torch.Tensor: A PyTorch tensor representing the sequence of images.
        """
        # Convert each image to a tensor and stack them along a new dimension
        tensor_sequence = [torch.FloatTensor(img).to(device).unsqueeze(0).unsqueeze(0).float() for img in img_sequence]
        
        # Stack the tensors along the sequence dimension
        tensor_sequence = torch.stack(tensor_sequence, dim=1)
        
        return tensor_sequence
    

    def convert_size(self, size_bytes):
        if size_bytes == 0:
            return "0B"
        size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return "%s %s" % (s, size_name[i])

    def act(self, state_sequence):
        state_sequence=torch.stack(state_sequence)
        state_sequence=state_sequence.permute(1,0,2,3,4)
        self.eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1.0 * self.steps_done / self.eps_decay
        )
        self.steps_done += 1
        if random.random() > self.eps_threshold:
            # print("greedy")
            if torch.cuda.is_available():
                action = np.argmax(self.policy(state_sequence).cpu().data.squeeze().numpy())
            else:
                action = np.argmax(self.policy(state_sequence).data.squeeze().numpy())
        else:
            action = random.randrange(0, 4)
        return int(action)

    # def append_sample(self, state, action, reward, next_state):
    #     next_state = self.transformToTensor(next_state)

    #     current_q = self.policy(state).squeeze().cpu().detach().numpy()[action]
    #     next_q = self.target(next_state).squeeze().cpu().detach().numpy()[action]
    #     expected_q = reward + (self.gamma * next_q)

    #     error = abs(current_q - expected_q),

    #     self.memory.add(error, state, action, reward, next_state)
    
    def memorize_sequence(self, state_sequence,action,reward,next_state_sequence):
        state_sequence = torch.stack(state_sequence)
        next_state_sequence = torch.stack(next_state_sequence)
        state_sequence=state_sequence.permute(1,0,2,3,4)
        next_state_sequence=next_state_sequence.permute(1,0,2,3,4)
        current_q = self.policy(state_sequence).squeeze().cpu().detach().numpy()[action]
        next_q = self.target(next_state_sequence).squeeze().cpu().detach().numpy()[action]
        expected_q = reward + (self.gamma * next_q)

        error = abs(current_q - expected_q)

        self.memory.add(error, state_sequence, action, reward, next_state_sequence)

    def learn(self):
        if self.memory.tree.n_entries < self.batch_size:
            return

        state_sequences, actions, rewards, next_state_sequences, idxs, is_weights = self.memory.sample(self.batch_size)

        # states = torch.stack([torch.stack(seq) for seq in state_sequences])
        # next_states = torch.stack([torch.stack(seq) for seq in next_state_sequences])
        # if isinstance(state_sequences, list):
        #     if isinstance(state_sequences[0], list):
        #         state_sequences = torch.stack([torch.stack(seq) for seq in state_sequences])
        #         next_state_sequences = torch.stack([torch.stack(seq) for seq in next_state_sequences])
        #         if state_sequences
        #     else:
        #         state_sequences = torch.stack(state_sequences)
        #         next_state_sequences = torch.stack(next_state_sequences)
        states = state_sequences.to(device)
        next_states = next_state_sequences.to(device)

        actions = np.asarray(actions)
        rewards = np.asarray(rewards)
        
        if len(states.shape) != 5:
            raise Exception("States shape is not 5, it is: ", states.shape)
        # current_q = self.policy(states).squeeze().cpu().detach().numpy()[actions]
        # next_q = self.target(next_states).squeeze().cpu().detach().numpy()[actions]
        current_q = self.policy(states).gather(1, torch.tensor(actions).unsqueeze(1).to(device)).squeeze(1)
        next_q =self.policy(next_states).gather(1, torch.tensor(actions).unsqueeze(1).to(device)).squeeze(1)
        rewards = torch.FloatTensor(rewards).to(device)
        expected_q = rewards + (self.gamma * next_q)
        # expected_q = rewards + (self.gamma * next_q)
        # error = abs(current_q - expected_q)
        errors = torch.abs(current_q.squeeze() - expected_q.squeeze()).cpu().detach().numpy()

        # update priority
        for i in range(self.batch_size):
            idx = idxs[i]
            self.memory.update(idx, errors[i])
        
        loss = (torch.FloatTensor(is_weights).to(device) * F.smooth_l1_loss(current_q, expected_q, reduction='none')).mean()
        loss = F.smooth_l1_loss(current_q.squeeze(), expected_q.squeeze())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self):
        print("Starting...")

        score_history = []
        reward_history = []

        if self.episode == -1:
            self.episode = 1
        
        for e in range(1, self.max_episodes + 1):
            start = time.time()
            state, _ = self.env.reset()
            steps = 0
            score = 0
            state_sequence = []
            next_state_sequence = []
            while True:
                if isinstance(state, np.ndarray):
                    state = self.transformToTensor(state)
                if len(state_sequence) == 0:
                    for _ in range(self.num_frames):
                        state_sequence.append(state)
                        next_state_sequence.append(state)

                action = self.act(state_sequence)
                next_state, reward, done, _ = self.env.step(action)

                if steps == self.max_steps:
                    done = 1
                
                state_sequence.append(state)
                if isinstance(next_state, np.ndarray):
                    next_state = self.transformToTensor(next_state)
                next_state_sequence.append(next_state)
                if len(state_sequence) > self.num_frames:  # Keep the sequence length fixed at 7 steps
                    state_sequence.pop(0)  # Remove oldest step if sequence exceeds length
                    next_state_sequence.pop(0)

                if len(state_sequence) == self.num_frames:  # If sequence is complete
                    self.memorize_sequence(state_sequence,action,reward,next_state_sequence)  # Upload sequence to memory
                    self.learn()
                #self.memorize(state, action, reward, next_state)
                # self.append_sample(state, action, reward, next_state)
                # self.learn()

                state = next_state
                steps += 1
                score += reward
                if done:
                    print("----------------------------------------------------------------------------------------")
                    if self.memory.tree.n_entries < self.batch_size:
                        print("Training will start after ", self.batch_size - self.memory.tree.n_entries, " steps.")
                        break

                    print(
                        "episode:{0}, reward: {1}, mean reward: {2}, score: {3}, epsilon: {4}, total steps: {5}".format(
                            self.episode, reward, round(score / steps, 2), score, self.eps_threshold, self.steps_done))
                    wandb.log({
                        "reward": reward,
                        "mean_reward": round(score / steps, 2),
                        "score": score,
                        "epsilon": self.eps_threshold,
                        "total_steps": self.steps_done
                    })
                    score_history.append(score)
                    reward_history.append(reward)
                    with open('log.txt', 'a') as file:
                        file.write(
                            "episode:{0}, reward: {1}, mean reward: {2}, score: {3}, epsilon: {4}, total steps: {5}\n".format(
                                self.episode, reward, round(score / steps, 2), score, self.eps_threshold,
                                self.steps_done))

                    if torch.cuda.is_available():
                        print('Total Memory:', self.convert_size(torch.cuda.get_device_properties(0).total_memory))
                        print('Allocated Memory:', self.convert_size(torch.cuda.memory_allocated(0)))
                        print('Cached Memory:', self.convert_size(torch.cuda.memory_reserved(0)))
                        print('Free Memory:', self.convert_size(torch.cuda.get_device_properties(0).total_memory - (
                                torch.cuda.max_memory_allocated() + torch.cuda.max_memory_reserved())))

                        # tensorboard --logdir=runs
                        memory_usage_allocated = np.float64(round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1))
                        memory_usage_cached = np.float64(round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1))

                        writer.add_scalar("memory_usage_allocated", memory_usage_allocated, self.episode)
                        writer.add_scalar("memory_usage_cached", memory_usage_cached, self.episode)

                    writer.add_scalar('epsilon_value', self.eps_threshold, self.episode)
                    writer.add_scalar('score_history', score, self.episode)
                    writer.add_scalar('reward_history', reward, self.episode)
                    writer.add_scalar('Total steps', self.steps_done, self.episode)
                    writer.add_scalars('General Look', {'score_history': score,
                                                        'reward_history': reward}, self.episode)

                    # save checkpoint
                    if self.episode % self.save_interval == 0:
                        checkpoint = {
                            'episode': self.episode,
                            'steps_done': self.steps_done,
                            'state_dict': self.policy.state_dict()
                        }
                        torch.save(checkpoint, self.save_dir + '//EPISODE{}.pt'.format(self.episode))

                    if self.episode % self.network_update_interval == 0:
                        self.updateNetworks()

                    self.episode += 1
                    end = time.time()
                    stopWatch = end - start
                    print("Episode is done, episode time: ", stopWatch)

                    if self.episode % self.test_interval == 0:
                        self.test()

                    break
        writer.close()

    def test(self):
        self.test_network.load_state_dict(self.target.state_dict())

        start = time.time()
        steps = 0
        score = 0
        image_array = []
        state, next_state_image = self.env.reset()
        image_array.append(next_state_image)
        state_sequence = []
        while True:
            if isinstance(state, np.ndarray):
                state = self.transformToTensor(state)
            if len(state_sequence) == 0:
                for _ in range(self.num_frames):
                    state_sequence.append(state)
    
            
            action = int(np.argmax(self.test_network(torch.stack(state_sequence).permute(1,0,2,3,4)).cpu().data.squeeze().numpy()))
            next_state, reward, done, next_state_image = self.env.step(action)
            image_array.append(next_state_image)

            if steps == self.max_steps:
                done = 1

            state_sequence.append(state)
            if len(state_sequence) > self.num_frames:  # Keep the sequence length fixed at 7 steps
                state_sequence.pop(0)  # Remove oldest step if sequence exceeds length


            state = next_state
            steps += 1
            score += reward

            if done:
                print("----------------------------------------------------------------------------------------")
                print("TEST, reward: {}, score: {}, total steps: {}".format(
                    reward, score, self.steps_done))

                with open('tests.txt', 'a') as file:
                    file.write("TEST, reward: {}, score: {}, total steps: {}\n".format(
                        reward, score, self.steps_done))

                writer.add_scalars('Test', {'score': score, 'reward': reward}, self.episode)

                end = time.time()
                stopWatch = end - start
                print("Test is done, test time: ", stopWatch)

                # Convert images to video
                frameSize = (256, 144)
                import cv2
                video = cv2.VideoWriter("videos\\test_video_episode_{}_score_{}.avi".format(self.episode, score), cv2.VideoWriter_fourcc(*'DIVX'), 7, frameSize)

                for img in image_array:
                    video.write(img)

                video.release()

                break