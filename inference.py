import time
import numpy as np
import cv2
from DQRN_net import QNetwork
from env import DroneEnv
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def transformToTensor(img):
    tensor = torch.FloatTensor(img).to(device)
    tensor = tensor.unsqueeze(0)
    tensor = tensor.unsqueeze(0)
    tensor = tensor.float()
    return tensor

def test_DQN(checkpoint_dict,max_steps):
    model_dict=checkpoint_dict["state_dict"]
    episode=checkpoint_dict["episode"]
    steps_done=checkpoint_dict["steps_done"]
    test_network=QNetwork()
    test_network.load_state_dict(model_dict)

    start = time.time()
    steps = 0
    score = 0
    image_array = []
    env = DroneEnv()
    state, next_state_image = env.reset()
    image_array.append(next_state_image)

    while True:
        state = transformToTensor(state)

        action = int(np.argmax(test_network(state).cpu().data.squeeze().numpy()))
        next_state, reward, done, next_state_image = env.step(action)
        image_array.append(next_state_image)

        if steps == max_steps:
            done = 1

        state = next_state
        steps += 1
        score += reward

        if done:
            print("----------------------------------------------------------------------------------------")
            print("TEST, reward: {}, score: {}, total steps: {}".format(
                reward, score, steps_done))

            with open('tests.txt', 'a') as file:
                file.write("TEST, reward: {}, score: {}, total steps: {}\n".format(
                    reward, score, steps_done))

            # writer.add_scalars('Test', {'score': score, 'reward': reward}, episode)

            end = time.time()
            stopWatch = end - start
            print("Test is done, test time: ", stopWatch)

            # Convert images to video
            frameSize = (256, 144)
            video = cv2.VideoWriter("videos\\test_video_episode_{}_score_{}.avi".format(episode, score), cv2.VideoWriter_fourcc(*'DIVX'), 7, frameSize)

            for img in image_array:
                video.write(img)

            video.release()

            break

def test_DQRN(checkpoint_dict,max_steps,num_frames=7):
        model_dict=checkpoint_dict["state_dict"]
        episode=checkpoint_dict["episode"]
        steps_done=checkpoint_dict["steps_done"]
        test_network=QNetwork()
        test_network.load_state_dict(model_dict)
        test_network.to(device)

        env = DroneEnv()
        start = time.time()
        steps = 0
        score = 0
        image_array = []
        state, next_state_image = env.reset()
        image_array.append(next_state_image)
        state_sequence = []
        while True:
            if isinstance(state, np.ndarray):
                state = transformToTensor(state)
            if len(state_sequence) == 0:
                for _ in range(num_frames):
                    state_sequence.append(state)
    

            action = int(np.argmax(test_network(torch.stack(state_sequence).permute(1,0,2,3,4)).cpu().data.squeeze().numpy()))
            next_state, reward, done, next_state_image = env.step(action)
            image_array.append(next_state_image)

            if steps == max_steps:
                done = 1

            state_sequence.append(state)
            if len(state_sequence) > 7:  # Keep the sequence length fixed at 7 steps
                state_sequence.pop(0)  # Remove oldest step if sequence exceeds length


            state = next_state
            steps += 1
            score += reward

            if done:
                print("----------------------------------------------------------------------------------------")
                print("TEST, reward: {}, score: {}, total steps: {}".format(
                    reward, score, steps_done))

                with open('tests.txt', 'a') as file:
                    file.write("TEST, reward: {}, score: {}, total steps: {}\n".format(
                        reward, score, steps_done))


                end = time.time()
                stopWatch = end - start
                print("Test is done, test time: ", stopWatch)

                # Convert images to video
                frameSize = (256, 144)
                import cv2
                video = cv2.VideoWriter("videos\\test_video_episode_{}_score_{}.avi".format(episode, score), cv2.VideoWriter_fourcc(*'DIVX'), 7, frameSize)

                for img in image_array:
                    video.write(img)

                video.release()

                break

if __name__ == "__main__":
    checkpoint = torch.load("EPISODE76.pt")
    test_DQRN(checkpoint,34)