# -*- coding: utf-8 -*-
"""
Created on Fri May 20 13:00:33 2022

@author: jfell
46XbZHut3gF8NPN
"""
import numpy as np
import time
import gym
import tensorflow as tf
from tensorflow import keras

N_INPUTS = 4 # number of variables in the observation space
HIDDEN_LAYER = 5 # neurons in hidden layer
N_OUTPUTS = 1 # number of outputs is 1 since action space is binary
 
class CartPoleRL:
    def __init__(self, learning_rate = 1e-6):
        # setup cart pole environment
        self.env = gym.make('CartPole-v1')
        self.env.reset(seed=42)
        self.model = keras.models.Sequential([
            keras.layers.Dense(HIDDEN_LAYER, activation='elu', input_shape=[N_INPUTS]),
            keras.layers.Dense(N_OUTPUTS, activation='sigmoid'),
            ])
        self.optimizer = keras.optimizers.Adam(learning_rate = learning_rate)
        #self.loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
        self.loss_fn = keras.losses.binary_crossentropy
                
    def play_one_step(self, env, obs, model, loss_fn):
        
        with tf.GradientTape() as tape:
            left_proba = model(obs[np.newaxis])
            action = (tf.random.uniform([1,1]) > left_proba)
            y_target = tf.constant([[1.]]) - tf.cast(action,tf.float32)
            loss = tf.reduce_mean(loss_fn(y_target, left_proba))
            
        grads = tape.gradient(loss, model.trainable_variables)
        obs, reward, done, info = env.step(int(action[0,0].numpy()))
        env.render()
        time.sleep(0.01)
        return obs, reward, done, grads


    def play_multiple_episodes(self, env, n_episodes, n_max_steps,
                            model, loss_fn):
        all_rewards = []
        all_grads = []
        for episode in range(n_episodes):
            current_rewards = []
            current_grads = []
            obs = env.reset()
            #print(f'start episode {episode}')
            for step in range(n_max_steps):
                obs, reward, done, grads = self.play_one_step(
                    env, obs, model, loss_fn)
                current_rewards.append(reward)
                current_grads.append(grads)
                if done:
                    break
            all_rewards.append(current_rewards)
            all_grads.append(current_grads)
        return all_rewards, all_grads
    
    def discount_rewards(self, rewards, discount_factor):
        discounted = np.array(rewards)
        for step in range(len(rewards) - 2, -1, -1):
            discounted[step] += discounted[step + 1] * discount_factor
            return discounted
            
    def discount_and_normalize_rewards(self, all_rewards, 
                                       discount_factor):
        all_discounted_rewards = [self.discount_rewards(rewards, discount_factor)
                                  for rewards in all_rewards]
        flat_rewards = np.concatenate(all_discounted_rewards)
        reward_mean = flat_rewards.mean()
        reward_std = flat_rewards.std()
        return [(discounted_rewards - reward_mean)/reward_std
                for discounted_rewards in all_discounted_rewards]
    def animate(self):
        self.axes.clear()
        self.axes.plot(self.interations,self.mean_epidsode_rewards,color='blue',alpha=0.5)
        
        
    def play(self,n_iterations = 150, n_episodes_per_update = 10,
             n_max_steps = 200, discount_factor = 0.9):
  
        for iteration in range(n_iterations):
            all_rewards, all_grads = self.play_multiple_episodes(
                self.env, n_episodes_per_update, n_max_steps,
                self.model, self.loss_fn)
            
            total_rewards = sum(map(sum, all_rewards))                     # Not shown in the book
            print("\rIteration: {}, mean rewards: {:.1f}".format(          # Not shown
                iteration, total_rewards / n_episodes_per_update),end='')         # Not shown
 
            all_final_rewards = self.discount_and_normalize_rewards(all_rewards,
                                                                    discount_factor)            
            all_mean_grads = []
            for var_index in range(len(self.model.trainable_variables)):
                mean_grads = tf.reduce_mean(
                    [final_reward * all_grads[episode_index][step][var_index]
                    for episode_index, final_rewards in enumerate(all_final_rewards)
                        for step, final_reward in enumerate(final_rewards)], axis = 0)
                all_mean_grads.append(mean_grads)
                self.optimizer.apply_gradients(zip(all_mean_grads,
                                                    self.model.trainable_variables))

                
            
            
# Run the model
if __name__ == '__main__':
    cp = CartPoleRL()
    cp.play()
    cp.env.close()
            
            
            
        
        