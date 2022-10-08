
import gym 
import numpy as np
from gym import spaces
from gym import utils 
import random
from utils import *
from Preprocessing import preprocessing

class Knowledgegraph_gym(gym.Env):
	"""knowledge graph environment definition"""
	def __init__(self, task=None):

		# Knowledge Graph for path finding
		f = open(dataPath + 'kb_env_rl.txt')
		kb_all = f.readlines()
		f.close()

		self.kb = []
		if task != None:
			relation = task.split()[2]
			for line in kb_all:
				rel = line.split()[2]
				if rel != relation and rel != relation + '_inv':
					self.kb.append(line)

		self.die = 0 # record how many times does the agent choose an invalid path
	        self.observation_space = spaces.Box(low=-inf, high=inf, shape=(,200), dtype=np.float64)
		self.action_space = spaces.Discrete(400)
		self.state = None
        
		
	

	def step(self, state, action):
		'''
		This function process the interact from the agent
		state: is [current_position, target_position] 
		action: an integer
		return: (reward, [new_postion, target_position], done)
		'''
		terminated = 0 # Whether the episode has finished
		curr_pos, target_pos,self.die = self.state
		chosed_relation = self.relations[action]
		choices = []
		for line in self.kb:
			triple = line.rsplit()
			e1_idx = preprocessing.entity2id_[triple[0]]
			
			if curr_pos == e1_idx and triple[2] == chosed_relation and triple[1] in preprocessing.entity2id_:
				choices.append(triple)
		if len(choices) == 0:
			reward = -1
			self.die += 1
			self.state = (curr_pos, target_pos, self.die) # stay in the initial state
			return np.array(self.state,dtype=np.float32),reward, terminated, False, {}
		else: # find a valid step
			path = random.choice(choices)
			self.path.append(path[2] + ' -> ' + path[1])
			self.path_relations.append(path[2])
			# print('Find a valid step', path)
			# print('Action index', action)
			self.die = 0
			new_pos = preprocessing.entity2id_[path[1]]
			reward = 0
			self.state = (new_pos, target_pos, self.die)

			if new_pos == target_pos:
				print('Find a path:', self.path)
				terminated = 1
				reward = 1
				self.state = None
			return np.array(self.state,dtype=np.float32),reward, terminated, False, {}


	def get_valid_actions(self, entityID):
		actions = set()
		for line in self.kb:
			triple = line.split()
			e1_idx = self.entity2id_[triple[0]]
			if e1_idx == entityID:
				actions.add(self.relation2id_[triple[2]])
		return np.array(list(actions))

	def path_embedding(self, path):
		embeddings = [self.relation2vec[self.relation2id_[relation],:] for relation in path]
		embeddings = np.reshape(embeddings, (-1,embedding_dim))
		path_encoding = np.sum(embeddings, axis=0)
		return np.reshape(path_encoding,(-1, embedding_dim))


