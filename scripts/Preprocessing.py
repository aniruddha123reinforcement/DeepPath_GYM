import numpy as np
import random
from utils import *

class Preprocessing(object):
	"""knowledge graph environment definition"""
	def __init__(self, dataPath):
		f1 = open(dataPath + 'entity2id.txt')
		f2 = open(dataPath + 'relation2id.txt')
		self.entity2id = f1.readlines()
		self.relation2id = f2.readlines()
		f1.close()
		f2.close()
		self.entity2id_ = {}
		self.relation2id_ = {}
		self.relations = []
		for line in self.entity2id:
			self.entity2id_[line.split()[0]] =int(line.split()[1])
		for line in self.relation2id:
			self.relation2id_[line.split()[0]] = int(line.split()[1])
			self.relations.append(line.split()[0])
		self.entity2vec = np.loadtxt(dataPath + 'entity2vec.bern')
		self.relation2vec = np.loadtxt(dataPath + 'relation2vec.bern')


		self.path = []
		self.path_relations = []

