from matcher import *
from Snippext_public.snippext.model import MultiTaskNet
from logger import Logger
import time

sentence_pairs = []
lines = []
for line in open("data/company/test2.txt", 'r', encoding='utf-8'):
    line = line.replace("\n", "")
    sentence_pairs.append((line.split("\t")[0], line.split("\t")[1]))
    lines.append(line)

t_0 = time.time()

sys.path.insert(0, "Snippext_public")

config = {'name': 'ditto', 'task_type': 'classification', 'vocab': ['0', '1'],
          'trainset': 'data/company/train.txt',
          'validset': 'data/company/val.txt',
          'testset': 'data/company/test.txt'}

device = 'cpu'
lm = 'roberta'
max_len = 256


model =  MultiTaskNet([config], device, True, lm=lm)

# saved_state = torch.load('ditto_lm=roberta_da=swap_dk=None_su=False_size=None_id=0_dev.pt', map_location=lambda storage, loc: storage)
saved_state = torch.load('model-match-company_name.pt', map_location=lambda storage, loc: storage)
model.load_state_dict(saved_state, strict=False)

model = model.to(device)

model.eval()




predictions, logits = classify(sentence_pairs, config, model, lm=lm, max_len=max_len)

for i in range(len(lines)):
	print(lines[i] + "\t" + predictions[i])