import sys 
import os
na = os.path.realpath(__file__)

pna = os.path.dirname(na)
ppna = os.path.dirname(pna)
pppna = os.path.dirname(ppna)

sys.path.append(pppna)

order = "python3 ../../pymic/train_infer/train_infer.py train config/train_test.cfg"
os.system(order)