import argparse,subprocess,sys
p=argparse.ArgumentParser();p.add_argument('--samples',type=int,default=300);p.add_argument('--model',choices=['cnn','mobilenet','attention'],default='mobilenet');p.add_argument('--epochs',type=int,default=10);a=p.parse_args()
subprocess.run([sys.executable,'generate_dataset.py','--samples',str(a.samples)],check=True)
subprocess.run([sys.executable,'build_candidates.py'],check=True)
subprocess.run([sys.executable,'train.py','--model',a.model,'--epochs',str(a.epochs)],check=True)
subprocess.run([sys.executable,'evaluate.py'],check=True)
print('COMPLETE: dataset/ candidate_data/ models/model.pth results/')
