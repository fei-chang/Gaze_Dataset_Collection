import os

names=['train','val']
for name in names:
  txt_list=os.listdir("labels/"+name)
  for txt in txt_list:
    full_path=os.path.join("labels/"+name,txt)
    head_notes=[]
    with open(full_path,'r') as f:
      lines=f.readlines()
    for line in lines:
      if line[0]=='2':
        line='0'+line[1:]
        head_notes.append(line)
    with open(full_path,'w') as f:
      for head in head_notes:
        f.write(head)
  
      
