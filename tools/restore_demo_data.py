#!/usr/bin/env python3
"""Restore selected cases only, with a hard transfer cap on a separate filesystem."""
import os
os.environ['HF_HOME']='/var/tmp/cancer-model-eval/hf-cache'
import ast,io,json,random,hashlib,tarfile,time,threading,shutil
from pathlib import Path
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor,as_completed
import requests,nibabel as nib
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from huggingface_hub import HfApi,hf_hub_download
ROOT=Path('/var/tmp/data_reservoir');AUDIT=Path('/var/tmp/cancer-model-eval')
APP=Path('/workspace/repos/flowstudio-server-snapshot/server.py')
MAX_TRANSFER=8*2**30;MIN_FREE=64*2**30
lock=threading.Lock();reserved=0
TASKS={'msd_spleen_task09':'Task09_Spleen','msd_prostate_task05':'Task05_Prostate','msd_lung_task06':'Task06_Lung','msd_colon_task10':'Task10_Colon','msd_pancreas_task07':'Task07_Pancreas','msd_liver_task03':'Task03_Liver','msd_brain_task01':'Task01_BrainTumour'}

def guard(size=0):
 global reserved
 if not ROOT.resolve().is_relative_to(Path('/var/tmp')):raise RuntimeError('Destination must be under /var/tmp')
 if os.stat('/var/tmp').st_dev==os.stat('/workspace').st_dev:raise RuntimeError('Destination must not share the workspace filesystem')
 if shutil.disk_usage('/var/tmp').free < MIN_FREE+size:raise RuntimeError('Insufficient free space outside workspace')
 with lock:
  if reserved+size>MAX_TRANSFER:raise RuntimeError('8 GiB transfer cap reached; no more downloads')
  reserved+=size

def client():
 s=requests.Session();s.mount('https://',HTTPAdapter(max_retries=Retry(total=2,backoff_factor=.5,status_forcelist=[429,500,502,503,504])));return s

class RemoteTar(io.RawIOBase):
 def __init__(self,url):
  super().__init__();self.url=url;self.session=client();self.pos=0;self.cache=OrderedDict();self.transferred=0
  guard(512)
  with self.session.get(url,headers={'Range':'bytes=0-511'},stream=True,timeout=45) as r:
   r.raise_for_status()
   if r.status_code!=206 or not r.headers.get('Content-Range','').startswith('bytes 0-511/'):raise RuntimeError('Range support required; full archive download forbidden')
   self.size=int(r.headers['Content-Range'].split('/')[-1]);self.etag=r.headers.get('ETag');r.raw.read(512)
 def readable(self):return True
 def seekable(self):return True
 def tell(self):return self.pos
 def seek(self,offset,whence=0):
  self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
  if self.pos<0:raise ValueError('negative seek')
  return self.pos
 def fetch(self,start,length):
  end=min(self.size-1,start+length-1);expected=end-start+1
  guard(expected);headers={'Range':f'bytes={start}-{end}'}
  if self.etag:headers['If-Match']=self.etag
  with self.session.get(self.url,headers=headers,stream=True,timeout=90) as r:
   r.raise_for_status()
   if r.status_code!=206 or not r.headers.get('Content-Range','').startswith(f'bytes {start}-{end}/'):raise RuntimeError('Unexpected range response; refusing full download')
   data=r.raw.read(expected+1)
  if len(data)!=expected:raise IOError('Incomplete range')
  self.transferred+=expected;return data
 def read(self,n=-1):
  if n<0:raise RuntimeError('Unbounded reads forbidden')
  n=min(n,self.size-self.pos)
  if n<=0:return b''
  if n<=16384:
   block=self.pos//16384*16384
   if block not in self.cache:
    self.cache[block]=self.fetch(block,16384)
    if len(self.cache)>8:self.cache.popitem(last=False)
   offset=self.pos-block;data=self.cache[block][offset:offset+n]
   if len(data)<n:data=self.fetch(self.pos,n)
  else:
   if n>4*1024*1024:raise RuntimeError('Read exceeds buffer cap')
   data=self.fetch(self.pos,n)
  self.pos+=len(data);return data
 def close(self):self.session.close();super().close()

def inspect_file(path):
 if str(path).endswith(('.nii','.nii.gz')):
  shape=list(nib.load(str(path)).shape)
  if len(shape) not in (3,4) or min(shape)<=0:raise ValueError('Invalid NIfTI shape')
 else:shape=None
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return {'path':str(path),'bytes':path.stat().st_size,'sha256':h.hexdigest(),'shape':shape}

def cases():
 node=next(n for n in ast.parse(APP.read_text()).body if isinstance(n,ast.AnnAssign) and getattr(n.target,'id','')=='MULTI_ORGAN_COHORTS')
 return ast.literal_eval(node.value)

def msd(key,task,cohort):
 start=time.monotonic();url=f'https://msd-for-monai.s3-us-west-2.amazonaws.com/{task}.tar';index_path=AUDIT/f'{key}-tar-index.json'
 with RemoteTar(url) as remote:
  saved=json.loads(index_path.read_text()) if index_path.exists() else None
  if saved and saved.get('etag')==remote.etag:index=saved['members']
  else:
   index={}
   with tarfile.open(fileobj=remote,mode='r:') as archive:
    for m in archive:
     if m.isfile():index[m.name.lstrip('./')]={'offset':m.offset_data,'size':m.size}
   index_path.write_text(json.dumps({'url':url,'etag':remote.etag,'members':index}))
  name=f'{task}/dataset.json';entry=index[name];remote.seek(entry['offset']);descriptor=json.loads(remote.read(entry['size']))
  tr=list(descriptor['training']);random.Random(44).shuffle(tr);selected=tr[:int(len(tr)*.15)][:5];wanted={name};validation=[]
  for e in selected:
   names={k:f'{task}/{e[k].lstrip("./")}' for k in ['image','label']};wanted.update(names.values());validation.append({'id':Path(e['image']).name,'image_path':str(ROOT/key/names['image']),'mask_path':str(ROOT/key/names['label'])})
  for c in cohort:
   for field in ['image_path','mask_path','secondary_path']:
    p=c.get(field)
    if p and str(p).startswith(str(ROOT/key)+'/'):wanted.add(str(Path(p).relative_to(ROOT/key)))
  planned=sum(index[n]['size'] for n in wanted)
  if planned>2*2**30:raise RuntimeError('Selected dataset files exceed 2 GiB per-task cap')
  files=[]
  for name in sorted(wanted):
   info=index[name];target=ROOT/key/name;target.parent.mkdir(parents=True,exist_ok=True)
   if not target.is_file() or target.stat().st_size!=info['size']:
    guard();remote.seek(info['offset']);remaining=info['size'];part=target.with_name(target.name+'.partial')
    try:
     with part.open('wb') as out:
      while remaining:
       chunk=remote.read(min(4*1024*1024,remaining))
       if not chunk:raise IOError('Unexpected archive EOF')
       out.write(chunk);remaining-=len(chunk)
     part.replace(target)
    except BaseException:part.unlink(missing_ok=True);raise
   files.append(inspect_file(target))
  result={'dataset':key,'url':url,'etag':remote.etag,'files':files,'validation_subset':validation,'split_recipe':'seed 44 shuffle of full dataset.json training list; first floor(15%) validation; first 5 cases','transferred_bytes':remote.transferred,'seconds':round(time.monotonic()-start,2)}
  (AUDIT/f'{key}-restoration.json').write_text(json.dumps(result,indent=2));print(json.dumps({'dataset':key,'files':len(files),'evaluation_cases':len(selected),'transfer_mb':round(remote.transferred/1e6,1),'seconds':result['seconds']}),flush=True);return result

def hf(key,repo,cohort,revision,listing):
 sizes={f['path']:f['size'] for f in listing};wanted=set();validation=[]
 if key=='brats2021_full':
  all_ids=sorted({p.split('/')[0] for p in sizes if p.startswith('BraTS2021_') and p.endswith('_seg.nii.gz')});random.Random(45).shuffle(all_ids);selected=all_ids[:int(len(all_ids)*.15)][:5]
  demos={Path(c['image_path']).parent.name for c in cohort if '/brats2021_full/' in c['image_path']}
  for subject in set(selected)|demos:wanted.update(f'{subject}/{subject}_{seq}.nii.gz' for seq in ['t1','t1ce','t2','flair','seg'])
  validation=[{'id':s,'image_path':str(ROOT/key/s/f'{s}_t1ce.nii.gz'),'secondary_path':str(ROOT/key/s/f'{s}_flair.nii.gz'),'mask_path':str(ROOT/key/s/f'{s}_seg.nii.gz')} for s in selected]
 else:
  all_ids=sorted(p[7:] for p in sizes if p.startswith('labels/') and sizes[p] and sizes.get('images/'+p[7:].replace('.nii.gz','_0000.nii.gz')));random.Random(46).shuffle(all_ids);selected=all_ids[:int(len(all_ids)*.15)][:5]
  for c in cohort:
   if '/deeplesion_medsam2/' in c['image_path']:wanted.update(str(Path(c[k]).relative_to(ROOT/key)) for k in ['image_path','mask_path'])
  for name in selected:
   image='images/'+name.replace('.nii.gz','_0000.nii.gz');label='labels/'+name;wanted.update([image,label]);validation.append({'id':name,'image_path':str(ROOT/key/image),'mask_path':str(ROOT/key/label)})
 if sum(sizes[n] for n in wanted)>2*2**30:raise RuntimeError('HF selected files exceed 2 GiB cap')
 files=[]
 for name in sorted(wanted):
  target=ROOT/key/name
  if not target.is_file() or target.stat().st_size!=sizes[name]:
   guard(2*sizes[name]);target=Path(hf_hub_download(repo_id=repo,repo_type='dataset',revision=revision,filename=name,local_dir=str(ROOT/key)))
  files.append(inspect_file(target))
 result={'dataset':key,'repo':repo,'revision':revision,'files':files,'validation_subset':validation,'split_recipe':f'seed {45 if key=="brats2021_full" else 46} shuffle of complete sorted subjects/pairs; first 15%; first 5 cases'}
 (AUDIT/f'{key}-restoration.json').write_text(json.dumps(result,indent=2));print(json.dumps({'dataset':key,'files':len(files),'evaluation_cases':len(validation)}),flush=True);return result

def main():
 guard();ROOT.mkdir(parents=True,exist_ok=True);AUDIT.mkdir(parents=True,exist_ok=True);cohort=cases();api=HfApi();hf_jobs=[]
 print(json.dumps({'destination':str(ROOT),'free_gib':round(shutil.disk_usage('/var/tmp').free/2**30,1),'hard_transfer_cap_gib':8,'minimum_free_gib':64,'workers':2}),flush=True)
 for key,repo in [('brats2021_full','rocky93/BraTS_segmentation'),('deeplesion_medsam2','wanglab/CT_DeepLesion-MedSAM2')]:
  info=api.dataset_info(repo,files_metadata=True);hf_jobs.append((key,repo,info.sha,[{'path':f.rfilename,'size':f.size} for f in info.siblings]))
 results=[];errors=[]
 with ThreadPoolExecutor(max_workers=2) as pool:
  jobs={pool.submit(msd,k,t,cohort):k for k,t in TASKS.items()}
  jobs.update({pool.submit(hf,k,r,cohort,rev,listing):k for k,r,rev,listing in hf_jobs})
  for future in as_completed(jobs):
   try:results.append(future.result())
   except Exception as e:errors.append({'dataset':jobs[future],'error':repr(e)});print(json.dumps(errors[-1]),flush=True)
 summary={'restored':results,'errors':errors,'reserved_bytes':reserved,'demo_availability':[{'case_id':c['case_id'],'image_exists':Path(c['image_path']).is_file(),'mask_exists':Path(c['mask_path']).is_file() if c.get('mask_path') else None} for c in cohort]}
 (AUDIT/'restoration-summary.json').write_text(json.dumps(summary,indent=2))
 if errors:raise SystemExit(1)
if __name__=='__main__':main()
