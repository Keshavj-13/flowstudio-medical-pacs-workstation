#!/usr/bin/env python3
"""Full-volume score audit: reference masks never enter the prediction call."""
import ast,hashlib,json,time,sys,gc,csv,os
from pathlib import Path
import numpy as np
import nibabel as nib
import torch
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from genuine_engine import GenuineInferenceManager
AUDIT=Path('/var/tmp/cancer-model-eval')
ORGANS={'msd_spleen_task09':'spleen','msd_prostate_task05':'prostate','msd_lung_task06':'lung','msd_colon_task10':'colon','msd_pancreas_task07':'pancreas','msd_liver_task03':'liver','msd_brain_task01':'brain','brats2021_full':'brain','deeplesion_medsam2':'pan_body'}

def metrics(prediction,reference,affine,spacing):
 p=np.asarray(prediction)>0;g=np.asarray(reference)>0
 if p.shape!=g.shape:raise ValueError('Prediction/reference grid mismatch')
 tp=int(np.count_nonzero(p&g));pn=int(p.sum());gn=int(g.sum());fp=pn-tp;fn=gn-tp
 result={'dice':2*tp/(pn+gn) if pn+gn else 1.,'iou':tp/(pn+gn-tp) if pn+gn-tp else 1.,'precision':tp/pn if pn else 0.,'recall':tp/gn if gn else None,'pred_voxels':pn,'gt_voxels':gn,'tp':tp,'fp':fp,'fn':fn,'hd95_mm':None,'mean_surface_distance_mm':None}
 voxel_ml=abs(float(np.linalg.det(affine[:3,:3])))/1000
 result.update(pred_volume_ml=pn*voxel_ml,gt_volume_ml=gn*voxel_ml,absolute_volume_error_ml=abs(pn-gn)*voxel_ml)
 if pn and gn:
  ps=np.argwhere(p&~binary_erosion(p));gs=np.argwhere(g&~binary_erosion(g))
  pw=nib.affines.apply_affine(affine,ps);gw=nib.affines.apply_affine(affine,gs)
  a=cKDTree(gw).query(pw,workers=2)[0];b=cKDTree(pw).query(gw,workers=2)[0]
  result['hd95_mm']=float(max(np.percentile(a,95),np.percentile(b,95)))
  result['mean_surface_distance_mm']=float((a.sum()+b.sum())/(len(a)+len(b)))
 return result

def read_cases():
 cases=[]
 for key,organ in ORGANS.items():
  path=AUDIT/f'{key}-restoration.json'
  if not path.exists():continue
  for entry in json.loads(path.read_text())['validation_subset']:
   cases.append({**entry,'case_id':entry['id'],'dataset':key,'organ':organ,'modality':'3d_mri' if organ in ['brain','prostate'] else '3d_ct','cohort':'fixed_validation_candidates'})
 node=next(n for n in ast.parse((ROOT/'server.py').read_text()).body if isinstance(n,ast.AnnAssign) and getattr(n.target,'id','')=='MULTI_ORGAN_COHORTS')
 for c in ast.literal_eval(node.value):
  if c['organ']=='breast' and Path(c['image_path']).is_file() and Path(c.get('mask_path','')).is_file():cases.append({**c,'dataset':'breast_demo','cohort':'demo_cases_not_proven_held_out'})
 return cases

def main():
 output=AUDIT/'segmentation-rows.jsonl';existing=[]
 if output.exists():existing=[json.loads(l) for l in output.read_text().splitlines() if l]
 done={(r['model'],r['dataset'],r['case_id']) for r in existing};engine=GenuineInferenceManager.get_instance();cases=read_cases()
 if not cases:raise RuntimeError('No restored evaluation cases')
 print('Cases currently available',len(cases),flush=True)
 for name in ['cice_beatnet','legacy_beatnet']:
  load=engine.load_beatnet if name=='cice_beatnet' else engine.load_legacy_beatnet;load()
  predict=engine.infer_beatnet if name=='cice_beatnet' else engine.infer_legacy_beatnet
  for c in cases:
   if (name,c['dataset'],c['case_id']) in done:continue
   row={k:c[k] for k in ['case_id','dataset','organ','modality','cohort']};row['model']=name
   try:
    image=nib.load(c['image_path']);vol=image.get_fdata(dtype=np.float32)
    # Mirror the deployed loader's first-volume selection for 4D data.
    if vol.ndim==4:vol=vol[...,0]
    reference_image=nib.load(c['mask_path']);reference=reference_image.get_fdata(dtype=np.float32)
    if reference.ndim==4:reference=reference[...,0]
    if reference.shape!=vol.shape or not np.allclose(reference_image.affine,image.affine,atol=1e-3):raise ValueError('Input and reference affines/shapes differ')
    secondary=None
    if len(image.shape)==4 and image.shape[-1]>1:
     secondary=np.asarray(image.dataobj[...,1],dtype=np.float32)
    if c.get('secondary_path') and Path(c['secondary_path']).is_file():
     secondary=nib.load(c['secondary_path']).get_fdata(dtype=np.float32)
     if secondary.ndim==4:secondary=secondary[...,0]
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    kwargs={'organ':c['organ'],'modality':c['modality'],'gt_mask':None,'pixdim':image.header.get_zooms()[:3]}
    if name=='cice_beatnet':kwargs['secondary_volume']=secondary
    prediction=predict(vol,**kwargs)['pred_mask']
    torch.cuda.synchronize();row['pipeline_seconds']=time.perf_counter()-start;row['peak_allocated_gib']=torch.cuda.max_memory_allocated()/2**30
    row.update(metrics(prediction,reference,image.affine,image.header.get_zooms()[:3]));row['status']='ok';row['reference_guides_inference']=False
    row['reference_labels']=[int(x) for x in np.unique(reference)]
    print(name,c['dataset'],c['case_id'],'Dice',round(row['dice'],4),'seconds',round(row['pipeline_seconds'],2),flush=True)
    del vol,reference,secondary,prediction;gc.collect();torch.cuda.empty_cache()
   except Exception as exc:
    row.update(status='error',error=repr(exc));print(name,c['case_id'],'ERROR',repr(exc),flush=True);gc.collect();torch.cuda.empty_cache()
   with output.open('a') as f:f.write(json.dumps(row)+'\n')
   done.add((name,c['dataset'],c['case_id']))
 print('Saved',output,flush=True)
if __name__=='__main__':main()
