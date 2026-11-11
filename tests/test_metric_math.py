from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from benchmark_segmentation import metrics

def test_known_overlap_and_physical_units():
 p=np.zeros((4,4,4),dtype=bool);g=p.copy();p[0,0,:2]=True;g[0,0,1:3]=True
 r=metrics(p,g,np.diag([2.,3.,4.,1.]),(2,3,4))
 assert r['dice']==.5 and abs(r['iou']-1/3)<1e-12
 assert r['precision']==.5 and r['recall']==.5
 assert abs(r['pred_volume_ml']-.048)<1e-10
 assert abs(r['hd95_mm']-3.8)<1e-12

def test_empty_prediction_is_not_a_perfect_score():
 p=np.zeros((4,4,4),dtype=bool);g=p.copy();g[0,0,0]=True
 r=metrics(p,g,np.eye(4),(1,1,1))
 assert r['dice']==0 and r['iou']==0 and r['recall']==0 and r['hd95_mm'] is None


def test_live_metrics_use_spacing_and_score_annotated_empty_masks():
 from genuine_engine import GenuineInferenceManager
 engine=GenuineInferenceManager.__new__(GenuineInferenceManager)
 empty=np.zeros((4,4,4),dtype="uint8")
 measured=engine._calculate_metrics(empty,empty,pixdim=(2.,3.,4.))
 assert measured["has_gt"] is True
 assert measured["dice_3d"]==1.
 prediction=empty.copy();prediction[0,0,0]=1
 measured=engine._calculate_metrics(prediction,empty,pixdim=(2.,3.,4.))
 assert measured["pred_volume_ml"]==0.02
 assert measured["dice_3d"]==0.
 assert measured["validation_expected_dice"] is None
