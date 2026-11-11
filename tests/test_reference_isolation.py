from pathlib import Path
import sys
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from genuine_engine import GenuineInferenceManager,CONCEPT_NAMES

@pytest.mark.parametrize('organ,modality',[('breast','3d_mri'),('spleen','3d_ct')])
def test_reference_mask_cannot_change_inference_crop(organ,modality):
    inputs=[]
    class Model:
        def __call__(self,x,**kwargs):
            inputs.append(x.detach().clone())
            return {'seg_logits':torch.full((1,1,*x.shape[2:]),-10.),'concept_probs':torch.zeros((1,len(CONCEPT_NAMES)))}
    engine=GenuineInferenceManager.__new__(GenuineInferenceManager)
    engine.device=torch.device('cpu')
    engine.load_beatnet=lambda:Model()
    volume=np.random.default_rng(7).normal(100,10,size=(160,160,48)).astype('float32')
    reference=np.zeros(volume.shape,dtype='uint8');reference[5:15,5:15,2:6]=1
    first=engine.infer_beatnet(volume,organ=organ,modality=modality,gt_mask=None)
    second=engine.infer_beatnet(volume,organ=organ,modality=modality,gt_mask=reference)
    assert torch.equal(inputs[0],inputs[1])
    np.testing.assert_array_equal(first['pred_mask'],second['pred_mask'])
