import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from .data import COLORS, decode_mask, pairs
from .metrics import confusion, summary, evaluate
from .io import save_png, save_conf, sha256
from .__main__ import valid_output

class AdapterTest(unittest.TestCase):
    def test_palette_and_background(self):
        ids=np.arange(15,dtype=np.uint8).reshape(3,5)
        self.assertTrue(np.array_equal(decode_mask(Image.fromarray(COLORS[ids])),ids))
        self.assertEqual(int(decode_mask(Image.fromarray(np.zeros((1,1,3),dtype=np.uint8)))[0,0]),0)
        with self.assertRaises(ValueError):
            decode_mask(Image.fromarray(np.array([[[1,2,3]]],dtype=np.uint8)))
    def test_pair_with_spaces_and_dots(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d); (r/'images').mkdir(); (r/'labels').mkdir()
            Image.new('RGB',(2,1)).save(r/'images/a.b c.jpg')
            Image.new('RGB',(2,1)).save(r/'labels/a.b c.jpg___fuse.png')
            self.assertEqual(len(pairs(r/'images',r/'labels')),1)
    def test_metrics_and_resume_output(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d); (r/'images').mkdir(); (r/'labels').mkdir()
            image=r/'images/a.jpg'; label=r/'labels/a.jpg___fuse.png'
            Image.new('RGB',(2,2)).save(image)
            gt=np.array([[0,1],[1,2]],dtype=np.uint8)
            Image.fromarray(COLORS[gt]).save(label)
            pred=np.array([[0,1],[2,2]],dtype=np.uint8)
            conf=np.array([[.9,.8],[.7,.6]],dtype=np.float32)
            save_png(r/'masks/a.jpg.png',pred)
            save_conf(r/'confidence/a.jpg.npz',conf)
            rec=dict(image=str(image),label=str(label),mask='masks/a.jpg.png',confidence='confidence/a.jpg.npz')
            self.assertTrue(valid_output(r,rec))
            digests={str(image):{'mask':sha256(r/rec['mask']),'confidence':sha256(r/rec['confidence'])}}
            self.assertTrue(valid_output(r,rec,digests))
            self.assertFalse(valid_output(r,rec,{str(image):{'mask':'bad','confidence':'bad'}}))
            result=evaluate([rec],r)
            self.assertEqual(result['evaluated_images'],1)
            self.assertAlmostEqual(result['pixel_accuracy'],.75)
            self.assertAlmostEqual(result['miou_15'],(1+.5+.5)/3)
            self.assertAlmostEqual(result['thresholds'][4]['coverage'],.25)
            self.assertEqual(confusion(gt,pred).sum(),4)
            self.assertIsNone(summary(np.zeros((15,15),dtype=int))[0]['miou_15'])
            (r/'confidence/a.jpg.npz').write_bytes(b'invalid')
            self.assertFalse(valid_output(r,rec))

if __name__=='__main__': unittest.main()
