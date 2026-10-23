"""Private greedy target variant: original backbone Graph, certified screened head after it."""
import copy
from e29_graph import FeatureGraphTarget


class ScreenedGraphTarget(FeatureGraphTarget):
    def __init__(self,weights,screen,capacity=2048):
        super().__init__(weights,capacity);self.screen=screen

    def compute_features(self,state,rows):
        torch,w,f=self.torch,self.w,self.forward;buf=self.buffers[rows];segs=[(state,0,rows)]
        f._embed(w,buf.ids[:rows],w.cfg.streams,buf.h[:rows]);pending=None;features=[]
        for layer in w.layers:
            pending=f.layer_forward(layer,w,segs,buf,rows,pending,context=self.capacity)
            if layer.index in (5,15,25,35,45):
                shadow=buf.h[:rows].clone();proxy=copy.copy(buf);proxy.pss=torch.empty_like(buf.pss)
                f._writeback(shadow,proxy,rows,w.cfg,pending);features.append(shadow.unflatten(-1,(4,2560)).mean(-2))
        # finish(logits=False) still performs original final writeback/mixer; commit does not consume logits.
        f.finish(w,w.mixer,buf,rows,pending,logits=False)
        return buf.mixed[:rows].clone(),torch.cat(features,-1)

    def run_predictions(self,state,ids,diagnostic=False):
        hidden,features,buf=self.run(state,ids)
        pred,info=self.screen.select(hidden,diagnostic=diagnostic)
        return pred,features,buf,hidden,info
