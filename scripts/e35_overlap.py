"""Private MoE fork/join: unchanged shared/routed arithmetic and output slots."""
import torch


class OverlappedMoE:
    def __init__(self, module):
        self.module=module
        self.stream=torch.cuda.Stream()
        self.calls=0

    def __call__(self,x,xs,router_rows,ex,buf,cfg):
        from tensorfold.cuda import moe as moe_mod
        nvx=self.module.nvx
        rows,top_k=x.shape[0],int(cfg.num_experts_per_tok)
        prompt=buf.y.dtype!=torch.float32
        main=torch.cuda.current_stream()
        # Both streams read x only; main will not overwrite it until after join.
        self.stream.wait_stream(main)
        with torch.cuda.stream(self.stream):
            h=ex.shared_act(x,prompt)
            y=self.module._shared(ex.shared.down,h,prompt,not prompt)
        moe_mod.router(x,router_rows,buf.logits[:rows])
        moe_mod.select(buf.logits[:rows],buf,top_k,ex.routed,nvx.PREFILL_TILE)
        nvx.gate_up(x,ex.routed_experts,buf.plan,buf.act.view(-1,ex.width),rows,skip=ex.routed)
        # The shared slot is skipped, so routed down never reads the pending h.
        nvx.down(buf.act.view(-1,ex.width),ex.routed_experts,buf.plan,
                 buf.y.view(-1,ex.dims),rows,skip=ex.routed)
        main.wait_stream(self.stream)
        buf.act[:rows,top_k]=h
        buf.y[:rows,top_k]=y
        # Eager temporaries allocated on the side stream must survive main's copies.
        h.record_stream(main)
        y.record_stream(main)
        self.calls+=1
        return buf
