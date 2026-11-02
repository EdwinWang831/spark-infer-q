"""CPU address-liveness model of the pinned QSA pool reader, not a GPU port.

_pool_block reads complete four-row blocks starting at floor(current_pos/4).
Old search reads POOLED, never the complete raw IKC history. Preserve prior
unfinished rows plus the new prepare window; rolled-back future is overwritten.
"""


def simulate(*,capacity=262160,ring_rows=2048,max_window=1024,ratio=4):
    if ring_rows<max_window+ratio-1:raise ValueError('ring cannot retain new window and previous incomplete block')
    full={};ring={};pooled={};position=0;epoch=0;windows=reads=wraps=0
    def prepare(rows,keep):
        nonlocal position,epoch,windows,reads,wraps
        if not 0<=keep<=rows<=max_window or position+rows>capacity:raise ValueError('invalid simulated window')
        old_complete=position//ratio
        saved={b:pooled[b] for b in range(old_complete)} if windows%64==0 else None
        for r in range(rows):
            at=position+r;value=(at*131+epoch*17)%65521
            full[at]=value;ring[at%ring_rows]=value
            wraps+=int(at>=ring_rows)
        for b in range(position//ratio,(position+rows)//ratio):
            expected=[full[b*ratio+k] for k in range(ratio)]
            actual=[ring[(b*ratio+k)%ring_rows] for k in range(ratio)]
            assert actual==expected,'raw pool input overwritten'
            pooled[b]=tuple(actual);reads+=ratio
        if saved is not None:assert all(pooled[b]==v for b,v in saved.items()),'old committed pooled blocks changed'
        position+=keep;epoch+=1;windows+=1
    while position+1024<=261120:prepare(1024,1024)
    assert position==261120
    # Include zero retention, rejected future, odd/even widths and all partial
    # commits. A new epoch changes recomputed raw keys at the same future row.
    for i in range(512):
        rows=i%8+1;keep=(0 if i%7==0 else 1 if i%5==0 else rows//2 if i%3==0 else rows)
        if position+rows>capacity:break
        prepare(rows,keep)
    return dict(windows=windows,pool_input_reads=reads,wrapped_writes=wraps,final_position=position,
        ring_rows=ring_rows,max_window=max_window,ratio=ratio,pool_inputs_exact=True,
        scope='CPU pool-address liveness only; no GPU ring, allocator, partial-state port or model quality validation')


if __name__=='__main__':
    import json
    print(json.dumps(simulate(),indent=2))
