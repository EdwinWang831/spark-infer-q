"""Admission arithmetic from measured resident bytes, not a concurrency claim."""
from dataclasses import dataclass


@dataclass(frozen=True)
class LongCapacityBudget:
    shared_bytes: int
    slot_bytes: int
    transient_bytes: int
    allocation_limit_bytes: int
    host_reserve_bytes: int

    def __post_init__(self):
        values=vars(self)
        if any(type(v) is not int or v<0 for v in values.values()) or self.slot_bytes==0:
            raise ValueError('nonnegative byte counts and a positive slot size required')

    def evaluate(self,slots,*,allocated_bytes,reserved_bytes,host_available_bytes):
        if type(slots) is not int or not 1<=slots<=8:
            raise ValueError('one to eight request slots required')
        if any(type(v) is not int or v<0 for v in (allocated_bytes,reserved_bytes,host_available_bytes)):
            raise ValueError('nonnegative memory snapshot required')
        if reserved_bytes<allocated_bytes:
            raise ValueError('reserved memory cannot be below allocated memory')
        forecast=self.shared_bytes+slots*self.slot_bytes+self.transient_bytes
        additional=max(0,forecast-allocated_bytes)
        # MemAvailable and allocator slack describe overlapping unified RAM.
        # Do not add them. The admission check conservatively charges all new
        # live allocation to the host; reserved slack may be fragmented/private.
        host_remaining=host_available_bytes-additional
        allocation_ok=forecast<=self.allocation_limit_bytes
        host_ok=host_remaining>=self.host_reserve_bytes
        return dict(slots=slots,forecast_peak_bytes=forecast,additional_live_bytes=additional,
            process_deficit_bytes=max(0,forecast-self.allocation_limit_bytes),
            conservative_host_remaining_bytes=host_remaining,
            process_limit_fits=allocation_ok,host_reserve_fits=host_ok,
            admission_fits=allocation_ok and host_ok,execution_validated=False)

    def require(self,slots,**snapshot):
        result=self.evaluate(slots,**snapshot)
        if not result['admission_fits']:
            raise RuntimeError('long-context memory admission refused: '+str(result))
        return result
