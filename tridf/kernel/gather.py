from tridf.kernel import *


@triton.jit
def materialize_kernel(
    inp_ptr, 
    bitmap_ptr,
    block_count_nonzero_ptr,
    outp_ptr,
    size,
    BLOCK_SIZE: tl.constexpr
):
    offset = get_offset_kernel(BLOCK_SIZE)
    mask = offset < size
    bitmap = tl.load(bitmap_ptr + offset, mask=mask)
    mask = mask & bitmap.cast(tl.int1)
    inp = tl.load(inp_ptr + offset, mask=mask)
    cumsum = tl.cumsum(bitmap, axis=0)
    block_offset = cumsum.cast(tl.int32)
    block_idx = tl.program_id(axis=0)
    if block_idx != 0:
        block_offset += tl.load(block_count_nonzero_ptr + block_idx - 1)
    tl.store(outp_ptr + block_offset - 1, inp, mask=mask)
