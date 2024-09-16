from tridf.kernel import *


@triton.jit
def count_nonzero_kernel(
    bitmap_ptr,
    scalar_ptr,
    size,
    BLOCK_SIZE: tl.constexpr,
):
    offset = get_offset_kernel(BLOCK_SIZE)
    mask = offset < size
    bitmap = tl.load(bitmap_ptr + offset, mask=mask)
    block_count = tl.sum(bitmap)
    tl.atomic_add(scalar_ptr, block_count)
    

@triton.jit
def block_count_nonzero_kernel(
    bitmap_ptr,
    block_ptr,
    size,
    BLOCK_SIZE: tl.constexpr,
):
    offset = get_offset_kernel(BLOCK_SIZE)
    mask = offset < size
    bitmap = tl.load(bitmap_ptr + offset, mask=mask)
    block_count = tl.sum(bitmap)
    tl.store(block_ptr + tl.program_id(axis=0), block_count)
    