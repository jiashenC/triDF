from tridf.kernel import *


@triton.jit
def filter_kernel(
    inp_ptr, 
    bitmap_ptr,
    cond_op: tl.constexpr,
    val,
    size,
    BLOCK_SIZE: tl.constexpr,
):
    offset = get_offset_kernel(BLOCK_SIZE)
    mask = offset < size

    bitmap = tl.load(bitmap_ptr + offset, mask=mask)
    mask = mask and bitmap.cast(tl.int1)

    inp = tl.load(inp_ptr + offset, mask=mask)
    if cond_op == "=":
        res = (inp == val)
    elif cond_op == "!=":
        res = (inp != val)
    elif cond_op == "<=":
        res = (inp <= val)
    elif cond_op == "<":
        res = (inp < val)
    elif cond_op == ">":
        res = (inp > val)
    elif cond_op == ">=":
        res = (inp >= val)
    tl.store(bitmap_ptr + offset, res, mask=mask)
