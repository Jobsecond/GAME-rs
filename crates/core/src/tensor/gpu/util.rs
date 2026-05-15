use crate::{Error, Result};

pub(super) fn invalid_arg(message: impl Into<String>) -> Error {
    Error::message(message.into())
}

pub(super) fn checked_num_elements(shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| invalid_arg(format!("tensor shape {:?} is too large", shape)))
    })
}

pub(super) fn plain_num_elements(shape: &[usize]) -> usize {
    shape.iter().copied().product()
}

pub(super) fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![0; shape.len()];
    let mut stride = 1usize;
    for axis in (0..shape.len()).rev() {
        strides[axis] = stride;
        stride = stride.saturating_mul(shape[axis]);
    }
    strides
}

pub(super) fn validate_axis(axis: usize, rank: usize, op_name: &str) -> Result<()> {
    if axis >= rank {
        return Err(invalid_arg(format!(
            "{op_name} axis {} is out of bounds for rank {}",
            axis, rank
        )));
    }
    Ok(())
}

pub(super) fn normalize_axis(axis: isize, rank: usize, op_name: &str) -> Result<usize> {
    if rank == 0 {
        return Err(invalid_arg(format!(
            "{op_name} requires a tensor with at least one dimension"
        )));
    }

    let rank_isize = isize::try_from(rank).map_err(|_| invalid_arg("rank overflow"))?;
    let normalized = if axis < 0 { rank_isize + axis } else { axis };
    if normalized < 0 || normalized >= rank_isize {
        return Err(invalid_arg(format!(
            "{op_name} axis {} is out of bounds for rank {}",
            axis, rank
        )));
    }

    usize::try_from(normalized).map_err(|_| invalid_arg("axis overflow"))
}

pub(super) fn validate_rope_shape(
    shape: &[usize],
    positions_len: usize,
    head_dim: usize,
    num_heads: usize,
    op_name: &str,
) -> Result<()> {
    if shape.len() != 3 {
        return Err(invalid_arg(format!(
            "{op_name} expects a rank-3 tensor shaped [num_heads, seq_len, head_dim], got {:?}",
            shape
        )));
    }
    if shape[0] != num_heads {
        return Err(invalid_arg(format!(
            "{op_name} expected num_heads={}, got shape {:?}",
            num_heads, shape
        )));
    }
    if shape[1] != positions_len {
        return Err(invalid_arg(format!(
            "{op_name} expected seq_len={}, got shape {:?}",
            positions_len, shape
        )));
    }
    if shape[2] != head_dim {
        return Err(invalid_arg(format!(
            "{op_name} expected head_dim={}, got shape {:?}",
            head_dim, shape
        )));
    }
    Ok(())
}

pub(super) fn normalize_rope_dims(
    head_dim: usize,
    rope_dims: usize,
    op_name: &str,
    mixed: bool,
) -> Result<usize> {
    if head_dim == 0 {
        return Err(invalid_arg(format!("{op_name} requires head_dim > 0")));
    }

    let dims = if rope_dims == 0 { head_dim } else { rope_dims };
    if dims > head_dim {
        return Err(invalid_arg(format!(
            "{op_name} rope_dims {} exceeds head_dim {}",
            dims, head_dim
        )));
    }
    if mixed {
        if dims % 4 != 0 {
            return Err(invalid_arg(format!(
                "{op_name} requires rope_dims divisible by 4 for mixed RoPE, got {}",
                dims
            )));
        }
    } else if dims % 2 != 0 {
        return Err(invalid_arg(format!(
            "{op_name} requires an even rope_dims, got {}",
            dims
        )));
    }

    Ok(dims)
}

pub(super) fn broadcast_shape(lhs: &[usize], rhs: &[usize]) -> Result<Vec<usize>> {
    let rank = lhs.len().max(rhs.len());
    let mut out = vec![1usize; rank];

    for axis in 0..rank {
        let lhs_dim = lhs
            .len()
            .checked_sub(rank - axis)
            .and_then(|index| lhs.get(index))
            .copied()
            .unwrap_or(1);
        let rhs_dim = rhs
            .len()
            .checked_sub(rank - axis)
            .and_then(|index| rhs.get(index))
            .copied()
            .unwrap_or(1);

        if lhs_dim != rhs_dim && lhs_dim != 1 && rhs_dim != 1 {
            return Err(invalid_arg(format!(
                "cannot broadcast shapes {:?} and {:?}",
                lhs, rhs
            )));
        }
        out[axis] = lhs_dim.max(rhs_dim);
    }

    Ok(out)
}

pub(super) fn for_each_index(shape: &[usize], mut f: impl FnMut(&[usize], usize)) {
    let len = plain_num_elements(shape);
    if len == 0 {
        return;
    }
    if shape.is_empty() {
        f(&[], 0);
        return;
    }

    let mut coords = vec![0usize; shape.len()];
    for flat in 0..len {
        f(&coords, flat);
        for axis in (0..coords.len()).rev() {
            coords[axis] += 1;
            if coords[axis] < shape[axis] {
                break;
            }
            coords[axis] = 0;
        }
    }
}

pub(super) fn bytes_for_elements(elements: usize) -> Result<u64> {
    let elements = elements.max(1);
    u64::try_from(
        elements
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| invalid_arg("buffer size overflow"))?,
    )
    .map_err(|_| invalid_arg("buffer size overflow"))
}

pub(super) fn bytes_for_offset(elements: usize) -> Result<u64> {
    u64::try_from(
        elements
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| invalid_arg("buffer offset overflow"))?,
    )
    .map_err(|_| invalid_arg("buffer offset overflow"))
}

pub(super) fn usize_to_u32(value: usize, label: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| {
        invalid_arg(format!(
            "{label} {value} exceeds u32::MAX for the GPU backend"
        ))
    })
}
