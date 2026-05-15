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

pub(super) fn broadcast_offset(
    coords: &[usize],
    shape: &[usize],
    strides: &[usize],
    out_rank: usize,
) -> usize {
    if shape.is_empty() {
        return 0;
    }

    let rank_diff = out_rank - shape.len();
    let mut offset = 0usize;
    for (out_axis, &coord) in coords.iter().enumerate() {
        if out_axis < rank_diff {
            continue;
        }
        let axis = out_axis - rank_diff;
        if shape[axis] != 1 {
            offset += coord * strides[axis];
        }
    }
    offset
}

pub(super) fn trailing_feature_broadcast_dim(
    lhs: &[usize],
    rhs: &[usize],
    out: &[usize],
) -> Option<usize> {
    if out.is_empty() {
        return None;
    }

    let feature_dim = *out.last()?;
    if feature_dim == 0 {
        return Some(0);
    }

    let lhs_feature = *lhs.last().unwrap_or(&1);
    let rhs_feature = *rhs.last().unwrap_or(&1);
    if lhs_feature != feature_dim || rhs_feature != feature_dim {
        return None;
    }

    let lhs_matches = lhs.len() == 1 || lhs == out;
    let rhs_matches = rhs.len() == 1 || rhs == out;
    if lhs_matches && rhs_matches && (lhs.len() == 1 || rhs.len() == 1) {
        Some(feature_dim)
    } else {
        None
    }
}

pub(super) fn suffix_broadcast_block_len(
    lhs: &[usize],
    rhs: &[usize],
    out: &[usize],
) -> Option<usize> {
    if lhs != out || rhs.len() >= out.len() || rhs.is_empty() {
        return None;
    }

    let rank_diff = out.len() - rhs.len();
    if out[..rank_diff].iter().any(|&dim| dim == 0) {
        return None;
    }
    if out[rank_diff..] != rhs[..] {
        return None;
    }

    Some(rhs.iter().copied().product())
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

pub(super) fn should_parallelize(len: usize) -> bool {
    len >= 16_384 && rayon::current_num_threads() > 1
}
