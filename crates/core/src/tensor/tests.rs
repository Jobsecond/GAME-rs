use super::{CpuDevice, CpuTensor, Tensor};
use crate::Result;

pub(crate) fn tensor<T: Tensor>(shape: &[usize], data: &[f32], device: &T::Device) -> T {
    T::from_data(data, shape, device).expect("tensor should build")
}

pub(crate) fn export<T: Tensor>(tensor: &T) -> Vec<f32> {
    let mut out = vec![0.0; tensor.shape().iter().copied().product()];
    tensor.export(&mut out).expect("export should succeed");
    out
}

pub(crate) fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "length mismatch: actual={actual:?} expected={expected:?}"
    );
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let diff = (actual - expected).abs();
        assert!(
            diff <= 1e-4,
            "value mismatch at index {index}: actual={actual} expected={expected} diff={diff}"
        );
    }
}

pub(crate) fn assert_tensor<T: Tensor>(tensor: &T, shape: &[usize], expected: &[f32]) {
    assert_eq!(tensor.shape(), shape);
    let actual = export(tensor);
    assert_close(&actual, expected);
}

fn cpu_tensor(shape: &[usize], data: &[f32]) -> CpuTensor {
    CpuTensor::from_data(data, shape, &CpuDevice).expect("CPU tensor should build")
}

fn assert_roundtrip<T: Tensor>(shape: &[usize], data: &[f32], device: &T::Device) {
    let tensor = tensor::<T>(shape, data, device);
    assert_tensor(&tensor, shape, data);
}

pub(crate) fn run_roundtrip<T: Tensor>(device: &T::Device) {
    assert_roundtrip::<T>(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], device);
}

pub(crate) fn run_layout_ops_preserve_view_semantics<T: Tensor>(device: &T::Device) {
    let base = tensor::<T>(&[3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], device);
    let head = base.clone().slice(0, 0, 1).unwrap();
    let tail = base.clone().slice(0, 1, 3).unwrap();
    let transposed = tail.clone().transpose(0, 1).unwrap();
    let compact = transposed.clone().contiguous().unwrap();
    let reshaped = compact.reshape(&[4]).unwrap();
    let joined = T::concat(&[&head, &tail], 0).unwrap();

    assert_tensor(&tail, &[2, 2], &[3.0, 4.0, 5.0, 6.0]);
    assert_tensor(&transposed, &[2, 2], &[3.0, 5.0, 4.0, 6.0]);
    assert_tensor(&reshaped, &[4], &[3.0, 5.0, 4.0, 6.0]);
    assert_tensor(&joined, &[3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let left = tensor::<T>(&[2, 1], &[10.0, 20.0], device);
    let mid = tensor::<T>(&[2, 2], &[30.0, 40.0, 50.0, 60.0], device);
    let right = tensor::<T>(&[2, 1], &[70.0, 80.0], device);
    let axis1 = T::concat(&[&left, &mid, &right], 1).unwrap();
    assert_tensor(
        &axis1,
        &[2, 4],
        &[10.0, 30.0, 40.0, 70.0, 20.0, 50.0, 60.0, 80.0],
    );

    let middle_cols = axis1.clone().slice(1, 1, 3).unwrap();
    assert_tensor(&middle_cols, &[2, 2], &[30.0, 40.0, 50.0, 60.0]);

    let transposed_col = base
        .clone()
        .transpose(0, 1)
        .unwrap()
        .slice(1, 1, 2)
        .unwrap();
    assert_tensor(&transposed_col, &[2, 1], &[3.0, 4.0]);

    let kv = tensor::<T>(
        &[2, 4],
        &[1.0, 2.0, 10.0, 20.0, 3.0, 4.0, 30.0, 40.0],
        device,
    );
    let (k, v) = kv.split_last_dim_two_for_attention_heads(1, 2).unwrap();
    assert_tensor(&k, &[1, 2, 2], &[1.0, 2.0, 3.0, 4.0]);
    assert_tensor(&v, &[1, 2, 2], &[10.0, 20.0, 30.0, 40.0]);

    let qkv = tensor::<T>(
        &[2, 6],
        &[
            1.0, 2.0, 10.0, 20.0, 100.0, 200.0, 3.0, 4.0, 30.0, 40.0, 300.0, 400.0,
        ],
        device,
    );
    let (q, k, v) = qkv.split_last_dim_three_for_attention_heads(1, 2).unwrap();
    assert_tensor(&q, &[1, 2, 2], &[1.0, 2.0, 3.0, 4.0]);
    assert_tensor(&k, &[1, 2, 2], &[10.0, 20.0, 30.0, 40.0]);
    assert_tensor(&v, &[1, 2, 2], &[100.0, 200.0, 300.0, 400.0]);
}

pub(crate) fn run_broadcast_add_and_mul_match_expected_values<T: Tensor>(device: &T::Device) {
    let lhs = tensor::<T>(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], device);
    let rhs = tensor::<T>(&[3], &[10.0, 20.0, 30.0], device);
    let mul = tensor::<T>(&[3], &[1.0, 2.0, 3.0], device);

    let added = lhs.clone().add(&rhs).unwrap();
    let multiplied = lhs.mul(&mul).unwrap();

    assert_tensor(&added, &[2, 3], &[11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    assert_tensor(&multiplied, &[2, 3], &[1.0, 4.0, 9.0, 4.0, 10.0, 18.0]);
}

pub(crate) fn run_matmul_supports_2d_and_batched_3d_inputs<T: Tensor>(device: &T::Device) {
    let lhs = tensor::<T>(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], device);
    let rhs = tensor::<T>(&[3, 2], &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], device);
    let product = lhs.matmul(&rhs).unwrap();
    assert_tensor(&product, &[2, 2], &[58.0, 64.0, 139.0, 154.0]);

    let batched_lhs = tensor::<T>(
        &[2, 2, 2],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        device,
    );
    let batched_rhs = tensor::<T>(&[2, 2, 1], &[10.0, 20.0, 30.0, 40.0], device);
    let batched = batched_lhs.matmul(&batched_rhs).unwrap();
    assert_tensor(&batched, &[2, 2, 1], &[50.0, 110.0, 390.0, 530.0]);
}

pub(crate) fn run_matmul_handles_views_and_rejects_unsupported_batch_shapes<T: Tensor>(
    device: &T::Device,
) {
    let lhs = tensor::<T>(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], device);
    let rhs_source = tensor::<T>(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], device);
    let rhs_view = rhs_source.transpose(0, 1).unwrap();
    let product = lhs.matmul(&rhs_view).unwrap();
    assert_tensor(&product, &[2, 2], &[14.0, 32.0, 32.0, 77.0]);

    let empty_lhs = tensor::<T>(&[2, 0], &[], device);
    let empty_rhs = tensor::<T>(&[0, 3], &[], device);
    let empty_product = empty_lhs.matmul(&empty_rhs).unwrap();
    assert_tensor(&empty_product, &[2, 3], &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

    let higher_rank_lhs = tensor::<T>(
        &[2, 3, 1, 2],
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        device,
    );
    let flattened_batch_rhs = tensor::<T>(
        &[6, 2, 1],
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        device,
    );
    assert!(higher_rank_lhs.matmul(&flattened_batch_rhs).is_err());
}

pub(crate) fn run_linear_applies_weight_rows_and_optional_bias<T: Tensor>(device: &T::Device) {
    let x = tensor::<T>(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], device);
    let weight = tensor::<T>(
        &[4, 3],
        &[
            1.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, //
            0.0, 0.0, 1.0, //
            1.0, 1.0, 1.0,
        ],
        device,
    );
    let bias = tensor::<T>(&[4], &[0.5, -0.5, 1.0, 2.0], device);

    let out = x.linear(&weight, Some(&bias)).unwrap();
    assert_tensor(&out, &[2, 4], &[1.5, 1.5, 4.0, 8.0, 4.5, 4.5, 7.0, 17.0]);
}

pub(crate) fn run_normalization_and_activation_ops_match_reference_values<T: Tensor>(
    device: &T::Device,
) {
    let norm_x = tensor::<T>(&[2, 2], &[1.0, 2.0, 3.0, 4.0], device);
    let norm_weight = tensor::<T>(&[2], &[1.0, 2.0], device);
    let normed = norm_x.rms_norm(&norm_weight, 0.0).unwrap();
    assert_tensor(
        &normed,
        &[2, 2],
        &[0.6324555, 2.529822, 0.84852815, 2.2627418],
    );

    let sigmoid = tensor::<T>(&[3], &[-1.0, 0.0, 1.0], device)
        .sigmoid()
        .unwrap();
    assert_tensor(&sigmoid, &[3], &[0.26894143, 0.5, 0.7310586]);

    let gelu = tensor::<T>(&[3], &[-1.0, 0.0, 1.0], device).gelu().unwrap();
    assert_tensor(&gelu, &[3], &[-0.15865529, 0.0, 0.8413447]);

    let softmax = tensor::<T>(&[2, 2], &[1.0, 2.0, 3.0, 4.0], device)
        .softmax(-1)
        .unwrap();
    assert_tensor(
        &softmax,
        &[2, 2],
        &[0.26894143, 0.7310586, 0.26894143, 0.7310586],
    );
}

pub(crate) fn run_rope_rotates_each_head_using_global_positions<T: Tensor>(device: &T::Device) {
    let x = tensor::<T>(
        &[1, 2, 4],
        &[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
        device,
    );
    let y = x.rope(&[0, 1], 4, 1, 4, 10_000.0).unwrap();

    let expected = vec![
        1.0,
        0.0,
        0.0,
        1.0,
        1.0f32.cos(),
        1.0f32.sin(),
        -0.01f32.sin(),
        0.01f32.cos(),
    ];
    assert_tensor(&y, &[1, 2, 4], &expected);
}

pub(crate) fn run_region_rope_splits_global_and_region_rotation_halves<T: Tensor>(
    device: &T::Device,
) {
    let x = tensor::<T>(
        &[1, 2, 4],
        &[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        device,
    );
    let y = x.region_rope(&[0, 1], &[2, 3], 4, 1, 4, 10_000.0).unwrap();

    let expected = vec![
        1.0,
        0.0,
        2.0f32.cos(),
        2.0f32.sin(),
        1.0f32.cos(),
        1.0f32.sin(),
        3.0f32.cos(),
        3.0f32.sin(),
    ];
    assert_tensor(&y, &[1, 2, 4], &expected);
}

pub(crate) fn run_depthwise_conv_applies_per_channel_kernels<T: Tensor>(device: &T::Device) {
    let input = tensor::<T>(
        &[4, 2],
        &[1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0],
        device,
    );
    let kernel = tensor::<T>(&[2, 3], &[1.0, 0.0, -1.0, 1.0, 1.0, 1.0], device);
    let bias = tensor::<T>(&[2], &[1.0, -1.0], device);

    let out = input.conv1d_dw(&kernel, Some(&bias), 1, 1).unwrap();
    assert_tensor(
        &out,
        &[4, 2],
        &[-1.0, 29.0, -1.0, 59.0, -1.0, 89.0, 4.0, 69.0],
    );
}

pub(crate) fn run_embedding_and_repeat_return_expected_rows<T: Tensor>(device: &T::Device) {
    let table = tensor::<T>(&[3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], device);
    let embedded = T::embedding(&table, &[2, 0]).unwrap();
    assert_tensor(&embedded, &[2, 2], &[5.0, 6.0, 1.0, 2.0]);

    let repeated = embedded.clone().repeat(0, 3).unwrap();
    assert_tensor(
        &repeated,
        &[6, 2],
        &[5.0, 6.0, 1.0, 2.0, 5.0, 6.0, 1.0, 2.0, 5.0, 6.0, 1.0, 2.0],
    );

    let axis1 = embedded.repeat(1, 2).unwrap();
    assert_tensor(&axis1, &[2, 4], &[5.0, 6.0, 5.0, 6.0, 1.0, 2.0, 1.0, 2.0]);
}

pub(crate) fn run_gpu_against_cpu_reference<T: Tensor>(device: &T::Device) -> Result<()> {
    let shape = [2, 3];
    let data = [0.5, -1.0, 2.5, 3.0, -4.0, 1.5];
    let rhs = [1.0, 0.5, -2.0];

    let cpu = cpu_tensor(&shape, &data);
    let cpu_rhs = cpu_tensor(&[3], &rhs);
    let cpu_out = cpu.clone().add(&cpu_rhs)?.softmax(-1)?;

    let gpu = tensor::<T>(&shape, &data, device);
    let gpu_rhs = tensor::<T>(&[3], &rhs, device);
    let gpu_out = gpu.add(&gpu_rhs)?.softmax(-1)?;

    assert_tensor(&gpu_out, cpu_out.shape(), &cpu_out.to_vec()?);
    Ok(())
}

pub(crate) fn run_fused_attention_matches_reference<T: Tensor>(device: &T::Device) {
    let q = tensor::<T>(
        &[2, 3, 2],
        &[
            1.0, 0.0, 0.5, 0.5, -0.2, 0.8, //
            0.3, 0.7, 0.6, 0.4, 0.9, -0.1,
        ],
        device,
    );
    let k = tensor::<T>(
        &[2, 4, 2],
        &[
            0.9, 0.1, 0.1, 0.9, 0.8, 0.2, 0.3, 0.7, //
            0.2, 0.8, 0.7, 0.3, 0.4, 0.6, 0.6, 0.4,
        ],
        device,
    );
    let v = tensor::<T>(
        &[2, 4, 2],
        &[
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, //
            0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1,
        ],
        device,
    );
    let mask = tensor::<T>(
        &[2, 3, 4],
        &[
            0.0, 0.0, -10_000.0, -10_000.0, //
            0.0, 0.0, 0.0, -10_000.0, //
            0.0, -10_000.0, 0.0, 0.0, //
            0.0, -10_000.0, 0.0, 0.0, //
            -10_000.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 0.0,
        ],
        device,
    );

    let scale = 1.0 / (q.shape()[2] as f32).sqrt();
    let fused = T::fused_attention(&q, &k, &v, Some(&mask), scale).unwrap();
    let k_t = k.clone().transpose(1, 2).unwrap();
    let scores = T::attention_score_softmax(&q, &k_t, Some(&mask), scale).unwrap();
    let reference = T::attention_value_matmul(&scores, &v).unwrap();

    assert_eq!(fused.shape(), reference.shape());
    let fused_data = export(&fused);
    let reference_data = export(&reference);
    assert_close(&fused_data, &reference_data);
}
