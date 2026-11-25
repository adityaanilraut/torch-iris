#include <ATen/autocast_mode.h>

TORCH_LIBRARY_IMPL(_, AutocastPrivateUse1, m) {
    m.fallback(torch::CppFunction::makeFallthrough());
}

TORCH_LIBRARY_IMPL(aten, AutocastPrivateUse1, m) {
    KERNEL_PRIVATEUSEONE(mm, lower_precision_fp)
    KERNEL_PRIVATEUSEONE(addmm, lower_precision_fp)
    KERNEL_PRIVATEUSEONE(bmm, lower_precision_fp)
    KERNEL_PRIVATEUSEONE(matmul, lower_precision_fp)
    KERNEL_PRIVATEUSEONE(linear, lower_precision_fp)
    KERNEL_PRIVATEUSEONE(convolution, lower_precision_fp)
    KERNEL_PRIVATEUSEONE(_convolution, lower_precision_fp)
    KERNEL_PRIVATEUSEONE(conv1d, lower_precision_fp)
    KERNEL_PRIVATEUSEONE(conv2d, lower_precision_fp)
    KERNEL_PRIVATEUSEONE(conv3d, lower_precision_fp)
    KERNEL_PRIVATEUSEONE(conv_transpose1d, lower_precision_fp)
    KERNEL_PRIVATEUSEONE(conv_transpose2d, input, lower_precision_fp)
    KERNEL_PRIVATEUSEONE(conv_transpose3d, input, lower_precision_fp)
    KERNEL_PRIVATEUSEONE(scaled_dot_product_attention, lower_precision_fp)
    KERNEL_PRIVATEUSEONE(layer_norm, lower_precision_fp)
    KERNEL_PRIVATEUSEONE(group_norm, lower_precision_fp)
    KERNEL_PRIVATEUSEONE(rms_norm, lower_precision_fp)
    KERNEL_PRIVATEUSEONE(mse_loss, fp32)
    KERNEL_PRIVATEUSEONE(cross_entropy_loss, fp32)
    KERNEL_PRIVATEUSEONE(binary_cross_entropy_with_logits, fp32)
    KERNEL_PRIVATEUSEONE(nll_loss_forward, fp32)
    KERNEL_PRIVATEUSEONE(nll_loss_nd, fp32)
    KERNEL_PRIVATEUSEONE(softmax, int, fp32_set_opt_dtype)
    KERNEL_PRIVATEUSEONE(log_softmax, int, fp32_set_opt_dtype)
}
