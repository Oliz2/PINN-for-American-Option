#pragma once
#include <torch/torch.h>
#include <vector>

namespace it::univr::utils {

/**
 * Copies a tensor of any shape into a std::vector<double>, row by row.
 * Used by the output classes to print or save tensors.
 *
 * @author Alberto Oliva Medin
 */
inline std::vector<double> toVector(const torch::Tensor& x) {
    auto c = x.detach().to(torch::kDouble).contiguous().view({-1});
    return std::vector<double>(c.data_ptr<double>(), c.data_ptr<double>() + c.numel());
}

} // namespace it::univr::utils
