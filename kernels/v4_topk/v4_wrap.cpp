

#include <torch/extension.h>
#include <optional>
void deepseek_v4_topk_transform_512(const at::Tensor&, const at::Tensor&,
    const at::Tensor&, at::Tensor&, int64_t, std::optional<at::Tensor>);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.def("run", &deepseek_v4_topk_transform_512, "v4 topk512",
        py::arg("scores"), py::arg("seq_lens"), py::arg("page_table"),
        py::arg("page_indices"), py::arg("page_size"), py::arg("raw_indices"));
}
