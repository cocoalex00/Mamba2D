#include <torch/extension.h>

void wf_fwd_launcher(torch::Tensor& hs,
                     const torch::Tensor& x,
                     const torch::Tensor& deltaT,
                     const torch::Tensor& deltaL,
                     const torch::Tensor& BT,
                     const torch::Tensor& BL,
                     const torch::Tensor& AT,
                     const torch::Tensor& AL
                    );

void wf_bwd_launcher(const torch::Tensor& hs,
                     const torch::Tensor& x,
                     const torch::Tensor& deltaT,
                     const torch::Tensor& deltaL,
                     const torch::Tensor& AT,
                     const torch::Tensor& AL,
                     torch::Tensor& grad_output,
                     torch::Tensor& dDAT,
                     torch::Tensor& dDAL,
                     torch::Tensor& omega);

void wf_fwd(torch::Tensor hs,
            const torch::Tensor x,
            const torch::Tensor deltaT,
            const torch::Tensor deltaL,
            const torch::Tensor BT,
            const torch::Tensor BL,
            const torch::Tensor AT,
            const torch::Tensor AL
          ) {
    wf_fwd_launcher(hs, x, deltaT, deltaL, BT, BL, AT, AL);
}

void wf_bwd(const torch::Tensor hs,
            const torch::Tensor x,
            const torch::Tensor deltaT,
            const torch::Tensor deltaL,
            const torch::Tensor AT,
            const torch::Tensor AL,
            torch::Tensor grad_output,
            torch::Tensor dDAT,
            torch::Tensor dDAL,
            torch::Tensor omega) {
    wf_bwd_launcher(hs, x, deltaT, deltaL, AT, AL, grad_output, dDAT, dDAL, omega);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("wf_fwd", &wf_fwd, "Wavefront Forward");
  m.def("wf_bwd", &wf_bwd, "Wavefront Backward");
}
