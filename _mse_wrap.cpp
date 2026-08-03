
// This is a small wrapper to expose the CUDA function via torch.compile
torch::Tensor mse_loss_cuda(torch::Tensor pred, torch::Tensor target);
