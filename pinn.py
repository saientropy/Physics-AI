import torch
import torch.nn as nn
import torch.autograd as autograd

class PINN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=50, output_dim=1, num_hidden_layers=3):
        super(PINN, self).__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def compute_pde_residual(model, x, alpha=0.01):
    """
    Computes the residual for the 1D heat equation: u_t = alpha * u_xx.
    x is assumed to have two components: time and space [t, x].
    """
    x.requires_grad = True
    u = model(x)
    
    # Compute first derivatives
    grad_u = autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_t = grad_u[:, 0:1]
    u_x = grad_u[:, 1:2]
    
    # Compute second derivative w.r.t x
    grad_u_x = autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_xx = grad_u_x[:, 1:2]
    
    # Residual of the heat equation
    residual = u_t - alpha * u_xx
    return residual

if __name__ == '__main__':
    # Quick test for the PINN module
    model = PINN()
    # Example input: [t, x]
    x = torch.tensor([[0.0, 0.0],
                      [0.1, 0.1],
                      [0.2, 0.2]], requires_grad=True)
    u = model(x)
    residual = compute_pde_residual(model, x)
    print("Output u:", u)
    print("PDE Residual:", residual)
