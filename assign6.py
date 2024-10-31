import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def create_graph(n, graph_type='cycle'):
    """Create communication graph and return its mixing matrix."""
    if graph_type == 'cycle':
        G = nx.cycle_graph(n)
    elif graph_type == 'complete':
        G = nx.complete_graph(n)
    elif graph_type == 'line':
        G = nx.path_graph(n)

    # Get adjacency matrix
    A = nx.adjacency_matrix(G).toarray()

    # Create mixing matrix (Metropolis weights)
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j and A[i, j] == 1:
                W[i, j] = 1 / (max(G.degree(i), G.degree(j)) + 1)

    # Set diagonal to make row-stochastic
    for i in range(n):
        W[i, i] = 1 - np.sum(W[i, :])

    return W, nx.algebraic_connectivity(G)


def generate_local_optima(n, d=2, dissimilarity='low'):
    """Generate local optima with controlled dissimilarity."""
    if dissimilarity == 'low':
        radius = 0.1
    elif dissimilarity == 'medium':
        radius = 1.0
    else:  # high
        radius = 5.0

    # Generate random angles
    angles = np.linspace(0, 2 * np.pi, n)

    # Create points in circular pattern with given radius
    points = np.zeros((n, d))
    points[:, 0] = radius * np.cos(angles)
    points[:, 1] = radius * np.sin(angles)

    return points

def compute_optimal_point(local_optima):
    """Compute the optimal point (average of local optima)."""
    return np.mean(local_optima, axis=0)

def distributed_gradient_descent(W, local_optima, T=100, gamma=0.05, alpha=0.1):
    """Run DGD algorithm with explicit gossip communication."""
    n, d = local_optima.shape
    theta = np.random.randn(n, d)  # Random initialization
    losses = []
    optimal_point = np.mean(local_optima, axis=0)  # Compute optimal consensus point

    # Convert W to an adjacency matrix A if using neighbor-based communication
    A = (W > 0).astype(int)

    for t in range(T):
        # Compute local gradients
        gradients = theta - local_optima
        # Local gradient descent step
        theta_half = theta - gamma * gradients

        # Gossip consensus step (explicitly using alpha)
        theta_next = theta_half.copy()
        for i in range(n):
            neighbor_sum = np.sum([theta[j] - theta[i] for j in range(n) if A[i, j] == 1], axis=0)
            theta_next[i] += alpha * neighbor_sum

        theta = theta_next  # Update theta for the next iteration

        # Compute global loss with respect to optimal point
        global_loss = 0.5 * np.mean(np.sum((theta - optimal_point) ** 2, axis=1))
        losses.append(global_loss)

    return np.array(losses)



# Run experiments
n = 4  # number of clients
T = 100  # number of iterations
gamma = 0.05  # learning rate

# Test different graph connectivities
graph_types = ['line', 'cycle', 'complete']
dissimilarities = ['low', 'medium', 'high']

# Plot results for different graph connectivities
plt.figure(figsize=(12, 5))

# Experiment 1: Impact of algebraic connectivity
plt.subplot(1, 2, 1)
for graph_type in graph_types:
    W, alg_conn = create_graph(n, graph_type)
    local_optima = generate_local_optima(n, dissimilarity='medium')
    losses = distributed_gradient_descent(W, local_optima, T, gamma)
    plt.semilogy(losses, label=f'{graph_type} (μ={alg_conn:.3f})')

plt.xlabel('Iteration')
plt.ylabel('Global Loss')
plt.title('Impact of Algebraic Connectivity')
plt.legend()
plt.grid(True)

# Experiment 2: Impact of gradient dissimilarity
plt.subplot(1, 2, 2)
W, _ = create_graph(n, 'cycle')  # Fix graph type
for dissim in dissimilarities:
    local_optima = generate_local_optima(n, dissimilarity=dissim)
    losses = distributed_gradient_descent(W, local_optima, T, gamma)
    plt.semilogy(losses, label=f'{dissim} dissimilarity')

plt.xlabel('Iteration')
plt.ylabel('Global Loss')
plt.title('Impact of Gradient Dissimilarity')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# Calculate convergence rates
def calculate_convergence_rate(losses, window=100):
    """Calculate empirical convergence rate over last window iterations."""
    final_losses = losses[-window:]
    iterations = np.arange(len(losses))[-window:]
    # Fit log(y) = a*log(x) + b
    a, _ = np.polyfit(np.log(iterations + 1), np.log(final_losses), 1)
    return a


print("\nEmpirical Convergence Rates:")

# For different graph types
print("\nEffect of Algebraic Connectivity:")
for graph_type in graph_types:
    W, alg_conn = create_graph(n, graph_type)
    local_optima = generate_local_optima(n, dissimilarity='medium')
    losses = distributed_gradient_descent(W, local_optima, T, gamma)
    rate = calculate_convergence_rate(losses)
    print(f"{graph_type} graph (μ={alg_conn:.3f}): {rate:.3f}")

# For different dissimilarities
# print("\nEffect of Gradient Dissimilarity:")
# W, _ = create_graph(n, 'cycle')
# for dissim in dissimilarities:
#     local_optima = generate_local_optima(n, dissimilarity=dissim)
#     losses = distributed_gradient_descent(W, local_optima, T, gamma)
#     rate = calculate_convergence_rate(losses)
#     print(f"{dissim} dissimilarity: {rate:.3f}")




print("\nEffect of Gradient Dissimilarity:")
W, _ = create_graph(n, 'cycle')
for dissim in dissimilarities:
    local_optima = generate_local_optima(n, dissimilarity=dissim)
    losses = distributed_gradient_descent(W, local_optima, T, gamma)
    optimal_point = compute_optimal_point(local_optima)
    grad_dissim = np.sqrt(np.mean(np.sum((local_optima - optimal_point) ** 2, axis=1)))
    rate = calculate_convergence_rate(losses)
    print(f"ζ={grad_dissim:.2f}: {rate:.3f}")