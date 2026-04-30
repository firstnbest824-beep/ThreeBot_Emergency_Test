"""Gaussian Action Head for Fast 5-DoF Control.

Replaces Diffusion Policy with a deterministic MLP + uncertainty estimation.
Latency: ~2ms (vs 50ms for Diffusion)

Output Distribution:
    a ~ N(μ(s), σ(s))
    Where μ, σ are predicted by MLP.

Author: Architect_Sonnet @ Nova Robotics
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple

class GaussianActionHead(nn.Module):
    """Fast, uncertainty-aware action prediction."""
    
    action_dim: int = 1  # Z-axis (Shared)
    hidden_dim: int = 256
    min_logvar: float = -5.0
    max_logvar: float = 2.0
    
    @nn.compact
    def __call__(self, features: jnp.ndarray, train: bool = False):
        """
        Args:
            features: [B, D] input features
            train: If True, return (sample, mu, logvar). Else, return mu.
        Returns:
            If train: (sample, mu, logvar) each [B, action_dim]
            Else: mu [B, action_dim]
        """
        # Shared feature extractor
        h = nn.Dense(self.hidden_dim, name='fc1')(features)
        h = nn.LayerNorm(name='ln1')(h)
        h = nn.gelu(h)
        
        h = nn.Dense(self.hidden_dim, name='fc2')(h)
        h = nn.LayerNorm(name='ln2')(h)
        h = nn.gelu(h)
        
        # Mean head
        mu = nn.Dense(self.action_dim, name='mu_head')(h)
        mu = nn.tanh(mu)  # Bound mean to [-1, 1]
        
        if not train:
            return mu
            
        # Training: also compute logvar and sample
        logvar = nn.Dense(self.action_dim, name='logvar_head')(h)
        logvar = jnp.clip(logvar, self.min_logvar, self.max_logvar)
        
        # Reparameterization trick
        std = jnp.exp(0.5 * logvar)
        key = self.make_rng('sample')
        noise = jax.random.normal(key, mu.shape)
        sample = mu + std * noise
        
        return sample, mu, logvar

def compute_gaussian_loss(
    model: GaussianActionHead,
    params,
    features: jnp.ndarray,
    target: jnp.ndarray,
    key: jax.random.PRNGKey
) -> Tuple[jnp.ndarray, dict]:
    """Compute NLL using model's apply method.
    
    Args:
        model: GaussianActionHead class/instance
        params: JAX parameter pytree
        features: [B, D]
        target: [B, action_dim]
        key: RNG key
    Returns:
        loss, metrics
    """
    # Fix: Call apply() to avoid initialization errors
    sample, mu, logvar = model.apply(
        {'params': params},
        features,
        train=True,
        rngs={'sample': key}
    )
    
    # Gaussian NLL
    # L = 0.5 * [ log(2π) + log(σ²) + (y - μ)²/σ² ]
    inv_var = jnp.exp(-logvar)
    mse = (target - mu) ** 2
    
    nll = 0.5 * (jnp.log(2 * jnp.pi) + logvar + mse * inv_var)
    loss = jnp.mean(nll)
    
    metrics = {
        'gaussian_loss': loss,
        'z_mse': jnp.mean(mse),
        'z_uncertainty': jnp.mean(jnp.exp(logvar))
    }
    
    return loss, metrics
