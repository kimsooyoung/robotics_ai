import time
import numpy as np
import jax.numpy as jnp

from jax import random
from jax import device_put
from jax import grad, jit, vmap


key = random.PRNGKey(0)
x = random.normal(key, (10,))
print(x)

a = random.normal(key, (10, 20), dtype=jnp.float32)
b = random.normal(key, (20, 3), dtype=jnp.float32)
y = jnp.dot(a, b).block_until_ready()  # GPU에서 실행
print(a.shape, b.shape, y.shape)

size = 3000
x = random.normal(key, (size, size), dtype=jnp.float32)

start = time.time()
y = jnp.dot(x, x.T).block_until_ready()  # GPU에서 실행
end = time.time()
print(f"[Jax Matmul] Execution time: {end - start:.4f} seconds")

# JAX NumPy 함수는 일반 NumPy 배열에서 작동합니다.
# 매번 데이터를 GPU로 전송해야 하기 때문에 더 느립니다.
x_np = np.random.normal(size=(size, size)).astype(np.float32)
x_np = device_put(x_np)

start = time.time()
y_np = jnp.dot(x_np, x_np.T).block_until_ready()
end = time.time()
print(f"[Numpy Matmul] Execution time: {end - start:.4f} seconds")

# jit()을 사용하여 함수 속도 향상
def selu(x, alpha=1.67, lmbda=1.05):
    return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

x = random.normal(key, (1000000,))

start = time.time()
y = selu(x).block_until_ready()
end = time.time()
print(f"[Normal SeLU] Execution time: {end - start:.4f} seconds")

selu_jit = jit(selu)

start = time.time()
y = selu_jit(x).block_until_ready()
end = time.time()
print(f"[SeLU Jit First Trial] Execution time: {end - start:.4f} seconds")

start = time.time()
y = selu_jit(x).block_until_ready()
end = time.time()
print(f"[SeLU Jit Second Trial] Execution time: {end - start:.4f} seconds")

# grad()를 사용한 미분 계산
def sum_logistic(x):
    return jnp.sum(1.0 / (1.0 + jnp.exp(-x)))

x_small = jnp.arange(3.)
derivative_fn = grad(sum_logistic)
print(derivative_fn(x_small))
print(grad(jit(grad(jit(grad(sum_logistic)))))(1.0))

from jax import jacfwd, jacrev

def hessian(fun):
    return jit(jacfwd(jacrev(fun)))

print(hessian(sum_logistic)(1.0))

# vmap()을 사용한 자동 벡터화
a = jnp.stack([random.normal(key, (3,)), random.normal(key, (3,))])
print(a, a.shape)

mat = random.normal(key, (150, 100))
batched_x = random.normal(key, (10, 100))

def apply_matrix(v):
    return jnp.dot(mat, v) # (150,)

# apply_matrix와 같은 함수가 주어졌을 때, 
# 파이썬에서 배치 차원을 따라 반복할 수 있지만, 
# 보통 그런 작업은 성능이 좋지 않습니다.
def naively_batched_apply_matrix(v_batched):
    return jnp.stack([apply_matrix(v) for v in v_batched])

start = time.time()
y = naively_batched_apply_matrix(batched_x).block_until_ready()
end = time.time()
print(f"[Naively batched] Execution time: {end - start:.4f} seconds")

@jit
def vmap_batched_apply_matrix(v_batched):
  return vmap(apply_matrix)(v_batched)

start = time.time()
y = vmap_batched_apply_matrix(batched_x).block_until_ready()
end = time.time()
print(f"[Auto-vectorized with vmap First trial] Execution time: {end - start:.4f} seconds")

start = time.time()
y = vmap_batched_apply_matrix(batched_x).block_until_ready()
end = time.time()
print(f"[Auto-vectorized with vmap Second trial] Execution time: {end - start:.4f} seconds")

# Check equality
y_test = jnp.dot(batched_x, mat.T).block_until_ready()
print("y.shape:", y.shape, y)
print("y_test.shape:", y_test.shape, y_test)
diff = jnp.abs(y - y_test)
max_diff_idx = jnp.unravel_index(jnp.argmax(diff), diff.shape)
print(f"Index of maximum difference: {max_diff_idx}")
print(f"Value in y at max_diff_idx: {y[max_diff_idx]}")
print(f"Value in y_test at max_diff_idx: {y_test[max_diff_idx]}")