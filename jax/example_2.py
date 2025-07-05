import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

# numpy로 할 수 있는 거의 모든 것이 jax.numpy로도 할 수 있습니다:
x_np = np.linspace(0, 10, 1000)
y_np = 2 * np.sin(x_np) * np.cos(x_np)

x_jnp = jnp.linspace(0, 10, 1000)
y_jnp = 2 * jnp.sin(x_jnp) * jnp.cos(x_jnp)
plt.plot(x_jnp, y_jnp)
plt.show()

print(f"{type(x_np)=}, {type(x_jnp)=}")

# JAX 배열과 NumPy 배열 사이에는 하나의 중요한 차이가 있습니다: 
# JAX 배열은 불변입니다, 
# 즉 생성된 후 그 내용을 변경할 수 없다는 의미입니다.

# NumPy: 가변 배열
x = np.arange(10)
x[0] = 10
print(x)

# JAX: 불변 배열 => 오류가 발생합니다:
x = jnp.arange(10)
# x[0] = 10

# 개별 요소를 업데이트하기 위해, 
# JAX는 업데이트된 복사본을 반환하는 인덱스 업데이트 문법을 제공합니다.
y = x.at[0].set(10)
print(x)
print(y)

# NumPy, lax & XLA: JAX API 계층화
